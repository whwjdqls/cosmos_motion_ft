"""Batched native UniPC inference, with independent reproducible case seeds."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from motion_expert_phase3_edge_joint.model import EdgePhase3
from motion_expert_phase3_edge_joint.settings import RECIPE, add_paths, atomic_json, require_compute_node
from motion_expert_phase3_edge_joint.adapters import configure_checkpoint, load_trainable_state
from motion_expert_phase3_edge_joint.runtime import sampling_execution


def case_generators(seed, batch, device):
    seeds = [int(seed) + i for i in range(batch)] if isinstance(seed, int) else list(seed)
    if len(seeds) != batch:
        raise ValueError("One sampling seed is required per case")
    return [torch.Generator(device=device).manual_seed(int(value)) for value in seeds]


def case_noise(template, generators):
    # Drawing [1,...] per generator preserves the historical batch-one stream,
    # independently of sharding, batch size, padding and OOM retries.
    return torch.cat([torch.randn((1, *template.shape[1:]), device=template.device,
                      dtype=torch.float32, generator=g) for g in generators])


def case_captions(caption, batch):
    captions = [caption] * batch if isinstance(caption, str) else list(caption)
    if len(captions) != batch or not all(isinstance(c, str) for c in captions):
        raise ValueError("One caption is required per case")
    return captions


@torch.no_grad()
@sampling_execution
def sample(model, *, mode, caption, motion, neutral, video, seed=0, steps=35, guidance=1.0):
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler

    if mode == "joint":
        return sample_joint(model, caption=caption, motion=motion, neutral=neutral,
                            video=video, seed=seed, steps=steps, guidance=guidance)
    if mode not in RECIPE["tasks"]:
        raise ValueError(mode)
    device = motion.device
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
    scheduler.set_timesteps(steps, device=device, shift=3.0 if mode == "video2motion" else 10.0)
    batch = len(motion)
    generators = case_generators(seed, batch, device)
    captions = case_captions(caption, batch)
    if steps < 1 or len(neutral) != batch or len(video) != batch:
        raise ValueError("Invalid sampling steps or batch geometry")
    state = case_noise(motion if mode == "video2motion" else video, generators)
    if mode == "motimg2video":
        state[:, :, 0] = video[:, :, 0]
    model.eval()
    for index, timestep in enumerate(scheduler.timesteps):
        sigma = timestep.float().reshape(1).expand(batch) / 1000.0
        inputs = {
            "motion": state if mode == "video2motion" else motion,
            "video": video if mode == "video2motion" else state,
            "neutral": neutral, "motion_sigma": sigma if mode == "video2motion" else sigma * 0,
            "video_sigma": sigma if mode == "motimg2video" else sigma * 0,
            "captions": [""] * batch if mode == "video2motion" else captions, "modes": [mode] * batch,
        }
        key = "motion" if mode == "video2motion" else "video"
        prediction = model(**inputs)[key].float()
        if guidance != 1.0 and mode == "motimg2video":
            null = model(**{**inputs, "captions": [""] * batch})[key].float()
            prediction = null + guidance * (prediction - null)
        if mode == "video2motion":
            actual_sigma = scheduler.sigmas[index].to(device=device, dtype=torch.float32).clamp_min(1e-6)
            prediction = (state - prediction) / actual_sigma
        state = scheduler.step(prediction, timestep, state, return_dict=False)[0].float()
        if mode == "motimg2video":
            state[:, :, 0] = video[:, :, 0]
    if not torch.isfinite(state).all():
        raise FloatingPointError("Non-finite sampled output")
    return state



@torch.no_grad()
@sampling_execution
def sample_joint(model, *, caption, motion, neutral, video, seed=0, steps=35, guidance=1.0):
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    device = motion.device
    schedulers = {}
    for key, shift in (("motion", 3.0), ("video", 10.0)):
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        scheduler.set_timesteps(steps, device=device, shift=shift)
        schedulers[key] = scheduler
    batch = len(motion)
    generators = case_generators(seed, batch, device)
    captions = case_captions(caption, batch)
    states = {"motion": case_noise(motion, generators), "video": case_noise(video, generators)}
    states["video"][:, :, 0] = video[:, :, 0]
    model.eval()
    for index in range(steps):
        ms = schedulers["motion"].timesteps[index].float().reshape(1).expand(batch) / 1000.0
        vs = schedulers["video"].timesteps[index].float().reshape(1).expand(batch) / 1000.0
        inputs = dict(motion=states["motion"], video=states["video"], neutral=neutral,
                      motion_sigma=ms, video_sigma=vs, captions=captions, modes=["joint"] * batch)
        predictions = model(**inputs)
        if guidance != 1.0:
            null = model(**{**inputs, "captions": [""] * batch})
            predictions = {key: null[key].float() + guidance * (predictions[key].float() - null[key].float())
                           for key in ("motion", "video")}
        for key, scheduler in schedulers.items():
            pred = predictions[key].float()
            if key == "motion":
                sigma = scheduler.sigmas[index].to(device=device, dtype=torch.float32).clamp_min(1e-6)
                pred = (states[key] - pred) / sigma
            states[key] = scheduler.step(pred, scheduler.timesteps[index], states[key],
                return_dict=False)[0].float()
        states["video"][:, :, 0] = video[:, :, 0]
    if not all(torch.isfinite(value).all() for value in states.values()):
        raise FloatingPointError("Non-finite joint sample")
    return states


def batched_samples(model, records, *, mode, contract, batch_size=32, device="cuda", steps=35, guidance=1.):
    """Yield CPU results in input order, preserving case seeds across retries."""
    from motion_expert_phase3_edge.quantitative import check_bundle
    if batch_size < 1:
        raise ValueError("Sampling batch size must be positive")
    offset, current = 0, batch_size
    while offset < len(records):
        selected = records[offset:offset + current]
        arrays = [check_bundle(row, contract) for row in selected]
        try:
            inputs = {name: torch.from_numpy(np.stack([a[i] for a in arrays])).to(device).float()
                      for i, name in enumerate(("motion", "neutral", "video"))}
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                output = sample(model, mode=mode, caption=[r["caption"] for r in selected], **inputs,
                    seed=[int(r.get("seed", 0)) for r in selected], steps=steps, guidance=guidance)
            predictions = ({k: v.float().cpu().numpy() for k, v in output.items()} if isinstance(output, dict)
                           else output.float().cpu().numpy())
            del inputs, output
        except torch.cuda.OutOfMemoryError:
            inputs, output = None, None
            gc.collect()
            torch.cuda.empty_cache()
            if len(selected) == 1:
                raise
            current = max(1, len(selected) // 2)
            print(f"[phase3/sample] CUDA OOM: retrying batch={current}", flush=True)
            continue
        for i, (row, values) in enumerate(zip(selected, arrays, strict=True)):
            predicted = {k: v[i] for k, v in predictions.items()} if isinstance(predictions, dict) else predictions[i]
            yield row, values, predicted
        offset += len(selected)


def main():
    parser = add_paths(argparse.ArgumentParser())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True,
                        help="JSON list of fixed held-out {id, caption, bundle, seed} records")
    parser.add_argument("--mode", choices=list(RECIPE["tasks"]), required=True)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Cases per GPU, halved on CUDA OOM with unchanged case seeds")
    parser.add_argument("--weights", choices=["raw", "ema"], default="raw",
                        help="EMA selects the matched bridge and additional-LoRA weights")
    args = parser.parse_args()
    require_compute_node()
    torch.cuda.set_device(0)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("schema") not in (1, 2):
        raise RuntimeError("Not a phase-3 bridge checkpoint")
    from motion_expert_phase3_edge_joint.settings import configure_checkpoint_conditioning
    import sys
    configure_checkpoint_conditioning(payload,
        attention=args.conditioning_attention if "--conditioning-attention" in sys.argv else None,
        clean_motion_timestep=args.clean_motion_timestep if "--clean-motion-timestep" in sys.argv else None)
    configure_checkpoint(payload, bridge_parameterization=args.bridge_parameterization,
                         bridge_attention_backend=args.bridge_attention_backend, bridge_precision=args.bridge_precision)
    model = EdgePhase3(args.phase1, args.phase2, device="cuda", activation_checkpointing=False)
    saved = payload["sources"]
    if (saved["phase1_metadata_sha256"] != model.provenance["phase1_metadata_sha256"]
            or saved["phase2_contract"] != model.provenance["phase2_contract"]):
        raise RuntimeError("Bridge checkpoint references different frozen specialists")
    load_trainable_state(model, payload, weights=args.weights)
    records = json.loads(args.cases.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    completed = []
    for record, arrays, output in batched_samples(model, records, mode=args.mode,
            contract=model.provenance["phase2_contract"], batch_size=args.batch_size,
            steps=args.steps, guidance=args.guidance):
        # Bundle: already normalized with this motion specialist's exact stats,
        # centered neutral joints, and cached Wan latents. No online recanonicalization.
        motion, neutral, video = arrays
        name = Path(str(record["id"])).name
        if name != record["id"] or name in {"", ".", ".."}:
            raise ValueError("Case IDs must be safe basenames")
        path = args.out / (name + ".npz")
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing inference: {path}")
        predicted = ({"generated_motion": output["motion"], "generated_video": output["video"]}
                     if args.mode == "joint" else {"generated": output})
        np.savez(path, **predicted, gt_motion=motion, gt_video_latents=video, neutral=neutral)
        completed.append({**record, "output": str(path)})
        atomic_json(args.out / "manifest.json", {"cases": completed, "mode": args.mode,
                    "checkpoint": str(args.checkpoint), "step": payload["step"],
                    "weights": args.weights,
                    "requested_batch_size": args.batch_size,
                    "bridge_execution": RECIPE.get("bridge_execution", {"backend": "dense", "precision": "fp32"}),
                    "steps": args.steps, "guidance": args.guidance,
                    "shift": {"motion": 3.0, "video": 10.0} if args.mode == "joint" else (3.0 if args.mode == "video2motion" else 10.0),
                    "sources": model.provenance,
                    "metric_policy": "Report generated and matching GT; exclude conditioning frame 0 in video metrics"})
        print(f"[phase3/sample] {len(completed)}/{len(records)}: {path}", flush=True)


if __name__ == "__main__":
    main()
