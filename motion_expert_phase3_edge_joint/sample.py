"""Offline UniPC inference. No sampling or rendering inside training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_expert_phase3_edge_joint.model import EdgePhase3
from motion_expert_phase3_edge_joint.settings import RECIPE, add_paths, atomic_json, require_compute_node


@torch.no_grad()
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
    generator = torch.Generator(device=device).manual_seed(seed)
    target_shape = motion.shape if mode == "video2motion" else video.shape
    state = torch.randn(target_shape, device=device, dtype=torch.float32, generator=generator)
    if mode == "motimg2video":
        state[:, :, 0] = video[:, :, 0]
    model.eval()
    for index, timestep in enumerate(scheduler.timesteps):
        sigma = timestep.float().reshape(1) / 1000.0
        inputs = {
            "motion": state if mode == "video2motion" else motion,
            "video": video if mode == "video2motion" else state,
            "neutral": neutral, "motion_sigma": sigma if mode == "video2motion" else sigma * 0,
            "video_sigma": sigma if mode == "motimg2video" else sigma * 0,
            "captions": ["" if mode == "video2motion" else caption], "modes": [mode],
        }
        key = "motion" if mode == "video2motion" else "video"
        prediction = model(**inputs)[key].float()
        if guidance != 1.0 and mode == "motimg2video":
            null = model(**{**inputs, "captions": [""]})[key].float()
            prediction = null + guidance * (prediction - null)
        if mode == "video2motion":
            actual_sigma = scheduler.sigmas[index].to(device=device, dtype=torch.float32).clamp_min(1e-6)
            prediction = (state - prediction) / actual_sigma
        state = scheduler.step(prediction, timestep, state, return_dict=False, generator=generator)[0].float()
        if mode == "motimg2video":
            state[:, :, 0] = video[:, :, 0]
    if not torch.isfinite(state).all():
        raise FloatingPointError("Non-finite sampled output")
    return state



@torch.no_grad()
def sample_joint(model, *, caption, motion, neutral, video, seed=0, steps=35, guidance=1.0):
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    device = motion.device
    schedulers = {}
    for key, shift in (("motion", 3.0), ("video", 10.0)):
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        scheduler.set_timesteps(steps, device=device, shift=shift)
        schedulers[key] = scheduler
    generator = torch.Generator(device=device).manual_seed(seed)
    states = {"motion": torch.randn(motion.shape, device=device, dtype=torch.float32, generator=generator),
              "video": torch.randn(video.shape, device=device, dtype=torch.float32, generator=generator)}
    states["video"][:, :, 0] = video[:, :, 0]
    model.eval()
    for index in range(steps):
        ms = schedulers["motion"].timesteps[index].float().reshape(1) / 1000.0
        vs = schedulers["video"].timesteps[index].float().reshape(1) / 1000.0
        inputs = dict(motion=states["motion"], video=states["video"], neutral=neutral,
                      motion_sigma=ms, video_sigma=vs, captions=[caption], modes=["joint"])
        predictions = model(**inputs)
        if guidance != 1.0:
            null = model(**{**inputs, "captions": [""]})
            predictions = {key: null[key].float() + guidance * (predictions[key].float() - null[key].float())
                           for key in ("motion", "video")}
        for key, scheduler in schedulers.items():
            pred = predictions[key].float()
            if key == "motion":
                sigma = scheduler.sigmas[index].to(device=device, dtype=torch.float32).clamp_min(1e-6)
                pred = (states[key] - pred) / sigma
            states[key] = scheduler.step(pred, scheduler.timesteps[index], states[key],
                return_dict=False, generator=generator)[0].float()
        states["video"][:, :, 0] = video[:, :, 0]
    if not all(torch.isfinite(value).all() for value in states.values()):
        raise FloatingPointError("Non-finite joint sample")
    return states


def main():
    parser = add_paths(argparse.ArgumentParser())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True,
                        help="JSON list of fixed held-out {id, caption, bundle, seed} records")
    parser.add_argument("--mode", choices=list(RECIPE["tasks"]), required=True)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=1.0)
    args = parser.parse_args()
    require_compute_node()
    torch.cuda.set_device(0)
    model = EdgePhase3(args.phase1, args.phase2, device="cuda", activation_checkpointing=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("schema") != 1:
        raise RuntimeError("Not a phase-3 bridge checkpoint")
    saved = payload["sources"]
    if (saved["phase1_metadata_sha256"] != model.provenance["phase1_metadata_sha256"]
            or saved["phase2_contract"] != model.provenance["phase2_contract"]):
        raise RuntimeError("Bridge checkpoint references different frozen specialists")
    model.bridges.load_state_dict(payload["bridges"], strict=True)
    records = json.loads(args.cases.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    completed = []
    for record in records:
        # Bundle: already normalized with this motion specialist's exact stats,
        # centered neutral joints, and cached Wan latents. No online recanonicalization.
        with np.load(record["bundle"], allow_pickle=False) as bundle:
            expected_hash = model.provenance["phase2_contract"]["motion_mean"]
            if str(bundle["motion_mean_sha256"].item()) != expected_hash:
                raise RuntimeError("Held-out bundle uses different generator normalization")
            if str(bundle["motion_std_sha256"].item()) != model.provenance["phase2_contract"]["motion_std"]:
                raise RuntimeError("Held-out bundle uses different generator std")
            motion = torch.as_tensor(bundle["motion"].copy(), device="cuda").float()[None]
            neutral = torch.as_tensor(bundle["neutral"].copy(), device="cuda").float()[None]
            video = torch.as_tensor(bundle["video"].copy(), device="cuda").float()[None]
        output = sample(model, mode=args.mode, caption=record["caption"], motion=motion,
                        neutral=neutral, video=video, seed=int(record.get("seed", 0)),
                        steps=args.steps, guidance=args.guidance)
        name = Path(str(record["id"])).name
        if name != record["id"] or name in {"", ".", ".."}:
            raise ValueError("Case IDs must be safe basenames")
        path = args.out / (name + ".npz")
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing inference: {path}")
        predicted = ({"generated_motion": output["motion"][0].cpu().numpy(),
                      "generated_video": output["video"][0].cpu().numpy()} if args.mode == "joint"
                     else {"generated": output[0].cpu().numpy()})
        np.savez(path, **predicted, gt_motion=motion[0].cpu().numpy(),
                 gt_video_latents=video[0].cpu().numpy(), neutral=neutral[0].cpu().numpy())
        completed.append({**record, "output": str(path)})
        atomic_json(args.out / "manifest.json", {"cases": completed, "mode": args.mode,
                    "checkpoint": str(args.checkpoint), "step": payload["step"],
                    "steps": args.steps, "guidance": args.guidance,
                    "shift": {"motion": 3.0, "video": 10.0} if args.mode == "joint" else (3.0 if args.mode == "video2motion" else 10.0),
                    "sources": model.provenance,
                    "metric_policy": "Report generated and matching GT; exclude conditioning frame 0 in video metrics"})
        print(f"[phase3/sample] {len(completed)}/{len(records)}: {path}", flush=True)


if __name__ == "__main__":
    main()
