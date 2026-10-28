"""Paired, inference-only image-conditioning diagnostic for an Edge checkpoint.

Use run.sh on an owned compute allocation. Preparation freezes one window per
recording; workers never substitute cases. Training and checkpoint files are
read-only. Correct/wrong images retain identical caption/token positions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch

VARIANTS = ("correct", "wrong1", "wrong2", "no_image")
SIGMAS = (0.25, 0.5, 0.75, 1.0)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(args):
    import config
    from checkpoint import configure_attention_from_checkpoint
    from data import build_phase2_dataset

    if (args.out / "configuration.json").exists():
        raise RuntimeError("Prepared cohort already exists; do not replace it")
    contract = configure_attention_from_checkpoint(args.checkpoint)
    for key, path in (("motion_mean", config.MOTION_STATS_MEAN),
                      ("motion_std", config.MOTION_STATS_STD)):
        if sha(path) != contract[key]:
            raise RuntimeError(f"Checkpoint statistics mismatch: {key}")
    ds = build_phase2_dataset(split="test", train=False, num_frames=200,
                             ti2m_frames=97, task_weights={"textimg2motion": 1.0},
                             bones_frac=0, cfg_dropout=0, seed=args.seed)
    groups = defaultdict(list)
    for index, entry in enumerate(ds._index):
        groups[entry["uuid"]].append((index, entry))
    rng = random.Random(args.seed)
    recordings = sorted(groups)
    rng.shuffle(recordings)
    selected, rejected = [], []
    (args.out / "inputs").mkdir(parents=True, exist_ok=True)
    for uuid in recordings:
        candidates = list(groups[uuid])
        rng.shuffle(candidates)
        for index, entry in candidates:
            try:
                motion, neutral, pad = ds._load_motion(
                    entry["uni"], entry["s"], entry["off"], avail=entry["avail"])
                motion = motion[~pad]
                if len(motion) != 97 or not np.isfinite(motion).all():
                    raise ValueError("Expected 97 finite valid frames")
                image = ds._load_reasoner_image(entry["vis"], entry["s"])
                caption = ds._apply_caption_policy("textimg2motion", entry["cap"])
                if not caption.strip():
                    raise ValueError("Empty caption")
            except Exception as error:
                rejected.append({"uuid": uuid, "start": entry["s"], "error": str(error)})
                continue
            slot = len(selected)
            np.savez_compressed(args.out / "inputs" / f"case_{slot:03d}.npz",
                                motion=motion, neutral=neutral,
                                image=image.cpu().numpy())
            selected.append({"slot": slot, "uuid": uuid, "start": entry["s"],
                             "dataset_index": index, "caption": caption,
                             "video_path": entry["vis"], "motion_path": entry["uni"],
                             "floor_offset": entry["off"], "valid_frames": 97})
            print(f"[prepare] {slot + 1}/{args.cases}: {uuid}@{entry['s']}", flush=True)
            break
        if len(selected) == args.cases:
            break
    if len(selected) != args.cases or len(selected) < 3:
        raise RuntimeError(f"Only prepared {len(selected)} cases")
    for row in selected:
        row["wrong1_slot"] = (row["slot"] + 1) % len(selected)
        row["wrong2_slot"] = (row["slot"] + len(selected) // 2) % len(selected)
    atomic_json(args.out / "cases.json", selected)
    atomic_json(args.out / "preparation_rejections.json", rejected)
    atomic_json(args.out / "configuration.json", {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha(args.checkpoint),
        "checkpoint_contract": contract, "cohort_seed": args.seed, "cases": len(selected),
        "cohort": "One random valid TI2M window per distinct held-out recording",
        "sampling_seeds": [0, 1, 2], "sigmas": list(SIGMAS), "variants": list(VARIANTS),
        "sampling": {"steps": 35, "guidance": 2.0, "shift": 3.0, "frames": 97,
                     "fps": 20, "null_branch": "Drop text; retain the variant's image"},
        "control": "All variants share caption, neutral shape, GT, length and initial noise. "
                   "Wrong images come from different recordings with equal visual token layout. "
                   "No-image uses native T2M packing and is not a layout-matched image ablation.",
        "interpretation": "Denoiser errors measure paired x0 prediction. Sampled GT errors are "
                          "diagnostic only: a first-person image and caption do not specify a unique motion.",
        "statistics": {"mean": str(config.MOTION_STATS_MEAN), "std": str(config.MOTION_STATS_STD)},
        "input_hashes": {p.name: sha(p) for p in sorted((args.out / "inputs").glob("*.npz"))},
    })


def physics_setup():
    from shape_tmr_eval_common import add_bundle_python_paths
    add_bundle_python_paths(os.environ["SHAPE_TMR_BUNDLE"])
    from kimodo.metrics import FootSkateFromHeight, FootSkateRatio, compute_metrics
    from kimodo.skeleton import SOMASkeleton30
    from phase2_quantitative import floor_metrics
    skel = SOMASkeleton30().cuda()
    metrics = [FootSkateFromHeight(skeleton=skel, fps=20), FootSkateRatio(skeleton=skel, fps=20)]

    def measure(joints, raw):
        lengths = torch.tensor([joints.shape[1]], device=joints.device)
        values = compute_metrics(metrics, {"posed_joints": joints,
                                 "foot_contacts": raw[..., 279:283] > 0.5, "lengths": lengths})
        values.update(floor_metrics(joints, lengths, [int(x) for x in skel.foot_joint_idx]))
        return {k: float(v.float().mean().item()) for k, v in values.items()}
    return measure


@torch.inference_mode()
def worker(args):
    import config
    from checkpoint import configure_attention_from_checkpoint, load_checkpoint
    from model import EdgePhase2MotionExpert
    from flow import sample_x0_unipc
    from decode_uniego_torch import decode_joints
    from eval_motion_recon import joint_metrics

    cfg = json.loads((args.out / "configuration.json").read_text())
    if sha(args.checkpoint) != cfg["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed")
    configure_attention_from_checkpoint(args.checkpoint)
    model = EdgePhase2MotionExpert(device="cuda", dtype=torch.bfloat16)
    payload = load_checkpoint(args.checkpoint, model=model)
    if payload["step"] != 200000:
        raise RuntimeError("Expected the Phase3 specialist at step 200k")
    del payload
    model.eval().requires_grad_(False)
    measure_physics = physics_setup()
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).cuda().float()
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).cuda().float()
    cases = json.loads((args.out / "cases.json").read_text())
    started = time.time()
    work = args.out / f"shard_{args.shard}"
    work.mkdir(exist_ok=True)
    for case in cases[args.shard::args.shards]:
        slot = case["slot"]
        case_dir = work / f"case_{slot:03d}"
        if (case_dir / "results.json").exists():
            continue
        case_dir.mkdir(exist_ok=True)
        data = np.load(args.out / "inputs" / f"case_{slot:03d}.npz")
        gt = torch.from_numpy(data["motion"]).cuda().float()[None]
        neutral = torch.from_numpy(data["neutral"]).cuda().float()[None].expand(4, -1, -1)
        pad = torch.zeros((4, 97), device="cuda", dtype=torch.bool)
        images = [torch.from_numpy(data["image"])]
        for key in ("wrong1_slot", "wrong2_slot"):
            donor = np.load(args.out / "inputs" / f"case_{case[key]:03d}.npz")
            images.append(torch.from_numpy(donor["image"]))
        images.append(None)
        modes = ["textimg2motion"] * 3 + ["text2motion"]
        cond = model.prepare_conditions([case["caption"]] * 4, modes=modes, reasoner_images=images)
        null = model.prepare_conditions([""] * 4, modes=modes, reasoner_images=images)
        for group in (cond, null):
            for j in (1, 2):
                for key in ("input_ids", "position_ids", "visual_pos_mask"):
                    assert torch.equal(group[0][key], group[j][key]), (slot, key)
                assert not torch.equal(group[0]["inputs_embeds"], group[j]["inputs_embeds"])
                mask = group[0]["visual_pos_mask"].bool()
                assert torch.equal(group[0]["inputs_embeds"][~mask], group[j]["inputs_embeds"][~mask])
        audit = {v: {"tokens": len(c["input_ids"]),
                     "image_tokens": int(c.get("visual_pos_mask", torch.zeros(1)).sum()),
                     "next_position_id": c["next_position_id"]}
                 for v, c in zip(VARIANTS, cond)}
        audit["visual_embedding_swap_rms"] = [float((cond[0]["inputs_embeds"].float() -
                                                   cond[j]["inputs_embeds"].float()).square().mean().sqrt())
                                              for j in (1, 2)]
        gt_raw = gt * std + mean
        gt_j = decode_joints(gt_raw)
        gt_np = gt_j[0].cpu().numpy()
        gt_metrics = joint_metrics(gt_np, gt_np)
        gt_metrics.update(feat_mse=0.0, **measure_physics(gt_j, gt_raw))
        rows = []

        def predict(conditions):
            return lambda x, sigma: model(reasoner_inputs=conditions, x_sigma=x, sigma=sigma,
                                           neutral_joints=neutral, motion_pad_mask=pad)

        pred = predict(cond)
        for seed in cfg["sampling_seeds"]:
            # Same CPU-generated noise is copied to every condition, probe and sample.
            noise = torch.randn((1, 97, 283), generator=torch.Generator().manual_seed(seed)).cuda()
            for sigma in SIGMAS:
                noisy = ((1 - sigma) * gt + sigma * noise).expand(4, -1, -1).contiguous()
                levels = torch.full((4,), sigma, device="cuda")
                estimate = pred(noisy, levels)
                assert torch.isfinite(estimate).all()
                if seed == 0 and sigma == 1.0:
                    repeated = pred(noisy, levels)
                    audit["same_input_bitwise_replay"] = bool(torch.equal(estimate, repeated))
                    audit["same_input_replay_max_abs"] = float((estimate - repeated).abs().max())
                for j, variant in enumerate(VARIANTS):
                    rows.append({"kind": "denoiser", "variant": variant, "seed": seed, "sigma": sigma,
                                 "feat_mse": float((estimate[j:j+1] - gt).square().mean()),
                                 "swap_rms": float((estimate[j] - estimate[0]).square().mean().sqrt())})
            sampled = sample_x0_unipc(pred, predict_null=predict(null), T=97, motion_dim=283,
                                     steps=35, guidance=2, batch=4, device="cuda",
                                     initial_noise=noise.expand(4, -1, -1).contiguous())
            assert torch.isfinite(sampled).all()
            raw = sampled * std + mean
            joints = decode_joints(raw)
            arrays = {"gt_features": gt_raw[0].cpu().numpy(), "gt_joints": gt_np}
            for j, variant in enumerate(VARIANTS):
                row = joint_metrics(joints[j].cpu().numpy(), gt_np)
                row.update(measure_physics(joints[j:j+1], raw[j:j+1]))
                row.update(kind="sample", variant=variant, seed=seed,
                           feat_mse=float((sampled[j:j+1] - gt).square().mean()),
                           first_frame_mpjpe_m=float((joints[j, 0] - gt_j[0, 0]).norm(dim=-1).mean()),
                           swap_rms=float((sampled[j] - sampled[0]).square().mean().sqrt()),
                           swap_mpjpe_m=float((joints[j] - joints[0]).norm(dim=-1).mean()))
                rows.append(row)
                arrays[variant + "_features"] = raw[j].cpu().numpy()
                arrays[variant + "_joints"] = joints[j].cpu().numpy()
            np.savez_compressed(case_dir / f"seed_{seed}.npz", **arrays)
        atomic_json(case_dir / "results.json", {"case": case, "audit": audit,
                                                "GT": gt_metrics, "measurements": rows})
        progress = {"completed_case": slot, "shard": args.shard, "seconds": time.time() - started,
                    "cases_complete": len(list(work.glob("case_*/results.json")))}
        atomic_json(work / "progress.json", progress)
        print("[image-diagnostic] " + json.dumps(progress), flush=True)
    atomic_json(work / "complete.json", {"complete": True, "seconds": time.time() - started})


def summarize(args):
    cfg = json.loads((args.out / "configuration.json").read_text())
    results = [json.loads(p.read_text()) for p in sorted(args.out.glob("shard_*/case_*/results.json"))]
    if len(results) != cfg["cases"] or len({r['case']['slot'] for r in results}) != cfg["cases"]:
        raise RuntimeError(f"Incomplete/duplicate cohort: {len(results)}/{cfg['cases']}")
    groups = defaultdict(list)
    by_case = defaultdict(dict)
    for result in results:
        slot = result["case"]["slot"]
        groups[("GT", "GT", None)].append(result["GT"])
        local = defaultdict(list)
        for row in result["measurements"]:
            local[(row["kind"], row["variant"], row.get("sigma"))].append(row)
        for key, rows in local.items():
            values = {k: float(np.mean([r[k] for r in rows])) for k, v in rows[0].items()
                      if isinstance(v, (float, int)) and k not in {"seed", "sigma"}}
            groups[key].append(values)
            by_case[key][slot] = values
    aggregate = [{"kind": key[0], "variant": key[1], "sigma": key[2], "cases": len(rows),
                  "metrics": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}}
                 for key, rows in groups.items()]
    comparisons = []
    rng = np.random.default_rng(20261028)
    for kind, sigmas, metrics in (("denoiser", SIGMAS, ["feat_mse"]),
                                  ("sample", [None], ["feat_mse", "mpjpe_m", "pa_mpjpe_m", "first_frame_mpjpe_m"])):
        for sigma in sigmas:
            correct = by_case[(kind, "correct", sigma)]
            for variant in VARIANTS[1:]:
                other = by_case[(kind, variant, sigma)]
                for metric in metrics:
                    diffs = np.array([other[i][metric] - correct[i][metric] for i in sorted(correct)])
                    boot = diffs[rng.integers(len(diffs), size=(5000, len(diffs)))].mean(axis=1)
                    comparisons.append({"kind": kind, "sigma": sigma, "other": variant, "metric": metric,
                                        "other_minus_correct": float(diffs.mean()),
                                        "recording_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(),
                                        "correct_win_fraction": float((diffs > 0).mean())})
    atomic_json(args.out / "summary.json", {"cases": len(results), "seeds": cfg["sampling_seeds"],
                                            "aggregate": aggregate, "paired_comparisons": comparisons,
                                            "all_bitwise_replays": all(r['audit']['same_input_bitwise_replay'] for r in results),
                                            "interpretation": cfg['interpretation']})
    import csv
    rows = [{"slot": r['case']['slot'], "uuid": r['case']['uuid'], **x}
            for r in results for x in r['measurements']]
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (args.out / "per_case.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys);writer.writeheader();writer.writerows(rows)
    lines = ["# Phase 2 image-conditioning diagnostic", "", f"{len(results)} distinct test recordings; three paired seeds; frozen 200k checkpoint.",
             "", "## Single-denoiser normalized feature MSE (lower is better)", "",
             "| Sigma | Correct image | Wrong image 1 | Wrong image 2 | No image |", "|---:|---:|---:|---:|---:|"]
    def item(kind, variant, sigma=None):
        return next(r['metrics'] for r in aggregate if (r['kind'],r['variant'],r['sigma']) == (kind,variant,sigma))
    for sigma in SIGMAS:
        lines.append(f"| {sigma} | " + " | ".join(f"{item('denoiser',v,sigma)['feat_mse']:.6f}" for v in VARIANTS) + " |")
    lines += ["", "GT-self feature MSE: 0 (computed from the identical normalized inputs). Sigma 1 is pure noise.",
              "", "## UniPC35 / CFG2 / shift3 samples", "",
              "| Variant | MPJPE m | PA-MPJPE m | Frame-0 MPJPE m | Feature MSE |", "|---|---:|---:|---:|---:|"]
    for v in ('GT', *VARIANTS):
        d = item('GT' if v == 'GT' else 'sample', v)
        lines.append(f"| {v} | {d['mpjpe_m']:.5f} | {d['pa_mpjpe_m']:.5f} | {d.get('first_frame_mpjpe_m',0):.5f} | {d['feat_mse']:.5f} |")
    lines += ["", cfg['control'], "", cfg['interpretation'], "",
              "Physical GT/model metrics and recording-level paired bootstrap intervals are in summary.json. No training or checkpoint changes."]
    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def render(args):
    from PIL import Image, ImageDraw
    from render_motion import render_single, head_rotation_from_features, write_mp4
    cases = json.loads((args.out / "cases.json").read_text())
    gallery = args.out / "gallery";gallery.mkdir(exist_ok=True)
    for case in cases[:6]:
        slot = case['slot']
        result = next(args.out.glob(f"shard_*/case_{slot:03d}/seed_0.npz"))
        data = np.load(result)
        panels = []
        for name, title, color in [('gt','GT','tab:blue'),('correct','Correct image','tab:green'),
                                   ('wrong1','Wrong image','tab:orange'),('no_image','No image','tab:red')]:
            panels.append(render_single(data[name+'_joints'], caption=title, color=color,
                          width=360, height=360, frame_stride=2, max_frames=97,
                          head_rotation0=head_rotation_from_features(data[name+'_features'])))
        motion = np.concatenate(panels, axis=2)
        banner = Image.new('RGB', (1440, 240), (245,245,245))
        draw = ImageDraw.Draw(banner)
        draw.text((15,15), case['caption'][:190], fill='black')
        for donor, x, label, color in [(slot,390,'Correct observed image','green'),
                                      (case['wrong1_slot'],750,'Mismatched observed image','orange')]:
            image = np.load(args.out/'inputs'/f'case_{donor:03d}.npz')['image'].transpose(1,2,0)
            im = Image.fromarray(image).resize((180,180))
            banner.paste(im,(x,55));draw.rectangle((x-3,52,x+182,237),outline=color,width=3)
            draw.text((x,37),label,fill=color)
        frames = np.concatenate([np.broadcast_to(np.array(banner),(len(motion),240,1440,3)),motion],axis=1)
        write_mp4(frames,str(gallery/f'case_{slot:03d}.mp4'),fps=10)
        Image.fromarray(frames[0]).save(gallery/f'case_{slot:03d}.png')
        print(f'[render] case {slot}',flush=True)
    import html
    sections = [f'<h2>Case {c["slot"]}: {html.escape(c["caption"])}</h2><video controls width="1200" src="case_{c["slot"]:03d}.mp4"></video>' for c in cases[:6]]
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Image conditioning</title><h1>GT | Correct image | Wrong image | No image</h1><p>Fixed first six cases, seed 0; original coordinates and floor.</p>'+'\n'.join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','worker','summarize','render'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    host = socket.gethostname().split('.')[0]
    if 'login' in host or host in {'cs-gpu-01','node37'} or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('This diagnostic must run in an owned compute allocation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
