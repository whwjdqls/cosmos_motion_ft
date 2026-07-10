"""Per-feature mean/std for the shape-aware TMR's 186-d TMRMotionRep normalization.

Adapted from TAP `tmr_g1/training/build_stats.py`: streams random train-split windows
of the BONES-SEED **proportional** uniego tree through decode -> SOMASkeleton30
TMRMotionRep (CANONICALIZED — must match training), and writes the split-layout
stats `MotionRepBase` requires:

    <out>/global_root/{mean,std}.npy   <out>/local_root/{mean,std}.npy
    <out>/body/{mean,std}.npy          <out>/{mean,std}.npy

Usage (CPU, ~10 min):
  bash st_run.sh build_stats.py \
      --train-split /home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths.txt \
      --out /mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/stats_v0 --n-motions 5000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from kimodo.skeleton import SOMASkeleton30
from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep

from decode_uniego import decode_joints
from st_dataset import DATA_ROOT, SPLIT_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--train-split", default=f"{SPLIT_DIR}/train_split_paths.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--n-motions", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=200)  # 10 s @ 20 fps
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    data_root = Path(args.data_root)
    with open(args.train_split) as f:
        rels = [ln.strip() for ln in f if ln.strip()]
    rng.shuffle(rels)
    mr = TMRMotionRep(skeleton=SOMASkeleton30(), fps=args.fps, stats_path=None)

    n_total = mr.motion_rep_dim
    n_global_root = mr.global_root_dim
    n_local_root = mr.local_root_dim
    n_body = mr.body_dim
    print(f"[stats] rep dim={n_total} (global_root={n_global_root} body={n_body})")

    s = torch.zeros(n_total, dtype=torch.float64)
    ss = torch.zeros(n_total, dtype=torch.float64)
    count = 0
    n_taken = n_skip = 0

    for rel in rels:
        if n_taken >= args.n_motions:
            break
        npz_path = data_root / (rel + ".npz")
        if not npz_path.is_file():
            continue
        with np.load(npz_path, mmap_mode="r") as d:
            n_frames = int(d["features"].shape[0])
            if n_frames < 20:
                continue
            if n_frames > args.max_frames:
                off = int(rng.integers(0, n_frames - args.max_frames + 1))
                win = np.asarray(d["features"][off:off + args.max_frames]).astype(np.float32)
            else:
                win = np.asarray(d["features"]).astype(np.float32)
        if not np.isfinite(win).all():
            n_skip += 1
            continue
        T = win.shape[0]
        with torch.no_grad():
            joints = decode_joints(torch.from_numpy(win).unsqueeze(0))          # (1, T, 30, 3)
            feats = mr(
                posed_joints=joints,
                to_normalize=False,
                to_canonicalize=True,        # MUST match training
                lengths=torch.tensor([T]),
            )[0]                             # (T, 186)
        if not torch.isfinite(feats).all():
            n_skip += 1
            continue
        s += feats.double().sum(0)
        ss += feats.double().pow(2).sum(0)
        count += T
        n_taken += 1
        if n_taken % 200 == 0:
            print(f"[stats] {n_taken}/{args.n_motions}  frames={count}  skipped={n_skip}", flush=True)

    mean = (s / count).float()
    var = (ss / count).float() - mean.pow(2)
    std = var.clamp(min=1e-8).sqrt()
    print(f"[stats] {n_taken} motions, {count} frames, dim={n_total}, skipped={n_skip}")

    out = Path(args.out)
    for sub in ("global_root", "local_root", "body"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    np.save(out / "mean.npy", mean.numpy())
    np.save(out / "std.npy", std.numpy())
    np.save(out / "global_root" / "mean.npy", mean[:n_global_root].numpy())
    np.save(out / "global_root" / "std.npy", std[:n_global_root].numpy())
    np.save(out / "body" / "mean.npy", mean[n_global_root:n_global_root + n_body].numpy())
    np.save(out / "body" / "std.npy", std[n_global_root:n_global_root + n_body].numpy())
    # local_root: required by the stats-layout check, never consumed at retrieval time.
    np.save(out / "local_root" / "mean.npy", np.zeros(n_local_root, dtype=np.float32))
    np.save(out / "local_root" / "std.npy", np.ones(n_local_root, dtype=np.float32))
    print(f"[stats] wrote {out}")


if __name__ == "__main__":
    main()
