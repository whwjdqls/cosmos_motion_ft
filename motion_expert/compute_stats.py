"""Phase 0 (v3): 283-D UniEgoMotion normalization stats over GROUNDED train windows.

Computes stats on exactly what training sees: each train-pair window, grounded
(`ground_features` with its per-window ground_offset_y) + frame-0 canon reset. Grouped by
sequence so each uniego npz is read once. Constant channels (std<1e-6) → std=1.

Run (cosmos env):  python motion_expert/compute_stats.py
Outputs: motion_expert/stats/uniego283_{mean,std}.npy
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

from uniego_layout import FEAT_DIM, canonicalize_frame0, ground_features

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(HERE, "pairs_train.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "stats"))
    ap.add_argument("--const_eps", type=float, default=1e-6)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pairs)]
    by_seq = collections.defaultdict(list)
    for r in rows:
        by_seq[r["uniego_path"]].append(r)
    print(f"[stats] {len(rows)} train windows over {len(by_seq)} sequences (grounded + canon0)")

    n = 0
    s1 = np.zeros(FEAT_DIM, np.float64)
    s2 = np.zeros(FEAT_DIM, np.float64)
    for si, (path, ws) in enumerate(by_seq.items()):
        feats_all = np.load(path)["features"]
        for w in ws:
            f = feats_all[w["start"]:w["end"]].astype(np.float32)
            f = ground_features(f, w["ground_offset_y"])
            f = canonicalize_frame0(f)
            f64 = f.astype(np.float64)
            s1 += f64.sum(0)
            s2 += (f64 * f64).sum(0)
            n += f.shape[0]
        if (si + 1) % 100 == 0:
            print(f"  {si+1}/{len(by_seq)} seqs, {n} frames")

    mean = (s1 / n).astype(np.float32)
    std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0)).astype(np.float32)
    const = std < args.const_eps
    std[const] = 1.0

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "uniego283_mean.npy"), mean)
    np.save(os.path.join(args.out_dir, "uniego283_std.npy"), std)
    print(f"[stats] frames={n} const_channels={int(const.sum())} "
          f"finite={np.isfinite(mean).all() and np.isfinite(std).all()} "
          f"min_std={std.min():.4g} max_std={std.max():.4g}")
    print(f"[stats] wrote {args.out_dir}/uniego283_{{mean,std}}.npy")


if __name__ == "__main__":
    main()
