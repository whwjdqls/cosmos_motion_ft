"""v2: precompute frozen-reasoner H_R for all unique captions → fp16 cache (cosmos env).

The reasoner is frozen ⇒ H_R is deterministic per caption, so caching removes the 94%-of-step
on-the-fly cost. Sharded across GPUs for speed.

Global caption list = sorted(unique train+val captions) with "" at index 0. Shard k owns ids
where id % nshards == k, stored in order → packed [sum_Ti, 4096] fp16 + per-shard lens.npy.
Loader (hr_cache.py) reconstructs caption→id and id→(shard, offset).

Run (cosmos env), one process per GPU:
  for k in 0..N-1: CUDA_VISIBLE_DEVICES=k bash run.sh precompute_hr.py --shard k --nshards N &
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from reasoner import D_REASONER, FrozenReasoner

HERE = os.path.dirname(os.path.abspath(__file__))


def all_captions() -> list[str]:
    caps = set()
    for split in ("pairs_train.jsonl", "pairs_val.jsonl"):
        for l in open(os.path.join(HERE, split)):
            caps.add(json.loads(l)["caption"])
    caps.discard("")
    return [""] + sorted(caps)        # "" at index 0 (null/CFG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out_dir", default="/weka/jungbin/cosmos_motion_ft_runs/hr_cache")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    caps = all_captions()
    if args.shard == 0:
        json.dump(caps, open(os.path.join(args.out_dir, "captions.json"), "w"))
    my_ids = list(range(args.shard, len(caps), args.nshards))
    print(f"[hr] shard {args.shard}/{args.nshards}: {len(my_ids)} captions (of {len(caps)})")

    reasoner = FrozenReasoner(dtype=torch.bfloat16, device="cuda", verbose=(args.shard == 0))
    lens, chunks = [], []
    for j, cid in enumerate(my_ids):
        H = reasoner._one(caps[cid])                      # [Ti,4096] bf16
        chunks.append(H.to(torch.float16).cpu().numpy())
        lens.append(H.shape[0])
        if (j + 1) % 2000 == 0:
            print(f"  shard {args.shard}: {j+1}/{len(my_ids)}")
    packed = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, D_REASONER), np.float16)
    np.save(os.path.join(args.out_dir, f"shard{args.shard}_hr.npy"), packed)
    np.save(os.path.join(args.out_dir, f"shard{args.shard}_lens.npy"), np.array(lens, np.int64))
    print(f"[hr] shard {args.shard} done: {packed.shape[0]} tokens, "
          f"{packed.nbytes/1e9:.1f} GB -> {args.out_dir}/shard{args.shard}_hr.npy")


if __name__ == "__main__":
    main()
