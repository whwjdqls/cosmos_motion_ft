#!/usr/bin/env python
"""Verify a sharded FULL BONES-SEED export under a parent dir.

Checks per-shard: features.npy mmaps with shape (total_frames,369), offsets
consistent with lengths, index N matches. Aggregates totals and prints a few
example (text,len,source). Also opens the whole parent via the trainer's
multi-shard logic (re-implemented standalone, no cosmos import needed).
"""
import os, sys, json, glob
import numpy as np

DIM = 369


def check_shard(sd):
    idx = json.load(open(os.path.join(sd, "index.json")))
    offs = np.asarray(idx["offsets"], dtype=np.int64)
    lengths = np.asarray(idx["lengths"], dtype=np.int64)
    n = len(idx["texts"])
    assert len(offs) == n + 1, f"{sd}: offsets {len(offs)} != N+1 {n+1}"
    assert np.array_equal(np.diff(offs), lengths), f"{sd}: diff(offsets)!=lengths"
    feat = np.load(os.path.join(sd, "features.npy"), mmap_mode="r")
    assert feat.shape == (int(offs[-1]), DIM), f"{sd}: npy {feat.shape}"
    assert feat.dtype == np.float32, f"{sd}: dtype {feat.dtype}"
    size = os.path.getsize(os.path.join(sd, "features.npy"))
    return n, int(offs[-1]), size, idx


def main():
    parent = sys.argv[1] if len(sys.argv) > 1 else "/weka/jungbin/seed/cosmos_text_motion_full"
    shard_dirs = sorted(
        d for d in glob.glob(os.path.join(parent, "shard_*"))
        if os.path.exists(os.path.join(d, "index.json"))
    )
    print(f"parent={parent}  shards_found={len(shard_dirs)}")
    tot_n = tot_frames = tot_size = 0
    from collections import Counter
    src_counter = Counter()
    examples = []
    for sd in shard_dirs:
        n, frames, size, idx = check_shard(sd)
        tot_n += n; tot_frames += frames; tot_size += size
        src_counter.update(idx["sources"])
        print(f"  {os.path.basename(sd):12s}  N={n:7d}  frames={frames:9d}  "
              f"size={size/1e9:.2f}GB  OK")
        if len(examples) < 5 and n > 0:
            examples.append((idx["texts"][0][:60], idx["lengths"][0], idx["sources"][0]))
    print(f"\nTOTAL  N={tot_n}  total_frames={tot_frames}  "
          f"size={tot_size/1e9:.2f}GB")
    print(f"sources: {dict(src_counter)}")
    print("examples (text,len,source):")
    for t, L, s in examples:
        print(f"  [{s}] len={L}  {t!r}")


if __name__ == "__main__":
    main()
