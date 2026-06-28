"""One-time coverage check: every BONES-SEED caption (for a split) must be in the llm2vec cache.

Replaces the (unneeded) offline pairs file: the dataset builds its index in-memory, and this
asserts the text side lines up with the cache so training never KeyErrors on a lookup. CPU-only.

  PYTHONPATH=/home/jungbin_cho/kimodo_open python bs_check_cache.py --split <split.txt>
"""
from __future__ import annotations

import argparse
import os

from bs_dataset import BonesSeedUniegoDataset, SPLIT_DIR
from bs_text_cache import LLM2VecCache, DEFAULT_CACHE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(SPLIT_DIR, "train_split_paths.txt"))
    ap.add_argument("--cache_path", default=DEFAULT_CACHE)
    ap.add_argument("--cache_index", default=None, help="optional JSON to cache/reuse the segment index")
    args = ap.parse_args()

    cache = LLM2VecCache(args.cache_path, device="cpu")
    ds = BonesSeedUniegoDataset(args.split, cache_index=args.cache_index, train=True, seed=0)

    total = missing = 0
    examples = []
    print(f"[check] cache: {len(cache)} captions")
    for src in ds.SOURCES:
        n = m = 0
        for e in ds._pools[src]:
            n += 1
            if e.text not in cache:
                m += 1
                if len(examples) < 5:
                    examples.append((src, e.text[:60]))
        total += n; missing += m
        print(f"  {src:8s}: {n:7d} entries, {m} missing")
    print(f"[check] total={total} missing={missing}")
    if missing:
        for src, t in examples:
            print(f"    MISSING [{src}] {t!r}")
        raise SystemExit(f"FAIL: {missing} caption(s) not in cache")
    print("[check] PASS — all captions covered")


if __name__ == "__main__":
    main()
