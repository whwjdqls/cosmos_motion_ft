#!/usr/bin/env python
"""Verify a packed BONES-SEED text->motion export (features.npy + index.json)."""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)

    features = np.load(out / "features.npy", mmap_mode="r")
    with open(out / "index.json") as f:
        idx = json.load(f)

    offsets = np.asarray(idx["offsets"], dtype=np.int64)
    texts = idx["texts"]
    lengths = idx["lengths"]
    filenames = idx["filenames"]
    sources = idx["sources"]
    N = len(texts)

    print("=== shape / contract assertions ===")
    print("features.shape =", features.shape, "dtype =", features.dtype)
    assert features.shape[1] == 369, "feature dim != 369"
    assert offsets[-1] == features.shape[0], (
        f"offsets[-1]={offsets[-1]} != features rows {features.shape[0]}")
    assert len(offsets) == N + 1, "offsets length != N+1"
    assert len(lengths) == N and len(filenames) == N and len(sources) == N
    assert all(int(offsets[i + 1] - offsets[i]) == lengths[i] for i in range(N)), \
        "offset diffs != lengths"
    print("OK: shape[1]==369, offsets[-1]==rows, len(texts)==N, offset diffs==lengths")

    print("\n=== summary ===")
    print("N (samples)     =", N)
    print("total_frames    =", int(features.shape[0]))
    print("meta            =", idx["meta"])

    L = np.asarray(lengths)
    print("\n=== length stats (frames) ===")
    print(f"min={L.min()} max={L.max()} mean={L.mean():.1f} median={np.median(L):.0f}")
    edges = [10, 25, 50, 75, 100, 125, 150, 175, 200, 201]
    hist, _ = np.histogram(L, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        print(f"  [{lo:3d},{hi:3d}): {c}")

    print("\n=== source distribution ===")
    from collections import Counter
    for s, c in sorted(Counter(sources).items()):
        print(f"  {s:8s}: {c}")

    print("\n=== normalization check (sample of frames) ===")
    # Sample up to 200k frames to estimate per-array mean/std.
    n = min(features.shape[0], 200_000)
    sub = np.asarray(features[:n], dtype=np.float64)
    print(f"  over first {n} frames: mean={sub.mean():.4f} std={sub.std():.4f}")
    print(f"  per-channel mean range [{sub.mean(0).min():.3f}, {sub.mean(0).max():.3f}]")
    print(f"  per-channel std  range [{sub.std(0).min():.3f}, {sub.std(0).max():.3f}]")

    print("\n=== 5 example (text, length, source) ===")
    for i in range(min(5, N)):
        t = texts[i]
        t = t if len(t) <= 90 else t[:87] + "..."
        print(f"  [{i}] len={lengths[i]:3d} src={sources[i]:8s} file={filenames[i]}")
        print(f"        text: {t!r}")


if __name__ == "__main__":
    main()
