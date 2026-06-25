#!/usr/bin/env python
"""Export (raw_text, motion[T,369]) pairs from the kimodo BONES-SEED dataset.

Instantiates :class:`SOMABonesSeedDataset` with the SAME paths/args the real
training uses (configs/training/bones_seed_full.yaml + submit_train_bones_seed.sh),
with the on-disk-real ``/weka/jungbin/seed/...`` paths substituted for the
``/home/jungbin_cho/seed/...`` placeholders in the config, then iterates the
dataset and writes a packed format the cosmos env can mmap:

  features.npy  -- float32 [total_frames, 369], all clips concatenated along time
  index.json    -- offsets/texts/lengths/filenames/sources/meta (see DESIGN.md)

Sample i is recovered as ``features[offsets[i]:offsets[i+1]]`` -> [T_i, 369].

The text is kept as a RAW STRING (NOT embedded) -- Cosmos3 tokenizes it itself.
Run on a compute node (CPU only is fine; dataset __getitem__ needs no GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# kimodo package import (run from repo root or rely on installed pkg).
KIMODO_ROOT = "/home/jungbin_cho/kimodo_open"
if KIMODO_ROOT not in sys.path:
    sys.path.insert(0, KIMODO_ROOT)

import torch  # noqa: E402
from kimodo.data import SOMABonesSeedDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Paths -- the SAME args the real training uses, but with the data placeholder
# /home/jungbin_cho/seed/... rewritten to its real on-disk location /weka/...
# (config line: data_root=/home/jungbin_cho/seed/... -> /weka/jungbin/seed/...).
# ---------------------------------------------------------------------------
DATA_ROOT = "/weka/jungbin/seed/soma_uniform_motions_20fps"
NATURAL_CSV_PATH = "/weka/jungbin/seed/metadata/seed_metadata_v004.csv"
TEMPORAL_LABELS_PATH = "/weka/jungbin/seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
MULTI_TIMELINE_PATH = "/weka/jungbin/seed/multi_timeline.jsonl"
# submit script uses /home/.../Kimodo-Motion-Gen-Benchmark/splits/train_split_paths.txt;
# real location on weka:
TRAIN_SPLIT_PATH = "/weka/jungbin/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths.txt"
# DESIGN.md normalization stats (also what the task specifies):
STATS_PATH = "/weka/jungbin/seed/stats/soma_uniform_motions_20fps/"

# Numeric dataset args -- straight from bones_seed_full.yaml.
FPS = 20
MIN_FRAMES = 10
MAX_CLIP_SEC = 10.0       # -> max_frames = 200
MAX_SEGMENT_SEC = 15.0
INCLUDE_MIRRORED = True

DEFAULT_OUT = "/weka/jungbin/seed/cosmos_text_motion_subset"

LAYOUT = "smooth_root3|heading2|jpos90|rot6d180|vel90|footc4"


def build_dataset(seed: int) -> SOMABonesSeedDataset:
    """Build the BONES-SEED dataset, deterministic (no aug / no random offset)."""
    return SOMABonesSeedDataset(
        data_root=DATA_ROOT,
        natural_csv_path=NATURAL_CSV_PATH,
        temporal_labels_path=TEMPORAL_LABELS_PATH,
        multi_timeline_path=MULTI_TIMELINE_PATH,
        train_split_path=TRAIN_SPLIT_PATH,
        fps=FPS,
        max_clip_sec=MAX_CLIP_SEC,
        max_segment_sec=MAX_SEGMENT_SEC,
        # Deterministic export: no random start offset, no random heading aug.
        rand_offset_max_sec=0.0,
        min_frames=MIN_FRAMES,
        include_mirrored=INCLUDE_MIRRORED,
        stats_path=STATS_PATH,
        normalize=True,
        random_heading_aug=False,
        cache_index=None,
        seed=seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument(
        "--max-samples", type=int, default=4000,
        help="number of samples to export; -1 for all",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] building SOMABonesSeedDataset (this can take ~10 min the "
          f"first time)...", flush=True)
    ds = build_dataset(args.seed)
    n_total = len(ds)
    n_export = n_total if args.max_samples < 0 else min(args.max_samples, n_total)
    print(f"[export] dataset virtual length = {n_total}; exporting {n_export} "
          f"samples to {out_dir}", flush=True)

    # First pass: pull every sample, hold motion tensors + metadata in RAM so we
    # know total_frames before writing the single concatenated features.npy.
    # (4000 clips * <=200 frames * 369 * 4 bytes <= ~1.2 GB worst case -- fine.)
    motions: list[np.ndarray] = []
    texts: list[str] = []
    lengths: list[int] = []
    filenames: list[str] = []
    sources: list[str] = []

    total_frames = 0
    for i in range(n_export):
        item = ds[i]
        motion = item["motion"]                       # torch float32 [T, 369]
        if isinstance(motion, torch.Tensor):
            motion = motion.detach().cpu().contiguous().numpy()
        motion = np.ascontiguousarray(motion, dtype=np.float32)
        T = int(motion.shape[0])
        assert motion.shape[1] == 369, f"sample {i} dim {motion.shape[1]} != 369"

        motions.append(motion)
        texts.append(str(item["text"]))
        lengths.append(T)
        filenames.append(str(item["filename"]))
        sources.append(str(item["source"]))
        total_frames += T

        if (i + 1) % 500 == 0:
            print(f"[export] {i + 1}/{n_export} samples, {total_frames} frames "
                  f"so far", flush=True)

    print(f"[export] collected {len(motions)} samples, {total_frames} total "
          f"frames; writing features.npy...", flush=True)

    # Preallocate the single concatenated array and fill it in place.
    features = np.empty((total_frames, 369), dtype=np.float32)
    offsets = np.empty(len(motions) + 1, dtype=np.int64)
    offsets[0] = 0
    cursor = 0
    for j, m in enumerate(motions):
        T = m.shape[0]
        features[cursor:cursor + T] = m
        cursor += T
        offsets[j + 1] = cursor
    assert cursor == total_frames

    feat_path = out_dir / "features.npy"
    np.save(feat_path, features)

    index = {
        "offsets": offsets.tolist(),
        "texts": texts,
        "lengths": lengths,
        "filenames": filenames,
        "sources": sources,
        "meta": {
            "fps": FPS,
            "dim": 369,
            "normalized": True,
            "layout": LAYOUT,
            "source_dataset": "bones_seed soma_uniform_motions_20fps",
        },
    }
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"[export] DONE")
    print(f"[export]   features.npy : {feat_path}  shape={features.shape} "
          f"({feat_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[export]   index.json   : {index_path} "
          f"({index_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[export]   N={len(texts)}  total_frames={total_frames}")


if __name__ == "__main__":
    main()
