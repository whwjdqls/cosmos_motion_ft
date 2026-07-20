"""BONES-SEED (text, uniego-motion, skeleton) dataset for the in-context POC.

``BonesSeedUniegoDataset`` subclasses kimodo's ``SOMABonesSeedDataset`` so we inherit its
in-memory (text, segment) index over the three BONES-SEED text sources (natural/single/multi,
1:1:1), the split restriction, the ``_resolve_segment`` windowing (random offset + cap to
``max_clip_sec``), and the ``cache_index`` JSON fast-startup. We override only the motion I/O,
because our data is the **precomputed 283-D UniEgoMotion** tree, not raw rotations:

  * ``data_root`` points at ``…/soma_proportional_uniegomotion_20fps`` (so ``entry.motion_path``
    resolves straight to the uniego npz; no raw motion tree exists on this box).
  * ``_build_natural_pool`` reads a clip's frame count from ``features.shape[0]`` (the parent
    reads ``local_rot_mats``, which our npz doesn't have).
  * ``__getitem__`` loads ``features[sf:ef]`` + per-actor ``neutral_joints`` from the npz,
    canonicalizes frame 0, and normalizes with the selected Mean/Std pair.

No grounding / no heading aug (bones-seed uniego is already floor-grounded; frame-0
canonicalization removes global yaw). Text-drop for CFG is applied in the train loop, not here.

Runs in the ``kimodo`` env: ``PYTHONPATH=/home/jungbin_cho/kimodo_open``.
"""
from __future__ import annotations

import csv
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset  # noqa: F401  (re-export convenience)

from kimodo.data.soma_text_motion import SOMABonesSeedDataset, _BSEntry, _truthy

from uniego_layout import FEAT_DIM, canonicalize_frame0

log = logging.getLogger(__name__)

# ---- defaults for THIS box (A100; /weka absent) ----
DATA_ROOT = "/mnt/shared/jungbin_cho/seed/soma_proportional_uniegomotion_20fps"
MEAN_PATH = os.path.join(DATA_ROOT, "Mean_uniego.npy")
STD_PATH = os.path.join(DATA_ROOT, "Std_uniego.npy")
NATURAL_CSV = "/home/jungbin_cho/seed/metadata/seed_metadata_v004.csv"
TEMPORAL_JSONL = "/home/jungbin_cho/seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
MULTI_JSONL = "/home/jungbin_cho/seed/multi_timeline.jsonl"
SPLIT_DIR = "/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits"


class BonesSeedUniegoDataset(SOMABonesSeedDataset):
    def __init__(
        self,
        train_split_path: str,
        data_root: str = DATA_ROOT,
        natural_csv_path: str = NATURAL_CSV,
        temporal_labels_path: str = TEMPORAL_JSONL,
        multi_timeline_path: str = MULTI_JSONL,
        mean_path: str = MEAN_PATH,
        std_path: str = STD_PATH,
        fps: int = 20,
        max_clip_sec: float = 10.0,
        max_segment_sec: float = 15.0,
        rand_offset_max_sec: float = 2.0,
        min_frames: int = 10,
        include_mirrored: bool = True,
        cache_index: str | None = None,
        train: bool = True,
        seed: int | None = 0,
    ):
        self.mean = np.load(mean_path).astype(np.float32)
        self.std = np.load(std_path).astype(np.float32)
        assert self.mean.shape[0] == FEAT_DIM and self.std.shape[0] == FEAT_DIM, (
            f"stats must be ({FEAT_DIM},); got {self.mean.shape}"
        )
        self.train = bool(train)
        super().__init__(
            data_root=data_root,
            natural_csv_path=natural_csv_path,
            temporal_labels_path=temporal_labels_path,
            multi_timeline_path=multi_timeline_path,
            train_split_path=train_split_path,
            fps=fps,
            max_clip_sec=max_clip_sec,
            max_segment_sec=max_segment_sec,
            rand_offset_max_sec=rand_offset_max_sec,
            min_frames=min_frames,
            include_mirrored=include_mirrored,
            normalize=False,           # we normalize with uniego stats in __getitem__
            random_heading_aug=False,  # uniego frame-0 canon already drops global yaw
            cache_index=cache_index,
            seed=seed,
        )

    # -- override: natural-pool frame counts come from the uniego `features`, not `local_rot_mats` --
    def _build_natural_pool(self, out, path_by_name):
        desc_cols = (
            "content_natural_desc_1", "content_natural_desc_2",
            "content_natural_desc_3", "content_natural_desc_4",
        )
        descs_by_file: dict = {}
        no_motion = mirror = no_text = 0
        with open(self.natural_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                if not self.include_mirrored and (_truthy(row.get("is_mirror")) or fname.endswith("_M")):
                    mirror += 1
                    continue
                if fname not in path_by_name:
                    no_motion += 1
                    continue
                descs, seen = [], set()
                for k in desc_cols:
                    v = row.get(k, "")
                    if isinstance(v, str):
                        v = v.strip()
                    if v and v not in seen:
                        seen.add(v)
                        descs.append(v)
                if not descs:
                    no_text += 1
                    continue
                descs_by_file[fname] = descs

        long_seg = short_seg = bad_read = 0
        for fname, descs in descs_by_file.items():
            p = path_by_name[fname]
            try:
                with np.load(p, mmap_mode="r") as data:
                    n_total = int(data["features"].shape[0])   # <-- uniego npz (vs local_rot_mats)
            except (OSError, KeyError, ValueError, EOFError) as e:
                log.warning("natural pool: failed to read %s: %s", p, e)
                bad_read += 1
                continue
            t_sec = n_total / float(self.fps)
            if t_sec > self.max_segment_sec:
                long_seg += 1
                continue
            if t_sec * self.fps < self.min_frames:
                short_seg += 1
                continue
            for text in descs:
                out.append(_BSEntry(
                    source="natural", motion_path=p, filename=fname,
                    seg_start_sec=0.0, seg_end_sec=t_sec, text=text,
                ))
        log.info(
            "Pool 'natural' (uniego): %d entries (skipped no_motion=%d mirror=%d no_text=%d "
            "long>%.1fs=%d short=%d bad_read=%d)",
            len(out), no_motion, mirror, no_text, self.max_segment_sec, long_seg, short_seg, bad_read,
        )

    # -- override: load precomputed 283-D uniego features + per-actor neutrals --
    def __getitem__(self, index: int) -> dict:
        source = self.SOURCES[index % 3]
        pool = self._pools[source]
        entry = pool[(index // 3) % len(pool)]

        sf, ef = self._resolve_segment(entry)
        with np.load(entry.motion_path, mmap_mode="r") as d:
            n_total = int(d["features"].shape[0])
            sf = max(0, min(sf, n_total))
            ef = max(sf, min(ef, n_total))
            ef = min(ef, sf + self.max_frames)
            feats = np.asarray(d["features"][sf:ef]).astype(np.float32)        # [n, 283]
            nj = np.asarray(d["neutral_joints"]).astype(np.float32)            # [30, 3]

        n = feats.shape[0]
        # Drop too-short OR NaN-tainted windows — the proportional bones-seed tree has some
        # NaN-tainted files (kimodo's shape-aware dataset drops ~679 via a nan_audit); a NaN GT
        # window poisons the loss. Skip deterministically to the next index.
        if n < self.min_frames or not np.isfinite(feats).all() or not np.isfinite(nj).all():
            return self[(index + 1) % len(self)]

        feats = canonicalize_frame0(feats)                                     # window starts canonically
        feats = (feats - self.mean) / self.std                                 # configured-stat normalize
        nj = nj - nj.mean(axis=0, keepdims=True)                               # center (keep scale = size cue)

        return {
            "motion": torch.from_numpy(feats),          # [n, 283]
            "length": int(n),
            "text": entry.text,                         # raw caption (drop handled in train loop)
            "neutral_joints": torch.from_numpy(nj),     # [30, 3]
            "source": entry.source,
        }


def collate(batch: list) -> dict:
    """Pad to the batch-max length; motion_pad_mask True = padded frame."""
    lengths = [b["length"] for b in batch]
    max_t = int(max(lengths))
    bsz = len(batch)
    dim = batch[0]["motion"].shape[-1]
    motion = torch.zeros(bsz, max_t, dim, dtype=torch.float32)
    pad = torch.ones(bsz, max_t, dtype=torch.bool)   # True = pad
    for i, b in enumerate(batch):
        ln = b["length"]
        motion[i, :ln] = b["motion"]
        pad[i, :ln] = False
    return {
        "motion": motion,
        "motion_pad_mask": pad,
        "length": torch.tensor(lengths, dtype=torch.long),
        "text": [b["text"] for b in batch],
        "neutral_joints": torch.stack([b["neutral_joints"] for b in batch], dim=0),
        "source": [b["source"] for b in batch],
    }
