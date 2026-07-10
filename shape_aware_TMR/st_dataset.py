"""BONES-SEED (text, raw-joints motion, skeleton) dataset for the shape-aware TMR.

``ShapeTMRDataset`` subclasses kimodo's ``SOMABonesSeedDataset`` to inherit the in-memory
(text, segment) index over the three BONES-SEED text sources (natural/overview, single,
multi timeline), split restriction, windowing (``_resolve_segment``: <=10 s, random <=2 s
offset), and cache_index. Differences:

  * ``sources=(...)`` — only build/serve the listed pools (the parent raises if ANY of the
    3 pools is empty; test splits have ZERO multi entries, so this override is mandatory).
  * motion I/O: load the raw (unnormalized) uniego window from the PROPORTIONAL tree,
    ``decode_uniego.decode_joints`` -> raw posed_joints (T,30,3), then kimodo
    ``TMRMotionRep(posed_joints=..., to_canonicalize=True, to_normalize=True)`` -> (T,186).
    Canonicalization happens ONCE, inside the rep (same convention for GT and, later, for
    generated motions at eval time). Raw joints in, TAP-style features out.
  * per-actor ``neutral_joints (30,3)`` (centered — identical conditioning to the
    shape-aware generation model) is returned for the shape token.
  * train-time temporal jitter: both window boundaries jittered by +-``time_jitter_sec``
    (0.3 s like TAP's ``aug_time_jitter_sec``) in ``_resolve_segment``.

Collate emits ``mask`` with **True = VALID** (TAP/kimodo encoder convention — the OPPOSITE
of motion_expert/bs_dataset's ``motion_pad_mask`` where True = pad).

Runs in the ``kimodo`` env: PYTHONPATH=/home/jungbin_cho/kimodo_open.
"""
from __future__ import annotations

import csv
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from kimodo.data.soma_text_motion import SOMABonesSeedDataset, _BSEntry, _truthy

from decode_uniego import decode_joints

log = logging.getLogger(__name__)

# ---- defaults for this box ----
DATA_ROOT = "/mnt/shared/jungbin_cho/seed/soma_proportional_uniegomotion_20fps"
NATURAL_CSV = "/home/jungbin_cho/seed/metadata/seed_metadata_v004.csv"
TEMPORAL_JSONL = "/home/jungbin_cho/seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
MULTI_JSONL = "/home/jungbin_cho/seed/multi_timeline.jsonl"
SPLIT_DIR = "/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits"


def resample_joints_time(posed_joints: torch.Tensor, src_fps: float, dst_fps: float) -> torch.Tensor:
    """Linear time resample for raw joint positions.

    Accepts [B,T,J,3] or [T,J,3] and preserves the input rank.
    """
    if abs(float(src_fps) - float(dst_fps)) < 1e-6:
        return posed_joints
    squeeze = posed_joints.dim() == 3
    x = posed_joints.unsqueeze(0) if squeeze else posed_joints
    b, t, j, c = x.shape
    new_t = max(1, round(t * float(dst_fps) / float(src_fps)))
    y = x.reshape(b, t, j * c).permute(0, 2, 1)
    y = F.interpolate(y, size=new_t, mode="linear", align_corners=True)
    y = y.permute(0, 2, 1).reshape(b, new_t, j, c)
    return y.squeeze(0) if squeeze else y


class ShapeTMRDataset(SOMABonesSeedDataset):
    def __init__(
        self,
        split_path: str,
        motion_rep,                          # kimodo TMRMotionRep (CPU, with stats for normalize)
        sources=("natural", "single", "multi"),
        data_root: str = DATA_ROOT,
        natural_csv_path: str = NATURAL_CSV,
        temporal_labels_path: str = TEMPORAL_JSONL,
        multi_timeline_path: str = MULTI_JSONL,
        fps: int = 20,
        rep_fps: float | None = None,
        max_clip_sec: float = 10.0,
        max_segment_sec: float = 15.0,
        rand_offset_max_sec: float = 2.0,
        min_frames: int = 10,
        include_mirrored: bool = True,
        cache_index: str | None = None,
        time_jitter_sec: float = 0.0,        # train-time +- boundary jitter (sec); 0=off
        normalize: bool = True,              # z-score with the rep's stats (False for build_stats)
        train: bool = True,
        seed: int | None = 0,
        natural_desc_cols=("content_natural_desc_1", "content_natural_desc_2",
                           "content_natural_desc_3", "content_natural_desc_4"),
    ):
        # NOTE: the benchmark testsuite's OVERVIEW prompts are exactly `content_natural_desc_4`
        # (verified 400/400 string matches) — pass ("content_natural_desc_4",) to train the
        # natural pool on that style only.
        self.natural_desc_cols = tuple(natural_desc_cols)
        self.SOURCES = tuple(sources)        # instance attr shadows the class attr everywhere
        self.tmr_rep = motion_rep
        self.data_fps = float(fps)
        self.rep_fps = float(rep_fps if rep_fps is not None else fps)
        self.time_jitter_sec = float(time_jitter_sec)
        self.tmr_normalize = bool(normalize)
        self.train = bool(train)
        try:
            super().__init__(
                data_root=data_root,
                natural_csv_path=natural_csv_path,
                temporal_labels_path=temporal_labels_path,
                multi_timeline_path=multi_timeline_path,
                train_split_path=split_path,
                fps=fps,
                max_clip_sec=max_clip_sec,
                max_segment_sec=max_segment_sec,
                rand_offset_max_sec=(rand_offset_max_sec if train else 0.0),
                min_frames=min_frames,
                include_mirrored=include_mirrored,
                motion_rep=motion_rep,       # parent stores it; its Kimodo-rep paths are never used
                normalize=False,             # parent's (kimodo-rep) normalize machinery: OFF
                random_heading_aug=False,    # TMRMotionRep canonicalizes heading itself
                cache_index=cache_index,
                seed=seed,
            )
        except KeyError:
            # The parent's FINAL log line hardcodes all 3 pool names; with a subset `sources`
            # it KeyErrors AFTER all state (_pools/_pool_lens/_cycle_len) is set. Harmless —
            # but verify the state really is complete so a real failure can't slip through.
            if not (hasattr(self, "_pools") and hasattr(self, "_cycle_len")):
                raise
        log.info("ShapeTMRDataset: sources=%s pools=%s cycle_len=%d",
                 self.SOURCES, {s: len(self._pools[s]) for s in self.SOURCES}, self._cycle_len)

    # -- only build the pools for the requested sources (parent hardcodes all 3) --
    def _build_index(self, cache_index):
        cached = self._load_index_cache(cache_index)
        if cached is not None:
            return cached
        path_by_name = self._build_path_index()
        log.info("Path index: %d motions resolved.", len(path_by_name))
        pools = {s: [] for s in self.SOURCES}
        if "single" in pools:
            self._build_single_timeline_pool(pools["single"], path_by_name)
        if "multi" in pools:
            self._build_multi_timeline_pool(pools["multi"], path_by_name)
        if "natural" in pools:
            self._build_natural_pool(pools["natural"], path_by_name)
        if cache_index is not None:
            self._save_index_cache(cache_index, pools)
        return pools

    # -- natural-pool frame counts come from the uniego `features`, not `local_rot_mats` --
    def _build_natural_pool(self, out, path_by_name):
        desc_cols = self.natural_desc_cols
        meta_by_file: dict = {}
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
                try:
                    nframes = int(float(row["move_duration_frames"]))
                except (KeyError, TypeError, ValueError):
                    no_text += 1
                    continue
                meta_by_file[fname] = (descs, nframes)

        long_seg = short_seg = bad_read = 0
        for fname, (descs, n_total) in meta_by_file.items():
            p = path_by_name[fname]
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

    # -- train-time temporal jitter on BOTH boundaries (TAP's aug_time_jitter_sec) --
    def _resolve_segment(self, entry):
        sf, ef = super()._resolve_segment(entry)
        if self.train and self.time_jitter_sec > 0:
            j = int(round(self.time_jitter_sec * self.fps))       # 0.3 s @ 20 fps = 6 frames
            sf = sf + self._rng.randint(-j, j)
            ef = ef + self._rng.randint(-j, j)
            sf = max(0, sf)
            ef = max(sf + self.min_frames, ef)                    # keep a valid window
        return sf, ef

    def __len__(self) -> int:
        return len(self.SOURCES) * self._cycle_len

    def __getitem__(self, index: int) -> dict:
        source = self.SOURCES[index % len(self.SOURCES)]
        pool = self._pools[source]
        entry = pool[(index // len(self.SOURCES)) % len(pool)]

        sf, ef = self._resolve_segment(entry)
        with np.load(entry.motion_path, mmap_mode="r") as d:
            n_total = int(d["features"].shape[0])
            sf = max(0, min(sf, n_total))
            ef = max(sf, min(ef, n_total))
            ef = min(ef, sf + self.max_frames)
            feats_uniego = np.asarray(d["features"][sf:ef]).astype(np.float32)   # [T, 283] RAW
            nj = np.asarray(d["neutral_joints"]).astype(np.float32)              # [30, 3]

        T = feats_uniego.shape[0]
        # drop too-short or NaN-tainted windows (~0.4% of the proportional tree)
        if T < self.min_frames or not np.isfinite(feats_uniego).all() or not np.isfinite(nj).all():
            return self[(index + 1) % len(self)]

        with torch.no_grad():
            joints = decode_joints(torch.from_numpy(feats_uniego).unsqueeze(0))  # [1, T, 30, 3] RAW
            joints = resample_joints_time(joints, self.data_fps, self.rep_fps)
            T_rep = int(joints.shape[1])
            feats = self.tmr_rep(
                posed_joints=joints,
                to_normalize=self.tmr_normalize,
                to_canonicalize=True,               # ONE canonicalization convention, in the rep
                lengths=torch.tensor([T_rep]),
            )[0]                                    # [T, 186]
        if not torch.isfinite(feats).all():
            return self[(index + 1) % len(self)]

        nj = nj - nj.mean(axis=0, keepdims=True)    # center; scale = the size cue (as in generation)

        return {
            "features": feats.float(),              # [T, 186]
            "length": int(feats.shape[0]),
            "text": entry.text,
            "neutral_joints": torch.from_numpy(nj), # [30, 3]
            "source": entry.source,
        }


def collate_st(batch: list) -> dict:
    """Pad to batch-max; ``mask`` True = VALID frame (TAP/kimodo encoder convention)."""
    lengths = [b["length"] for b in batch]
    max_t = int(max(lengths))
    bsz = len(batch)
    dim = batch[0]["features"].shape[-1]
    feats = torch.zeros(bsz, max_t, dim, dtype=torch.float32)
    mask = torch.zeros(bsz, max_t, dtype=torch.bool)     # True = valid
    for i, b in enumerate(batch):
        n = b["length"]
        feats[i, :n] = b["features"]
        mask[i, :n] = True
    return {
        "features": feats,
        "mask": mask,
        "length": torch.tensor(lengths, dtype=torch.long),
        "text": [b["text"] for b in batch],
        "neutral_joints": torch.stack([b["neutral_joints"] for b in batch], dim=0),
        "source": [b["source"] for b in batch],
    }
