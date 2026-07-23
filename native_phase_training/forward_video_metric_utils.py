#!/usr/bin/env python
"""Shared contracts for advanced native Phase-1 forward-video metrics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from native_phase_training.evaluate_inverse_forward import (
    _base_name,
    _load_successful_output,
    _read_jsonl,
    _read_video_rgb,
    _resize_gt_like_native,
)


TOTAL_RGB_FRAMES = 97
CONDITIONED_RGB_FRAMES = 1
SUFFIX_RGB_INDICES = np.arange(CONDITIONED_RGB_FRAMES, TOTAL_RGB_FRAMES, dtype=np.int64)
HORIZON_SUFFIX_SLICES = {
    "early_frames_1_32": slice(0, 32),
    "middle_frames_33_64": slice(32, 64),
    "late_frames_65_96": slice(64, 96),
}


CDFVD_FRAME_INDICES = {
    "full_suffix_frames_1_96": np.arange(1, 97, dtype=np.int64),
    "early_frames_1_32": np.arange(1, 33, dtype=np.int64),
    "middle_frames_33_64": np.arange(33, 65, dtype=np.int64),
    "late_frames_65_96": np.arange(65, 97, dtype=np.int64),
}


@dataclass(frozen=True)
class ForwardVideoPair:
    name: str
    gt: np.ndarray
    generated: np.ndarray


def load_forward_records(
    eval_root: Path,
    expected_count: int,
    max_samples: int,
) -> list[dict[str, Any]]:
    records = _read_jsonl(eval_root / "fd_input.jsonl", "forward_dynamics")
    if expected_count and len(records) != expected_count:
        raise ValueError(f"expected {expected_count} forward records, got {len(records)}")
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    return records[:max_samples] if max_samples else records


def iter_forward_video_pairs(
    inference_root: Path,
    eval_root: Path,
    records: list[dict[str, Any]],
) -> Iterator[ForwardVideoPair]:
    for record in records:
        sample_dir, _ = _load_successful_output(inference_root, record)
        base_name = _base_name(record["name"], "forward_dynamics")
        gt_path = eval_root / "samples" / base_name / "gt_clip.mp4"
        generated_path = sample_dir / "vision.mp4"
        gt = _read_video_rgb(gt_path)
        generated = _read_video_rgb(generated_path)
        if len(gt) != TOTAL_RGB_FRAMES or len(generated) != TOTAL_RGB_FRAMES:
            raise ValueError(
                f"{base_name}: expected {TOTAL_RGB_FRAMES} frames, "
                f"got GT={len(gt)} generated={len(generated)}"
            )
        gt = _resize_gt_like_native(gt, generated.shape[1], generated.shape[2])
        if gt.shape != generated.shape:
            raise ValueError(f"{base_name}: GT/generated shape mismatch {gt.shape} vs {generated.shape}")
        if gt.dtype != np.uint8 or generated.dtype != np.uint8:
            raise ValueError(f"{base_name}: expected uint8 videos, got {gt.dtype} and {generated.dtype}")
        yield ForwardVideoPair(base_name, gt, generated)


def aggregate_scalars(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("metric aggregation requires a non-empty finite vector")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def records_fingerprint(records: list[dict[str, Any]]) -> str:
    names = [record["name"] for record in records]
    return hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
