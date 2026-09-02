#!/usr/bin/env python3
"""Fit and validate an absolute camera-origin lever for ``camhead_v1``.

``camhead_v1`` replaces the fitted SOMA Head orientation with the synchronized
upright-RGB camera orientation (after the train-fitted Head/camera axis change),
but it retains the historical lever that was estimated from *relative actions*.
That lever was never fit to the absolute optical-center positions.  Once the
Head orientation changes, rotating the old lever around the unchanged Head
joint can therefore increase absolute camera-origin error.

This script performs a separate, leakage-audited experiment:

1. reconstruct the exact T97 quality-filter population from the video manifest;
2. use only retained training windows to express ``p_camera - p_head`` in the
   corrected Head frame;
3. fit global frame-weighted, robust frame-weighted, sequence-balanced, and
   train-actor candidate levers;
4. evaluate absolute optical-center error and relative translation-action error
   on train and held-out test sequences; and
5. report a test-sequence oracle only as a clearly labeled diagnostic floor.

The motion corpus and historical calibration are never modified.  The emitted
experimental calibration keeps the existing train-only Head/camera rotation and
changes only ``camera_origin_in_head_m``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

try:
    from .audit_nymeria_camera_motion import _decode_uniego_head
    from .camera_head_recanonicalization import ARIA_Z_UP_TO_KIMODO_Y_UP
except ImportError:  # pragma: no cover - direct script execution
    from audit_nymeria_camera_motion import _decode_uniego_head
    from camera_head_recanonicalization import ARIA_Z_UP_TO_KIMODO_Y_UP


WEKA_ROOT = Path(os.environ.get("WEKA_ROOT", "/mnt/projects/ll/jungbinc/weka"))
DATA_ROOT = WEKA_ROOT / "nymeriaplus_kimodo_proportional"
RUN_ROOT = Path(os.environ.get("RUN_ROOT", WEKA_ROOT / "cosmos_motion_ft_runs"))
DEFAULT_MOTION_ROOT = DATA_ROOT / "uniego_rep_camhead_v1"
DEFAULT_ORIGINAL_MOTION_ROOT = DATA_ROOT / "uniego_rep"
DEFAULT_CAMERA_ROOT = DATA_ROOT / "camera_rgb"
DEFAULT_MANIFEST = DATA_ROOT / "video/manifest_video.jsonl"
DEFAULT_SPLIT = DATA_ROOT / "train_test_split.json"
DEFAULT_QUALITY_FILTER = DATA_ROOT / "metadata/camera_motion_quality_filter_v1_T97.json"
DEFAULT_ROTATION_CALIBRATION = Path(__file__).with_name("head_camera_calibration_train.json")
DEFAULT_REFERENCE_REPORT = (
    RUN_ROOT
    / "nymeria_camera_head_recanonicalization_v1/quantitative/comparison_report.json"
)
DEFAULT_OUTPUT = (
    RUN_ROOT
    / "nymeria_camera_head_recanonicalization_v1/absolute_lever_refit"
)
DEFAULT_QUALITATIVE_MANIFEST = (
    RUN_ROOT
    / "nymeria_camera_head_recanonicalization_v1/qualitative/gallery_manifest.json"
)

ABSOLUTE_THRESHOLDS_M = (0.005, 0.01, 0.02, 0.05, 0.10)
RELATIVE_THRESHOLDS_M = (0.001, 0.005, 0.01, 0.05)
ROTATION_THRESHOLDS_DEG = (0.01, 0.05, 0.1, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def geometric_median(
    points: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the Euclidean geometric median using deterministic Weiszfeld steps."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"points must be non-empty [N,3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("geometric-median points contain non-finite values")
    estimate = np.median(points, axis=0)
    converged = False
    movement = math.inf
    for iteration in range(max_iterations):
        distances = np.linalg.norm(points - estimate, axis=1)
        coincident = distances <= tolerance
        if np.any(coincident):
            # A coincident point is the optimum only when the remaining unit-vector
            # resultant has norm <= the coincident multiplicity.  Otherwise use the
            # modified Weiszfeld step, which avoids division by zero.
            others = ~coincident
            if not np.any(others):
                converged = True
                movement = 0.0
                break
            resultant = np.sum(
                (points[others] - estimate) / distances[others, None], axis=0
            )
            if np.linalg.norm(resultant) <= int(coincident.sum()):
                converged = True
                movement = 0.0
                break
            distances = np.maximum(distances, tolerance)
        weights = 1.0 / distances
        updated = np.sum(points * weights[:, None], axis=0) / weights.sum()
        movement = float(np.linalg.norm(updated - estimate))
        estimate = updated
        if movement <= tolerance:
            converged = True
            break
    return estimate, {
        "points": int(len(points)),
        "iterations": int(iteration + 1),
        "converged": bool(converged),
        "final_movement_m": float(movement),
        "tolerance_m": float(tolerance),
        "max_iterations": int(max_iterations),
    }


def _rotation_angle_deg(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def _relative_translation(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    inverse = np.swapaxes(rotation[:-1], -1, -2)
    return np.einsum(
        "tij,tj->ti", inverse, position[1:] - position[:-1], optimize=True
    )


def _clean_mask(length: int, windows: list[tuple[int, int]]) -> np.ndarray:
    """Union half-open windows into one frame mask without double-weighting overlap."""
    delta = np.zeros(length + 1, dtype=np.int32)
    for start, end in windows:
        start = max(0, int(start))
        end = min(length, int(end))
        if end <= start:
            continue
        delta[start] += 1
        delta[end] -= 1
    return np.cumsum(delta[:-1]) > 0


def _discover_clean_windows(
    manifest_path: Path,
    split_path: Path,
    quality_filter_path: Path,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, Any]]:
    """Rebuild and validate the quality-filter's exact physical T97 population."""
    split_payload = json.loads(split_path.read_text())
    split_for = {
        uuid: split
        for split in ("train", "test")
        for uuid in split_payload[split]
    }
    quality = json.loads(quality_filter_path.read_text())
    if quality.get("kind") != "nymeria_camera_motion_quality_filter":
        raise ValueError(f"unexpected quality-filter kind in {quality_filter_path}")
    num_frames = int(quality["num_frames"])
    multiplicity: Counter[tuple[str, str, int, int]] = Counter()
    for record in _read_jsonl(manifest_path):
        uuid = record.get("uuid")
        split = split_for.get(uuid)
        if split is None or not record.get("camera_path") or not record.get("vision_path"):
            continue
        frame_count = int(record.get("nb_frames", 0))
        for caption_window in record.get("t2w_windows", []):
            if not caption_window.get("usable", False) or not caption_window.get("caption"):
                continue
            start = int(caption_window["start_frame"])
            end = min(int(caption_window["end_frame"]), frame_count)
            while start + num_frames <= end:
                multiplicity[(split, uuid, start, start + num_frames)] += 1
                start += num_frames

    excluded = {
        (
            str(row["split"]),
            str(row["uuid"]),
            int(row["start"]),
            int(row["end"]),
        )
        for row in quality["excluded_windows"]
    }
    unknown_exclusions = excluded - set(multiplicity)
    if unknown_exclusions:
        raise ValueError(
            f"quality filter contains {len(unknown_exclusions)} unknown physical windows"
        )

    by_uuid: dict[str, list[tuple[int, int]]] = defaultdict(list)
    observed: dict[str, Any] = {}
    for split in ("train", "test"):
        keys = [key for key in multiplicity if key[0] == split]
        kept = [key for key in keys if key not in excluded]
        observed[split] = {
            "input_dataset_rows": int(sum(multiplicity[key] for key in keys)),
            "input_unique_physical_windows": int(len(keys)),
            "excluded_dataset_rows": int(
                sum(multiplicity[key] for key in keys if key in excluded)
            ),
            "excluded_unique_physical_windows": int(sum(key in excluded for key in keys)),
            "kept_dataset_rows": int(sum(multiplicity[key] for key in kept)),
            "kept_unique_physical_windows": int(len(kept)),
        }
        expected = quality["summary_by_split"][split]
        for key, value in observed[split].items():
            if int(expected[key]) != int(value):
                raise ValueError(
                    f"quality-filter population mismatch for {split}.{key}: "
                    f"observed={value} expected={expected[key]}"
                )
        for _, uuid, start, end in kept:
            by_uuid[uuid].append((start, end))

    for uuid in by_uuid:
        by_uuid[uuid] = sorted(set(by_uuid[uuid]))
    return dict(by_uuid), {
        "num_frames": num_frames,
        "observed_counts": observed,
        "quality_filter_kind": quality["kind"],
        "quality_filter_version": quality.get("version"),
        "quality_filter_sha256": _sha256(quality_filter_path),
        "duplicate_windows_are_unioned_for_frame_metrics": True,
    }


def _load_sequence(
    uuid: str,
    motion_root: Path,
    camera_root: Path,
) -> dict[str, np.ndarray]:
    subject, sequence = uuid.split("/", 1)
    motion_path = motion_root / subject / f"{sequence}.npz"
    camera_path = camera_root / subject / f"{sequence}.npz"
    with np.load(motion_path, allow_pickle=False) as motion:
        features = motion["features"].astype(np.float64)
        motion_timestamps = motion["timestamps_us"]
    with np.load(camera_path, allow_pickle=False) as camera:
        camera_position = camera["cam_world_pos_upright"].astype(np.float64)
        camera_rotation = camera["cam_world_rot_upright"].astype(np.float64)
        camera_action = camera["cam_action_upright_k1"].astype(np.float64)
        camera_timestamps = camera["timestamps_us"]
    if not np.array_equal(motion_timestamps, camera_timestamps):
        raise ValueError(f"timestamp mismatch for {uuid}")
    head_rotation, head_position, delta_off_axis = _decode_uniego_head(features)
    if len(head_position) != len(camera_position) or len(camera_action) != len(head_position) - 1:
        raise ValueError(
            f"length mismatch for {uuid}: head={len(head_position)} "
            f"camera={len(camera_position)} action={len(camera_action)}"
        )
    camera_position = np.einsum(
        "ij,tj->ti", ARIA_Z_UP_TO_KIMODO_Y_UP, camera_position, optimize=True
    )
    camera_rotation = np.einsum(
        "ij,tjk->tik", ARIA_Z_UP_TO_KIMODO_Y_UP, camera_rotation, optimize=True
    )
    return {
        "head_rotation": head_rotation,
        "head_position": head_position,
        "camera_rotation": camera_rotation,
        "camera_position": camera_position,
        "camera_action": camera_action,
        "delta_off_axis_max": np.asarray(delta_off_axis, dtype=np.float64),
    }


def _fit_one(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        uuid,
        split,
        windows,
        motion_root_text,
        camera_root_text,
        rotation_values,
        fit_stride,
    ) = payload
    data = _load_sequence(uuid, Path(motion_root_text), Path(camera_root_text))
    head_rotation = data["head_rotation"]
    head_position = data["head_position"]
    camera_rotation = data["camera_rotation"]
    camera_position = data["camera_position"]
    clean = _clean_mask(len(head_position), windows)
    clean_indices = np.flatnonzero(clean)
    if len(clean_indices) == 0:
        raise ValueError(f"{uuid}: no retained quality-filter frames")
    offset_world = camera_position - head_position
    offsets = np.einsum(
        "tji,tj->ti", head_rotation, offset_world, optimize=True
    )
    sampled_indices = clean_indices[::fit_stride]
    sampled = offsets[sampled_indices]
    center, center_fit = geometric_median(sampled)
    rotation_head_to_camera = np.asarray(rotation_values, dtype=np.float64)
    implied_camera_rotation = head_rotation[sampled_indices] @ rotation_head_to_camera
    rotation_error = _rotation_angle_deg(
        np.swapaxes(implied_camera_rotation, -1, -2)
        @ camera_rotation[sampled_indices]
    )
    return {
        "uuid": uuid,
        "actor": uuid.split("/", 1)[0],
        "split": split,
        "frames": int(len(head_position)),
        "clean_frames": int(len(clean_indices)),
        "fit_samples": int(len(sampled)),
        "offset_sum": offsets[clean].sum(axis=0),
        "sequence_geometric_median": center,
        "sequence_fit": center_fit,
        "sampled_offsets": sampled,
        "head_camera_rotation_error_mean_deg": float(rotation_error.mean()),
        "head_camera_rotation_error_max_deg": float(rotation_error.max()),
        "head_camera_separation_mean_m": float(np.linalg.norm(offset_world[clean], axis=-1).mean()),
        "delta_off_axis_max": float(data["delta_off_axis_max"]),
    }


@dataclass
class MetricAccumulator:
    thresholds: tuple[float, ...]
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    threshold_above: dict[float, int] = field(default_factory=dict)
    samples: list[np.ndarray] = field(default_factory=list)

    def update(self, values: np.ndarray, *, sample_stride: int) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("metric contains non-finite values")
        if len(values) == 0:
            return
        self.count += int(len(values))
        self.total += float(values.sum(dtype=np.float64))
        self.total_square += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        for threshold in self.thresholds:
            self.threshold_above[threshold] = self.threshold_above.get(threshold, 0) + int(
                np.count_nonzero(values > threshold)
            )
        self.samples.append(values[::sample_stride].astype(np.float32))

    def merge_state(self, state: dict[str, Any]) -> None:
        if int(state["count"]) == 0:
            return
        self.count += int(state["count"])
        self.total += float(state["total"])
        self.total_square += float(state["total_square"])
        self.minimum = min(self.minimum, float(state["minimum"]))
        self.maximum = max(self.maximum, float(state["maximum"]))
        for key, value in state["threshold_above"].items():
            threshold = float(key)
            self.threshold_above[threshold] = self.threshold_above.get(threshold, 0) + int(value)
        sample = np.asarray(state["samples"], dtype=np.float32)
        if len(sample):
            self.samples.append(sample)

    def state(self) -> dict[str, Any]:
        sample = np.concatenate(self.samples) if self.samples else np.empty(0, np.float32)
        return {
            "count": int(self.count),
            "total": float(self.total),
            "total_square": float(self.total_square),
            "minimum": None if self.count == 0 else float(self.minimum),
            "maximum": None if self.count == 0 else float(self.maximum),
            "threshold_above": {str(k): int(v) for k, v in self.threshold_above.items()},
            "samples": sample,
        }

    def summary(self, *, sample_stride: int) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        samples = np.concatenate(self.samples).astype(np.float64)
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": int(self.count),
            "mean_exact": float(mean),
            "rmse_exact": float(math.sqrt(self.total_square / self.count)),
            "std_exact": float(math.sqrt(variance)),
            "min_exact": float(self.minimum),
            "max_exact": float(self.maximum),
            "quantile_sample_stride_per_sequence": int(sample_stride),
            "quantile_sample_count": int(len(samples)),
            "median_sampled": float(np.median(samples)),
            "p90_sampled": float(np.quantile(samples, 0.90)),
            "p95_sampled": float(np.quantile(samples, 0.95)),
            "p99_sampled": float(np.quantile(samples, 0.99)),
            "threshold_counts_exact": {
                str(threshold): {
                    "above": int(self.threshold_above.get(threshold, 0)),
                    "fraction_above": float(
                        self.threshold_above.get(threshold, 0) / self.count
                    ),
                }
                for threshold in self.thresholds
            },
        }


def _metric_state(
    values: np.ndarray,
    thresholds: tuple[float, ...],
    sample_stride: int,
) -> dict[str, Any]:
    accumulator = MetricAccumulator(thresholds)
    accumulator.update(values, sample_stride=sample_stride)
    return accumulator.state()


def _metric_one(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        uuid,
        split,
        windows,
        motion_root_text,
        original_motion_root_text,
        camera_root_text,
        rotation_values,
        fixed_levers,
        actor_lever_values,
        sequence_oracle_values,
        sample_stride,
    ) = payload
    data = _load_sequence(uuid, Path(motion_root_text), Path(camera_root_text))
    original_data = _load_sequence(
        uuid, Path(original_motion_root_text), Path(camera_root_text)
    )
    head_rotation = data["head_rotation"]
    head_position = data["head_position"]
    measured_rotation = data["camera_rotation"]
    measured_position = data["camera_position"]
    target_translation = data["camera_action"][:, :3]
    rotation_head_to_camera = np.asarray(rotation_values, dtype=np.float64)
    predicted_rotation = head_rotation @ rotation_head_to_camera
    corrected_rotation_error = _rotation_angle_deg(
        np.swapaxes(predicted_rotation, -1, -2) @ measured_rotation
    )
    pose_translation = _relative_translation(measured_rotation, measured_position)
    pose_action_reproduction = np.linalg.norm(
        pose_translation - target_translation, axis=-1
    )
    clean = _clean_mask(len(head_position), windows)
    clean_transition = clean[:-1] & clean[1:]
    cohorts = {
        "all": (np.ones(len(clean), dtype=bool), np.ones(len(clean) - 1, dtype=bool)),
        "quality_filter_clean": (clean, clean_transition),
    }
    candidate_levers = {
        name: np.asarray(value, dtype=np.float64) for name, value in fixed_levers.items()
    }
    candidate_levers["absolute_train_actor_geomedian"] = np.asarray(
        actor_lever_values, dtype=np.float64
    )
    candidate_levers["absolute_sequence_oracle"] = np.asarray(
        sequence_oracle_values, dtype=np.float64
    )
    candidate_specs = {
        name: (head_rotation, head_position, lever)
        for name, lever in candidate_levers.items()
    }
    candidate_specs["original_representation_historical_lever"] = (
        original_data["head_rotation"],
        original_data["head_position"],
        candidate_levers["historical_relative_lever"],
    )
    metrics: dict[str, Any] = {}
    per_sequence: dict[str, Any] = {
        "uuid": uuid,
        "actor": uuid.split("/", 1)[0],
        "split": split,
        "frames": int(len(head_position)),
        "quality_filter_clean_frames": int(clean.sum()),
        "quality_filter_clean_transitions": int(clean_transition.sum()),
        "candidates": {},
    }
    for name, (candidate_head_rotation, candidate_head_position, lever) in (
        candidate_specs.items()
    ):
        candidate_camera_rotation = candidate_head_rotation @ rotation_head_to_camera
        candidate_rotation_error = _rotation_angle_deg(
            np.swapaxes(candidate_camera_rotation, -1, -2) @ measured_rotation
        )
        predicted_position = candidate_head_position + np.einsum(
            "tij,j->ti", candidate_head_rotation, lever, optimize=True
        )
        absolute_error = np.linalg.norm(
            predicted_position - measured_position, axis=-1
        )
        predicted_translation = _relative_translation(
            candidate_camera_rotation, predicted_position
        )
        relative_error = np.linalg.norm(
            predicted_translation - target_translation, axis=-1
        )
        per_sequence["candidates"][name] = {
            "lever_m": lever.tolist(),
            "absolute_clean_mean_m": float(absolute_error[clean].mean()),
            "absolute_clean_rmse_m": float(
                np.sqrt(np.mean(np.square(absolute_error[clean])))
            ),
            "relative_clean_mean_m": float(relative_error[clean_transition].mean()),
            "relative_clean_rmse_m": float(
                np.sqrt(np.mean(np.square(relative_error[clean_transition])))
            ),
            "rotation_clean_mean_deg": float(candidate_rotation_error[clean].mean()),
            "rotation_clean_rmse_deg": float(
                np.sqrt(np.mean(np.square(candidate_rotation_error[clean])))
            ),
        }
        metrics[name] = {}
        for cohort, (frame_mask, transition_mask) in cohorts.items():
            metrics[name][cohort] = {
                "absolute_translation_m": _metric_state(
                    absolute_error[frame_mask], ABSOLUTE_THRESHOLDS_M, sample_stride
                ),
                "relative_translation_m": _metric_state(
                    relative_error[transition_mask], RELATIVE_THRESHOLDS_M, sample_stride
                ),
                "camera_rotation_deg": _metric_state(
                    candidate_rotation_error[frame_mask],
                    ROTATION_THRESHOLDS_DEG,
                    sample_stride,
                ),
            }
    common = {}
    for cohort, (frame_mask, transition_mask) in cohorts.items():
        common[cohort] = {
            "head_camera_rotation_deg": _metric_state(
                corrected_rotation_error[frame_mask],
                ROTATION_THRESHOLDS_DEG,
                sample_stride,
            ),
            "stored_pose_action_translation_reproduction_m": _metric_state(
                pose_action_reproduction[transition_mask], RELATIVE_THRESHOLDS_M, sample_stride
            ),
        }
    return {
        "uuid": uuid,
        "split": split,
        "metrics": metrics,
        "common": common,
        "per_sequence": per_sequence,
    }


def _summarize_per_sequence(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    absolute = np.asarray(
        [row["candidates"][candidate]["absolute_clean_mean_m"] for row in rows],
        dtype=np.float64,
    )
    relative = np.asarray(
        [row["candidates"][candidate]["relative_clean_mean_m"] for row in rows],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [row["candidates"][candidate]["rotation_clean_mean_deg"] for row in rows],
        dtype=np.float64,
    )
    return {
        "sequences": int(len(rows)),
        "absolute_translation_sequence_mean_m": {
            "mean": float(absolute.mean()),
            "median": float(np.median(absolute)),
            "p90": float(np.quantile(absolute, 0.90)),
            "max": float(absolute.max()),
        },
        "relative_translation_sequence_mean_m": {
            "mean": float(relative.mean()),
            "median": float(np.median(relative)),
            "p90": float(np.quantile(relative, 0.90)),
            "max": float(relative.max()),
        },
        "camera_rotation_sequence_mean_deg": {
            "mean": float(rotation.mean()),
            "median": float(np.median(rotation)),
            "p90": float(np.quantile(rotation, 0.90)),
            "max": float(rotation.max()),
        },
    }


def _plot_results(
    output: Path,
    aggregate_internal: dict[str, Any],
    fit_rows: list[dict[str, Any]],
    levers: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = (
        "original_representation_historical_lever",
        "historical_relative_lever",
        "absolute_global_frame_geomedian",
        "absolute_train_actor_geomedian",
        "absolute_sequence_oracle",
    )
    labels = {
        "original_representation_historical_lever": "original Head + historical lever",
        "historical_relative_lever": "historical relative lever",
        "absolute_global_frame_geomedian": "train-global absolute lever",
        "absolute_train_actor_geomedian": "train-actor absolute lever",
        "absolute_sequence_oracle": "test-sequence oracle (leaky)",
    }
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for name in selected:
        state = aggregate_internal["test"][name]["quality_filter_clean"][
            "absolute_translation_m"
        ]
        values = np.concatenate(state.samples).astype(np.float64)
        values = np.sort(values)
        cdf = np.arange(1, len(values) + 1) / len(values)
        axis.plot(values * 100.0, cdf, lw=2, label=labels[name])
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.set_xlabel("absolute optical-center error [cm]")
    axis.set_ylabel("held-out clean-frame CDF")
    axis.set_title("Corrected Head frame: camera-origin lever comparison")
    axis.legend(fontsize=8)
    fig.savefig(output / "heldout_clean_absolute_translation_cdf.png", dpi=180)
    plt.close(fig)

    candidates = list(selected)
    absolute_means = [
        aggregate_internal["test"][name]["quality_filter_clean"][
            "absolute_translation_m"
        ].summary(sample_stride=1)["mean_exact"]
        * 100.0
        for name in candidates
    ]
    relative_means = [
        aggregate_internal["test"][name]["quality_filter_clean"][
            "relative_translation_m"
        ].summary(sample_stride=1)["mean_exact"]
        * 1000.0
        for name in candidates
    ]
    rotation_means = [
        aggregate_internal["test"][name]["quality_filter_clean"][
            "camera_rotation_deg"
        ].summary(sample_stride=1)["mean_exact"]
        for name in candidates
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    x = np.arange(len(candidates))
    short = [
        "original",
        "corrected +\nhistorical",
        "corrected +\nglobal",
        "corrected +\ntrain actor",
        "corrected +\ntest oracle",
    ]
    colors = ("tab:purple", "0.55", "tab:green", "tab:blue", "tab:orange")
    axes[0].bar(x, absolute_means, color=colors)
    axes[0].set_xticks(x, short, rotation=18)
    axes[0].set_ylabel("mean absolute error [cm]")
    axes[0].set_title("Optical-center location")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, relative_means, color=colors)
    axes[1].set_xticks(x, short, rotation=18)
    axes[1].set_ylabel("mean relative-action error [mm]")
    axes[1].set_title("Frame-to-frame translation")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, rotation_means, color=colors)
    axes[2].set_xticks(x, short, rotation=18)
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-3, 30.0)
    axes[2].set_ylabel("mean rotation error [deg, log scale]")
    axes[2].set_title("Camera orientation")
    axes[2].grid(axis="y", alpha=0.25)
    for index, value in enumerate(rotation_means):
        axes[2].text(
            index,
            value * 1.15,
            f"{value:.4g}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.suptitle("Held-out quality-filter-clean comparison")
    fig.savefig(output / "heldout_clean_absolute_vs_relative.png", dpi=180)
    plt.close(fig)

    train_centers = np.asarray(
        [row["sequence_geometric_median"] for row in fit_rows if row["split"] == "train"]
    )
    test_centers = np.asarray(
        [row["sequence_geometric_median"] for row in fit_rows if row["split"] == "test"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, dims, names in (
        (axes[0], (0, 2), ("x", "z")),
        (axes[1], (1, 2), ("y", "z")),
    ):
        axis.scatter(
            train_centers[:, dims[0]] * 100,
            train_centers[:, dims[1]] * 100,
            s=9,
            alpha=0.35,
            label="train sequence centers",
        )
        axis.scatter(
            test_centers[:, dims[0]] * 100,
            test_centers[:, dims[1]] * 100,
            s=14,
            alpha=0.55,
            label="test sequence centers",
        )
        for name, color in (
            ("historical_relative_lever", "black"),
            ("absolute_global_frame_geomedian", "tab:green"),
            ("absolute_global_sequence_geomedian", "tab:red"),
        ):
            value = levers[name]
            axis.scatter(
                [value[dims[0]] * 100],
                [value[dims[1]] * 100],
                marker="X",
                s=100,
                color=color,
                label=name,
            )
        axis.set_xlabel(f"corrected-Head {names[0]} [cm]")
        axis.set_ylabel(f"corrected-Head {names[1]} [cm]")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-sequence optical-center lever centers")
    fig.savefig(output / "sequence_lever_centers.png", dpi=180)
    plt.close(fig)


def _plot_qualitative_lever_comparison(
    output: Path,
    manifest_path: Path,
    original_motion_root: Path,
    corrected_motion_root: Path,
    camera_root: Path,
    historical_lever: np.ndarray,
    selected_lever: np.ndarray,
) -> dict[str, Any] | None:
    """Overlay the two lever choices on the existing diverse held-out gallery."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not manifest_path.is_file():
        return None
    gallery = json.loads(manifest_path.read_text())
    cases = gallery.get("cases", [])
    if not cases:
        return None
    columns = 3
    rows = int(math.ceil(len(cases) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 4.5 * rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    case_summaries = []
    for axis, case in zip(axes, cases):
        uuid = str(case["uuid"])
        original = _load_sequence(uuid, original_motion_root, camera_root)
        corrected = _load_sequence(uuid, corrected_motion_root, camera_root)
        start = int(case["start"])
        end = min(
            start + int(case.get("frames", 97)),
            len(corrected["head_position"]),
        )
        selector = slice(start, end)
        measured = corrected["camera_position"][selector]
        original_position = original["head_position"][selector] + np.einsum(
            "tij,j->ti",
            original["head_rotation"][selector],
            historical_lever,
            optimize=True,
        )
        corrected_historical = corrected["head_position"][selector] + np.einsum(
            "tij,j->ti",
            corrected["head_rotation"][selector],
            historical_lever,
            optimize=True,
        )
        corrected_refit = corrected["head_position"][selector] + np.einsum(
            "tij,j->ti",
            corrected["head_rotation"][selector],
            selected_lever,
            optimize=True,
        )
        named_curves = (
            ("measured camera", measured, "black", "-", 2.3),
            ("original Head + historical lever", original_position, "tab:purple", "-", 1.5),
            ("corrected Head + historical lever", corrected_historical, "0.5", "--", 1.5),
            ("corrected Head + refit lever", corrected_refit, "tab:green", "-", 1.8),
        )
        origin = measured[0]
        centered = []
        for label, curve, color, style, width in named_curves:
            relative = curve - origin
            centered.append(relative)
            axis.plot(
                relative[:, 0],
                relative[:, 2],
                color=color,
                ls=style,
                lw=width,
                label=label,
            )
        axis.scatter(0.0, 0.0, color="black", marker="o", s=18, zorder=5)
        horizontal = np.concatenate([curve[:, (0, 2)] for curve in centered], axis=0)
        center = (horizontal.min(axis=0) + horizontal.max(axis=0)) * 0.5
        radius = max(float(np.ptp(horizontal, axis=0).max()) * 0.56, 0.08)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.22)
        axis.set_xlabel("world x relative to measured frame 0 [m]")
        axis.set_ylabel("world z relative to measured frame 0 [m]")
        errors_cm = {
            "original": float(
                np.linalg.norm(original_position - measured, axis=-1).mean() * 100.0
            ),
            "corrected_historical": float(
                np.linalg.norm(corrected_historical - measured, axis=-1).mean()
                * 100.0
            ),
            "corrected_refit": float(
                np.linalg.norm(corrected_refit - measured, axis=-1).mean() * 100.0
            ),
        }
        axis.set_title(
            f"{case['category']}\n{uuid}@{start}\n"
            f"mean absolute error: {errors_cm['original']:.2f} / "
            f"{errors_cm['corrected_historical']:.2f} / "
            f"{errors_cm['corrected_refit']:.2f} cm",
            fontsize=9,
        )
        case_summaries.append(
            {
                "category": str(case["category"]),
                "uuid": uuid,
                "start": start,
                "end": end,
                "mean_absolute_error_cm": errors_cm,
            }
        )
    for axis in axes[len(cases):]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.002, 0.5),
        ncol=1,
        fontsize=9,
    )
    fig.suptitle(
        "Diverse held-out trajectories: original, corrected-historical, and corrected-refit",
        y=1.01,
        fontsize=16,
    )
    path = output / "diverse_absolute_lever_trajectory_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "source_gallery_manifest": str(manifest_path.resolve()),
        "source_gallery_manifest_sha256": _sha256(manifest_path),
        "cases": int(len(case_summaries)),
        "case_summaries": case_summaries,
        "trajectory_comparison": str(path.resolve()),
        "unchanged_human_motion_keyframes": gallery.get("artifacts", {}).get(
            "human_motion_keyframes"
        ),
        "unchanged_motion_video_directory": gallery.get("artifacts", {}).get(
            "video_directory"
        ),
    }


def _reference_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    out: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "contract": "existing exhaustive camhead_v1 comparison; copied for context only",
        "groups": {},
    }
    for group in ("all", "train", "test"):
        metrics = payload["groups"][group]["metrics"]
        out["groups"][group] = {
            name: {
                "mean_exact": metrics[name]["mean_exact"],
                "rmse_exact": metrics[name]["rmse_exact"],
                "median_sampled": metrics[name]["median_sampled"],
                "p90_sampled": metrics[name]["p90_sampled"],
            }
            for name in (
                "absolute_camera_translation_old_m",
                "absolute_camera_translation_new_m",
                "camera_action_translation_old_m",
                "camera_action_translation_new_m",
            )
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION_ROOT)
    parser.add_argument(
        "--original-motion-root", type=Path, default=DEFAULT_ORIGINAL_MOTION_ROOT
    )
    parser.add_argument("--camera-root", type=Path, default=DEFAULT_CAMERA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--quality-filter", type=Path, default=DEFAULT_QUALITY_FILTER)
    parser.add_argument(
        "--rotation-calibration", type=Path, default=DEFAULT_ROTATION_CALIBRATION
    )
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT)
    parser.add_argument(
        "--qualitative-manifest", type=Path, default=DEFAULT_QUALITATIVE_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fit-stride", type=int, default=20)
    parser.add_argument("--percentile-stride", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.fit_stride < 1 or args.percentile_stride < 1:
        raise SystemExit("workers and strides must be positive")
    output = args.output_dir
    report_path = output / "absolute_lever_report.json"
    if report_path.exists() and not args.overwrite:
        raise SystemExit(f"{report_path} exists; pass --overwrite to replace this experiment")
    output.mkdir(parents=True, exist_ok=True)

    split_payload = json.loads(args.split_file.read_text())
    clean_windows, window_contract = _discover_clean_windows(
        args.manifest, args.split_file, args.quality_filter
    )
    calibration = json.loads(args.rotation_calibration.read_text())
    if calibration.get("split") != "train":
        raise ValueError("rotation calibration must declare split='train'")
    rotation_head_to_camera = np.asarray(
        calibration["rotation_head_to_upright_camera"], dtype=np.float64
    )
    historical_lever = np.asarray(
        calibration["camera_origin_in_head_m"], dtype=np.float64
    )

    sequence_rows = [
        (uuid, split)
        for split in ("train", "test")
        for uuid in split_payload[split]
        if uuid in clean_windows
    ]
    fit_payloads = [
        (
            uuid,
            split,
            clean_windows[uuid],
            str(args.motion_root),
            str(args.camera_root),
            rotation_head_to_camera.tolist(),
            args.fit_stride,
        )
        for uuid, split in sequence_rows
    ]
    print(
        f"[absolute-lever] fit pass: {len(fit_payloads)} sequences, workers={args.workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        fit_rows = list(executor.map(_fit_one, fit_payloads, chunksize=1))
    fit_rows.sort(key=lambda row: row["uuid"])
    train_rows = [row for row in fit_rows if row["split"] == "train"]
    test_rows = [row for row in fit_rows if row["split"] == "test"]
    if len(train_rows) != len(split_payload["train"]) or len(test_rows) != len(
        split_payload["test"]
    ):
        raise ValueError(
            f"fit coverage mismatch train={len(train_rows)}/{len(split_payload['train'])} "
            f"test={len(test_rows)}/{len(split_payload['test'])}"
        )

    train_samples = np.concatenate([row["sampled_offsets"] for row in train_rows])
    frame_geomedian, frame_geomedian_fit = geometric_median(train_samples)
    sequence_centers = np.stack(
        [row["sequence_geometric_median"] for row in train_rows]
    )
    sequence_geomedian, sequence_geomedian_fit = geometric_median(sequence_centers)
    exact_offset_sum = np.sum(
        np.stack([row["offset_sum"] for row in train_rows]), axis=0
    )
    exact_offset_count = sum(row["clean_frames"] for row in train_rows)
    frame_mean = exact_offset_sum / exact_offset_count

    actor_samples: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in train_rows:
        actor_samples[row["actor"]].append(row["sampled_offsets"])
    actor_levers: dict[str, np.ndarray] = {}
    actor_fit: dict[str, Any] = {}
    for actor in sorted(actor_samples):
        actor_levers[actor], actor_fit[actor] = geometric_median(
            np.concatenate(actor_samples[actor])
        )

    fixed_levers = {
        "historical_relative_lever": historical_lever,
        "absolute_global_frame_mean": frame_mean,
        "absolute_global_frame_geomedian": frame_geomedian,
        "absolute_global_sequence_geomedian": sequence_geomedian,
    }
    sequence_oracles = {
        row["uuid"]: np.asarray(row["sequence_geometric_median"], dtype=np.float64)
        for row in fit_rows
    }
    metric_payloads = [
        (
            uuid,
            split,
            clean_windows[uuid],
            str(args.motion_root),
            str(args.original_motion_root),
            str(args.camera_root),
            rotation_head_to_camera.tolist(),
            {name: value.tolist() for name, value in fixed_levers.items()},
            actor_levers.get(uuid.split("/", 1)[0], frame_geomedian).tolist(),
            sequence_oracles[uuid].tolist(),
            args.percentile_stride,
        )
        for uuid, split in sequence_rows
    ]
    print(
        f"[absolute-lever] metric pass: {len(metric_payloads)} sequences, workers={args.workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        metric_results = list(executor.map(_metric_one, metric_payloads, chunksize=1))
    metric_results.sort(key=lambda row: row["uuid"])

    candidate_names = ["original_representation_historical_lever"] + list(fixed_levers) + (
        ["absolute_train_actor_geomedian", "absolute_sequence_oracle"]
    )
    aggregate_internal: dict[str, Any] = {}
    common_internal: dict[str, Any] = {}
    for split in ("train", "test", "all"):
        aggregate_internal[split] = {}
        common_internal[split] = {}
        selected = [
            result
            for result in metric_results
            if split == "all" or result["split"] == split
        ]
        for name in candidate_names:
            aggregate_internal[split][name] = {}
            for cohort in ("all", "quality_filter_clean"):
                aggregate_internal[split][name][cohort] = {}
                for metric_name, thresholds in (
                    ("absolute_translation_m", ABSOLUTE_THRESHOLDS_M),
                    ("relative_translation_m", RELATIVE_THRESHOLDS_M),
                    ("camera_rotation_deg", ROTATION_THRESHOLDS_DEG),
                ):
                    accumulator = MetricAccumulator(thresholds)
                    for result in selected:
                        accumulator.merge_state(
                            result["metrics"][name][cohort][metric_name]
                        )
                    aggregate_internal[split][name][cohort][metric_name] = accumulator
        for cohort in ("all", "quality_filter_clean"):
            common_internal[split][cohort] = {}
            for metric_name, thresholds in (
                ("head_camera_rotation_deg", ROTATION_THRESHOLDS_DEG),
                ("stored_pose_action_translation_reproduction_m", RELATIVE_THRESHOLDS_M),
            ):
                accumulator = MetricAccumulator(thresholds)
                for result in selected:
                    accumulator.merge_state(result["common"][cohort][metric_name])
                common_internal[split][cohort][metric_name] = accumulator

    per_sequence = [result["per_sequence"] for result in metric_results]
    aggregates: dict[str, Any] = {}
    for split in ("train", "test", "all"):
        rows = [row for row in per_sequence if split == "all" or row["split"] == split]
        aggregates[split] = {
            "sequences": int(len(rows)),
            "candidates": {},
            "common": {},
        }
        for name in candidate_names:
            aggregates[split]["candidates"][name] = {
                cohort: {
                    metric_name: accumulator.summary(
                        sample_stride=args.percentile_stride
                    )
                    for metric_name, accumulator in metrics.items()
                }
                for cohort, metrics in aggregate_internal[split][name].items()
            }
            aggregates[split]["candidates"][name]["quality_filter_clean_sequence_balanced"] = (
                _summarize_per_sequence(rows, name)
            )
        aggregates[split]["common"] = {
            cohort: {
                metric_name: accumulator.summary(
                    sample_stride=args.percentile_stride
                )
                for metric_name, accumulator in metrics.items()
            }
            for cohort, metrics in common_internal[split].items()
        }

    selected_name = "absolute_global_frame_geomedian"
    selected_lever = frame_geomedian
    created = datetime.now(timezone.utc).isoformat()
    fit_centers_path = output / "fit_sequence_centers.jsonl"
    fit_lines = []
    for row in fit_rows:
        serializable = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in row.items()
            if key not in {"sampled_offsets", "offset_sum"}
        }
        serializable["offset_sum_m"] = row["offset_sum"].tolist()
        fit_lines.append(json.dumps(serializable, sort_keys=True))
    _atomic_write_text(fit_centers_path, "\n".join(fit_lines) + "\n")

    per_sequence_path = output / "per_sequence_metrics.jsonl"
    _atomic_write_text(
        per_sequence_path,
        "\n".join(json.dumps(row, sort_keys=True) for row in per_sequence) + "\n",
    )

    calibration_path = output / "head_camera_calibration_train_absolute_lever_v1.json"
    experimental_calibration = {
        "schema_version": 2,
        "kind": "experimental_camhead_absolute_lever_calibration",
        "created_utc": created,
        "status": "experimental_opt_in_not_for_historical_checkpoints",
        "split": "train",
        "rotation_head_to_upright_camera": rotation_head_to_camera.tolist(),
        "camera_origin_in_head_m": selected_lever.tolist(),
        "coordinate_contract": {
            "motion": "camhead_v1 SOMA-30 Head joint index 6, Kimodo Y-up",
            "camera": "upright RGB/OpenCV optical center in the shared metric world",
            "equation": "p_world_camera = p_world_head + R_world_head @ camera_origin_in_head",
            "absolute_translation_used": True,
            "rotation_source": (
                "unchanged train-only rotation from "
                "head_camera_calibration_train.json"
            ),
        },
        "fit": {
            "selected_candidate": selected_name,
            "fit_stride": int(args.fit_stride),
            "train_sequences": int(len(train_rows)),
            "train_clean_union_frames": int(exact_offset_count),
            "train_sampled_frames": int(len(train_samples)),
            "geometric_median": frame_geomedian_fit,
            "quality_filter": str(args.quality_filter.resolve()),
            "quality_filter_sha256": window_contract["quality_filter_sha256"],
        },
        "historical_relative_lever_m": historical_lever.tolist(),
        "selected_minus_historical_m": (selected_lever - historical_lever).tolist(),
        "heldout_quality_filter_clean": aggregates["test"]["candidates"][selected_name][
            "quality_filter_clean"
        ],
        "leakage_contract": {
            "uses_test_to_fit_selected_lever": False,
            "train_actor_candidate_uses_train_only": True,
            "train_actor_candidate_is_not_selected": True,
            "test_sequence_oracle_is_report_only_and_leaky": True,
        },
    }
    _atomic_write_json(calibration_path, experimental_calibration)

    fit_summary = {
        "historical_relative_lever_m": historical_lever.tolist(),
        "absolute_global_frame_mean_m": frame_mean.tolist(),
        "absolute_global_frame_geomedian_m": frame_geomedian.tolist(),
        "absolute_global_sequence_geomedian_m": sequence_geomedian.tolist(),
        "selected_candidate": selected_name,
        "selected_lever_m": selected_lever.tolist(),
        "selected_minus_historical_m": (selected_lever - historical_lever).tolist(),
        "selected_norm_m": float(np.linalg.norm(selected_lever)),
        "historical_norm_m": float(np.linalg.norm(historical_lever)),
        "frame_geometric_median_fit": frame_geomedian_fit,
        "sequence_geometric_median_fit": sequence_geomedian_fit,
        "actor_levers_train_only_m": {
            actor: actor_levers[actor].tolist() for actor in sorted(actor_levers)
        },
        "actor_fit": actor_fit,
        "train_sequence_center_distribution_m": {
            "component_median": np.median(sequence_centers, axis=0).tolist(),
            "component_p10": np.quantile(sequence_centers, 0.10, axis=0).tolist(),
            "component_p90": np.quantile(sequence_centers, 0.90, axis=0).tolist(),
            "distance_to_selected": {
                "mean": float(np.linalg.norm(sequence_centers - selected_lever, axis=1).mean()),
                "median": float(
                    np.median(np.linalg.norm(sequence_centers - selected_lever, axis=1))
                ),
                "p90": float(
                    np.quantile(
                        np.linalg.norm(sequence_centers - selected_lever, axis=1),
                        0.90,
                    )
                ),
            },
        },
    }
    report = {
        "schema_version": 1,
        "kind": "camhead_absolute_camera_origin_lever_refit",
        "created_utc": created,
        "status": "complete_experimental_calibration",
        "conclusion_guard": (
            "The selected global lever is train-only. Actor candidates use only train data. "
            "The per-test-sequence oracle uses test camera labels and is diagnostic only."
        ),
        "sources": {
            "motion_root": str(args.motion_root.resolve()),
            "original_motion_root": str(args.original_motion_root.resolve()),
            "camera_root": str(args.camera_root.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _sha256(args.manifest),
            "split_file": str(args.split_file.resolve()),
            "split_file_sha256": _sha256(args.split_file),
            "quality_filter": str(args.quality_filter.resolve()),
            "quality_filter_sha256": _sha256(args.quality_filter),
            "rotation_calibration": str(args.rotation_calibration.resolve()),
            "rotation_calibration_sha256": _sha256(args.rotation_calibration),
        },
        "population_contract": window_contract,
        "fit_contract": {
            "fit_split": "train",
            "fit_stride_per_sequence": int(args.fit_stride),
            "percentile_stride_per_sequence": int(args.percentile_stride),
            "clean_frame_weighting": "union of retained physical T97 windows; overlaps count once",
            "global_frame_mean": "exact mean over every retained train union frame",
            "global_frame_geomedian": (
                "geometric median over deterministic every-Nth retained train frame"
            ),
            "global_sequence_geomedian": "geometric median of per-sequence geometric medians",
            "train_actor_geomedian": (
                "geometric median of retained sampled train frames per Sxx actor"
            ),
            "test_sequence_oracle": "same-sequence held-out labels; leaky lower-bound diagnostic",
        },
        "fit": fit_summary,
        "aggregates": aggregates,
        "reference_camhead_v1_report": _reference_metrics(args.reference_report),
        "artifacts": {
            "experimental_calibration": str(calibration_path.resolve()),
            "fit_sequence_centers": str(fit_centers_path.resolve()),
            "per_sequence_metrics": str(per_sequence_path.resolve()),
            "heldout_clean_absolute_translation_cdf": str(
                (output / "heldout_clean_absolute_translation_cdf.png").resolve()
            ),
            "heldout_clean_absolute_vs_relative": str(
                (output / "heldout_clean_absolute_vs_relative.png").resolve()
            ),
            "sequence_lever_centers": str((output / "sequence_lever_centers.png").resolve()),
        },
    }
    _atomic_write_json(report_path, report)
    _plot_results(
        output,
        aggregate_internal,
        fit_rows,
        {
            **fixed_levers,
            "absolute_global_frame_geomedian": frame_geomedian,
            "absolute_global_sequence_geomedian": sequence_geomedian,
        },
    )
    qualitative = _plot_qualitative_lever_comparison(
        output,
        args.qualitative_manifest,
        args.original_motion_root,
        args.motion_root,
        args.camera_root,
        historical_lever,
        selected_lever,
    )
    report["qualitative_contract"] = qualitative
    if qualitative is not None:
        report["artifacts"]["diverse_absolute_lever_trajectory_comparison"] = (
            qualitative["trajectory_comparison"]
        )
    # Rewrite after plots so a complete report never points at missing figures.
    report["artifacts_sha256"] = {
        name: _sha256(Path(path))
        for name, path in report["artifacts"].items()
        if name != "experimental_calibration" and Path(path).is_file()
    }
    report["artifacts_sha256"]["experimental_calibration"] = _sha256(calibration_path)
    _atomic_write_json(report_path, report)
    print(f"[absolute-lever] wrote {report_path}", flush=True)
    print(
        "[absolute-lever] selected "
        f"{selected_name}={np.array2string(selected_lever, precision=8)} m",
        flush=True,
    )
    test_clean = aggregates["test"]["candidates"]
    for name in candidate_names:
        absolute = test_clean[name]["quality_filter_clean"]["absolute_translation_m"]
        relative = test_clean[name]["quality_filter_clean"]["relative_translation_m"]
        print(
            f"[absolute-lever] heldout clean {name}: "
            f"absolute={absolute['mean_exact'] * 100:.3f} cm "
            f"relative={relative['mean_exact'] * 1000:.3f} mm",
            flush=True,
        )


if __name__ == "__main__":
    main()
