#!/usr/bin/env python3
"""Exhaustive quantitative comparison of original and camera-aligned UniEgo data.

Every available frame and transition contributes to means, RMSE values, extrema,
and threshold counts.  To keep memory bounded, global percentiles and plots use a
deterministic every-Nth sample; the report labels that sampling stride explicitly.
Per-sequence metric shards make the CPU-heavy decode resumable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from camera_head_recanonicalization import (
    ARIA_Z_UP_TO_KIMODO_Y_UP,
    DELTA_END,
    HEAD_JOINT_IDX,
    N_JOINTS,
    camera_rotations_to_kimodo,
    canonical_frame_from_head,
    cont6d_to_matrix,
    decode_uniego,
    load_rotation_head_to_camera,
    rotation_angle_deg,
)
from uniego_layout import FOOT_JOINT_IDX


WEKA_ROOT = Path(os.environ.get("WEKA_ROOT", "/mnt/projects/ll/jungbinc/weka"))
DATA_ROOT = WEKA_ROOT / "nymeriaplus_kimodo_proportional"
RUN_ROOT = Path(os.environ.get("RUN_ROOT", WEKA_ROOT / "cosmos_motion_ft_runs"))
DEFAULT_OLD_ROOT = DATA_ROOT / "uniego_rep"
DEFAULT_NEW_ROOT = DATA_ROOT / "uniego_rep_camhead_v1"
DEFAULT_CAMERA_ROOT = DATA_ROOT / "camera_rgb"
DEFAULT_SPLIT_FILE = DATA_ROOT / "train_test_split.json"
DEFAULT_MANIFEST = DATA_ROOT / "video" / "manifest_video.jsonl"
DEFAULT_FLOOR_CALIBRATION = DATA_ROOT / "metadata" / "floor_calibration.json"
DEFAULT_CALIBRATION = Path(__file__).with_name("head_camera_calibration_train.json")
DEFAULT_OUTPUT = RUN_ROOT / "nymeria_camera_head_recanonicalization_v1" / "quantitative"

# SOMA-30 parents.  Physical preservation uses every real bone, including bones
# omitted from the display-only stick figure.
PARENTS = (-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18,
           19, 19, 0, 22, 23, 24, 0, 26, 27, 28)

FRAME_METRICS = (
    "absolute_camera_rotation_old_deg",
    "absolute_camera_rotation_new_deg",
    "absolute_camera_translation_old_m",
    "absolute_camera_translation_new_m",
    "head_correction_deg",
    "decoded_position_preservation_max_joint_m",
    "decoded_foot_position_preservation_max_m",
    "decoded_foot_height_preservation_max_m",
    "decoded_bone_length_preservation_max_m",
    "decoded_nonhead_rotation_preservation_max_joint_deg",
    "decoded_new_head_target_error_deg",
    "new_canonical_frame_error_deg",
    "feature_mean_absolute_change",
)
TRANSITION_METRICS = (
    "camera_action_rotation_old_deg",
    "camera_action_rotation_new_deg",
    "camera_action_translation_old_m",
    "camera_action_translation_new_m",
    "camera_pose_vs_action_rotation_error_deg",
    "camera_pose_vs_action_translation_error_m",
    "head_angular_step_old_deg",
    "head_angular_step_new_deg",
    "camera_angular_step_deg",
    "head_vs_camera_step_magnitude_old_deg",
    "head_vs_camera_step_magnitude_new_deg",
    "root_speed_mps",
    "body_relative_joint_speed_mps",
    "camera_speed_mps",
)
PHYSICAL_WINDOW_METRICS = (
    "contact_foot_skate_old_mps",
    "contact_foot_skate_new_mps",
    "height_toe_skate_old_mps",
    "height_toe_skate_new_mps",
    "foot_skate_ratio_old",
    "foot_skate_ratio_new",
    "contact_foot_height_abs_mean_old_m",
    "contact_foot_height_abs_mean_new_m",
    "contact_floating_fraction_gt10cm_old",
    "contact_floating_fraction_gt10cm_new",
    "contact_penetration_fraction_lt_minus5cm_old",
    "contact_penetration_fraction_lt_minus5cm_new",
    "frame_min_foot_height_mean_old_m",
    "frame_min_foot_height_mean_new_m",
    "frame_penetration_fraction_lt_minus5cm_old",
    "frame_penetration_fraction_lt_minus5cm_new",
    "frame_deep_penetration_fraction_lt_minus20cm_old",
    "frame_deep_penetration_fraction_lt_minus20cm_new",
    "frame_all_feet_above15cm_fraction_old",
    "frame_all_feet_above15cm_fraction_new",
)
ALL_METRICS = FRAME_METRICS + TRANSITION_METRICS + PHYSICAL_WINDOW_METRICS

THRESHOLDS: dict[str, tuple[float, ...]] = {
    "absolute_camera_rotation_old_deg": (1.0, 5.0, 10.0, 20.0),
    "absolute_camera_rotation_new_deg": (0.01, 0.05, 0.1, 1.0),
    "absolute_camera_translation_old_m": (0.01, 0.02, 0.05, 0.10),
    "absolute_camera_translation_new_m": (0.01, 0.02, 0.05, 0.10),
    "camera_action_rotation_old_deg": (0.1, 0.5, 1.0, 5.0),
    "camera_action_rotation_new_deg": (0.01, 0.05, 0.1, 1.0),
    "camera_action_translation_old_m": (0.001, 0.005, 0.01, 0.05),
    "camera_action_translation_new_m": (0.001, 0.005, 0.01, 0.05),
    "decoded_position_preservation_max_joint_m": (1e-6, 1e-5, 1e-4, 1e-3),
    "decoded_foot_position_preservation_max_m": (1e-6, 1e-5, 1e-4, 1e-3),
    "decoded_foot_height_preservation_max_m": (1e-6, 1e-5, 1e-4, 1e-3),
    "decoded_bone_length_preservation_max_m": (1e-6, 1e-5, 1e-4, 1e-3),
    "decoded_nonhead_rotation_preservation_max_joint_deg": (1e-4, 1e-3, 1e-2, 0.1),
    "decoded_new_head_target_error_deg": (0.01, 0.05, 0.1, 1.0),
    "new_canonical_frame_error_deg": (0.001, 0.01, 0.1, 1.0),
}


def _physical_window_metrics(
    positions: np.ndarray,
    contacts: np.ndarray,
    windows: list[tuple[int, int, float]],
    fps: float,
) -> dict[str, np.ndarray]:
    """Compute established skate metrics plus calibrated floating/penetration checks.

    The skating definitions match Kimodo's benchmark metrics: four-foot 3D speed
    under stored contacts, toe speed below 5 cm, and the fraction of below-5-cm
    toe transitions moving faster than 0.2 m/s.  Height metrics subtract the exact
    calibrated per-window floor offset used by the training loader.

    ``frame_all_feet_above15cm_fraction`` is descriptive, not automatically a
    defect: a jump can be legitimately airborne.  Contact-labelled floating is the
    stronger failure signal.
    """
    contacts = np.asarray(contacts, dtype=bool)
    values: dict[str, list[float]] = {
        "contact_foot_skate_mps": [],
        "height_toe_skate_mps": [],
        "foot_skate_ratio": [],
        "contact_foot_height_abs_mean_m": [],
        "contact_floating_fraction_gt10cm": [],
        "contact_penetration_fraction_lt_minus5cm": [],
        "frame_min_foot_height_mean_m": [],
        "frame_penetration_fraction_lt_minus5cm": [],
        "frame_deep_penetration_fraction_lt_minus20cm": [],
        "frame_all_feet_above15cm_fraction": [],
    }
    foot_indices = np.asarray(FOOT_JOINT_IDX, dtype=np.int64)
    for start, end, floor_offset in windows:
        start = max(0, int(start))
        end = min(int(end), len(positions))
        if end - start < 2:
            continue
        feet = positions[start:end, foot_indices].astype(np.float64, copy=True)
        feet[..., 1] -= float(floor_offset)
        window_contacts = contacts[start:end]
        foot_speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * float(fps)
        contact_transition = window_contacts[:-1]
        contact_count = int(contact_transition.sum())
        values["contact_foot_skate_mps"].append(
            float((foot_speed * contact_transition).sum() / (contact_count + 1e-6))
        )

        toes = feet[:, (1, 3)]
        toe_speed = np.linalg.norm(np.diff(toes, axis=0), axis=-1) * float(fps)
        toe_below_at_start = toes[:-1, :, 1] < 0.05
        below_count = int(toe_below_at_start.sum())
        values["height_toe_skate_mps"].append(
            float((toe_speed * toe_below_at_start).sum() / (below_count + 1e-6))
        )
        toe_below_both = (toes[:-1, :, 1] < 0.05) & (toes[1:, :, 1] < 0.05)
        values["foot_skate_ratio"].append(
            float(((toe_speed > 0.2) & toe_below_both).sum() / (toe_below_both.sum() + 1e-6))
        )

        contact_heights = feet[..., 1][window_contacts]
        if len(contact_heights):
            values["contact_foot_height_abs_mean_m"].append(float(np.abs(contact_heights).mean()))
            values["contact_floating_fraction_gt10cm"].append(float((contact_heights > 0.10).mean()))
            values["contact_penetration_fraction_lt_minus5cm"].append(
                float((contact_heights < -0.05).mean())
            )
        else:
            values["contact_foot_height_abs_mean_m"].append(0.0)
            values["contact_floating_fraction_gt10cm"].append(0.0)
            values["contact_penetration_fraction_lt_minus5cm"].append(0.0)

        minimum_height = feet[..., 1].min(axis=-1)
        values["frame_min_foot_height_mean_m"].append(float(minimum_height.mean()))
        values["frame_penetration_fraction_lt_minus5cm"].append(
            float((minimum_height < -0.05).mean())
        )
        values["frame_deep_penetration_fraction_lt_minus20cm"].append(
            float((minimum_height < -0.20).mean())
        )
        values["frame_all_feet_above15cm_fraction"].append(
            float((minimum_height > 0.15).mean())
        )
    return {name: np.asarray(metric_values, dtype=np.float32) for name, metric_values in values.items()}


def _relative_transforms(
    rotations: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.swapaxes(rotations[:-1], -1, -2)
    relative_rotation = inverse @ rotations[1:]
    relative_translation = np.einsum(
        "tij,tj->ti", inverse, positions[1:] - positions[:-1], optimize=True
    )
    return relative_rotation, relative_translation


def _rotation_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return rotation_angle_deg(np.swapaxes(predicted, -1, -2) @ target)


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "rmse": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _atomic_savez_compressed(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metric_one(
    payload: tuple[
        str,
        str,
        str,
        str,
        str,
        list[list[float]],
        list[float],
        list[tuple[int, int, float]],
        bool,
    ]
) -> dict[str, Any]:
    (
        relative,
        old_root_text,
        new_root_text,
        camera_root_text,
        shard_root_text,
        rotation_values,
        lever_values,
        physical_windows,
        overwrite,
    ) = payload
    started = time.time()
    old_path = Path(old_root_text) / relative
    new_path = Path(new_root_text) / relative
    camera_path = Path(camera_root_text) / relative
    shard_path = Path(shard_root_text) / relative
    record: dict[str, Any] = {"relative_path": relative, "shard_path": str(shard_path)}
    try:
        if shard_path.is_file() and not overwrite:
            with np.load(shard_path, allow_pickle=False) as shard:
                if not all(metric in shard.files for metric in ALL_METRICS):
                    raise ValueError("existing shard is incomplete")
                summary_text = str(np.asarray(shard["sequence_summary_json"]).item())
            summary = json.loads(summary_text)
            record.update(status="validated_existing", seconds=time.time() - started, **summary)
            return record

        with np.load(old_path, allow_pickle=False) as archive:
            old_arrays = {key: archive[key] for key in archive.files}
        with np.load(new_path, allow_pickle=False) as archive:
            new_arrays = {key: archive[key] for key in archive.files}
        with np.load(camera_path, allow_pickle=False) as camera:
            measured_camera_rotation = camera_rotations_to_kimodo(
                camera["cam_world_rot_upright"]
            )
            measured_camera_position = np.einsum(
                "ij,tj->ti",
                ARIA_Z_UP_TO_KIMODO_Y_UP,
                camera["cam_world_pos_upright"].astype(np.float64),
                optimize=True,
            )
            camera_action = camera["cam_action_upright_k1"].astype(np.float64)
            camera_timestamps = camera["timestamps_us"]
            fps = float(np.asarray(camera["fps"]).item())

        if old_arrays.keys() != new_arrays.keys():
            raise ValueError("old/new NPZ keys differ")
        if not np.array_equal(old_arrays["timestamps_us"], camera_timestamps):
            raise ValueError("motion/camera timestamps differ")
        changed_members = [
            key
            for key in old_arrays
            if key != "features" and not np.array_equal(old_arrays[key], new_arrays[key])
        ]
        if changed_members:
            raise ValueError(f"new NPZ changed preserved members: {changed_members}")

        old = decode_uniego(old_arrays["features"])
        new = decode_uniego(new_arrays["features"])
        if len(old.world_positions) != len(measured_camera_position):
            raise ValueError("decoded motion/camera frame counts differ")
        rotation_head_to_camera = np.asarray(rotation_values, dtype=np.float64)
        lever = np.asarray(lever_values, dtype=np.float64)

        old_head_rotation = old.world_rotations[:, HEAD_JOINT_IDX]
        new_head_rotation = new.world_rotations[:, HEAD_JOINT_IDX]
        head_position = old.world_positions[:, HEAD_JOINT_IDX]
        old_camera_rotation = old_head_rotation @ rotation_head_to_camera
        new_camera_rotation = new_head_rotation @ rotation_head_to_camera
        old_camera_position = head_position + np.einsum(
            "tij,j->ti", old_head_rotation, lever, optimize=True
        )
        new_camera_position = head_position + np.einsum(
            "tij,j->ti", new_head_rotation, lever, optimize=True
        )

        old_action_rotation, old_action_translation = _relative_transforms(
            old_camera_rotation, old_camera_position
        )
        new_action_rotation, new_action_translation = _relative_transforms(
            new_camera_rotation, new_camera_position
        )
        pose_action_rotation, pose_action_translation = _relative_transforms(
            measured_camera_rotation, measured_camera_position
        )
        target_action_rotation = cont6d_to_matrix(camera_action[:, 3:9])
        target_action_translation = camera_action[:, :3]
        if len(target_action_rotation) != len(old_action_rotation):
            raise ValueError("camera action length does not equal frames-1")

        nonhead = np.arange(N_JOINTS) != HEAD_JOINT_IDX
        position_preservation = np.linalg.norm(
            new.world_positions - old.world_positions, axis=-1
        ).max(axis=-1)
        old_feet = old.world_positions[:, FOOT_JOINT_IDX]
        new_feet = new.world_positions[:, FOOT_JOINT_IDX]
        foot_position_preservation = np.linalg.norm(new_feet - old_feet, axis=-1).max(axis=-1)
        foot_height_preservation = np.abs(new_feet[..., 1] - old_feet[..., 1]).max(axis=-1)
        bone_children = np.asarray(
            [joint for joint, parent in enumerate(PARENTS) if parent >= 0], dtype=np.int64
        )
        bone_parents = np.asarray(
            [parent for parent in PARENTS if parent >= 0], dtype=np.int64
        )
        old_bone_lengths = np.linalg.norm(
            old.world_positions[:, bone_children] - old.world_positions[:, bone_parents], axis=-1
        )
        new_bone_lengths = np.linalg.norm(
            new.world_positions[:, bone_children] - new.world_positions[:, bone_parents], axis=-1
        )
        bone_length_preservation = np.abs(new_bone_lengths - old_bone_lengths).max(axis=-1)
        nonhead_rotation_preservation = _rotation_error(
            new.world_rotations[:, nonhead], old.world_rotations[:, nonhead]
        ).max(axis=-1)
        corrected_head_target = measured_camera_rotation @ rotation_head_to_camera.T
        expected_canonical_rotation, _ = canonical_frame_from_head(
            new.world_rotations, new.world_positions
        )

        old_head_step, _ = _relative_transforms(old_head_rotation, head_position)
        new_head_step, _ = _relative_transforms(new_head_rotation, head_position)
        camera_step, _ = _relative_transforms(
            measured_camera_rotation, measured_camera_position
        )
        old_head_step_angle = rotation_angle_deg(old_head_step)
        new_head_step_angle = rotation_angle_deg(new_head_step)
        camera_step_angle = rotation_angle_deg(camera_step)
        root_velocity = np.diff(old.world_positions[:, 0], axis=0) * fps
        body_relative = old.world_positions - old.world_positions[:, :1]
        body_relative_velocity = np.diff(body_relative, axis=0) * fps
        camera_velocity = np.diff(measured_camera_position, axis=0) * fps

        arrays: dict[str, np.ndarray] = {
            "absolute_camera_rotation_old_deg": _rotation_error(
                old_camera_rotation, measured_camera_rotation
            ),
            "absolute_camera_rotation_new_deg": _rotation_error(
                new_camera_rotation, measured_camera_rotation
            ),
            "absolute_camera_translation_old_m": np.linalg.norm(
                old_camera_position - measured_camera_position, axis=-1
            ),
            "absolute_camera_translation_new_m": np.linalg.norm(
                new_camera_position - measured_camera_position, axis=-1
            ),
            "head_correction_deg": _rotation_error(old_head_rotation, corrected_head_target),
            "decoded_position_preservation_max_joint_m": position_preservation,
            "decoded_foot_position_preservation_max_m": foot_position_preservation,
            "decoded_foot_height_preservation_max_m": foot_height_preservation,
            "decoded_bone_length_preservation_max_m": bone_length_preservation,
            "decoded_nonhead_rotation_preservation_max_joint_deg": nonhead_rotation_preservation,
            "decoded_new_head_target_error_deg": _rotation_error(
                new_head_rotation, corrected_head_target
            ),
            "new_canonical_frame_error_deg": _rotation_error(
                new.canonical_rotations, expected_canonical_rotation
            ),
            "feature_mean_absolute_change": np.mean(
                np.abs(
                    new_arrays["features"].astype(np.float64)
                    - old_arrays["features"].astype(np.float64)
                ),
                axis=-1,
            ),
            "camera_action_rotation_old_deg": _rotation_error(
                old_action_rotation, target_action_rotation
            ),
            "camera_action_rotation_new_deg": _rotation_error(
                new_action_rotation, target_action_rotation
            ),
            "camera_action_translation_old_m": np.linalg.norm(
                old_action_translation - target_action_translation, axis=-1
            ),
            "camera_action_translation_new_m": np.linalg.norm(
                new_action_translation - target_action_translation, axis=-1
            ),
            "camera_pose_vs_action_rotation_error_deg": _rotation_error(
                pose_action_rotation, target_action_rotation
            ),
            "camera_pose_vs_action_translation_error_m": np.linalg.norm(
                pose_action_translation - target_action_translation, axis=-1
            ),
            "head_angular_step_old_deg": old_head_step_angle,
            "head_angular_step_new_deg": new_head_step_angle,
            "camera_angular_step_deg": camera_step_angle,
            "head_vs_camera_step_magnitude_old_deg": np.abs(
                old_head_step_angle - camera_step_angle
            ),
            "head_vs_camera_step_magnitude_new_deg": np.abs(
                new_head_step_angle - camera_step_angle
            ),
            "root_speed_mps": np.linalg.norm(root_velocity, axis=-1),
            "body_relative_joint_speed_mps": np.linalg.norm(
                body_relative_velocity, axis=-1
            ).mean(axis=-1),
            "camera_speed_mps": np.linalg.norm(camera_velocity, axis=-1),
        }
        contacts = old_arrays["features"][:, DELTA_END:] > 0.5
        old_physical = _physical_window_metrics(
            old.world_positions, contacts, physical_windows, fps
        )
        new_physical = _physical_window_metrics(
            new.world_positions, contacts, physical_windows, fps
        )
        for base_name, old_values in old_physical.items():
            new_values = new_physical[base_name]
            arrays[f"{base_name.removesuffix('_mps').removesuffix('_m')}_old" + (
                "_mps" if base_name.endswith("_mps") else "_m" if base_name.endswith("_m") else ""
            )] = old_values
            arrays[f"{base_name.removesuffix('_mps').removesuffix('_m')}_new" + (
                "_mps" if base_name.endswith("_mps") else "_m" if base_name.endswith("_m") else ""
            )] = new_values
        # Keep the public metric names straightforward and explicit.  The generic
        # construction above preserves units at the suffix; assert it matched the
        # declared report schema before writing a shard.
        missing_physical = [name for name in PHYSICAL_WINDOW_METRICS if name not in arrays]
        if missing_physical:
            raise AssertionError(f"internal physical metric naming error: {missing_physical}")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("a computed metric contains non-finite values")

        summary: dict[str, Any] = {
            "frames": int(len(old.world_positions)),
            "transitions": int(len(old.world_positions) - 1),
            "physical_windows": int(len(physical_windows)),
            "metadata_preserved": True,
            "feature_contacts_preserved": bool(
                np.array_equal(
                    old_arrays["features"][:, DELTA_END:],
                    new_arrays["features"][:, DELTA_END:],
                )
            ),
            "path_length_camera_m": float(np.linalg.norm(
                np.diff(measured_camera_position, axis=0), axis=-1
            ).sum()),
            "net_displacement_camera_m": float(np.linalg.norm(
                measured_camera_position[-1] - measured_camera_position[0]
            )),
        }
        for name, values in arrays.items():
            metric_summary = _summary(values)
            summary[f"{name}_mean"] = metric_summary["mean"]
            summary[f"{name}_p95"] = metric_summary["p95"]
            summary[f"{name}_max"] = metric_summary["max"]
            arrays[name] = np.asarray(values, dtype=np.float32)
        arrays["sequence_summary_json"] = np.asarray(json.dumps(summary, sort_keys=True))
        _atomic_savez_compressed(shard_path, arrays)
        record.update(status="computed", seconds=time.time() - started, **summary)
    except Exception as error:  # noqa: BLE001
        record.update(
            status="error",
            error=f"{type(error).__name__}: {error}",
            seconds=time.time() - started,
        )
    return record


class _StreamingMetric:
    def __init__(self, sample_stride: int, thresholds: tuple[float, ...]) -> None:
        self.sample_stride = sample_stride
        self.thresholds = thresholds
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.above = {threshold: 0 for threshold in thresholds}
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) == 0:
            return
        self.count += len(values)
        self.total += float(values.sum(dtype=np.float64))
        self.total_square += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        for threshold in self.thresholds:
            self.above[threshold] += int(np.count_nonzero(values > threshold))
        self.samples.append(values[:: self.sample_stride].astype(np.float32))

    def finish(self) -> tuple[dict[str, Any], np.ndarray]:
        samples = np.concatenate(self.samples) if self.samples else np.empty(0, np.float32)
        if self.count == 0:
            result: dict[str, Any] = {
                "count": 0,
                "mean_exact": None,
                "rmse_exact": None,
                "std_exact": None,
                "min_exact": None,
                "max_exact": None,
                "quantile_sample_stride": self.sample_stride,
                "quantile_sample_count": 0,
                "median_sampled": None,
                "p90_sampled": None,
                "p95_sampled": None,
                "p99_sampled": None,
            }
            if self.thresholds:
                result["threshold_counts_exact"] = {
                    str(threshold): {"above": 0, "fraction_above": None}
                    for threshold in self.thresholds
                }
            return result, samples
        result: dict[str, Any] = {
            "count": self.count,
            "mean_exact": self.total / self.count,
            "rmse_exact": math.sqrt(self.total_square / self.count),
            "std_exact": math.sqrt(max(self.total_square / self.count - (self.total / self.count) ** 2, 0.0)),
            "min_exact": self.minimum,
            "max_exact": self.maximum,
            "quantile_sample_stride": self.sample_stride,
            "quantile_sample_count": int(len(samples)),
            "median_sampled": float(np.median(samples)),
            "p90_sampled": float(np.quantile(samples, 0.90)),
            "p95_sampled": float(np.quantile(samples, 0.95)),
            "p99_sampled": float(np.quantile(samples, 0.99)),
        }
        if self.thresholds:
            result["threshold_counts_exact"] = {
                str(threshold): {
                    "above": self.above[threshold],
                    "fraction_above": self.above[threshold] / self.count,
                }
                for threshold in self.thresholds
            }
        return result, samples


def _split_lookup(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    payload = json.loads(path.read_text())
    lookup: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "test"):
        values = payload.get(split, [])
        counts[split] = len(values)
        for uuid in values:
            if uuid in lookup:
                raise ValueError(f"duplicate split UUID {uuid}")
            lookup[uuid] = split
    return lookup, counts


def _load_physical_windows(
    manifest_path: Path,
    floor_calibration_path: Path,
    available_uuids: set[str],
) -> tuple[dict[str, list[tuple[int, int, float]]], dict[str, Any]]:
    """Mirror the training loader's calibrated, floor-filtered caption windows."""
    calibration = json.loads(floor_calibration_path.read_text())
    deltas = {uuid: float(value) for uuid, value in calibration.get("deltas", {}).items()}
    global_delta = float(calibration.get("global_delta", 0.0))
    dropped = {
        uuid: {(int(entry[0]), int(entry[1])): str(entry[2]) for entry in entries}
        for uuid, entries in calibration.get("dropped_windows", {}).items()
    }
    by_uuid: dict[str, list[tuple[int, int, float]]] = {
        uuid: [] for uuid in available_uuids
    }
    counts = {
        "manifest_records": 0,
        "usable_caption_windows": 0,
        "floor_filter_dropped": 0,
        "missing_ground_offset": 0,
        "short_or_empty": 0,
        "kept_windows": 0,
    }
    drop_reasons: dict[str, int] = {}
    with manifest_path.open() as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            uuid = record.get("uuid")
            if uuid not in available_uuids:
                continue
            counts["manifest_records"] += 1
            frame_count = int(record.get("nb_frames", 0))
            for window in record.get("t2w_windows", []):
                if not window.get("usable", False) or not window.get("caption"):
                    continue
                counts["usable_caption_windows"] += 1
                start = int(window["start_frame"])
                raw_end = int(window["end_frame"])
                reason = dropped.get(uuid, {}).get((start, raw_end))
                if reason is not None:
                    counts["floor_filter_dropped"] += 1
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                    continue
                offset = window.get("ground_offset_y")
                if offset is None:
                    counts["missing_ground_offset"] += 1
                    continue
                end = min(raw_end, frame_count)
                if end - start < 2:
                    counts["short_or_empty"] += 1
                    continue
                total_offset = float(offset) + deltas.get(uuid, global_delta)
                by_uuid[uuid].append((start, end, total_offset))
                counts["kept_windows"] += 1
    counts["drop_reasons"] = drop_reasons
    counts["sequences_with_windows"] = sum(bool(windows) for windows in by_uuid.values())
    return by_uuid, counts


def _aggregate(
    records: list[dict[str, Any]], split_lookup: dict[str, str], sample_stride: int
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    groups = ("all", "train", "test", "other")
    accumulators = {
        group: {
            metric: _StreamingMetric(sample_stride, THRESHOLDS.get(metric, ()))
            for metric in ALL_METRICS
        }
        for group in groups
    }
    group_sequences = {group: 0 for group in groups}
    group_frames = {group: 0 for group in groups}
    group_transitions = {group: 0 for group in groups}
    metadata_failures: list[str] = []

    for index, record in enumerate(records, 1):
        if record["status"] not in {"computed", "validated_existing"}:
            continue
        uuid = record["relative_path"][:-4]
        split = split_lookup.get(uuid, "other")
        selected_groups = ("all", split)
        group_sequences["all"] += 1
        group_sequences[split] += 1
        group_frames["all"] += int(record["frames"])
        group_frames[split] += int(record["frames"])
        group_transitions["all"] += int(record["transitions"])
        group_transitions[split] += int(record["transitions"])
        if not record.get("metadata_preserved") or not record.get("feature_contacts_preserved"):
            metadata_failures.append(record["relative_path"])
        with np.load(record["shard_path"], allow_pickle=False) as shard:
            for metric in ALL_METRICS:
                values = shard[metric]
                for group in selected_groups:
                    accumulators[group][metric].update(values)
        if index % 50 == 0 or index == len(records):
            print(f"[camhead-compare] aggregated {index}/{len(records)} shards", flush=True)

    report: dict[str, Any] = {"metadata_or_contact_failures": metadata_failures, "groups": {}}
    samples_by_group: dict[str, dict[str, np.ndarray]] = {}
    for group in groups:
        if group_sequences[group] == 0:
            continue
        group_report: dict[str, Any] = {
            "sequences": group_sequences[group],
            "frames": group_frames[group],
            "transitions": group_transitions[group],
            "metrics": {},
        }
        group_samples: dict[str, np.ndarray] = {}
        for metric, accumulator in accumulators[group].items():
            metric_report, samples = accumulator.finish()
            group_report["metrics"][metric] = metric_report
            group_samples[metric] = samples
        report["groups"][group] = group_report
        samples_by_group[group] = group_samples
    return report, samples_by_group


def _paired_summary(group: dict[str, Any], old_name: str, new_name: str) -> dict[str, float]:
    old = group["metrics"][old_name]
    new = group["metrics"][new_name]
    old_mean = float(old["mean_exact"])
    new_mean = float(new["mean_exact"])
    return {
        "old_mean": old_mean,
        "new_mean": new_mean,
        "absolute_mean_change": new_mean - old_mean,
        "relative_mean_change_fraction": (new_mean - old_mean) / old_mean if old_mean else math.nan,
        "old_p95_sampled": float(old["p95_sampled"]),
        "new_p95_sampled": float(new["p95_sampled"]),
    }


def _render_aggregate_plots(
    output: Path, report: dict[str, Any], samples: dict[str, np.ndarray]
) -> None:
    pairs = (
        ("Absolute camera rotation", "absolute_camera_rotation_old_deg", "absolute_camera_rotation_new_deg", "degrees"),
        ("Relative camera-action rotation", "camera_action_rotation_old_deg", "camera_action_rotation_new_deg", "degrees"),
        ("Absolute camera-origin translation", "absolute_camera_translation_old_m", "absolute_camera_translation_new_m", "metres"),
        ("Relative camera-action translation", "camera_action_translation_old_m", "camera_action_translation_new_m", "metres"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (title, old_name, new_name, unit) in zip(axes.flat, pairs):
        old = np.sort(samples[old_name])
        new = np.sort(samples[new_name])
        # The 99.5th percentile keeps a handful of source jumps from flattening the plot.
        limit = max(float(np.quantile(old, 0.995)), float(np.quantile(new, 0.995)), 1e-8)
        for values, label, color in ((old, "original", "tab:orange"), (new, "camera-aligned v1", "tab:blue")):
            clipped = values[values <= limit]
            y = np.linspace(0.0, 1.0, len(clipped), endpoint=True)
            axis.plot(clipped, y, label=label, color=color, lw=2)
        old_mean = report["metrics"][old_name]["mean_exact"]
        new_mean = report["metrics"][new_name]["mean_exact"]
        axis.set_title(f"{title}\nmean {old_mean:.5g} → {new_mean:.5g} {unit}")
        axis.set_xlabel(unit)
        axis.set_ylabel("empirical CDF (deterministic sample)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Nymeria original vs camera-aligned Head — all available sequences", fontsize=15)
    fig.savefig(output / "aggregate_old_vs_new_cdf.png", dpi=170)
    plt.close(fig)

    preservation = (
        ("Position max across joints [m]", "decoded_position_preservation_max_joint_m"),
        ("Foot position max [m]", "decoded_foot_position_preservation_max_m"),
        ("Foot height max [m]", "decoded_foot_height_preservation_max_m"),
        ("Bone-length max [m]", "decoded_bone_length_preservation_max_m"),
        ("Non-Head rotation max [deg]", "decoded_nonhead_rotation_preservation_max_joint_deg"),
        ("New Head target error [deg]", "decoded_new_head_target_error_deg"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for axis, (title, name) in zip(axes.flat, preservation):
        values = np.maximum(samples[name].astype(np.float64), 1e-12)
        axis.hist(np.log10(values), bins=80, color="tab:green", alpha=0.82)
        metric = report["metrics"][name]
        axis.set_title(f"{title}\nmax={metric['max_exact']:.4g}")
        axis.set_xlabel("log10(error)")
        axis.set_ylabel("sampled frames")
        axis.grid(alpha=0.2)
    fig.suptitle("Lossless re-canonicalization preservation checks", fontsize=15)
    fig.savefig(output / "preservation_checks.png", dpi=170)
    plt.close(fig)

    physical_pairs = (
        ("Contact foot skate", "contact_foot_skate_old_mps", "contact_foot_skate_new_mps", "m/s"),
        ("Height toe skate", "height_toe_skate_old_mps", "height_toe_skate_new_mps", "m/s"),
        ("Foot skate ratio", "foot_skate_ratio_old", "foot_skate_ratio_new", "fraction"),
        (
            "Contact foot |height|",
            "contact_foot_height_abs_mean_old_m",
            "contact_foot_height_abs_mean_new_m",
            "m",
        ),
        (
            "Contact floating >10 cm",
            "contact_floating_fraction_gt10cm_old",
            "contact_floating_fraction_gt10cm_new",
            "fraction",
        ),
        (
            "Contact penetration <-5 cm",
            "contact_penetration_fraction_lt_minus5cm_old",
            "contact_penetration_fraction_lt_minus5cm_new",
            "fraction",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for axis, (title, old_name, new_name, unit) in zip(axes.flat, physical_pairs):
        old_values = samples[old_name].astype(np.float64)
        new_values = samples[new_name].astype(np.float64)
        limit = max(
            float(np.quantile(old_values, 0.995)),
            float(np.quantile(new_values, 0.995)),
            1e-8,
        )
        bins = np.linspace(0.0, limit, 70)
        axis.hist(old_values, bins=bins, histtype="step", lw=2, color="tab:orange", label="original")
        axis.hist(new_values, bins=bins, histtype="step", lw=1.5, color="tab:blue", ls="--", label="camera-aligned v1")
        old_mean = report["metrics"][old_name]["mean_exact"]
        new_mean = report["metrics"][new_name]["mean_exact"]
        axis.set_title(f"{title}\nmean {old_mean:.6g} → {new_mean:.6g} {unit}")
        axis.set_xlabel(unit)
        axis.set_ylabel("window count (sampled)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Physical motion quality on calibrated, floor-filtered caption windows",
        fontsize=15,
    )
    fig.savefig(output / "physical_motion_old_vs_new.png", dpi=170)
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--camera-root", type=Path, default=DEFAULT_CAMERA_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--floor-calibration", type=Path, default=DEFAULT_FLOOR_CALIBRATION
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite-shards", action="store_true")
    parser.add_argument("--sample-stride", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.workers <= 0 or args.sample_stride <= 0 or args.progress_every <= 0:
        parser.error("workers, sample-stride, and progress-every must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    shard_root = args.output / "metric_shards"
    rotation, calibration = load_rotation_head_to_camera(args.calibration)
    lever = np.asarray(calibration["camera_origin_in_head_m"], dtype=np.float64)
    if lever.shape != (3,) or not np.isfinite(lever).all():
        raise ValueError("invalid camera_origin_in_head_m calibration")

    new_files = sorted(args.new_root.glob("S*/*.npz"))
    if args.limit is not None:
        new_files = new_files[: args.limit]
    if not new_files:
        raise SystemExit(f"no corrected NPZs under {args.new_root}")
    relative_paths = [str(path.relative_to(args.new_root)) for path in new_files]
    available_uuids = {relative[:-4] for relative in relative_paths}
    physical_windows, physical_window_contract = _load_physical_windows(
        args.manifest, args.floor_calibration, available_uuids
    )
    missing_inputs = [
        relative
        for relative in relative_paths
        if not (args.old_root / relative).is_file() or not (args.camera_root / relative).is_file()
    ]
    if missing_inputs:
        raise SystemExit(f"{len(missing_inputs)} corrected files lack old/camera input")
    print(
        f"[camhead-compare] sequences={len(relative_paths)} workers={args.workers} "
        f"output={args.output}",
        flush=True,
    )

    work = [
        (
            relative,
            str(args.old_root),
            str(args.new_root),
            str(args.camera_root),
            str(shard_root),
            rotation.tolist(),
            lever.tolist(),
            physical_windows.get(relative[:-4], []),
            args.overwrite_shards,
        )
        for relative in relative_paths
    ]
    records: list[dict[str, Any]] = []
    started = time.time()
    if args.workers == 1:
        iterator = map(_metric_one, work)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(_metric_one, work, chunksize=1)
    try:
        for index, record in enumerate(iterator, 1):
            records.append(record)
            if index % args.progress_every == 0 or index == len(work):
                counts: dict[str, int] = {}
                for item in records:
                    counts[item["status"]] = counts.get(item["status"], 0) + 1
                print(
                    f"[camhead-compare] metrics {index}/{len(work)} {counts} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    errors = [record for record in records if record["status"] == "error"]
    if errors:
        _write_json(args.output / "metric_errors.json", {"errors": errors})
        raise SystemExit(f"metric computation failed for {len(errors)} sequence(s)")

    split_lookup, expected_split_counts = _split_lookup(args.split_file)
    aggregate, samples = _aggregate(records, split_lookup, args.sample_stride)
    all_group = aggregate["groups"]["all"]
    paired = {
        "absolute_camera_rotation": _paired_summary(
            all_group, "absolute_camera_rotation_old_deg", "absolute_camera_rotation_new_deg"
        ),
        "camera_action_rotation": _paired_summary(
            all_group, "camera_action_rotation_old_deg", "camera_action_rotation_new_deg"
        ),
        "absolute_camera_translation": _paired_summary(
            all_group, "absolute_camera_translation_old_m", "absolute_camera_translation_new_m"
        ),
        "camera_action_translation": _paired_summary(
            all_group, "camera_action_translation_old_m", "camera_action_translation_new_m"
        ),
        "head_vs_camera_step_magnitude": _paired_summary(
            all_group,
            "head_vs_camera_step_magnitude_old_deg",
            "head_vs_camera_step_magnitude_new_deg",
        ),
        "contact_foot_skate": _paired_summary(
            all_group, "contact_foot_skate_old_mps", "contact_foot_skate_new_mps"
        ),
        "height_toe_skate": _paired_summary(
            all_group, "height_toe_skate_old_mps", "height_toe_skate_new_mps"
        ),
        "foot_skate_ratio": _paired_summary(
            all_group, "foot_skate_ratio_old", "foot_skate_ratio_new"
        ),
        "contact_foot_height_abs": _paired_summary(
            all_group,
            "contact_foot_height_abs_mean_old_m",
            "contact_foot_height_abs_mean_new_m",
        ),
        "contact_floating_fraction_gt10cm": _paired_summary(
            all_group,
            "contact_floating_fraction_gt10cm_old",
            "contact_floating_fraction_gt10cm_new",
        ),
        "contact_penetration_fraction_lt_minus5cm": _paired_summary(
            all_group,
            "contact_penetration_fraction_lt_minus5cm_old",
            "contact_penetration_fraction_lt_minus5cm_new",
        ),
    }
    report = {
        "schema_version": 1,
        "kind": "nymeria_camera_head_recanonicalization_quantitative_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "old_root": str(args.old_root.resolve()),
        "new_root": str(args.new_root.resolve()),
        "camera_root": str(args.camera_root.resolve()),
        "calibration": str(args.calibration.resolve()),
        "split_file": str(args.split_file.resolve()),
        "manifest": str(args.manifest.resolve()),
        "floor_calibration": str(args.floor_calibration.resolve()),
        "physical_window_contract": {
            "selection": (
                "usable captioned manifest windows after the same floor-calibration "
                "drop map and per-sequence offset used by the training loader"
            ),
            "foot_joints": list(FOOT_JOINT_IDX),
            "contact_skate": "mean 3D speed of four feet under stored contacts",
            "height_skate": "mean 3D toe speed when calibrated toe height at transition start is <5cm",
            "skate_ratio": "fraction of toe transitions below 5cm at both ends with speed >0.2m/s",
            "floating_note": (
                "contact-labelled foot height >10cm is a defect signal; all-feet-above-15cm "
                "is descriptive because legitimate jumps are airborne"
            ),
            **physical_window_contract,
        },
        "expected_split_sequence_counts": expected_split_counts,
        "percentile_contract": (
            f"deterministic every-{args.sample_stride}th value within every sequence; "
            "means, RMSE, extrema, counts, and threshold fractions use every value"
        ),
        "paired_old_vs_new_all": paired,
        **aggregate,
        "records": records,
    }
    _write_json(args.output / "comparison_report.json", report)

    fieldnames = sorted({key for record in records for key in record})
    with (args.output / "per_sequence.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    _render_aggregate_plots(args.output, all_group, samples["all"])
    print(json.dumps({
        "output": str(args.output),
        "sequences": all_group["sequences"],
        "frames": all_group["frames"],
        "transitions": all_group["transitions"],
        "metadata_or_contact_failures": aggregate["metadata_or_contact_failures"],
        "paired_old_vs_new_all": paired,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
