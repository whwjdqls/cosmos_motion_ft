#!/usr/bin/env python3
"""Audit NymeriaPlus camera/motion alignment from source artifacts.

This is intentionally independent of the Phase-3 model and calibration files.  It checks the
stored proportional motion, UniEgo representation, raw MPS camera sidecar, upright-RGB camera
sidecar, source SMPL timestamps/rotations, and video metadata against one another.

The most important diagnostic is performed in the original shared world frame: UniEgo's decoded
SOMA Head position is compared directly with the upright RGB camera position after only the fixed
Aria-Z-up -> Kimodo-Y-up basis change.  No fitted Head->camera transform and no frame-0 alignment
is used for that check.

Run on a compute node (CPU is sufficient), for example:

  CUDA_VISIBLE_DEVICES= python audit_nymeria_camera_motion.py --workers 4
  CUDA_VISIBLE_DEVICES= python audit_nymeria_camera_motion.py --split test \
      --source-smpl-orientation --workers 2
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


MOTION_ROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")
SOURCE_ROOT = Path("/weka/jungbin/nymeriaplus")
DEFAULT_OUTPUT = Path(
    "/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit"
)
HEAD_JOINT_IDX = 6
N_FOOT = 4
ARIA_Z_UP_TO_KIMODO_Y_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# SMPL-24 chain from pelvis (0) to Head (15). body_pose stores joints 1..23.
SMPL_HEAD_CHAIN = (3, 6, 9, 12, 15)


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: math.nan for k in ("mean", "median", "p90", "p95", "p99", "max")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _cont6d_to_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_axis = x[..., :3]
    y_raw = x[..., 3:6]
    x_axis /= np.maximum(np.linalg.norm(x_axis, axis=-1, keepdims=True), 1e-12)
    z_axis = np.cross(x_axis, y_raw)
    z_axis /= np.maximum(np.linalg.norm(z_axis, axis=-1, keepdims=True), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=-1)


def _axis_angle_to_matrix(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = v / np.maximum(theta, 1e-12)
    x, y, z = np.moveaxis(axis, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), axis=-1
    ).reshape(v.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), skew.shape)
    return eye + np.sin(theta)[..., None] * skew + (1.0 - np.cos(theta))[..., None] * (
        skew @ skew
    )


def _project_so3(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def _rotation_angle_deg(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def _fit_fixed_relation(a_world: np.ndarray, b_world: np.ndarray, stride: int) -> dict[str, Any]:
    """Fit X in R_b = R_a X and summarize framewise deviations."""
    a_sample = a_world[::stride]
    b_sample = b_world[::stride]
    relation = np.swapaxes(a_sample, -1, -2) @ b_sample
    fitted = _project_so3(relation.mean(axis=0))
    error = _rotation_angle_deg(fitted.T @ relation)
    predicted = a_sample @ fitted
    predicted_forward = predicted[:, :, 2]
    actual_forward = b_sample[:, :, 2]
    predicted_heading = np.arctan2(predicted_forward[:, 0], predicted_forward[:, 2])
    actual_heading = np.arctan2(actual_forward[:, 0], actual_forward[:, 2])
    heading_delta = np.arctan2(
        np.sin(predicted_heading - actual_heading),
        np.cos(predicted_heading - actual_heading),
    )
    forward_angle = np.rad2deg(
        np.arccos(np.clip(np.sum(predicted_forward * actual_forward, axis=-1), -1.0, 1.0))
    )
    up_angle = np.rad2deg(
        np.arccos(
            np.clip(np.sum(predicted[:, :, 1] * b_sample[:, :, 1], axis=-1), -1.0, 1.0)
        )
    )
    return {
        "rotation": fitted,
        "deviation_deg": _stats(error),
        "horizontal_forward_heading_error_deg": _stats(np.abs(np.rad2deg(heading_delta))),
        "forward_axis_error_deg": _stats(forward_angle),
        "up_axis_error_deg": _stats(up_angle),
    }


def _window_relation_metrics(
    a_world: np.ndarray,
    b_world: np.ndarray,
    sequence_relation: np.ndarray,
    *,
    window_frames: int = 97,
    stride: int = 5,
) -> dict[str, Any]:
    """Separate within-window rigidity from drift of the fitted relation across a sequence."""
    deviation_means = []
    heading_means = []
    relation_vs_sequence = []
    for start in range(0, len(a_world) - window_frames + 1, window_frames):
        fit = _fit_fixed_relation(
            a_world[start:start + window_frames],
            b_world[start:start + window_frames],
            stride,
        )
        deviation_means.append(fit["deviation_deg"]["mean"])
        heading_means.append(fit["horizontal_forward_heading_error_deg"]["mean"])
        relation_vs_sequence.append(
            float(_rotation_angle_deg(sequence_relation.T @ fit["rotation"]))
        )
    return {
        "window_frames": window_frames,
        "window_count": len(deviation_means),
        "within_window_rotation_deviation_mean_deg": _stats(np.asarray(deviation_means)),
        "within_window_heading_error_mean_deg": _stats(np.asarray(heading_means)),
        "window_relation_vs_sequence_relation_deg": _stats(
            np.asarray(relation_vs_sequence)
        ),
    }


def _largest(values: np.ndarray, count: int = 8, *, transition: bool = False) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return []
    indices = np.argsort(values)[-count:][::-1]
    return [
        {
            ("from_frame" if transition else "frame"): int(index),
            **({"to_frame": int(index + 1)} if transition else {}),
            "value": float(values[index]),
        }
        for index in indices
    ]


def _true_ranges(mask: np.ndarray) -> list[dict[str, int]]:
    """Compress a frame mask into half-open contiguous ranges."""
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, breaks)
    return [
        {"start": int(group[0]), "end": int(group[-1] + 1), "count": int(len(group))}
        for group in groups
    ]


def _decode_uniego_head(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode only Head SE(3), using the exact structure written by kimodo_to_uniego."""
    features = np.asarray(features, dtype=np.float64)
    n_frames, feature_dim = features.shape
    n_joints = (feature_dim - 9 - N_FOOT) // 9
    if n_joints * 9 + 9 + N_FOOT != feature_dim:
        raise ValueError(f"invalid UniEgo feature width {feature_dim}")
    head = features[:, HEAD_JOINT_IDX * 9:(HEAD_JOINT_IDX + 1) * 9]
    delta = features[:, n_joints * 9:n_joints * 9 + 9]
    head_local_rotation = _cont6d_to_matrix(head[:, :6])
    head_local_position = head[:, 6:9]
    delta_rotation = _cont6d_to_matrix(delta[:, :6])
    delta_translation = delta[:, 6:9]

    # Canonical rotations are Y-only.  Vectorizing their cumulative yaw avoids a Python loop
    # over roughly thirteen million frames in a full-dataset audit.
    delta_yaw = np.arctan2(delta_rotation[:, 0, 2], delta_rotation[:, 2, 2])
    canonical_yaw = np.cumsum(delta_yaw)
    cosine, sine = np.cos(canonical_yaw), np.sin(canonical_yaw)
    canonical_rotation = np.zeros((n_frames, 3, 3), dtype=np.float64)
    canonical_rotation[:, 0, 0] = cosine
    canonical_rotation[:, 0, 2] = sine
    canonical_rotation[:, 1, 1] = 1.0
    canonical_rotation[:, 2, 0] = -sine
    canonical_rotation[:, 2, 2] = cosine

    increments = np.empty_like(delta_translation)
    increments[0] = delta_translation[0]
    if n_frames > 1:
        c_prev, s_prev = cosine[:-1], sine[:-1]
        step = delta_translation[1:]
        increments[1:, 0] = c_prev * step[:, 0] + s_prev * step[:, 2]
        increments[1:, 1] = step[:, 1]
        increments[1:, 2] = -s_prev * step[:, 0] + c_prev * step[:, 2]
    canonical_position = np.cumsum(increments, axis=0)

    head_rotation = canonical_rotation @ head_local_rotation
    head_position = canonical_position + (
        canonical_rotation @ head_local_position[..., None]
    )[..., 0]

    expected_y_only = np.zeros_like(delta_rotation)
    expected_y_only[:, 0, 0] = np.cos(delta_yaw)
    expected_y_only[:, 0, 2] = np.sin(delta_yaw)
    expected_y_only[:, 1, 1] = 1.0
    expected_y_only[:, 2, 0] = -np.sin(delta_yaw)
    expected_y_only[:, 2, 2] = np.cos(delta_yaw)
    delta_off_axis = float(np.max(np.abs(delta_rotation - expected_y_only)))
    return head_rotation, head_position, delta_off_axis


def _smpl_head_rotation(source_npz: Path, timestamps_us: np.ndarray) -> tuple[np.ndarray, bool]:
    with np.load(source_npz) as data:
        source_timestamps = data["timestamps"].astype(np.int64)
        indices = np.searchsorted(source_timestamps, timestamps_us)
        indices = np.clip(indices, 0, len(source_timestamps) - 1)
        exact = bool(np.array_equal(source_timestamps[indices], timestamps_us))
        global_orient = data["global_orient"][indices].astype(np.float64)
        body_pose = data["body_pose"][indices].astype(np.float64).reshape(-1, 23, 3)
    rotation = _axis_angle_to_matrix(global_orient)
    for joint_idx in SMPL_HEAD_CHAIN:
        rotation = rotation @ _axis_angle_to_matrix(body_pose[:, joint_idx - 1])
    return ARIA_Z_UP_TO_KIMODO_Y_UP @ rotation, exact


def _best_motion_lag(head_position: np.ndarray, camera_position: np.ndarray) -> dict[str, Any]:
    head_step = np.diff(head_position, axis=0)
    camera_step = np.diff(camera_position, axis=0)
    scores: dict[int, float] = {}
    for lag in range(-5, 6):
        if lag < 0:
            h, c = head_step[-lag:], camera_step[:lag]
        elif lag > 0:
            h, c = head_step[:-lag], camera_step[lag:]
        else:
            h, c = head_step, camera_step
        moving = (np.linalg.norm(h, axis=-1) + np.linalg.norm(c, axis=-1)) > 0.002
        if np.any(moving):
            scores[lag] = float(np.median(np.linalg.norm(h[moving] - c[moving], axis=-1)))
    best = min(scores, key=scores.get)
    zero = scores.get(0, math.nan)
    return {
        "best_lag_frames": int(best),
        "best_median_step_error_m": scores[best],
        "lag0_median_step_error_m": zero,
        "improvement_over_lag0_m": float(zero - scores[best]),
        "scores_m": {str(k): v for k, v in scores.items()},
    }


def _mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.is_file() else None


def _audit_one(payload: tuple[str, bool, int]) -> dict[str, Any]:
    uuid, source_smpl_orientation, orientation_stride = payload
    subject, sequence = uuid.split("/", 1)
    paths = {
        "proportional": MOTION_ROOT / subject / f"{sequence}.npz",
        "uniego": MOTION_ROOT / "uniego_rep" / subject / f"{sequence}.npz",
        "camera_raw": MOTION_ROOT / "camera" / subject / f"{sequence}.npz",
        "camera_rgb": MOTION_ROOT / "camera_rgb" / subject / f"{sequence}.npz",
        "video_meta": MOTION_ROOT / "video" / subject / f"{sequence}.json",
        "source_smpl": SOURCE_ROOT / subject / sequence / "body/xdata_smpl_neutral.npz",
        "source_soma": SOURCE_ROOT / subject / sequence / "body/xdata_soma.npz",
        "source_soma_meta": SOURCE_ROOT / subject / sequence / "body/xdata_soma_meta.json",
    }
    result: dict[str, Any] = {"uuid": uuid, "errors": [], "warnings": []}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        result["errors"].append(f"missing:{','.join(missing)}")
        return result

    try:
        with np.load(paths["proportional"], allow_pickle=True) as data:
            proportional_timestamps = data["timestamps_us"].astype(np.int64)
            proportional_frames = int(data["local_rot_mats"].shape[0])
            result["proportional"] = {
                "frames": proportional_frames,
                "fps": int(data["fps"]),
                "grounded": bool(data["grounded"]),
                "floor_offset": float(data["floor_offset"]),
            }
        with np.load(paths["uniego"]) as data:
            features = data["features"].astype(np.float64)
            uniego_timestamps = data["timestamps_us"].astype(np.int64)
            result["uniego"] = {
                "frames": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "grounded": bool(data["grounded"]),
            }
        with np.load(paths["camera_raw"]) as data:
            raw_position = data["cam_world_pos"].astype(np.float64)
            raw_rotation = data["cam_world_rot"].astype(np.float64)
            raw_timestamps = data["timestamps_us"].astype(np.int64)
            tdiff_ns = data["tdiff_ns"].astype(np.int64)
        with np.load(paths["camera_rgb"], allow_pickle=True) as data:
            camera_position = data["cam_world_pos_upright"].astype(np.float64)
            camera_rotation = data["cam_world_rot_upright"].astype(np.float64)
            camera_action = data["cam_action_upright_k1"].astype(np.float64)
            rgb_timestamps = data["timestamps_us"].astype(np.int64)
            t_device_rgb = data["T_device_rgb"].astype(np.float64)
            t_rgb_upright = data["T_rgb_upright"].astype(np.float64)

        lengths = {
            "proportional": proportional_frames,
            "uniego": len(features),
            "camera_raw": len(raw_position),
            "camera_rgb": len(camera_position),
            "camera_action_plus_one": len(camera_action) + 1,
        }
        result["lengths"] = lengths
        if len(set(lengths.values())) != 1:
            result["errors"].append("frame_length_mismatch")
        timestamp_equal = {
            "proportional_uniego": bool(np.array_equal(proportional_timestamps, uniego_timestamps)),
            "proportional_camera_raw": bool(np.array_equal(proportional_timestamps, raw_timestamps)),
            "proportional_camera_rgb": bool(np.array_equal(proportional_timestamps, rgb_timestamps)),
        }
        result["timestamp_equal"] = timestamp_equal
        if not all(timestamp_equal.values()):
            result["errors"].append("timestamp_mismatch")

        dt = np.diff(proportional_timestamps).astype(np.float64) / 1e6
        result["timing"] = {
            "dt_seconds": _stats(dt),
            "nonpositive_steps": int(np.count_nonzero(dt <= 0.0)),
            "mps_abs_tdiff_ms": _stats(np.abs(tdiff_ns).astype(np.float64) / 1e6),
        }
        if np.any(dt <= 0.0):
            result["errors"].append("non_monotonic_timestamps")

        head_rotation, head_position, delta_off_axis = _decode_uniego_head(features)
        result["uniego"]["canonical_delta_off_axis_max_abs"] = delta_off_axis
        if delta_off_axis > 2e-4:
            result["warnings"].append("uniego_delta_not_y_only")

        n = min(len(head_position), len(camera_position))
        head_position = head_position[:n]
        head_rotation = head_rotation[:n]
        camera_position_y = (ARIA_Z_UP_TO_KIMODO_Y_UP @ camera_position[:n].T).T
        camera_rotation_y = ARIA_Z_UP_TO_KIMODO_Y_UP @ camera_rotation[:n]
        head_camera_offset = camera_position_y - head_position
        offset_norm = np.linalg.norm(head_camera_offset, axis=-1)
        direct_step_error = np.linalg.norm(
            np.diff(camera_position_y, axis=0) - np.diff(head_position, axis=0), axis=-1
        )
        if n > 96:
            offset_change_97 = np.linalg.norm(head_camera_offset[96:] - head_camera_offset[:-96], axis=-1)
        else:
            offset_change_97 = np.empty(0, dtype=np.float64)
        head_frame_offset = (
            np.swapaxes(head_rotation, -1, -2) @ head_camera_offset[..., None]
        )[..., 0]
        distance_over_0p5 = offset_norm > 0.5
        direct_step_over_0p25 = direct_step_error > 0.25
        relation = _fit_fixed_relation(head_rotation, camera_rotation_y, orientation_stride)
        result["shared_world"] = {
            "head_camera_distance_m": _stats(offset_norm),
            "direct_step_translation_error_m": _stats(direct_step_error),
            "offset_change_over_97_frames_m": _stats(offset_change_97),
            "head_frame_offset_xyz_mean_m": head_frame_offset.mean(axis=0).tolist(),
            "head_frame_offset_residual_m": _stats(
                np.linalg.norm(head_frame_offset - head_frame_offset.mean(axis=0), axis=-1)
            ),
            "largest_head_camera_distances_m": _largest(offset_norm),
            "largest_direct_step_translation_errors_m": _largest(
                direct_step_error, transition=True
            ),
            "head_camera_distance_over_0p5m_frames": int(distance_over_0p5.sum()),
            "head_camera_distance_over_0p5m_ranges": _true_ranges(distance_over_0p5),
            "direct_step_error_over_0p25m_indices": np.flatnonzero(
                direct_step_over_0p25
            ).astype(int).tolist(),
            "fixed_head_camera_rotation": relation["rotation"].tolist(),
            "fixed_head_camera_rotation_deviation_deg": relation["deviation_deg"],
            "fixed_head_camera_heading_error_deg": relation[
                "horizontal_forward_heading_error_deg"
            ],
            "fixed_head_camera_forward_axis_error_deg": relation["forward_axis_error_deg"],
            "fixed_head_camera_up_axis_error_deg": relation["up_axis_error_deg"],
            "window_relation": _window_relation_metrics(
                head_rotation,
                camera_rotation_y,
                relation["rotation"],
                stride=orientation_stride,
            ),
            "best_step_lag": _best_motion_lag(head_position, camera_position_y),
        }
        if float(np.max(offset_norm)) > 0.5:
            result["errors"].append("shared_world_head_camera_distance_implausible")

        # Recompute the static device->upright-RGB transform and the relative action.
        sample = np.arange(0, n, max(1, orientation_stride))
        expected_rotation = raw_rotation[sample] @ t_device_rgb[:3, :3] @ t_rgb_upright[:3, :3]
        expected_position = raw_position[sample] + (
            raw_rotation[sample] @ t_device_rgb[:3, 3, None]
        )[..., 0]
        result["camera_preprocess"] = {
            "raw_to_rgb_position_max_abs_m": float(
                np.max(np.abs(expected_position - camera_position[sample]))
            ),
            "raw_to_rgb_rotation_max_abs": float(
                np.max(np.abs(expected_rotation - camera_rotation[sample]))
            ),
        }
        relative_rotation = np.swapaxes(camera_rotation[:-1], -1, -2) @ camera_rotation[1:]
        relative_translation = (
            np.swapaxes(camera_rotation[:-1], -1, -2)
            @ (camera_position[1:] - camera_position[:-1])[..., None]
        )[..., 0]
        action_rotation = _cont6d_to_matrix(camera_action[:, 3:9])
        camera_step_translation = np.linalg.norm(relative_translation, axis=-1)
        camera_step_rotation = _rotation_angle_deg(relative_rotation)
        head_relative_rotation = np.swapaxes(head_rotation[:-1], -1, -2) @ head_rotation[1:]
        head_relative_translation = (
            np.swapaxes(head_rotation[:-1], -1, -2)
            @ (head_position[1:] - head_position[:-1])[..., None]
        )[..., 0]
        head_step_translation = np.linalg.norm(head_relative_translation, axis=-1)
        head_step_rotation = _rotation_angle_deg(head_relative_rotation)
        bad_camera_translation = camera_step_translation >= 0.25
        bad_camera_rotation = camera_step_rotation >= 30.0
        bad_camera_step = bad_camera_translation | bad_camera_rotation
        bad_head_translation = head_step_translation >= 0.25
        bad_head_rotation = head_step_rotation >= 30.0
        bad_head_step = bad_head_translation | bad_head_rotation
        result["camera_preprocess"].update(
            {
                "action_translation_max_abs_m": float(
                    np.max(np.abs(relative_translation - camera_action[:, :3]))
                ),
                "action_rotation_error_deg": _stats(
                    _rotation_angle_deg(np.swapaxes(action_rotation, -1, -2) @ relative_rotation)
                ),
                "camera_step_translation_m": _stats(camera_step_translation),
                "camera_step_rotation_deg": _stats(camera_step_rotation),
                "largest_camera_step_translations_m": _largest(
                    camera_step_translation, transition=True
                ),
                "largest_camera_step_rotations_deg": _largest(
                    camera_step_rotation, transition=True
                ),
                "implausible_camera_steps": int(
                    np.count_nonzero(bad_camera_step)
                ),
                "implausible_camera_step_indices": np.flatnonzero(bad_camera_step).astype(int).tolist(),
                "camera_translation_step_over_0p25m_indices": np.flatnonzero(
                    bad_camera_translation
                ).astype(int).tolist(),
                "camera_rotation_step_over_30deg_indices": np.flatnonzero(
                    bad_camera_rotation
                ).astype(int).tolist(),
                "motion_head_step_translation_m": _stats(
                    head_step_translation
                ),
                "motion_head_step_rotation_deg": _stats(
                    head_step_rotation
                ),
                "implausible_motion_head_steps": int(np.count_nonzero(bad_head_step)),
                "implausible_motion_head_step_indices": np.flatnonzero(bad_head_step).astype(int).tolist(),
                "motion_head_translation_step_over_0p25m_indices": np.flatnonzero(
                    bad_head_translation
                ).astype(int).tolist(),
                "motion_head_rotation_step_over_30deg_indices": np.flatnonzero(
                    bad_head_rotation
                ).astype(int).tolist(),
            }
        )
        if result["camera_preprocess"]["implausible_camera_steps"]:
            result["warnings"].append("implausible_camera_steps")
        if result["camera_preprocess"]["implausible_motion_head_steps"]:
            result["warnings"].append("implausible_motion_head_steps")

        video_meta = json.loads(paths["video_meta"].read_text())
        result["video"] = {
            "nb_frames": int(video_meta["nb_frames"]),
            "valid_start": int(video_meta["valid_start"]),
            "valid_end": int(video_meta["valid_end"]),
            "n_invalid": int(video_meta["n_invalid"]),
            "length_matches": int(video_meta["nb_frames"]) == proportional_frames,
        }
        if not result["video"]["length_matches"]:
            result["errors"].append("video_length_mismatch")

        soma_meta = json.loads(paths["source_soma_meta"].read_text())
        result["source_soma_fit"] = {
            "script_version": str(soma_meta.get("script_version", "missing")),
            "created_at": soma_meta.get("created_at"),
            "per_vertex_error_cm": soma_meta.get("per_vertex_error_cm", {}),
        }
        mtimes = {name: _mtime(path) for name, path in paths.items()}
        result["mtime_unix"] = mtimes
        result["stale"] = {
            "proportional_older_than_source_soma": bool(
                mtimes["proportional"] < mtimes["source_soma"]
            ),
            "uniego_older_than_proportional": bool(mtimes["uniego"] < mtimes["proportional"]),
            "camera_raw_older_than_proportional": bool(
                mtimes["camera_raw"] < mtimes["proportional"]
            ),
            "camera_rgb_older_than_camera_raw": bool(
                mtimes["camera_rgb"] < mtimes["camera_raw"]
            ),
        }

        if source_smpl_orientation:
            smpl_head_rotation, exact = _smpl_head_rotation(
                paths["source_smpl"], proportional_timestamps
            )
            smpl_camera = _fit_fixed_relation(
                smpl_head_rotation, camera_rotation_y, orientation_stride
            )
            smpl_soma = _fit_fixed_relation(
                smpl_head_rotation, head_rotation, orientation_stride
            )
            result["source_smpl_orientation"] = {
                "timestamps_exact": exact,
                "fixed_smpl_head_camera_rotation": smpl_camera["rotation"].tolist(),
                "fixed_smpl_head_camera_rotation_deviation_deg": smpl_camera[
                    "deviation_deg"
                ],
                "fixed_smpl_head_camera_heading_error_deg": smpl_camera[
                    "horizontal_forward_heading_error_deg"
                ],
                "fixed_smpl_head_soma_head_rotation_deviation_deg": smpl_soma[
                    "deviation_deg"
                ],
            }
            if not exact:
                result["errors"].append("source_smpl_timestamp_mismatch")
    except Exception as exc:  # keep a full scan going and expose the failing sequence
        result["errors"].append(f"exception:{type(exc).__name__}:{exc}")
    return result


def _sequence_uuids(split: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return sorted(explicit)
    all_uuids = sorted(
        f"{path.parent.name}/{path.stem}" for path in MOTION_ROOT.glob("S*/*.npz")
    )
    if split == "all":
        return all_uuids
    split_data = json.loads((MOTION_ROOT / "train_test_split.json").read_text())
    key = "train" if split == "train" else "test"
    selected = set(split_data[key])
    return [uuid for uuid in all_uuids if uuid in selected]


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def values(*keys: str) -> np.ndarray:
        found = []
        for result in results:
            value: Any = result
            try:
                for key in keys:
                    value = value[key]
                found.append(float(value))
            except (KeyError, TypeError, ValueError):
                continue
        return np.asarray(found, dtype=np.float64)

    errors = Counter(error.split(":", 1)[0] for result in results for error in result["errors"])
    warnings = Counter(warning for result in results for warning in result["warnings"])
    lag_histogram = Counter(
        int(result["shared_world"]["best_step_lag"]["best_lag_frames"])
        for result in results
        if "shared_world" in result
    )
    versions = Counter(
        result["source_soma_fit"]["script_version"]
        for result in results
        if "source_soma_fit" in result
    )
    stale = Counter()
    for result in results:
        for key, value in result.get("stale", {}).items():
            if value:
                stale[key] += 1
    return {
        "sequence_count": len(results),
        "sequences_without_errors": sum(not result["errors"] for result in results),
        "error_counts": dict(errors),
        "warning_counts": dict(warnings),
        "soma_script_versions": dict(versions),
        "stale_counts": dict(stale),
        "best_step_lag_histogram": {str(k): v for k, v in sorted(lag_histogram.items())},
        "per_sequence_metric_distributions": {
            "median_head_camera_distance_m": _stats(
                values("shared_world", "head_camera_distance_m", "median")
            ),
            "mean_direct_step_translation_error_mm": _stats(
                1000.0 * values("shared_world", "direct_step_translation_error_m", "mean")
            ),
            "median_offset_change_over_97_frames_m": _stats(
                values("shared_world", "offset_change_over_97_frames_m", "median")
            ),
            "mean_fixed_soma_head_camera_rotation_deviation_deg": _stats(
                values(
                    "shared_world", "fixed_head_camera_rotation_deviation_deg", "mean"
                )
            ),
            "mean_fixed_soma_head_camera_heading_error_deg": _stats(
                values("shared_world", "fixed_head_camera_heading_error_deg", "mean")
            ),
            "mean_within_97frame_window_head_camera_rotation_deviation_deg": _stats(
                values(
                    "shared_world",
                    "window_relation",
                    "within_window_rotation_deviation_mean_deg",
                    "mean",
                )
            ),
            "mean_within_97frame_window_head_camera_heading_error_deg": _stats(
                values(
                    "shared_world",
                    "window_relation",
                    "within_window_heading_error_mean_deg",
                    "mean",
                )
            ),
            "median_mps_abs_tdiff_ms": _stats(
                values("timing", "mps_abs_tdiff_ms", "median")
            ),
            "source_soma_fit_median_cm": _stats(
                values("source_soma_fit", "per_vertex_error_cm", "median")
            ),
            "max_camera_step_translation_m": _stats(
                values("camera_preprocess", "camera_step_translation_m", "max")
            ),
            "max_camera_step_rotation_deg": _stats(
                values("camera_preprocess", "camera_step_rotation_deg", "max")
            ),
            "mean_fixed_source_smpl_head_camera_rotation_deviation_deg": _stats(
                values(
                    "source_smpl_orientation",
                    "fixed_smpl_head_camera_rotation_deviation_deg",
                    "mean",
                )
            ),
            "mean_fixed_source_smpl_head_soma_head_rotation_deviation_deg": _stats(
                values(
                    "source_smpl_orientation",
                    "fixed_smpl_head_soma_head_rotation_deviation_deg",
                    "mean",
                )
            ),
        },
        "worst_sequences": {
            "head_camera_distance": _worst(
                results, ("shared_world", "head_camera_distance_m", "median")
            ),
            "soma_head_camera_orientation": _worst(
                results,
                ("shared_world", "fixed_head_camera_rotation_deviation_deg", "mean"),
            ),
            "direct_step_translation": _worst(
                results, ("shared_world", "direct_step_translation_error_m", "mean")
            ),
            "mps_time_difference": _worst(
                results, ("timing", "mps_abs_tdiff_ms", "max")
            ),
            "source_soma_fit": _worst(
                results, ("source_soma_fit", "per_vertex_error_cm", "median")
            ),
        },
    }


def _worst(
    results: list[dict[str, Any]], keys: tuple[str, ...], count: int = 10
) -> list[dict[str, Any]]:
    ranked = []
    for result in results:
        value: Any = result
        try:
            for key in keys:
                value = value[key]
            value = float(value)
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            ranked.append({"uuid": result["uuid"], "value": value})
    return sorted(ranked, key=lambda item: item["value"], reverse=True)[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--uuid", action="append", help="Audit only this Sxx/sequence; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--orientation-stride", type=int, default=5)
    parser.add_argument("--source-smpl-orientation", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    uuids = _sequence_uuids(args.split, args.uuid)
    if args.limit:
        uuids = uuids[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"[audit] sequences={len(uuids)} split={args.split} workers={args.workers} "
        f"source_smpl_orientation={args.source_smpl_orientation}",
        flush=True,
    )
    payloads = [
        (uuid, args.source_smpl_orientation, args.orientation_stride) for uuid in uuids
    ]
    results = []
    if args.workers <= 1:
        iterator = map(_audit_one, payloads)
        for index, result in enumerate(iterator, 1):
            results.append(result)
            print(f"[{index}/{len(payloads)}] {result['uuid']} errors={result['errors']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_audit_one, payload): payload[0] for payload in payloads}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                if index % 20 == 0 or result["errors"] or index == len(payloads):
                    print(
                        f"[{index}/{len(payloads)}] {result['uuid']} errors={result['errors']}",
                        flush=True,
                    )
    results.sort(key=lambda item: item["uuid"])
    suffix = args.split + ("_source_smpl" if args.source_smpl_orientation else "")
    details_path = args.output / f"details_{suffix}.jsonl"
    with details_path.open("w") as handle:
        for result in results:
            handle.write(json.dumps(result, allow_nan=True) + "\n")
    summary = _aggregate(results)
    summary.update(
        {
            "split": args.split,
            "source_smpl_orientation": args.source_smpl_orientation,
            "orientation_stride": args.orientation_stride,
            "details": str(details_path),
        }
    )
    summary_path = args.output / f"summary_{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(f"[wrote] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
