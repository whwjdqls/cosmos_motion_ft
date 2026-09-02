#!/usr/bin/env python3
"""Select and visualize diverse original-vs-camera-aligned Nymeria windows.

The gallery deliberately includes both automatically ranked clean test windows and
known diagnostic cases.  Since re-canonicalization must not move the body, old and
new skeletons should overlap.  The informative visual difference is the Head-implied
camera orientation; foot-height/speed traces make physical preservation explicit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
import numpy as np

# The Cosmos environment ships imageio-ffmpeg's pinned binary but does not expose a
# system ``ffmpeg`` command.  Point every render worker at that exact executable.
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

from camera_head_recanonicalization import (
    ARIA_Z_UP_TO_KIMODO_Y_UP,
    DELTA_END,
    HEAD_JOINT_IDX,
    camera_rotations_to_kimodo,
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
DEFAULT_MANIFEST = DATA_ROOT / "video" / "manifest_video.jsonl"
DEFAULT_SPLIT = DATA_ROOT / "train_test_split.json"
DEFAULT_FLOOR = DATA_ROOT / "metadata" / "floor_calibration.json"
DEFAULT_QUALITY = DATA_ROOT / "metadata" / "camera_motion_quality_filter_v1_T97.json"
DEFAULT_CALIBRATION = Path(__file__).with_name("head_camera_calibration_train.json")
DEFAULT_OUTPUT = RUN_ROOT / "nymeria_camera_head_recanonicalization_v1" / "qualitative"

PARENTS = (-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18,
           19, 19, 0, 22, 23, 24, 0, 26, 27, 28)
SKIP_JOINTS = {14, 15, 20, 21}
STRUCTURAL_SOURCE_REASONS = {
    "camera_translation_jump",
    "camera_rotation_jump",
    "cross_modal_translation_jump",
    "head_camera_separation",
    "smooth_translation_nonrigid",
    "head_translation_jump",
}
KNOWN_CASES = (
    {
        "category": "known_worst_rotation",
        "uuid": "S04/20230711_s0_frederick_young_act1_2imlal",
        "start": 12779,
        "frames": 97,
    },
    {
        "category": "known_translation_drift",
        "uuid": "S09/20230620_s0_marie_vasquez_act4_dhbf58",
        "start": 7759,
        "frames": 97,
    },
    {
        "category": "known_source_jump_negative_control",
        "uuid": "S17/20230918_s0_kevin_shaw_act2_5g4k0z",
        "start": 1450,
        "frames": 150,
    },
    {
        "category": "known_william_clean_high_divergence",
        "uuid": "S01/20230705_s0_william_davis_act4_h94ovw",
        "start": 0,
        "frames": 97,
    },
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _rotation_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return rotation_angle_deg(np.swapaxes(predicted, -1, -2) @ target)


def _load_floor_contract(path: Path) -> tuple[dict[str, float], float, dict[str, dict[tuple[int, int], str]]]:
    payload = json.loads(path.read_text())
    deltas = {uuid: float(value) for uuid, value in payload.get("deltas", {}).items()}
    global_delta = float(payload.get("global_delta", 0.0))
    drops = {
        uuid: {(int(row[0]), int(row[1])): str(row[2]) for row in rows}
        for uuid, rows in payload.get("dropped_windows", {}).items()
    }
    return deltas, global_delta, drops


def _quality_reasons(path: Path) -> dict[tuple[str, int], set[str]]:
    payload = json.loads(path.read_text())
    return {
        (str(row["uuid"]), int(row["start"])): set(map(str, row.get("reasons", [])))
        for row in payload.get("excluded_windows", [])
    }


def _build_candidates(
    manifest_path: Path,
    split_path: Path,
    floor_path: Path,
    quality_path: Path,
    old_root: Path,
    new_root: Path,
    camera_root: Path,
    frames: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    test_uuids = set(json.loads(split_path.read_text())["test"])
    deltas, global_delta, drops = _load_floor_contract(floor_path)
    quality = _quality_reasons(quality_path)
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    all_windows: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    counts = {
        "test_sequences": len(test_uuids),
        "usable_caption_spans": 0,
        "floor_dropped_spans": 0,
        "unique_aligned_candidates": 0,
    }
    for record in _read_jsonl(manifest_path):
        uuid = record.get("uuid")
        if not uuid:
            continue
        all_windows[uuid] = list(record.get("t2w_windows", []))
        if uuid not in test_uuids:
            continue
        relative = Path(uuid + ".npz")
        if not ((old_root / relative).is_file() and (new_root / relative).is_file() and (camera_root / relative).is_file()):
            continue
        frame_count = int(record.get("nb_frames", 0))
        for caption_window in record.get("t2w_windows", []):
            if not caption_window.get("usable", False) or not caption_window.get("caption"):
                continue
            counts["usable_caption_spans"] += 1
            span_start = int(caption_window["start_frame"])
            raw_end = int(caption_window["end_frame"])
            if (span_start, raw_end) in drops.get(uuid, {}):
                counts["floor_dropped_spans"] += 1
                continue
            raw_offset = caption_window.get("ground_offset_y")
            if raw_offset is None:
                continue
            offset = float(raw_offset) + deltas.get(uuid, global_delta)
            end = min(raw_end, frame_count)
            start = span_start
            while start + frames <= end:
                key = (uuid, start)
                if key not in seen:
                    reasons = quality.get(key, set())
                    by_uuid.setdefault(uuid, []).append(
                        {
                            "uuid": uuid,
                            "start": start,
                            "frames": frames,
                            "end": start + frames,
                            "floor_offset": offset,
                            "caption": str(caption_window.get("caption", "")),
                            "quality_filter_reasons": sorted(reasons),
                            "structural_source_reasons": sorted(reasons & STRUCTURAL_SOURCE_REASONS),
                        }
                    )
                    seen.add(key)
                start += frames
    counts["unique_aligned_candidates"] = len(seen)
    counts["sequences_with_candidates"] = len(by_uuid)
    return by_uuid, all_windows, counts


def _load_sequence(
    uuid: str,
    old_root: Path,
    new_root: Path,
    camera_root: Path,
) -> dict[str, np.ndarray | float]:
    relative = Path(uuid + ".npz")
    with np.load(old_root / relative, allow_pickle=False) as archive:
        old_features = archive["features"]
    with np.load(new_root / relative, allow_pickle=False) as archive:
        new_features = archive["features"]
    with np.load(camera_root / relative, allow_pickle=False) as camera:
        camera_rotation = camera_rotations_to_kimodo(camera["cam_world_rot_upright"])
        camera_position = np.einsum(
            "ij,tj->ti",
            ARIA_Z_UP_TO_KIMODO_Y_UP,
            camera["cam_world_pos_upright"].astype(np.float64),
            optimize=True,
        )
        fps = float(np.asarray(camera["fps"]).item())
    return {
        "old_features": old_features,
        "new_features": new_features,
        "old": decode_uniego(old_features),
        "new": decode_uniego(new_features),
        "camera_rotation": camera_rotation,
        "camera_position": camera_position,
        "fps": fps,
    }


def _case_metrics(
    sequence: dict[str, Any],
    case: dict[str, Any],
    rotation_head_to_camera: np.ndarray,
) -> dict[str, float]:
    start = int(case["start"])
    end = min(start + int(case["frames"]), len(sequence["camera_position"]))
    selector = slice(start, end)
    old = sequence["old"]
    new = sequence["new"]
    camera_position = sequence["camera_position"][selector]
    camera_rotation = sequence["camera_rotation"][selector]
    old_head_rotation = old.world_rotations[selector, HEAD_JOINT_IDX]
    new_head_rotation = new.world_rotations[selector, HEAD_JOINT_IDX]
    old_camera_rotation = old_head_rotation @ rotation_head_to_camera
    new_camera_rotation = new_head_rotation @ rotation_head_to_camera
    camera_steps = np.linalg.norm(np.diff(camera_position, axis=0), axis=-1)
    camera_rotation_steps = rotation_angle_deg(
        np.swapaxes(camera_rotation[:-1], -1, -2) @ camera_rotation[1:]
    )
    path_length = float(camera_steps.sum())
    net_displacement = float(np.linalg.norm(camera_position[-1] - camera_position[0]))
    forward = camera_rotation[:, :, 2]
    heading = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    heading_turn = float(np.abs(np.diff(np.rad2deg(heading))).sum())
    positions = old.world_positions[selector]
    body_relative = positions - positions[:, :1]
    body_speed = np.linalg.norm(np.diff(body_relative, axis=0), axis=-1) * float(sequence["fps"])
    feet = positions[:, FOOT_JOINT_IDX]
    foot_speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * float(sequence["fps"])
    contacts = sequence["old_features"][selector, DELTA_END:] > 0.5
    contact_count = int(contacts[:-1].sum())
    contact_skate = float((foot_speed * contacts[:-1]).sum() / (contact_count + 1e-6))
    position_delta = np.linalg.norm(
        new.world_positions[selector] - old.world_positions[selector], axis=-1
    )
    return {
        "camera_path_length_m": path_length,
        "camera_net_displacement_m": net_displacement,
        "path_efficiency": net_displacement / max(path_length, 1e-8),
        "cumulative_abs_camera_heading_deg": heading_turn,
        "cumulative_camera_rotation_deg": float(camera_rotation_steps.sum()),
        "mean_body_relative_joint_speed_mps": float(body_speed.mean()),
        "mean_contact_foot_skate_mps": contact_skate,
        "max_camera_translation_step_m": float(camera_steps.max()),
        "max_camera_rotation_step_deg": float(camera_rotation_steps.max()),
        "old_camera_rotation_error_mean_deg": float(
            _rotation_error(old_camera_rotation, camera_rotation).mean()
        ),
        "new_camera_rotation_error_mean_deg": float(
            _rotation_error(new_camera_rotation, camera_rotation).mean()
        ),
        "decoded_position_delta_max_m": float(position_delta.max()),
    }


def _rank_candidates(
    candidates_by_uuid: dict[str, list[dict[str, Any]]],
    old_root: Path,
    new_root: Path,
    camera_root: Path,
    rotation_head_to_camera: np.ndarray,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for index, (uuid, cases) in enumerate(sorted(candidates_by_uuid.items()), 1):
        sequence = _load_sequence(uuid, old_root, new_root, camera_root)
        for case in cases:
            ranked.append({**case, **_case_metrics(sequence, case, rotation_head_to_camera)})
        if index % 10 == 0 or index == len(candidates_by_uuid):
            print(
                f"[camhead-viz] ranked {index}/{len(candidates_by_uuid)} test sequences; "
                f"windows={len(ranked)}",
                flush=True,
            )
    return ranked


def _zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    return (values - float(np.mean(values))) / max(scale, 1e-8)


def _choose_diverse(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = [
        row
        for row in ranked
        if not row["structural_source_reasons"]
        and row["max_camera_translation_step_m"] < 0.25
        and row["max_camera_rotation_step_deg"] < 30.0
    ]
    if len(clean) < 8:
        raise ValueError(f"too few structurally clean candidates: {len(clean)}")
    path = np.asarray([row["camera_path_length_m"] for row in clean])
    efficiency = np.asarray([row["path_efficiency"] for row in clean])
    turn = np.asarray([row["cumulative_abs_camera_heading_deg"] for row in clean])
    body = np.asarray([row["mean_body_relative_joint_speed_mps"] for row in clean])
    skate = np.asarray([row["mean_contact_foot_skate_mps"] for row in clean])
    orientation = np.asarray([row["old_camera_rotation_error_mean_deg"] for row in clean])
    used: set[tuple[str, int]] = set()
    selected: list[dict[str, Any]] = []

    def take(category: str, ordering: np.ndarray, eligible: np.ndarray | None = None) -> None:
        indices = np.argsort(ordering)[::-1]
        for index in indices:
            if eligible is not None and not bool(eligible[index]):
                continue
            row = clean[int(index)]
            key = (row["uuid"], int(row["start"]))
            if key in used:
                continue
            selected.append({**row, "category": category, "selection_population": "clean_test"})
            used.add(key)
            return
        raise ValueError(f"could not select a distinct {category} window")

    straight_eligible = (
        (path >= np.quantile(path, 0.70))
        & (efficiency >= 0.80)
        & (turn <= np.quantile(turn, 0.45))
    )
    straight_score = _zscore(path) + _zscore(efficiency) - 0.5 * _zscore(turn)
    take("straight_locomotion", straight_score, straight_eligible)
    take("turning_high_angular", turn)
    take("long_travel", path)
    take("high_human_articulation", body)
    take("high_contact_skate_stress_case", skate)
    take("worst_original_rotation_clean", orientation)
    low_motion_score = -(_zscore(path) + _zscore(body) + 0.25 * _zscore(turn))
    take("low_motion", low_motion_score)

    matrix = np.stack((_zscore(path), _zscore(efficiency), _zscore(turn), _zscore(body)), axis=-1)
    median = np.median(matrix, axis=0)
    typical_score = -np.linalg.norm(matrix - median, axis=-1)
    take("typical_median", typical_score)
    return selected


def _resolve_known_cases(
    known: tuple[dict[str, Any], ...],
    all_manifest_windows: dict[str, list[dict[str, Any]]],
    floor_path: Path,
    old_root: Path,
    new_root: Path,
    camera_root: Path,
    rotation_head_to_camera: np.ndarray,
) -> list[dict[str, Any]]:
    deltas, global_delta, _drops = _load_floor_contract(floor_path)
    resolved: list[dict[str, Any]] = []
    for template in known:
        uuid = str(template["uuid"])
        start = int(template["start"])
        containing = [
            window
            for window in all_manifest_windows.get(uuid, [])
            if int(window.get("start_frame", -1)) <= start < int(window.get("end_frame", -1))
            and window.get("ground_offset_y") is not None
        ]
        if not containing:
            raise ValueError(f"known case has no containing floor window: {uuid}@{start}")
        window = min(
            containing,
            key=lambda row: int(row["end_frame"]) - int(row["start_frame"]),
        )
        case = {
            **template,
            "end": start + int(template["frames"]),
            "floor_offset": float(window["ground_offset_y"]) + deltas.get(uuid, global_delta),
            "caption": str(window.get("caption", "")),
            "quality_filter_reasons": [],
            "structural_source_reasons": [],
            "selection_population": "manual_diagnostic",
        }
        sequence = _load_sequence(uuid, old_root, new_root, camera_root)
        resolved.append({**case, **_case_metrics(sequence, case, rotation_head_to_camera)})
    return resolved


def _equal_axes(axis: Any, arrays: list[np.ndarray]) -> None:
    points = np.concatenate([array[:, (0, 2)] for array in arrays], axis=0)
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    radius = max(float(np.ptp(points, axis=0).max()) * 0.58, 0.25)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_aspect("equal", adjustable="box")


def _draw_skeleton_2d(axis: Any, joints: np.ndarray, color: str = "0.2") -> None:
    centered = joints - joints[0]
    for child, parent in enumerate(PARENTS):
        if parent < 0 or child in SKIP_JOINTS or parent in SKIP_JOINTS:
            continue
        axis.plot(
            [centered[parent, 0], centered[child, 0]],
            [centered[parent, 1], centered[child, 1]],
            color=color,
            lw=1.5,
        )
    axis.scatter(centered[:, 0], centered[:, 1], s=6, color=color)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def _render_contact_sheet(
    cases: list[dict[str, Any]],
    output: Path,
    old_root: Path,
    new_root: Path,
    camera_root: Path,
    rotation_head_to_camera: np.ndarray,
    lever: np.ndarray,
) -> None:
    columns = 4
    rows = int(math.ceil(len(cases) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(18, 4.2 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, case in zip(axes, cases):
        sequence = _load_sequence(case["uuid"], old_root, new_root, camera_root)
        start, end = int(case["start"]), int(case["end"])
        selector = slice(start, end)
        old = sequence["old"]
        new = sequence["new"]
        head_position = old.world_positions[selector, HEAD_JOINT_IDX]
        old_head_rotation = old.world_rotations[selector, HEAD_JOINT_IDX]
        new_head_rotation = new.world_rotations[selector, HEAD_JOINT_IDX]
        measured = sequence["camera_position"][selector]
        old_derived = head_position + np.einsum("tij,j->ti", old_head_rotation, lever)
        new_derived = head_position + np.einsum("tij,j->ti", new_head_rotation, lever)
        origin = measured[0]
        curves = [curve - origin for curve in (measured, old_derived, new_derived, head_position)]
        axis.plot(curves[0][:, 0], curves[0][:, 2], color="tab:blue", lw=2.2, label="measured camera")
        axis.plot(curves[1][:, 0], curves[1][:, 2], color="tab:orange", lw=1.6, label="original Head→camera")
        axis.plot(curves[2][:, 0], curves[2][:, 2], color="tab:green", lw=1.5, ls="--", label="corrected Head→camera")
        axis.plot(curves[3][:, 0], curves[3][:, 2], color="0.35", lw=1.0, alpha=0.7, label="Head position")
        axis.scatter(curves[0][0, 0], curves[0][0, 2], marker="o", color="black", s=20)
        _equal_axes(axis, curves)
        axis.grid(alpha=0.22)
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world z [m]")
        axis.set_title(
            f"{case['category']}\n{case['uuid']}@{start}\n"
            f"path={case['camera_path_length_m']:.2f}m turn={case['cumulative_abs_camera_heading_deg']:.0f}° "
            f"rot {case['old_camera_rotation_error_mean_deg']:.1f}°→{case['new_camera_rotation_error_mean_deg']:.3f}°",
            fontsize=8.5,
        )
    for axis in axes[len(cases):]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10)
    fig.suptitle("Diverse old-vs-camera-aligned trajectories", fontsize=16)
    fig.savefig(output / "diverse_trajectory_contact_sheet.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_keyframes(
    cases: list[dict[str, Any]],
    output: Path,
    old_root: Path,
    new_root: Path,
    camera_root: Path,
) -> None:
    columns = 4
    fig, axes = plt.subplots(
        len(cases), columns, figsize=(12, 2.55 * len(cases)), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(len(cases), columns)
    for row_index, case in enumerate(cases):
        sequence = _load_sequence(case["uuid"], old_root, new_root, camera_root)
        start, end = int(case["start"]), int(case["end"])
        indices = np.linspace(start, end - 1, columns, dtype=int)
        for column, frame in enumerate(indices):
            axis = axes[row_index, column]
            _draw_skeleton_2d(axis, sequence["old"].world_positions[frame])
            delta = np.linalg.norm(
                sequence["new"].world_positions[frame]
                - sequence["old"].world_positions[frame],
                axis=-1,
            ).max()
            axis.set_title(
                (f"{case['category']}\n" if column == 0 else "")
                + f"t={(frame - start) / float(sequence['fps']):.1f}s  max Δpos={delta * 1000:.4f}mm",
                fontsize=8,
            )
    fig.suptitle(
        "Human-motion keyframes — original and corrected skeletons overlap (frontal X/Y view)",
        fontsize=15,
    )
    fig.savefig(output / "diverse_human_motion_keyframes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _draw_skeleton_3d(axis: Any, joints: np.ndarray) -> None:
    for child, parent in enumerate(PARENTS):
        if parent < 0 or child in SKIP_JOINTS or parent in SKIP_JOINTS:
            continue
        axis.plot(
            [joints[parent, 0], joints[child, 0]],
            [joints[parent, 2], joints[child, 2]],
            [joints[parent, 1], joints[child, 1]],
            color="0.16",
            lw=2.0,
        )
    keep = [joint for joint in range(len(joints)) if joint not in SKIP_JOINTS]
    axis.scatter(joints[keep, 0], joints[keep, 2], joints[keep, 1], color="0.1", s=10)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _render_animation(payload: tuple[dict[str, Any], str, str, str, str, list[list[float]], list[float], int]) -> dict[str, Any]:
    case, old_root_text, new_root_text, camera_root_text, output_text, rotation_values, lever_values, frame_stride = payload
    started = time.time()
    output = Path(output_text)
    try:
        sequence = _load_sequence(
            case["uuid"], Path(old_root_text), Path(new_root_text), Path(camera_root_text)
        )
        rotation_head_to_camera = np.asarray(rotation_values, dtype=np.float64)
        lever = np.asarray(lever_values, dtype=np.float64)
        start = int(case["start"])
        end = min(int(case["end"]), len(sequence["camera_position"]))
        selector = slice(start, end)
        old = sequence["old"]
        new = sequence["new"]
        floor_offset = float(case["floor_offset"])
        joints_old = old.world_positions[selector].copy()
        joints_new = new.world_positions[selector].copy()
        joints_old[..., 1] -= floor_offset
        joints_new[..., 1] -= floor_offset
        head_position_world = old.world_positions[selector, HEAD_JOINT_IDX]
        old_head_rotation = old.world_rotations[selector, HEAD_JOINT_IDX]
        new_head_rotation = new.world_rotations[selector, HEAD_JOINT_IDX]
        measured_rotation = sequence["camera_rotation"][selector]
        measured_position = sequence["camera_position"][selector].copy()
        old_camera_position = head_position_world + np.einsum("tij,j->ti", old_head_rotation, lever)
        new_camera_position = head_position_world + np.einsum("tij,j->ti", new_head_rotation, lever)
        measured_position[:, 1] -= floor_offset
        old_camera_position[:, 1] -= floor_offset
        new_camera_position[:, 1] -= floor_offset
        old_camera_rotation = old_head_rotation @ rotation_head_to_camera
        new_camera_rotation = new_head_rotation @ rotation_head_to_camera
        rotation_error_old = _rotation_error(old_camera_rotation, measured_rotation)
        rotation_error_new = _rotation_error(new_camera_rotation, measured_rotation)
        foot_indices = np.asarray(FOOT_JOINT_IDX)
        foot_height_old = joints_old[:, foot_indices, 1].min(axis=-1)
        foot_height_new = joints_new[:, foot_indices, 1].min(axis=-1)
        contacts = sequence["old_features"][selector, DELTA_END:] > 0.5
        foot_speed_old = np.linalg.norm(np.diff(joints_old[:, foot_indices], axis=0), axis=-1) * float(sequence["fps"])
        foot_speed_new = np.linalg.norm(np.diff(joints_new[:, foot_indices], axis=0), axis=-1) * float(sequence["fps"])
        contact_counts = contacts[:-1].sum(axis=-1)
        contact_mean_old = np.divide(
            (foot_speed_old * contacts[:-1]).sum(axis=-1),
            contact_counts,
            out=np.zeros(len(foot_speed_old), dtype=np.float64),
            where=contact_counts > 0,
        )
        contact_mean_new = np.divide(
            (foot_speed_new * contacts[:-1]).sum(axis=-1),
            contact_counts,
            out=np.zeros(len(foot_speed_new), dtype=np.float64),
            where=contact_counts > 0,
        )
        time_axis = np.arange(end - start) / float(sequence["fps"])
        origin = measured_position[0].copy()
        trajectories = [
            measured_position - origin,
            old_camera_position - origin,
            new_camera_position - origin,
        ]
        horizontal = np.concatenate([curve[:, (0, 2)] for curve in trajectories], axis=0)
        horizontal_center = (horizontal.min(axis=0) + horizontal.max(axis=0)) * 0.5
        horizontal_radius = max(float(np.ptp(horizontal, axis=0).max()) * 0.58, 0.35)
        arrow_length = min(max(horizontal_radius * 0.15, 0.12), 0.35)
        render_frames = list(range(0, end - start, frame_stride))
        if render_frames[-1] != end - start - 1:
            render_frames.append(end - start - 1)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.mp4")
        fig = plt.figure(figsize=(14, 9))
        writer = FFMpegWriter(
            fps=float(sequence["fps"]) / frame_stride,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "19"],
            metadata={"title": str(case["category"])},
        )
        # 14x9 inches at 100 dpi yields 1400x900, both even as required by
        # libx264/yuv420p.  An odd render height makes ffmpeg reject the stream.
        with writer.saving(fig, str(temporary), dpi=100):
            for frame in render_frames:
                fig.clear()
                grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
                axis_3d = fig.add_subplot(grid[0, 0], projection="3d")
                _draw_skeleton_3d(axis_3d, joints_old[frame])
                root = joints_old[frame, 0]
                view_half = 1.25
                axis_3d.set_xlim(root[0] - view_half, root[0] + view_half)
                axis_3d.set_ylim(root[2] - view_half, root[2] + view_half)
                axis_3d.set_zlim(min(-0.15, float(joints_old[..., 1].min()) - 0.05), max(2.2, float(joints_old[..., 1].max()) + 0.1))
                axis_3d.set_xlabel("x")
                axis_3d.set_ylabel("z")
                axis_3d.set_zlabel("y")
                axis_3d.view_init(elev=20, azim=-60)
                for position, rotation, color, label in (
                    (measured_position[frame], measured_rotation[frame], "tab:blue", "measured camera +Z"),
                    (old_camera_position[frame], old_camera_rotation[frame], "tab:orange", "original Head→camera +Z"),
                    (new_camera_position[frame], new_camera_rotation[frame], "tab:green", "corrected Head→camera +Z"),
                ):
                    direction = rotation[:, 2] * 0.28
                    axis_3d.quiver(
                        position[0], position[2], position[1],
                        direction[0], direction[2], direction[1],
                        color=color, linewidth=2.2, arrow_length_ratio=0.25, label=label,
                    )
                axis_3d.set_title("Human motion and camera-facing axes")
                axis_3d.legend(fontsize=6, loc="upper left")

                axis = fig.add_subplot(grid[0, 1])
                for curve, color, label in (
                    (trajectories[0], "tab:blue", "measured camera"),
                    (trajectories[1], "tab:orange", "original Head→camera"),
                    (trajectories[2], "tab:green", "corrected Head→camera"),
                ):
                    axis.plot(curve[:, 0], curve[:, 2], color=color, alpha=0.18, lw=1.2)
                    axis.plot(curve[:frame + 1, 0], curve[:frame + 1, 2], color=color, lw=2.0, label=label)
                    axis.scatter(curve[frame, 0], curve[frame, 2], color=color, s=30)
                for rotation, color in (
                    (measured_rotation[frame], "tab:blue"),
                    (old_camera_rotation[frame], "tab:orange"),
                    (new_camera_rotation[frame], "tab:green"),
                ):
                    direction = rotation[:, 2][(0, 2),]
                    norm = max(float(np.linalg.norm(direction)), 1e-8)
                    direction = direction / norm * arrow_length
                    point = trajectories[0 if color == "tab:blue" else 1 if color == "tab:orange" else 2][frame]
                    axis.arrow(point[0], point[2], direction[0], direction[1], color=color, width=0.005, length_includes_head=True)
                axis.set_xlim(horizontal_center[0] - horizontal_radius, horizontal_center[0] + horizontal_radius)
                axis.set_ylim(horizontal_center[1] - horizontal_radius, horizontal_center[1] + horizontal_radius)
                axis.set_aspect("equal", adjustable="box")
                axis.grid(alpha=0.22)
                axis.set_xlabel("x relative to camera frame 0 [m]")
                axis.set_ylabel("z relative to camera frame 0 [m]")
                axis.set_title("Top-view trajectories")
                axis.legend(fontsize=7)

                axis = fig.add_subplot(grid[1, 0])
                axis.plot(time_axis, rotation_error_old, color="tab:orange", lw=2, label="original")
                axis.plot(time_axis, rotation_error_new, color="tab:green", lw=2, label="corrected")
                axis.axvline(time_axis[frame], color="black", lw=1)
                axis.set_xlabel("time [s]")
                axis.set_ylabel("camera rotation error [deg]")
                axis.set_title("Head-implied camera orientation")
                axis.grid(alpha=0.22)
                axis.legend()

                axis = fig.add_subplot(grid[1, 1])
                axis.plot(time_axis, foot_height_old * 100.0, color="tab:orange", lw=2, label="min foot height original")
                axis.plot(time_axis, foot_height_new * 100.0, color="tab:green", lw=1.5, ls="--", label="min foot height corrected")
                axis.axhline(0.0, color="0.4", lw=1)
                axis.axvline(time_axis[frame], color="black", lw=1)
                axis.set_xlabel("time [s]")
                axis.set_ylabel("minimum foot height [cm]")
                axis.grid(alpha=0.22)
                speed_axis = axis.twinx()
                speed_axis.plot(time_axis[:-1], contact_mean_old * 100.0, color="tab:red", alpha=0.75, label="contact speed original")
                speed_axis.plot(time_axis[:-1], contact_mean_new * 100.0, color="tab:blue", alpha=0.75, ls="--", label="contact speed corrected")
                speed_axis.set_ylabel("contact-foot speed [cm/s]")
                handles_a, labels_a = axis.get_legend_handles_labels()
                handles_b, labels_b = speed_axis.get_legend_handles_labels()
                axis.legend(handles_a + handles_b, labels_a + labels_b, fontsize=6, loc="upper right")
                axis.set_title("Physical preservation: feet and floor")

                fig.suptitle(
                    f"{case['category']} | {case['uuid']}@{start} | t={time_axis[frame]:.2f}s\n"
                    f"old/new skeleton positions overlap; max Δ={case['decoded_position_delta_max_m'] * 1000:.4f} mm | "
                    f"rot {case['old_camera_rotation_error_mean_deg']:.2f}°→{case['new_camera_rotation_error_mean_deg']:.4f}°\n"
                    f"{case.get('caption', '')[:140]}",
                    fontsize=11,
                )
                writer.grab_frame()
        os.replace(temporary, output)
        return {
            "status": "ok",
            "category": case["category"],
            "output": str(output),
            "rendered_frames": len(render_frames),
            "seconds": time.time() - started,
        }
    except Exception as error:  # noqa: BLE001
        return {
            "status": "error",
            "category": case.get("category"),
            "output": str(output),
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.time() - started,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--camera-root", type=Path, default=DEFAULT_CAMERA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--floor-calibration", type=Path, default=DEFAULT_FLOOR)
    parser.add_argument("--quality-filter", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-frames", type=int, default=97)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()
    if args.window_frames < 2 or args.frame_stride <= 0 or args.render_workers <= 0:
        parser.error("window-frames >=2; frame-stride and render-workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rotation, calibration = load_rotation_head_to_camera(args.calibration)
    lever = np.asarray(calibration["camera_origin_in_head_m"], dtype=np.float64)

    candidates_by_uuid, all_manifest_windows, candidate_counts = _build_candidates(
        args.manifest,
        args.split_file,
        args.floor_calibration,
        args.quality_filter,
        args.old_root,
        args.new_root,
        args.camera_root,
        args.window_frames,
    )
    ranked = _rank_candidates(
        candidates_by_uuid, args.old_root, args.new_root, args.camera_root, rotation
    )
    automatic = _choose_diverse(ranked)
    known = _resolve_known_cases(
        KNOWN_CASES,
        all_manifest_windows,
        args.floor_calibration,
        args.old_root,
        args.new_root,
        args.camera_root,
        rotation,
    )
    cases = automatic + known
    _render_contact_sheet(
        cases, args.output, args.old_root, args.new_root, args.camera_root, rotation, lever
    )
    _render_keyframes(cases, args.output, args.old_root, args.new_root, args.camera_root)

    video_records: list[dict[str, Any]] = []
    if not args.skip_videos:
        payloads = []
        for case in cases:
            filename = (
                f"{_safe_name(case['category'])}__{_safe_name(case['uuid'])}__"
                f"{int(case['start']):06d}.mp4"
            )
            payloads.append(
                (
                    case,
                    str(args.old_root),
                    str(args.new_root),
                    str(args.camera_root),
                    str(args.output / "videos" / filename),
                    rotation.tolist(),
                    lever.tolist(),
                    args.frame_stride,
                )
            )
        with ProcessPoolExecutor(max_workers=args.render_workers) as executor:
            for index, record in enumerate(executor.map(_render_animation, payloads), 1):
                video_records.append(record)
                print(
                    f"[camhead-viz] rendered {index}/{len(payloads)}: "
                    f"{record['category']} status={record['status']}",
                    flush=True,
                )

    report = {
        "schema_version": 1,
        "kind": "nymeria_camera_head_recanonicalization_qualitative_gallery",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract": {
            "automatic_population": "held-out test T97 caption-aligned windows",
            "normal_case_filter": (
                "reject structural camera/translation source defects; retain old Head-rotation "
                "defects because those are the quantity corrected by camhead_v1"
            ),
            "automatic_categories": [case["category"] for case in automatic],
            "manual_categories": [case["category"] for case in known],
            "candidate_counts": candidate_counts,
        },
        "visual_contract": (
            "old/new skeleton positions should overlap; measured, original Head-implied, and "
            "corrected Head-implied camera paths/+Z axes are shown separately; calibrated foot "
            "height and contact-foot speed traces explicitly check physical preservation"
        ),
        "cases": cases,
        "videos": video_records,
        "artifacts": {
            "trajectory_sheet": str(args.output / "diverse_trajectory_contact_sheet.png"),
            "human_motion_keyframes": str(args.output / "diverse_human_motion_keyframes.png"),
            "video_directory": str(args.output / "videos"),
        },
    }
    _write_json(args.output / "gallery_manifest.json", report)
    errors = [record for record in video_records if record["status"] != "ok"]
    print(json.dumps({
        "output": str(args.output),
        "cases": len(cases),
        "videos_ok": len(video_records) - len(errors),
        "video_errors": errors,
        "categories": [case["category"] for case in cases],
    }, indent=2), flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} video render(s) failed")


if __name__ == "__main__":
    main()
