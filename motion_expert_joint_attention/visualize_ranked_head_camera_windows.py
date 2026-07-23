#!/usr/bin/env python3
"""Rank and visualize clean Nymeria Head/camera training windows.

The ranking population matches the Phase-2/3 aligned T97 index after its existing floor-window
filter, then additionally removes every window flagged by the source-level continuity audit.  The
primary score is the mean framewise SO(3) residual after fitting one fixed Head-to-camera rotation
inside each window::

    Q_t = R_head,t.T @ R_camera,t
    X_w = project_SO3(mean_t(Q_t))
    score_w = mean_t angle(X_w.T @ Q_t)

This separates within-window rotational non-rigidity from a stable actor/session-specific
extrinsic.  Translation is never used for ranking; direct position, step-agreement, and fitted
lever-arm diagnostics are shown alongside the rotational score.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from summarize_nymeria_alignment_quality import (
    DEFAULT_DETAILS,
    DEFAULT_FLOOR_CALIBRATION,
    MOTION_ROOT,
    _flags,
)
from visualize_nymeria_alignment_audit import _load_case


DEFAULT_OUTPUT = Path(
    "/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/"
    "ranked_clean_train_T97"
)
DEFAULT_CALIBRATION = Path(__file__).with_name("head_camera_calibration_train.json")
DEFAULT_IMPACT = DEFAULT_DETAILS.with_name("training_window_impact_T97.json")


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _project_so3_batched(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    reflected = np.linalg.det(rotation) < 0.0
    if np.any(reflected):
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        rotation = u @ vh
    return rotation


def _rotation_angle_deg_from_matrix(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def _heading_error_deg(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    predicted_forward = predicted[..., :, 2]
    actual_forward = actual[..., :, 2]
    predicted_heading = np.arctan2(predicted_forward[..., 0], predicted_forward[..., 2])
    actual_heading = np.arctan2(actual_forward[..., 0], actual_forward[..., 2])
    delta = np.arctan2(
        np.sin(predicted_heading - actual_heading),
        np.cos(predicted_heading - actual_heading),
    )
    return np.abs(np.rad2deg(delta))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _build_clean_train_windows(
    *,
    details_path: Path,
    manifest_path: Path,
    split_path: Path,
    floor_path: Path,
    num_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    audit = {row["uuid"]: row for row in _read_jsonl(details_path)}
    train_uuids = set(json.loads(split_path.read_text())["train"])
    floor_payload = json.loads(floor_path.read_text())
    floor_drops = {
        uuid: {(int(entry[0]), int(entry[1])) for entry in entries}
        for uuid, entries in floor_payload.get("dropped_windows", {}).items()
    }

    counts = defaultdict(int)
    clean: list[dict[str, Any]] = []
    for record in _read_jsonl(manifest_path):
        uuid = record.get("uuid")
        detail = audit.get(uuid)
        if uuid not in train_uuids or detail is None or "shared_world" not in detail:
            continue
        if not record.get("vision_path") or not record.get("camera_path"):
            continue
        frame_count = int(record.get("nb_frames", 0))
        for caption_window in record.get("t2w_windows", []):
            if not caption_window.get("usable", False) or not caption_window.get("caption"):
                continue
            raw_start = int(caption_window["start_frame"])
            raw_end = int(caption_window["end_frame"])
            if (raw_start, raw_end) in floor_drops.get(uuid, set()):
                counts["floor_dropped_caption_spans"] += 1
                continue
            end = min(raw_end, frame_count)
            start = raw_start
            while start + num_frames <= end:
                counts["floor_filtered_aligned_windows"] += 1
                defects = _flags(detail, start, start + num_frames)
                if any(defects.values()):
                    counts["source_quality_dropped_windows"] += 1
                else:
                    clean.append(
                        {
                            "uuid": uuid,
                            "start": start,
                            "end": start + num_frames,
                        }
                    )
                start += num_frames
    counts["clean_aligned_windows"] = len(clean)
    return clean, dict(counts)


def _rank_clean_windows(
    windows: list[dict[str, Any]],
    *,
    num_frames: int,
    global_rotation: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        grouped[row["uuid"]].append(row)

    ranked: list[dict[str, Any]] = []
    frame_offsets = np.arange(num_frames, dtype=np.int64)
    for sequence_index, uuid in enumerate(sorted(grouped)):
        data = _load_case(uuid)
        head_rotation = data["head_rotation"]
        head_position = data["head_position"]
        camera_rotation = data["camera_rotation"]
        camera_position = data["camera_position"]
        relation = np.swapaxes(head_rotation, -1, -2) @ camera_rotation
        raw_offset = camera_position - head_position
        camera_step_length = np.linalg.norm(np.diff(camera_position, axis=0), axis=-1)
        camera_path_prefix = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(camera_step_length))
        )
        rows = grouped[uuid]
        starts = np.asarray([row["start"] for row in rows], dtype=np.int64)

        # Prefix sums make all window means O(number of windows), then NumPy performs one batched
        # 3x3 SVD per sequence.  This avoids a Python SVD loop over roughly 112k windows.
        prefix = np.concatenate(
            (np.zeros((1, 3, 3), dtype=np.float64), np.cumsum(relation, axis=0)),
            axis=0,
        )
        relation_mean = (prefix[starts + num_frames] - prefix[starts]) / num_frames
        fitted = _project_so3_batched(relation_mean)

        # Chunk the frame gather to bound memory for long recordings with many caption windows.
        chunk_size = 2048
        for chunk_start in range(0, len(rows), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(rows))
            chunk_starts = starts[chunk_start:chunk_end]
            chunk_relation = relation[chunk_starts[:, None] + frame_offsets[None, :]]
            chunk_fitted = fitted[chunk_start:chunk_end]
            residual_rotation = (
                np.swapaxes(chunk_fitted, -1, -2)[:, None] @ chunk_relation
            )
            residual = _rotation_angle_deg_from_matrix(residual_rotation)
            global_residual = _rotation_angle_deg_from_matrix(
                global_rotation.T[None, None] @ chunk_relation
            )
            chunk_offset = raw_offset[chunk_starts[:, None] + frame_offsets[None, :]]
            # This is a raw shared-world trajectory comparison. Removing the initial constant
            # camera-to-Head offset is not a fitted transform: the remaining curve is exactly
            # (p_camera[t]-p_camera[0]) - (p_head[t]-p_head[0]).
            raw_trajectory_error = np.linalg.norm(
                chunk_offset - chunk_offset[:, :1], axis=-1
            )
            raw_step_error = np.linalg.norm(np.diff(chunk_offset, axis=1), axis=-1)
            for local_index in range(chunk_end - chunk_start):
                row = rows[chunk_start + local_index]
                values = residual[local_index]
                global_values = global_residual[local_index]
                ranked.append(
                    {
                        **row,
                        "window_fit_rotation": chunk_fitted[local_index].tolist(),
                        "window_fit_rotation_residual_mean_deg": float(values.mean()),
                        "window_fit_rotation_residual_median_deg": float(np.median(values)),
                        "window_fit_rotation_residual_p90_deg": float(
                            np.quantile(values, 0.90)
                        ),
                        "window_fit_rotation_residual_max_deg": float(values.max()),
                        "train_global_rotation_residual_mean_deg": float(
                            global_values.mean()
                        ),
                        "raw_relative_trajectory_rmse_m": float(
                            np.sqrt(np.mean(raw_trajectory_error[local_index] ** 2))
                        ),
                        "raw_relative_trajectory_mean_m": float(
                            raw_trajectory_error[local_index].mean()
                        ),
                        "raw_relative_trajectory_endpoint_m": float(
                            raw_trajectory_error[local_index, -1]
                        ),
                        "raw_relative_trajectory_max_m": float(
                            raw_trajectory_error[local_index].max()
                        ),
                        "raw_step_translation_error_mean_m": float(
                            raw_step_error[local_index].mean()
                        ),
                        "camera_path_length_m": float(
                            camera_path_prefix[
                                chunk_starts[local_index] + num_frames - 1
                            ]
                            - camera_path_prefix[chunk_starts[local_index]]
                        ),
                        "camera_net_displacement_m": float(
                            np.linalg.norm(
                                camera_position[
                                    chunk_starts[local_index] + num_frames - 1
                                ]
                                - camera_position[chunk_starts[local_index]]
                            )
                        ),
                    }
                )
        if (sequence_index + 1) % 50 == 0 or sequence_index + 1 == len(grouped):
            print(
                f"[rank] {sequence_index + 1}/{len(grouped)} sequences, "
                f"{len(ranked)}/{len(windows)} windows",
                flush=True,
            )
    return sorted(
        ranked,
        key=lambda row: (
            row["window_fit_rotation_residual_mean_deg"],
            row["uuid"],
            row["start"],
        ),
    )


def _selected_case_metrics(
    row: dict[str, Any],
    *,
    global_rotation: np.ndarray,
) -> dict[str, Any]:
    data = _load_case(row["uuid"])
    selector = slice(int(row["start"]), int(row["end"]))
    head_rotation = data["head_rotation"][selector]
    head_position = data["head_position"][selector]
    camera_rotation = data["camera_rotation"][selector]
    camera_position = data["camera_position"][selector]
    fitted_rotation = np.asarray(row["window_fit_rotation"], dtype=np.float64)

    relation = np.swapaxes(head_rotation, -1, -2) @ camera_rotation
    fitted_residual = _rotation_angle_deg_from_matrix(fitted_rotation.T @ relation)
    global_residual = _rotation_angle_deg_from_matrix(global_rotation.T @ relation)
    predicted_rotation = head_rotation @ fitted_rotation
    heading_residual = _heading_error_deg(predicted_rotation, camera_rotation)

    offset = camera_position - head_position
    raw_relative_trajectory_error = np.linalg.norm(offset - offset[:1], axis=-1)
    head_frame_offset = (np.swapaxes(head_rotation, -1, -2) @ offset[..., None])[..., 0]
    fitted_lever = head_frame_offset.mean(axis=0)
    predicted_camera_position = head_position + (
        head_rotation @ fitted_lever[:, None]
    )[..., 0]
    fitted_lever_residual = np.linalg.norm(
        predicted_camera_position - camera_position, axis=-1
    )
    distance = np.linalg.norm(offset, axis=-1)
    camera_step = np.diff(camera_position, axis=0)
    head_step = np.diff(head_position, axis=0)
    step_error = np.linalg.norm(camera_step - head_step, axis=-1)
    camera_path_length = float(np.linalg.norm(camera_step, axis=-1).sum())

    return {
        **row,
        "head_rotation": head_rotation,
        "head_position": head_position,
        "camera_rotation": camera_rotation,
        "camera_position": camera_position,
        "predicted_rotation": predicted_rotation,
        "predicted_camera_position": predicted_camera_position,
        "fitted_rotation_residual_deg": fitted_residual,
        "global_rotation_residual_deg": global_residual,
        "heading_residual_deg": heading_residual,
        "head_camera_distance_m": distance,
        "raw_relative_trajectory_error_m": raw_relative_trajectory_error,
        "direct_step_translation_error_m": step_error,
        "fitted_lever_residual_m": fitted_lever_residual,
        "fitted_lever_m": fitted_lever,
        "camera_path_length_m": camera_path_length,
        "camera_net_displacement_m": float(
            np.linalg.norm(camera_position[-1] - camera_position[0])
        ),
        "position_offset_change_m": float(np.linalg.norm(offset[-1] - offset[0])),
    }


def _equal_top_view(axis: Any, cases: list[np.ndarray], *, minimum_radius: float = 0.25) -> None:
    points = np.concatenate([case[:, (0, 2)] for case in cases], axis=0)
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    radius = max(float(np.ptp(points, axis=0).max()) * 0.58, minimum_radius)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_aspect("equal", adjustable="box")


def _draw_direction(
    axis: Any,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    color: str,
    length: float,
    alpha: float = 0.85,
) -> None:
    direction = rotation[:, 2][[0, 2]]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return
    direction = direction / norm * length
    axis.arrow(
        position[0],
        position[2],
        direction[0],
        direction[1],
        color=color,
        width=max(length * 0.035, 0.003),
        head_width=max(length * 0.22, 0.025),
        length_includes_head=True,
        alpha=alpha,
    )


def _ranking_text(case: dict[str, Any], ranking: str) -> str:
    if ranking == "rotation":
        return (
            f"window-fit SO(3) mean/p90/max="
            f"{case['window_fit_rotation_residual_mean_deg']:.2f}/"
            f"{case['window_fit_rotation_residual_p90_deg']:.2f}/"
            f"{case['window_fit_rotation_residual_max_deg']:.2f} deg"
        )
    if ranking == "raw_translation":
        return (
            f"raw relative trajectory RMSE/endpoint/max="
            f"{case['raw_relative_trajectory_rmse_m']:.3f}/"
            f"{case['raw_relative_trajectory_endpoint_m']:.3f}/"
            f"{case['raw_relative_trajectory_max_m']:.3f} m"
        )
    raise ValueError(f"unknown ranking {ranking!r}")


def _render_case(
    case: dict[str, Any],
    output: Path,
    *,
    cohort: str,
    rank: int,
    ranking: str,
) -> None:
    camera = case["camera_position"] - case["camera_position"][0]
    head = case["head_position"] - case["camera_position"][0]
    predicted = case["predicted_camera_position"] - case["camera_position"][0]
    time = np.arange(len(camera), dtype=np.float64) / 20.0

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.2), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(camera[:, 0], camera[:, 2], color="tab:blue", lw=2.4, label="upright RGB camera")
    axis.plot(head[:, 0], head[:, 2], color="tab:orange", lw=1.8, label="SOMA Head joint")
    axis.plot(
        predicted[:, 0],
        predicted[:, 2],
        color="tab:green",
        lw=1.5,
        label="Head + per-window fitted lever",
    )
    axis.scatter(camera[0, 0], camera[0, 2], color="black", s=26, zorder=5)
    axis.set_title("Direct shared-world position trajectories")
    axis.set_xlabel("world x relative to camera frame 0 [m]")
    axis.set_ylabel("world z relative to camera frame 0 [m]")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=9)
    _equal_top_view(axis, [camera, head, predicted])

    axis = axes[0, 1]
    axis.plot(camera[:, 0], camera[:, 2], color="tab:blue", lw=2.2, label="camera path")
    axis.plot(predicted[:, 0], predicted[:, 2], color="tab:orange", lw=1.8, label="motion-derived path")
    arrow_indices = np.linspace(0, len(camera) - 1, min(9, len(camera))).astype(int)
    points = np.concatenate((camera[:, (0, 2)], predicted[:, (0, 2)]), axis=0)
    arrow_length = min(max(float(np.ptp(points, axis=0).max()) * 0.08, 0.08), 0.22)
    for index in arrow_indices:
        _draw_direction(
            axis,
            camera[index],
            case["camera_rotation"][index],
            color="tab:blue",
            length=arrow_length,
        )
        _draw_direction(
            axis,
            predicted[index],
            case["predicted_rotation"][index],
            color="tab:orange",
            length=arrow_length,
        )
    axis.set_title("Camera +Z vs Head-implied camera +Z (per-window fixed rotation)")
    axis.set_xlabel("world x relative to camera frame 0 [m]")
    axis.set_ylabel("world z relative to camera frame 0 [m]")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=9)
    _equal_top_view(axis, [camera, predicted])

    axis = axes[1, 0]
    axis.plot(
        time,
        case["fitted_rotation_residual_deg"],
        color="tab:red",
        lw=2.0,
        label="residual to per-window fixed rotation",
    )
    axis.plot(
        time,
        case["global_rotation_residual_deg"],
        color="tab:purple",
        lw=1.5,
        alpha=0.82,
        label="residual to train-global rotation",
    )
    axis.plot(
        time,
        case["heading_residual_deg"],
        color="tab:green",
        lw=1.4,
        alpha=0.82,
        label="horizontal +Z residual (window fit)",
    )
    axis.set_xlabel("time [s]")
    axis.set_ylabel("rotation error [deg]")
    axis.set_title("Head-to-camera orientation inconsistency")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    axis.plot(
        time,
        case["head_camera_distance_m"],
        color="tab:blue",
        lw=1.8,
        label="camera-origin to Head distance",
    )
    axis.plot(
        time,
        case["raw_relative_trajectory_error_m"],
        color="tab:red",
        lw=1.6,
        label="raw relative-trajectory disagreement",
    )
    axis.plot(
        time,
        case["fitted_lever_residual_m"],
        color="tab:green",
        lw=1.6,
        label="per-window lever residual",
    )
    axis.set_xlabel("time [s]")
    axis.set_ylabel("position [m]")
    axis.grid(alpha=0.22)
    twin = axis.twinx()
    twin.plot(
        time[1:],
        1000.0 * case["direct_step_translation_error_m"],
        color="tab:orange",
        lw=1.4,
        alpha=0.78,
        label="world-step disagreement",
    )
    twin.set_ylabel("step disagreement [mm]", color="tab:orange")
    twin.tick_params(axis="y", labelcolor="tab:orange")
    lines, labels = axis.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    axis.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")
    axis.set_title("Translation remains a separate diagnostic")

    fig.suptitle(
        f"{cohort} #{rank:02d}: {case['uuid']} frames {case['start']}:{case['end']}\n"
        f"{_ranking_text(case, ranking)} | "
        f"global mean={case['train_global_rotation_residual_mean_deg']:.2f} deg | "
        f"d50={np.median(case['head_camera_distance_m']):.3f} m | "
        f"lever residual mean={case['fitted_lever_residual_m'].mean() * 1000.0:.1f} mm | "
        f"camera path={case['camera_path_length_m']:.2f} m",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _render_sheet(
    cases: list[dict[str, Any]], output: Path, *, cohort: str, ranking: str
) -> None:
    columns = 5
    rows = int(math.ceil(len(cases) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(22, 4.1 * rows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for rank, (axis, case) in enumerate(zip(axes_flat, cases), start=1):
        camera = case["camera_position"] - case["camera_position"][0]
        head = case["head_position"] - case["camera_position"][0]
        axis.plot(camera[:, 0], camera[:, 2], color="tab:blue", lw=2.1, label="camera")
        axis.plot(head[:, 0], head[:, 2], color="tab:orange", lw=1.6, label="Head")
        for index in np.linspace(0, len(camera) - 1, 5).astype(int):
            _draw_direction(
                axis,
                camera[index],
                case["camera_rotation"][index],
                color="tab:blue",
                length=0.10,
                alpha=0.70,
            )
            _draw_direction(
                axis,
                head[index],
                case["predicted_rotation"][index],
                color="tab:orange",
                length=0.10,
                alpha=0.70,
            )
        axis.set_title(
            f"#{rank:02d} {case['uuid'].split('/', 1)[0]} "
            f"{case['uuid'].rsplit('_', 1)[-1]}@{case['start']}\n"
            + (
                f"rot={case['window_fit_rotation_residual_mean_deg']:.1f} deg "
                f"global={case['train_global_rotation_residual_mean_deg']:.1f} deg\n"
                if ranking == "rotation"
                else f"raw RMSE={case['raw_relative_trajectory_rmse_m']:.3f}m "
                f"end={case['raw_relative_trajectory_endpoint_m']:.3f}m\n"
            )
            + f"d50={np.median(case['head_camera_distance_m']):.3f}m "
            f"step={case['direct_step_translation_error_m'].mean() * 1000.0:.1f}mm "
            f"path={case['camera_path_length_m']:.2f}m",
            fontsize=8.5,
        )
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=7)
        _equal_top_view(axis, [camera, head])
    for axis in axes_flat[len(cases):]:
        axis.axis("off")
    fig.suptitle(
        f"{cohort} clean train T97 windows by "
        + (
            "within-window fixed Head-to-camera SO(3) residual\n"
            if ranking == "rotation"
            else "raw origin-relative trajectory RMSE (no rotation/scale/lever fit)\n"
        )
        + "blue: upright RGB camera and +Z; orange: SOMA Head path and Head-implied camera +Z",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _render_montage_animation(
    cases: list[dict[str, Any]], output: Path, *, cohort: str, ranking: str
) -> None:
    columns = 5
    rows = int(math.ceil(len(cases) / columns))
    limits = []
    for case in cases:
        camera = case["camera_position"] - case["camera_position"][0]
        head = case["head_position"] - case["camera_position"][0]
        points = np.concatenate((camera[:, (0, 2)], head[:, (0, 2)]), axis=0)
        center = (points.min(axis=0) + points.max(axis=0)) * 0.5
        radius = max(float(np.ptp(points, axis=0).max()) * 0.60, 0.25)
        limits.append((center, radius))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20, 14))
    writer = FFMpegWriter(
        fps=20,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "20"],
        metadata={"title": f"{cohort} Head-camera windows"},
    )
    frame_count = min(len(case["camera_position"]) for case in cases)
    with writer.saving(fig, str(output), dpi=100):
        for frame in range(frame_count):
            fig.clear()
            axes = fig.subplots(rows, columns).reshape(-1)
            for rank, (axis, case, (center, radius)) in enumerate(
                zip(axes, cases, limits), start=1
            ):
                camera = case["camera_position"] - case["camera_position"][0]
                head = case["head_position"] - case["camera_position"][0]
                axis.plot(camera[:, 0], camera[:, 2], color="tab:blue", alpha=0.15, lw=1.0)
                axis.plot(head[:, 0], head[:, 2], color="tab:orange", alpha=0.15, lw=1.0)
                axis.plot(camera[:frame + 1, 0], camera[:frame + 1, 2], color="tab:blue", lw=1.8)
                axis.plot(head[:frame + 1, 0], head[:frame + 1, 2], color="tab:orange", lw=1.5)
                arrow_length = min(max(radius * 0.18, 0.08), 0.20)
                _draw_direction(
                    axis,
                    camera[frame],
                    case["camera_rotation"][frame],
                    color="tab:blue",
                    length=arrow_length,
                )
                _draw_direction(
                    axis,
                    head[frame],
                    case["predicted_rotation"][frame],
                    color="tab:orange",
                    length=arrow_length,
                )
                axis.set_xlim(center[0] - radius, center[0] + radius)
                axis.set_ylim(center[1] - radius, center[1] + radius)
                axis.set_aspect("equal", adjustable="box")
                axis.set_title(
                    f"#{rank:02d} {case['uuid'].split('/', 1)[0]}@{case['start']} | "
                    + (
                        f"rot {case['window_fit_rotation_residual_mean_deg']:.1f} deg"
                        if ranking == "rotation"
                        else f"raw RMSE {case['raw_relative_trajectory_rmse_m']:.3f} m"
                    ),
                    fontsize=8,
                )
                axis.grid(alpha=0.15)
                axis.tick_params(labelsize=6)
            for axis in axes[len(cases):]:
                axis.axis("off")
            fig.suptitle(
                f"{cohort} clean train T97 Head-camera windows | "
                f"frame {frame:02d}/{frame_count - 1} ({frame / 20.0:.2f}s)\n"
                "blue: upright RGB camera/+Z; orange: SOMA Head path and fitted Head-implied camera +Z\n"
                + (
                    "ranked by rotational non-rigidity"
                    if ranking == "rotation"
                    else "ranked by raw trajectory disagreement; paths have no fitted transform"
                ),
                fontsize=14,
            )
            fig.subplots_adjust(left=0.035, right=0.985, bottom=0.035, top=0.92, hspace=0.34)
            writer.grab_frame()
    plt.close(fig)


def _serializable_case(case: dict[str, Any]) -> dict[str, Any]:
    array_keys = {
        "head_rotation",
        "head_position",
        "camera_rotation",
        "camera_position",
        "predicted_rotation",
        "predicted_camera_position",
        "fitted_rotation_residual_deg",
        "global_rotation_residual_deg",
        "heading_residual_deg",
        "head_camera_distance_m",
        "raw_relative_trajectory_error_m",
        "direct_step_translation_error_m",
        "fitted_lever_residual_m",
    }
    payload = {key: value for key, value in case.items() if key not in array_keys}
    payload.update(
        {
            "fitted_lever_m": case["fitted_lever_m"].tolist(),
            "head_camera_distance_m": _stats(case["head_camera_distance_m"]),
            "raw_relative_trajectory_error_m": _stats(
                case["raw_relative_trajectory_error_m"]
            ),
            "direct_step_translation_error_m": _stats(
                case["direct_step_translation_error_m"]
            ),
            "fitted_lever_residual_m": _stats(case["fitted_lever_residual_m"]),
            "fitted_rotation_residual_deg": _stats(
                case["fitted_rotation_residual_deg"]
            ),
            "global_rotation_residual_deg": _stats(
                case["global_rotation_residual_deg"]
            ),
            "heading_residual_deg": _stats(case["heading_residual_deg"]),
        }
    )
    return payload


def _take_unique(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select unique physical windows while preserving ranking order."""
    selected = []
    seen = set()
    for row in rows:
        key = (row["uuid"], int(row["start"]), int(row["end"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == count:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument(
        "--manifest", type=Path, default=MOTION_ROOT / "video/manifest_video.jsonl"
    )
    parser.add_argument(
        "--split-file", type=Path, default=MOTION_ROOT / "train_test_split.json"
    )
    parser.add_argument(
        "--floor-calibration", type=Path, default=DEFAULT_FLOOR_CALIBRATION
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--impact-report", type=Path, default=DEFAULT_IMPACT)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--best-min-camera-path-m",
        type=float,
        default=0.5,
        help="Minimum raw camera path length for non-trivial best-case visualizations",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-animations", action="store_true")
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.best_min_camera_path_m < 0.0:
        raise ValueError("--best-min-camera-path-m must be non-negative")

    calibration = json.loads(args.calibration.read_text())
    global_rotation = np.asarray(
        calibration["rotation_head_to_upright_camera"], dtype=np.float64
    )
    clean_windows, counts = _build_clean_train_windows(
        details_path=args.details,
        manifest_path=args.manifest,
        split_path=args.split_file,
        floor_path=args.floor_calibration,
        num_frames=args.num_frames,
    )
    if args.impact_report.is_file():
        expected = json.loads(args.impact_report.read_text())[
            "phase23_aligned_T97_floor_filtered"
        ]["train"]
        if counts["floor_filtered_aligned_windows"] != int(expected["windows"]):
            raise RuntimeError(
                "reconstructed floor-filtered population differs from audit report: "
                f"{counts['floor_filtered_aligned_windows']} != {expected['windows']}"
            )
        if counts["source_quality_dropped_windows"] != int(expected["any_defect"]):
            raise RuntimeError(
                "reconstructed source-quality exclusions differ from audit report: "
                f"{counts['source_quality_dropped_windows']} != {expected['any_defect']}"
            )
    print(f"[population] {json.dumps(counts, sort_keys=True)}", flush=True)

    ranked = _rank_clean_windows(
        clean_windows,
        num_frames=args.num_frames,
        global_rotation=global_rotation,
    )
    count = min(args.count, len(ranked))
    moving_rotation_ranked = [
        row for row in ranked
        if row["camera_path_length_m"] >= args.best_min_camera_path_m
    ]
    rotation_best_rows = _take_unique(moving_rotation_ranked, count)
    rotation_worst_rows = _take_unique(list(reversed(ranked)), count)
    raw_ranked = sorted(
        ranked,
        key=lambda row: (
            row["raw_relative_trajectory_rmse_m"],
            row["uuid"],
            row["start"],
        ),
    )
    moving_raw_ranked = [
        row for row in raw_ranked
        if row["camera_path_length_m"] >= args.best_min_camera_path_m
    ]
    raw_best_rows = _take_unique(moving_raw_ranked, count)
    raw_worst_rows = _take_unique(list(reversed(raw_ranked)), count)

    cohorts = {
        "rotation_worst": (
            "ROTATION WORST",
            "rotation",
            rotation_worst_rows,
        ),
        "rotation_best": (
            "ROTATION BEST MOVING",
            "rotation",
            rotation_best_rows,
        ),
        "raw_translation_worst": (
            "RAW TRANSLATION WORST",
            "raw_translation",
            raw_worst_rows,
        ),
        "raw_translation_best": (
            "RAW TRANSLATION BEST MOVING",
            "raw_translation",
            raw_best_rows,
        ),
    }
    selected_cases = {
        name: [
            _selected_case_metrics(row, global_rotation=global_rotation) for row in rows
        ]
        for name, (_, _, rows) in cohorts.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    for cohort_name, (cohort_label, ranking, _) in cohorts.items():
        cases = selected_cases[cohort_name]
        case_dir = args.output / cohort_name
        for rank, case in enumerate(cases, start=1):
            name = f"{rank:02d}_{case['uuid'].replace('/', '__')}_{case['start']}.png"
            _render_case(
                case,
                case_dir / name,
                cohort=cohort_label,
                rank=rank,
                ranking=ranking,
            )
        _render_sheet(
            cases,
            args.output / f"{cohort_name}20_contact_sheet.png",
            cohort=cohort_label,
            ranking=ranking,
        )
        if not args.skip_animations:
            _render_montage_animation(
                cases,
                args.output / f"{cohort_name}20_montage.mp4",
                cohort=cohort_label,
                ranking=ranking,
            )

    rotation_scores = np.asarray(
        [row["window_fit_rotation_residual_mean_deg"] for row in ranked],
        dtype=np.float64,
    )
    raw_translation_scores = np.asarray(
        [row["raw_relative_trajectory_rmse_m"] for row in ranked],
        dtype=np.float64,
    )
    summary = {
        "contract": {
            "population": (
                "Exact Phase-2/3 train aligned-T97 construction after existing floor-caption "
                "exclusions, then excluding the conservative source-quality union from "
                "training_window_impact_T97.json. The active dataset currently applies only "
                "the first of these two filters."
            ),
            "primary_ranking": (
                "mean framewise SO(3) residual after fitting one fixed Head-to-upright-RGB-camera "
                "rotation independently within each 97-frame window"
            ),
            "raw_translation_ranking": (
                "RMSE of (p_camera[t]-p_camera[0])-(p_head[t]-p_head[0]) in the direct shared "
                "world frame; no rotation, scale, or lever fit is applied"
            ),
            "best_case_selection": (
                "lowest-error unique physical windows among clips whose raw camera path length "
                f"is at least {args.best_min_camera_path_m:.3f} m; this avoids ranking static "
                "clips as the most informative best examples"
            ),
            "position_contract": (
                "direct shared-world upright RGB camera origin and independently decoded UniEgo "
                "SOMA Head joint after only the fixed Aria-Z-up to Kimodo-Y-up basis change"
            ),
        },
        "inputs": {
            "details": str(args.details),
            "manifest": str(args.manifest),
            "split_file": str(args.split_file),
            "floor_calibration": str(args.floor_calibration),
            "calibration": str(args.calibration),
            "impact_report": str(args.impact_report),
        },
        "population_counts": counts,
        "best_visualization_min_camera_path_m": args.best_min_camera_path_m,
        "windows_meeting_best_movement_gate": len(moving_rotation_ranked),
        "rotation_score_distribution_deg": _stats(rotation_scores),
        "raw_translation_score_distribution_m": _stats(raw_translation_scores),
        "cohorts": {
            name: [_serializable_case(case) for case in selected_cases[name]]
            for name in cohorts
        },
        "outputs": {
            name: {
                "contact_sheet": str(args.output / f"{name}20_contact_sheet.png"),
                "montage": (
                    None
                    if args.skip_animations
                    else str(args.output / f"{name}20_montage.mp4")
                ),
                "case_dir": str(args.output / name),
            }
            for name in cohorts
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "population_counts": counts,
        "rotation_score_distribution_deg": summary["rotation_score_distribution_deg"],
        "raw_translation_score_distribution_m": summary[
            "raw_translation_score_distribution_m"
        ],
        "cohort_first": {
            name: rows[0] for name, rows in summary["cohorts"].items()
        },
        "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
