#!/usr/bin/env python3
"""Render corrected source-level camera/motion alignment diagnostics.

Unlike ``visualize_gt_head_camera_alignment.py``, the left panel here compares camera and motion
directly in their stored shared world frame.  A second panel separately shows what happens when the
motion-derived camera path is rotated to force its frame-0 orientation to the measured camera: this
is the operation that exposes the noisy/non-rigid Head-orientation relation used by the Phase-3
head-camera loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from audit_nymeria_camera_motion import (
    ARIA_Z_UP_TO_KIMODO_Y_UP,
    MOTION_ROOT,
    _decode_uniego_head,
    _rotation_angle_deg,
)


DEFAULT_DETAILS = Path(
    "/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/"
    "details_all.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/plots"
)
DEFAULT_CALIBRATION = Path(__file__).with_name("head_camera_calibration_train.json")
DEFAULT_CLEAN_WINDOWS = Path(
    "/weka/jungbin/cosmos_motion_ft_runs/"
    "ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/"
    "eval_full71_step200000_unipc30/motion_clean71_windows.json"
)


def _stats(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)), float(np.quantile(array, 0.90)), float(array.max())


def _load_case(uuid: str) -> dict[str, np.ndarray]:
    subject, sequence = uuid.split("/", 1)
    with np.load(MOTION_ROOT / "uniego_rep" / subject / f"{sequence}.npz") as data:
        features = data["features"].astype(np.float64)
    with np.load(MOTION_ROOT / "camera_rgb" / subject / f"{sequence}.npz") as data:
        camera_position = data["cam_world_pos_upright"].astype(np.float64)
        camera_rotation = data["cam_world_rot_upright"].astype(np.float64)
    head_rotation, head_position, _ = _decode_uniego_head(features)
    return {
        "head_rotation": head_rotation,
        "head_position": head_position,
        "camera_rotation": ARIA_Z_UP_TO_KIMODO_Y_UP @ camera_rotation,
        "camera_position": (ARIA_Z_UP_TO_KIMODO_Y_UP @ camera_position.T).T,
    }


def _heading_error_deg(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    predicted_forward = predicted[:, :, 2]
    actual_forward = actual[:, :, 2]
    predicted_heading = np.arctan2(predicted_forward[:, 0], predicted_forward[:, 2])
    actual_heading = np.arctan2(actual_forward[:, 0], actual_forward[:, 2])
    delta = np.arctan2(
        np.sin(predicted_heading - actual_heading),
        np.cos(predicted_heading - actual_heading),
    )
    return np.abs(np.rad2deg(delta))


def _equal_top_axes(axis: Any, arrays: list[np.ndarray]) -> None:
    points = np.concatenate([array[:, (0, 2)] for array in arrays], axis=0)
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    radius = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 0.25) * 0.58
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_aspect("equal", adjustable="box")


def _direct_window(uuid: str, start: int, frames: int) -> dict[str, np.ndarray]:
    data = _load_case(uuid)
    end = min(start + frames, len(data["head_position"]))
    if end - start < 2:
        raise ValueError(f"short direct-visualization window {uuid}@{start}:{end}")
    selector = slice(start, end)
    return {key: value[selector] for key, value in data.items()}


def _direct_metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    distance = np.linalg.norm(data["camera_position"] - data["head_position"], axis=-1)
    camera_step = np.linalg.norm(np.diff(data["camera_position"], axis=0), axis=-1)
    head_step = np.linalg.norm(np.diff(data["head_position"], axis=0), axis=-1)
    return {
        "median_head_camera_distance_m": float(np.median(distance)),
        "max_head_camera_distance_m": float(distance.max()),
        "max_camera_step_m": float(camera_step.max()),
        "max_head_step_m": float(head_step.max()),
    }


def render_clean71_sheet(windows_path: Path, output: Path) -> list[dict[str, Any]]:
    windows = json.loads(windows_path.read_text())
    columns = 8
    rows_count = int(np.ceil(len(windows) / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(24, 2.65 * rows_count),
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).reshape(-1)
    summaries: list[dict[str, Any]] = []
    for axis, window in zip(axes_flat, windows):
        uuid = str(window["uuid"])
        start = int(window["start"])
        frames = int(window.get("num_frames", 97))
        data = _direct_window(uuid, start, frames)
        camera = data["camera_position"] - data["camera_position"][0]
        head = data["head_position"] - data["camera_position"][0]
        metrics = _direct_metrics(data)
        summaries.append({"uuid": uuid, "start": start, "frames": len(camera), **metrics})
        axis.plot(camera[:, 0], camera[:, 2], color="tab:blue", lw=1.7)
        axis.plot(head[:, 0], head[:, 2], color="tab:orange", lw=1.3)
        axis.scatter(camera[0, 0], camera[0, 2], color="black", s=10, zorder=5)
        axis.set_title(
            f"{uuid.split('/', 1)[0]} {uuid.rsplit('_', 1)[-1]}@{start}\n"
            f"d50={metrics['median_head_camera_distance_m']:.3f}m "
            f"dmax={metrics['max_head_camera_distance_m']:.3f}m",
            fontsize=7.5,
        )
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=6)
        _equal_top_axes(axis, [camera, head])
    for axis in axes_flat[len(windows):]:
        axis.axis("off")
    fig.suptitle(
        "Motion-clean71 direct stored-world trajectories "
        "(blue: upright RGB camera origin, orange: SOMA Head joint)",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summaries


def render_direct_animation(
    uuid: str,
    start: int,
    frames: int,
    output: Path,
    *,
    label: str,
) -> dict[str, Any]:
    data = _direct_window(uuid, start, frames)
    camera_position = data["camera_position"]
    camera_rotation = data["camera_rotation"]
    head_position = data["head_position"]
    head_rotation = data["head_rotation"]
    origin = camera_position[0]
    camera = camera_position - origin
    head = head_position - origin
    time = np.arange(len(camera), dtype=np.float64) / 20.0
    distance = np.linalg.norm(camera_position - head_position, axis=-1)
    camera_step = np.linalg.norm(np.diff(camera_position, axis=0), axis=-1)
    head_step = np.linalg.norm(np.diff(head_position, axis=0), axis=-1)
    metrics = _direct_metrics(data)

    horizontal = np.concatenate((camera[:, (0, 2)], head[:, (0, 2)]), axis=0)
    horizontal_center = (horizontal.min(axis=0) + horizontal.max(axis=0)) * 0.5
    horizontal_radius = max(float(np.ptp(horizontal, axis=0).max()) * 0.58, 0.25)
    vertical_points = np.concatenate((camera, head), axis=0)
    y_min, y_max = float(vertical_points[:, 1].min()), float(vertical_points[:, 1].max())
    y_pad = max((y_max - y_min) * 0.12, 0.08)
    arrow_length = min(max(horizontal_radius * 0.14, 0.10), 0.28)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 8.8))
    writer = FFMpegWriter(
        fps=20,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        metadata={"title": label},
    )
    with writer.saving(fig, str(output), dpi=100):
        for frame in range(len(camera)):
            fig.clear()
            grid = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

            axis = fig.add_subplot(grid[0, 0])
            axis.plot(camera[:, 0], camera[:, 2], color="tab:blue", alpha=0.18, lw=1.2)
            axis.plot(head[:, 0], head[:, 2], color="tab:orange", alpha=0.18, lw=1.2)
            axis.plot(camera[:frame + 1, 0], camera[:frame + 1, 2], color="tab:blue", lw=2.2)
            axis.plot(head[:frame + 1, 0], head[:frame + 1, 2], color="tab:orange", lw=2.0)
            axis.plot(
                [camera[frame, 0], head[frame, 0]],
                [camera[frame, 2], head[frame, 2]],
                color="0.35",
                lw=1.0,
                ls="--",
            )
            for position, rotation, color, name in (
                (camera[frame], camera_rotation[frame], "tab:blue", "camera +Z"),
                (head[frame], head_rotation[frame], "tab:orange", "Head +Z"),
            ):
                direction = rotation[:, 2][(0, 2),]
                norm = float(np.linalg.norm(direction))
                if norm > 1e-8:
                    direction = direction / norm * arrow_length
                    axis.arrow(
                        position[0],
                        position[2],
                        direction[0],
                        direction[1],
                        color=color,
                        width=max(horizontal_radius * 0.006, 0.004),
                        head_width=max(horizontal_radius * 0.045, 0.035),
                        length_includes_head=True,
                        label=name,
                    )
            axis.scatter(camera[frame, 0], camera[frame, 2], color="tab:blue", s=45, zorder=6)
            axis.scatter(head[frame, 0], head[frame, 2], color="tab:orange", s=45, zorder=6)
            axis.set_xlim(horizontal_center[0] - horizontal_radius, horizontal_center[0] + horizontal_radius)
            axis.set_ylim(horizontal_center[1] - horizontal_radius, horizontal_center[1] + horizontal_radius)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("world x relative to camera frame 0 [m]")
            axis.set_ylabel("world z relative to camera frame 0 [m]")
            axis.set_title("Top view: paths and +Z directions")
            axis.grid(alpha=0.22)

            axis_3d = fig.add_subplot(grid[0, 1], projection="3d")
            axis_3d.plot(camera[:, 0], camera[:, 2], camera[:, 1], color="tab:blue", alpha=0.18)
            axis_3d.plot(head[:, 0], head[:, 2], head[:, 1], color="tab:orange", alpha=0.18)
            axis_3d.plot(
                camera[:frame + 1, 0],
                camera[:frame + 1, 2],
                camera[:frame + 1, 1],
                color="tab:blue",
                lw=2.2,
                label="camera",
            )
            axis_3d.plot(
                head[:frame + 1, 0],
                head[:frame + 1, 2],
                head[:frame + 1, 1],
                color="tab:orange",
                lw=2.0,
                label="SOMA Head",
            )
            axis_3d.plot(
                [camera[frame, 0], head[frame, 0]],
                [camera[frame, 2], head[frame, 2]],
                [camera[frame, 1], head[frame, 1]],
                color="0.35",
                ls="--",
            )
            axis_3d.set_xlim(horizontal_center[0] - horizontal_radius, horizontal_center[0] + horizontal_radius)
            axis_3d.set_ylim(horizontal_center[1] - horizontal_radius, horizontal_center[1] + horizontal_radius)
            axis_3d.set_zlim(y_min - y_pad, y_max + y_pad)
            axis_3d.set_xlabel("world x [m]")
            axis_3d.set_ylabel("world z [m]")
            axis_3d.set_zlabel("world y [m]")
            axis_3d.set_title("3D trajectories (Kimodo Y-up)")
            axis_3d.legend(loc="upper left", fontsize=8)
            axis_3d.view_init(elev=24, azim=-62)

            axis = fig.add_subplot(grid[1, 0])
            axis.plot(time, distance, color="tab:purple", lw=2.0)
            axis.axvline(time[frame], color="black", lw=1.2)
            axis.scatter(time[frame], distance[frame], color="tab:purple", s=35, zorder=5)
            axis.axhline(0.5, color="red", ls="--", lw=1.0, label="0.5 m audit gate")
            axis.set_xlim(time[0], time[-1])
            axis.set_ylim(0.0, max(float(distance.max()) * 1.12, 0.20))
            axis.set_xlabel("time [s]")
            axis.set_ylabel("camera origin to Head joint [m]")
            axis.set_title(f"Head-camera distance: {distance[frame]:.3f} m")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8)

            axis = fig.add_subplot(grid[1, 1])
            axis.plot(time[1:], camera_step, color="tab:blue", lw=1.8, label="camera step")
            axis.plot(time[1:], head_step, color="tab:orange", lw=1.6, label="Head step")
            axis.axhline(0.25, color="red", ls="--", lw=1.0, label="0.25 m audit gate")
            axis.axvline(time[frame], color="black", lw=1.2)
            axis.set_xlim(time[0], time[-1])
            axis.set_ylim(0.0, max(float(camera_step.max()), float(head_step.max()), 0.25) * 1.12)
            axis.set_xlabel("time [s]")
            axis.set_ylabel("20 FPS translation step [m]")
            axis.set_title("Per-frame translation")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8)

            fig.suptitle(
                f"{label}\n{uuid}, frame {start + frame} ({frame / 20.0:.2f}s) | "
                "no fitted transform and no frame-0 pose alignment",
                fontsize=13,
            )
            fig.subplots_adjust(left=0.07, right=0.96, bottom=0.07, top=0.86)
            writer.grab_frame()
    plt.close(fig)
    return {
        "uuid": uuid,
        "start": start,
        "frames": len(camera),
        "output": str(output),
        **metrics,
    }


def render_case(
    uuid: str,
    start: int,
    frames: int,
    calibration: dict[str, Any],
    output: Path,
    *,
    label: str,
) -> None:
    data = _load_case(uuid)
    end = min(start + frames, len(data["head_position"]))
    selector = slice(start, end)
    head_rotation = data["head_rotation"][selector]
    head_position = data["head_position"][selector]
    camera_rotation = data["camera_rotation"][selector]
    camera_position = data["camera_position"][selector]
    rotation = np.asarray(calibration["rotation_head_to_upright_camera"], dtype=np.float64)
    lever = np.asarray(calibration["camera_origin_in_head_m"], dtype=np.float64)

    derived_rotation = head_rotation @ rotation
    derived_position = head_position + (head_rotation @ lever[:, None])[..., 0]
    # Force the derived frame-0 orientation and origin to equal the measured camera.  This is the
    # accumulated-action comparison used by the earlier plot, kept separate from direct world data.
    left_rotation = camera_rotation[0] @ derived_rotation[0].T
    frame0_aligned_position = camera_position[0] + (
        left_rotation @ (derived_position - derived_position[0]).T
    ).T
    frame0_aligned_rotation = left_rotation @ derived_rotation

    origin = camera_position[0]
    camera_plot = camera_position - origin
    head_plot = head_position - origin
    derived_plot = derived_position - origin
    aligned_plot = frame0_aligned_position - origin
    time = np.arange(end - start) / 20.0
    distance = np.linalg.norm(camera_position - head_position, axis=-1)
    camera_step = np.linalg.norm(np.diff(camera_position, axis=0), axis=-1)
    head_step = np.linalg.norm(np.diff(head_position, axis=0), axis=-1)
    relation = np.swapaxes(head_rotation, -1, -2) @ camera_rotation
    relation_error = _rotation_angle_deg(rotation.T @ relation)
    heading_error = _heading_error_deg(derived_rotation, camera_rotation)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(camera_plot[:, 0], camera_plot[:, 2], lw=2.4, label="GT upright RGB camera")
    ax.plot(head_plot[:, 0], head_plot[:, 2], lw=1.8, label="GT SOMA Head joint")
    ax.plot(derived_plot[:, 0], derived_plot[:, 2], lw=1.4, label="Head + global lever")
    ax.scatter([0], [0], c="black", s=28, zorder=5)
    ax.set_title("Direct stored shared-world trajectories (translation only)")
    ax.set_xlabel("world x relative to camera frame 0 [m]")
    ax.set_ylabel("world z relative to camera frame 0 [m]")
    ax.grid(alpha=0.25)
    ax.legend()
    _equal_top_axes(ax, [camera_plot, head_plot, derived_plot])

    ax = axes[0, 1]
    ax.plot(camera_plot[:, 0], camera_plot[:, 2], lw=2.4, label="GT camera")
    ax.plot(
        aligned_plot[:, 0],
        aligned_plot[:, 2],
        lw=2.0,
        label="motion-derived camera after frame-0 pose alignment",
    )
    arrow_indices = np.linspace(0, len(camera_plot) - 1, min(9, len(camera_plot))).astype(int)
    for index in arrow_indices:
        for position, direction, color in (
            (camera_plot[index], camera_rotation[index, :, 2], "tab:blue"),
            (aligned_plot[index], frame0_aligned_rotation[index, :, 2], "tab:orange"),
        ):
            horizontal = direction[[0, 2]]
            norm = np.linalg.norm(horizontal)
            if norm > 1e-6:
                horizontal = horizontal / norm * 0.18
                ax.arrow(
                    position[0],
                    position[2],
                    horizontal[0],
                    horizontal[1],
                    color=color,
                    width=0.008,
                    head_width=0.06,
                    length_includes_head=True,
                    alpha=0.8,
                )
    ax.set_title("Global Head->camera rotation forced to the GT frame-0 camera pose")
    ax.set_xlabel("aligned x [m]")
    ax.set_ylabel("aligned z [m]")
    ax.grid(alpha=0.25)
    ax.legend()
    _equal_top_axes(ax, [camera_plot, aligned_plot])

    ax = axes[1, 0]
    ax.plot(time, distance, label="|camera origin - Head joint|", color="tab:purple")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("direct shared-world distance [m]", color="tab:purple")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    twin.plot(time, relation_error, label="full rotation", color="tab:red", alpha=0.85)
    twin.plot(time, heading_error, label="horizontal heading", color="tab:green", alpha=0.85)
    twin.set_ylabel("global-calibration mismatch [deg]")
    twin.legend(loc="upper right")
    ax.set_title("Direct Head-camera separation and orientation mismatch")

    ax = axes[1, 1]
    ax.plot(time[1:], camera_step, label="camera step", lw=1.8)
    ax.plot(time[1:], head_step, label="Head step", lw=1.5)
    ax.axhline(0.25, color="red", ls="--", lw=1.2, label="0.25 m quality threshold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("20 FPS translation step [m]")
    ax.set_title("Source continuity")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle(f"{label}: {uuid} frames {start}:{end}", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_aggregate(details: list[dict[str, Any]], output: Path) -> None:
    valid = [item for item in details if "shared_world" in item]
    full_rotation = [
        item["shared_world"]["fixed_head_camera_rotation_deviation_deg"]["mean"]
        for item in valid
    ]
    full_heading = [
        item["shared_world"]["fixed_head_camera_heading_error_deg"]["mean"]
        for item in valid
    ]
    window_rotation = [
        item["shared_world"]["window_relation"][
            "within_window_rotation_deviation_mean_deg"
        ]["mean"]
        for item in valid
    ]
    window_heading = [
        item["shared_world"]["window_relation"]["within_window_heading_error_mean_deg"][
            "mean"
        ]
        for item in valid
    ]
    max_camera_step = [
        item["camera_preprocess"]["camera_step_translation_m"]["max"] for item in valid
    ]
    max_camera_rotation = [
        item["camera_preprocess"]["camera_step_rotation_deg"]["max"] for item in valid
    ]
    max_distance = [item["shared_world"]["head_camera_distance_m"]["max"] for item in valid]
    direct_step = [
        item["shared_world"]["direct_step_translation_error_m"]["mean"] * 1000.0
        for item in valid
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].hist(full_rotation, bins=45, alpha=0.65, label="full SO(3)")
    axes[0, 0].hist(full_heading, bins=45, alpha=0.65, label="horizontal heading")
    axes[0, 0].set_title("Per-sequence fixed Head->camera relation residual")
    axes[0, 0].set_xlabel("mean error [deg]")
    axes[0, 0].legend()

    axes[0, 1].hist(window_rotation, bins=45, alpha=0.65, label="full SO(3)")
    axes[0, 1].hist(window_heading, bins=45, alpha=0.65, label="horizontal heading")
    axes[0, 1].set_title("Within non-overlapping 97-frame windows")
    axes[0, 1].set_xlabel("mean error [deg]")
    axes[0, 1].legend()

    axes[1, 0].scatter(max_camera_step, max_camera_rotation, s=16, alpha=0.55)
    axes[1, 0].axvline(0.25, color="red", ls="--", lw=1.2)
    axes[1, 0].axhline(30.0, color="red", ls="--", lw=1.2)
    axes[1, 0].set_xscale("symlog", linthresh=0.1)
    axes[1, 0].set_title("Per-sequence worst camera action")
    axes[1, 0].set_xlabel("maximum 20 FPS translation [m]")
    axes[1, 0].set_ylabel("maximum 20 FPS rotation [deg]")

    axes[1, 1].scatter(max_distance, direct_step, s=16, alpha=0.55)
    axes[1, 1].axvline(0.5, color="red", ls="--", lw=1.2)
    axes[1, 1].set_title("Direct shared-world consistency")
    axes[1, 1].set_xlabel("maximum camera-to-Head origin distance [m]")
    axes[1, 1].set_ylabel("mean step disagreement [mm]")
    axes[1, 1].set_xscale("symlog", linthresh=0.15)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    rotation_stats = _stats(full_rotation)
    window_stats = _stats(window_rotation)
    fig.suptitle(
        f"Nymeria camera/motion source audit, n={len(valid)} sequences | "
        f"sequence SO(3) median/p90/max={rotation_stats[0]:.1f}/{rotation_stats[1]:.1f}/"
        f"{rotation_stats[2]:.1f} deg | window={window_stats[0]:.1f}/{window_stats[1]:.1f}/"
        f"{window_stats[2]:.1f} deg",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--clean-windows", type=Path, default=DEFAULT_CLEAN_WINDOWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-animations", action="store_true")
    args = parser.parse_args()
    details = list(map(json.loads, args.details.read_text().splitlines()))
    calibration = json.loads(args.calibration.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    render_aggregate(details, args.output / "dataset_summary.png")
    render_case(
        "S01/20230705_s0_william_davis_act4_h94ovw",
        0,
        97,
        calibration,
        args.output / "direct_vs_frame0_aligned_william_start0.png",
        label="Motion-clean71 high trajectory-divergence case",
    )
    render_case(
        "S17/20230918_s0_kevin_shaw_act2_5g4k0z",
        1450,
        150,
        calibration,
        args.output / "source_discontinuity_kevin_frames1450_1600.png",
        label="Raw-source discontinuity",
    )
    clean71_rows = render_clean71_sheet(
        args.clean_windows,
        args.output / "heldout_motion_clean71_direct_head_camera.png",
    )
    animations = []
    if not args.skip_animations:
        animations.append(
            render_direct_animation(
                "S01/20230705_s0_william_davis_act4_h94ovw",
                0,
                97,
                args.output / "direct_head_camera_william_start0.mp4",
                label="Clean held-out case previously exaggerated by frame-0 pose alignment",
            )
        )
        animations.append(
            render_direct_animation(
                "S17/20230918_s0_kevin_shaw_act2_5g4k0z",
                1450,
                150,
                args.output / "direct_head_camera_kevin_source_jump_1450_1600.mp4",
                label="Raw source discontinuity",
            )
        )
    manifest = {
        "contract": (
            "Direct stored-world camera and decoded UniEgo SOMA-Head poses after only the fixed "
            "Aria-Z-up to Kimodo-Y-up basis change; no fitted Head-camera transform and no "
            "frame-0 pose alignment. Camera and Head arrows are their respective +Z axes."
        ),
        "clean_windows": str(args.clean_windows),
        "clean71_rows": clean71_rows,
        "animations": animations,
    }
    (args.output / "direct_head_camera_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
