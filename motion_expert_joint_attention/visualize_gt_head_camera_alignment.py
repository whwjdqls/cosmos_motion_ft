#!/usr/bin/env python3
"""Visualize the production global Head->camera relative mapping on synchronized GT.

This is not a direct overlay of the two stored world trajectories. The 2026-07-21 source audit
established that clean camera and UniEgo position streams normally share the same metric world
frame. Here the motion-derived camera ``H_t X`` is instead canonicalized with the training-global
Head->camera rotation, then one constant left transform makes its frame-0 pose equal the measured
camera. Subsequent divergence measures that production relative-frame approximation. Use
``visualize_nymeria_alignment_audit.py`` for direct shared-world overlays and source jumps.

Outputs include an all-window ranking, static trajectory/direction plots, animations with the GT
skeleton, and a machine-readable summary. This script never samples a model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config as C  # noqa: E402
from decode_uniego_torch import decode_transforms  # noqa: E402
from eval_all import build_full_index, load_gt_motion  # noqa: E402
from head_camera_alignment import (  # noqa: E402
    DEFAULT_CALIBRATION,
    HEAD_JOINT_IDX,
    load_head_camera_calibration,
    motion_to_camera_action,
)
from nymeria_joint_dataset import _load_rgb_cam, rel_action_from_window  # noqa: E402
from render_motion import PARENTS, SKIP_JOINTS  # noqa: E402


DEFAULT_EVAL_ROOT = Path(
    "/weka/jungbin/cosmos_motion_ft_runs/"
    "ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/"
    "eval_full71_step200000_unipc30"
)
DEFAULT_UNIEGO_ROOT = Path(
    "/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep"
)

GT_COLOR = "#1769aa"
MOTION_COLOR = "#d95f02"
ROOT_COLOR = "#555555"
SKELETON_COLOR = "#343434"
ARIA_Z_UP_TO_KIMODO_Y_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(temporary, path)


def _project_so3(rotation: np.ndarray) -> np.ndarray:
    """Project near-rotation matrices to SO(3) before pose integration/visualization."""
    rotation = np.asarray(rotation, dtype=np.float64)
    flat = rotation.reshape(-1, 3, 3)
    u, _, vt = np.linalg.svd(flat)
    projected = u @ vt
    reflected = np.linalg.det(projected) < 0.0
    if np.any(reflected):
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        projected = u @ vt
    return projected.reshape(rotation.shape)


def _pose_from_position_rotation(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.broadcast_to(np.eye(4, dtype=np.float64), (len(position), 4, 4)).copy()
    pose[:, :3, :3] = _project_so3(rotation)
    pose[:, :3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def _action_to_pose(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    x = action[:, 3:6]
    y_raw = action[:, 6:9]
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    z = np.cross(x, y_raw)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z], axis=-1)
    pose = np.broadcast_to(np.eye(4, dtype=np.float64), (len(action), 4, 4)).copy()
    pose[:, :3, :3] = rotation
    pose[:, :3, 3] = action[:, :3]
    return pose


def _relative_pose(pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(pose[:-1]) @ pose[1:]


def _rotation_angle_deg(rotation: np.ndarray) -> np.ndarray:
    cosine = np.clip(
        (np.trace(rotation, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _vector_angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-12)
    cosine = np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _similarity_ate(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return Umeyama Sim(3)-aligned RMSE and fitted source-to-target scale."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(source_centered ** 2, axis=-1)))
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    aligned = scale * (source_centered @ rotation.T) + target.mean(axis=0)
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=-1))))
    return rmse, scale


def _case_name(order: int, uuid: str, start: int) -> str:
    return f"c{order:02d}_{uuid.replace('/', '_')}_{int(start)}"


def _load_case(
    order: int,
    item: dict,
    *,
    num_frames: int,
    mean: np.ndarray,
    std: np.ndarray,
    calibration_rotation: torch.Tensor,
    calibration_lever: torch.Tensor,
) -> dict:
    gt_z, _ = load_gt_motion(
        item["uni"], item["s"], item["off"], num_frames, mean, std
    )
    motion = gt_z * std + mean
    motion_t = torch.from_numpy(np.ascontiguousarray(motion)).double().unsqueeze(0)
    with torch.no_grad():
        transforms = decode_transforms(motion_t)[0]
        head = transforms[:, HEAD_JOINT_IDX].cpu().numpy().copy()
        head[:, :3, :3] = _project_so3(head[:, :3, :3])
        joints = transforms[..., :3, 3].cpu().numpy()
        derived_action = motion_to_camera_action(
            motion_t,
            calibration_rotation.double(),
            calibration_lever.double(),
        )[0].cpu().numpy()

    x = np.eye(4, dtype=np.float64)
    x[:3, :3] = calibration_rotation.double().cpu().numpy()
    x[:3, 3] = calibration_lever.double().cpu().numpy()
    motion_camera = head @ x

    camera_position, camera_rotation = _load_rgb_cam(item["rgb"])
    start = int(item["s"])
    stop = start + num_frames
    source_camera = _pose_from_position_rotation(
        camera_position[start:stop], camera_rotation[start:stop]
    )
    if len(source_camera) != num_frames:
        raise ValueError(
            f"{item['uuid']}@{start}: short camera window {len(source_camera)}/{num_frames}"
        )

    # One constant world transform makes only frame 0 coincide. It does not alter any camera
    # relative action, and it is the only valid absolute alignment available for these streams.
    world_alignment = motion_camera[0] @ np.linalg.inv(source_camera[0])
    measured_camera = world_alignment[None] @ source_camera
    measured_action = rel_action_from_window(
        camera_position[start:stop], camera_rotation[start:stop]
    ).astype(np.float64)

    # This is the same absolute-orientation observation used to fit the train-global R_X. It does
    # not use absolute translation. A large, nearly constant deviation here rotates camera-frame
    # translation vectors coherently and can therefore create a large endpoint gap on fast clips.
    with np.load(item["uni"]) as motion_npz:
        # UniEgo stores an absolute canonical transform only at sequence frame 0; all later
        # canon-delta rows are relative to the previous frame. Decode the prefix before slicing.
        raw_motion_prefix = motion_npz["features"][:stop].astype(np.float64)
    with torch.no_grad():
        raw_head = (
            decode_transforms(
                torch.from_numpy(np.ascontiguousarray(raw_motion_prefix)).double().unsqueeze(0)
            )[0, start:stop, HEAD_JOINT_IDX]
            .cpu()
            .numpy()
        )
    raw_head_rotation = _project_so3(raw_head[:, :3, :3])
    camera_rotation_kimodo_world = _project_so3(
        ARIA_Z_UP_TO_KIMODO_Y_UP[None] @ camera_rotation[start:stop]
    )
    observed_head_camera_rotation = _project_so3(
        np.swapaxes(raw_head_rotation, -1, -2) @ camera_rotation_kimodo_world
    )
    calibration_rotation_np = calibration_rotation.double().cpu().numpy()
    head_camera_rotation_deviation = _rotation_angle_deg(
        calibration_rotation_np.T[None] @ observed_head_camera_rotation
    )

    with np.load(item["rgb"]) as camera_npz:
        stored_action = camera_npz["cam_action_upright_k1"][start:stop - 1].astype(
            np.float64
        )
    stored_action_translation_delta = float(
        np.max(np.abs(stored_action[:, :3] - measured_action[:, :3]))
    )
    stored_action_rotation_delta = float(
        np.max(
            _rotation_angle_deg(
                np.swapaxes(
                    _action_to_pose(stored_action)[:, :3, :3], -1, -2
                )
                @ _action_to_pose(measured_action)[:, :3, :3]
            )
        )
    )

    measured_relative = _relative_pose(measured_camera)
    motion_relative = _relative_pose(motion_camera)
    measured_from_action = _action_to_pose(measured_action)
    derived_from_action = _action_to_pose(derived_action)
    contracts = {
        "stored_vs_recomputed_translation_max_abs_m": stored_action_translation_delta,
        "stored_vs_recomputed_rotation_max_deg": stored_action_rotation_delta,
        "measured_pose_vs_action_translation_max_abs_m": float(
            np.max(np.abs(measured_relative[:, :3, 3] - measured_from_action[:, :3, 3]))
        ),
        "measured_pose_vs_action_rotation_max_deg": float(
            np.max(
                _rotation_angle_deg(
                    np.swapaxes(measured_relative[:, :3, :3], -1, -2)
                    @ measured_from_action[:, :3, :3]
                )
            )
        ),
        "motion_pose_vs_formula_translation_max_abs_m": float(
            np.max(np.abs(motion_relative[:, :3, 3] - derived_from_action[:, :3, 3]))
        ),
        "motion_pose_vs_formula_rotation_max_deg": float(
            np.max(
                _rotation_angle_deg(
                    np.swapaxes(motion_relative[:, :3, :3], -1, -2)
                    @ derived_from_action[:, :3, :3]
                )
            )
        ),
        "frame0_pose_max_abs": float(np.max(np.abs(measured_camera[0] - motion_camera[0]))),
    }

    local_translation = np.linalg.norm(
        derived_action[:, :3] - measured_action[:, :3], axis=-1
    )
    local_rotation = _rotation_angle_deg(
        np.swapaxes(derived_from_action[:, :3, :3], -1, -2)
        @ measured_from_action[:, :3, :3]
    )
    position_error = np.linalg.norm(
        motion_camera[:, :3, 3] - measured_camera[:, :3, 3], axis=-1
    )
    orientation_error = _rotation_angle_deg(
        np.swapaxes(motion_camera[:, :3, :3], -1, -2)
        @ measured_camera[:, :3, :3]
    )
    direction_error = _vector_angle_deg(
        motion_camera[:, :3, 2], measured_camera[:, :3, 2]
    )
    measured_steps = np.diff(measured_camera[:, :3, 3], axis=0)
    motion_steps = np.diff(motion_camera[:, :3, 3], axis=0)
    measured_displacement = float(
        np.linalg.norm(measured_camera[-1, :3, 3] - measured_camera[0, :3, 3])
    )
    motion_displacement = float(
        np.linalg.norm(motion_camera[-1, :3, 3] - motion_camera[0, :3, 3])
    )
    median_frame_deviation = float(np.median(head_camera_rotation_deviation))
    sim3_rmse, sim3_scale = _similarity_ate(
        motion_camera[:, :3, 3], measured_camera[:, :3, 3]
    )
    metrics = {
        "local_translation_mean_m": float(local_translation.mean()),
        "local_translation_median_m": float(np.median(local_translation)),
        "local_translation_p90_m": float(np.quantile(local_translation, 0.90)),
        "local_rotation_mean_deg": float(local_rotation.mean()),
        "local_rotation_median_deg": float(np.median(local_rotation)),
        "position_error_mean_m": float(position_error.mean()),
        "position_error_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "position_error_endpoint_m": float(position_error[-1]),
        "position_error_endpoint_xyz_m": (
            motion_camera[-1, :3, 3] - measured_camera[-1, :3, 3]
        ).tolist(),
        "orientation_error_mean_deg": float(orientation_error.mean()),
        "orientation_error_endpoint_deg": float(orientation_error[-1]),
        "forward_direction_error_mean_deg": float(direction_error.mean()),
        "forward_direction_error_endpoint_deg": float(direction_error[-1]),
        "head_camera_frame_rotation_deviation_mean_deg": float(
            head_camera_rotation_deviation.mean()
        ),
        "head_camera_frame_rotation_deviation_median_deg": median_frame_deviation,
        "head_camera_frame_rotation_deviation_p90_deg": float(
            np.quantile(head_camera_rotation_deviation, 0.90)
        ),
        "measured_path_length_m": float(np.linalg.norm(measured_steps, axis=-1).sum()),
        "motion_path_length_m": float(np.linalg.norm(motion_steps, axis=-1).sum()),
        "measured_net_displacement_m": measured_displacement,
        "motion_net_displacement_m": motion_displacement,
        "path_length_ratio_motion_over_measured": float(
            np.linalg.norm(motion_steps, axis=-1).sum()
            / max(np.linalg.norm(measured_steps, axis=-1).sum(), 1e-12)
        ),
        "orientation_chord_times_displacement_m": float(
            2.0
            * measured_displacement
            * math.sin(0.5 * math.radians(median_frame_deviation))
        ),
        "sim3_ate_rmse_m": sim3_rmse,
        "sim3_scale_motion_to_measured": sim3_scale,
    }
    return {
        "name": _case_name(order, item["uuid"], start),
        "order": order,
        "uuid": item["uuid"],
        "start": start,
        "caption": item["cap"],
        "joints": joints,
        "head": head,
        "measured_camera": measured_camera,
        "motion_camera": motion_camera,
        "local_translation": local_translation,
        "local_rotation": local_rotation,
        "position_error": position_error,
        "orientation_error": orientation_error,
        "direction_error": direction_error,
        "head_camera_rotation_deviation": head_camera_rotation_deviation,
        "metrics": metrics,
        "contracts": contracts,
    }


def _plot_xyz(ax, points: np.ndarray, *args, **kwargs):
    return ax.plot(points[..., 0], points[..., 2], points[..., 1], *args, **kwargs)


def _draw_skeleton(ax, joints: np.ndarray, *, color: str, alpha: float, linewidth: float) -> None:
    skipped = set(SKIP_JOINTS)
    for child, parent in enumerate(PARENTS):
        if parent < 0 or child in skipped or parent in skipped:
            continue
        segment = joints[[parent, child]]
        _plot_xyz(ax, segment, color=color, alpha=alpha, linewidth=linewidth)


def _draw_forward_arrows_3d(
    ax,
    poses: np.ndarray,
    indices: np.ndarray,
    *,
    color: str,
    length: float,
    alpha: float = 0.9,
) -> None:
    position = poses[indices, :3, 3]
    forward = poses[indices, :3, 2]
    ax.quiver(
        position[:, 0],
        position[:, 2],
        position[:, 1],
        forward[:, 0],
        forward[:, 2],
        forward[:, 1],
        length=length,
        normalize=True,
        color=color,
        alpha=alpha,
        linewidth=1.2,
        arrow_length_ratio=0.22,
    )


def _draw_forward_arrows_top(
    ax,
    poses: np.ndarray,
    indices: np.ndarray,
    *,
    color: str,
    length: float,
) -> None:
    position = poses[indices, :3, 3]
    forward = poses[indices, :3, 2]
    norm = np.maximum(np.linalg.norm(forward[:, [0, 2]], axis=-1), 1e-12)
    ax.quiver(
        position[:, 0],
        position[:, 2],
        length * forward[:, 0] / norm,
        length * forward[:, 2] / norm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        width=0.006,
    )


def _equal_3d(ax, points: np.ndarray) -> None:
    plot_points = points[:, [0, 2, 1]]
    minimum = plot_points.min(axis=0)
    maximum = plot_points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float(np.max(maximum - minimum)) * 0.55, 0.75)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(-0.1, center[2] - radius), center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _case_plot(case: dict, path: Path) -> None:
    measured = case["measured_camera"]
    motion = case["motion_camera"]
    joints = case["joints"]
    metrics = case["metrics"]
    t = np.arange(len(measured)) / 20.0
    arrow_indices = np.unique(np.linspace(0, len(measured) - 1, 9).round().astype(int))
    skeleton_indices = np.unique(np.linspace(0, len(measured) - 1, 5).round().astype(int))
    scene_points = np.concatenate(
        [joints.reshape(-1, 3), measured[:, :3, 3], motion[:, :3, 3]], axis=0
    )
    span = np.ptp(scene_points, axis=0)
    arrow_length = max(0.12, min(0.35, 0.12 * float(max(span[0], span[2], 1.0))))

    fig = plt.figure(figsize=(14, 10), dpi=130)
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    _plot_xyz(ax3d, measured[:, :3, 3], color=GT_COLOR, linewidth=2.8, label="Phase-1 GT camera")
    _plot_xyz(ax3d, motion[:, :3, 3], color=MOTION_COLOR, linewidth=2.4, label="GT motion -> camera")
    for rank, frame in enumerate(skeleton_indices):
        _draw_skeleton(
            ax3d,
            joints[frame],
            color=SKELETON_COLOR,
            alpha=0.18 + 0.13 * rank,
            linewidth=1.0,
        )
    _draw_forward_arrows_3d(
        ax3d, measured, arrow_indices, color=GT_COLOR, length=arrow_length
    )
    _draw_forward_arrows_3d(
        ax3d, motion, arrow_indices, color=MOTION_COLOR, length=arrow_length
    )
    _equal_3d(ax3d, scene_points)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("z [m]")
    ax3d.set_zlabel("y up [m]")
    ax3d.set_title("Frame-0-aligned trajectories and +Z optical directions")
    ax3d.view_init(elev=24, azim=-58)
    ax3d.legend(loc="upper left", fontsize=8)

    axtop = fig.add_subplot(2, 2, 2)
    axtop.plot(
        measured[:, 0, 3], measured[:, 2, 3], color=GT_COLOR, linewidth=2.8,
        label="Phase-1 GT camera",
    )
    axtop.plot(
        motion[:, 0, 3], motion[:, 2, 3], color=MOTION_COLOR, linewidth=2.4,
        label="GT motion -> camera",
    )
    root = joints[:, 0]
    axtop.plot(root[:, 0], root[:, 2], color=ROOT_COLOR, linewidth=1.2, alpha=0.7, label="GT pelvis")
    _draw_forward_arrows_top(
        axtop, measured, arrow_indices, color=GT_COLOR, length=arrow_length
    )
    _draw_forward_arrows_top(
        axtop, motion, arrow_indices, color=MOTION_COLOR, length=arrow_length
    )
    axtop.scatter(measured[0, 0, 3], measured[0, 2, 3], color="black", marker="o", s=32, label="start")
    axtop.scatter(measured[-1, 0, 3], measured[-1, 2, 3], color=GT_COLOR, marker="X", s=50)
    axtop.scatter(motion[-1, 0, 3], motion[-1, 2, 3], color=MOTION_COLOR, marker="X", s=50)
    axtop.set_aspect("equal", adjustable="datalim")
    axtop.grid(True, alpha=0.25)
    axtop.set_xlabel("x [m]")
    axtop.set_ylabel("z [m]")
    axtop.set_title("Top view; arrows are +Z optical forward")
    axtop.legend(fontsize=8)

    axerr = fig.add_subplot(2, 2, 3)
    axerr.plot(t, 100.0 * case["position_error"], color="#7b3294", linewidth=2.0)
    axerr.set_xlabel("time [s]")
    axerr.set_ylabel("accumulated position gap [cm]", color="#7b3294")
    axerr.tick_params(axis="y", labelcolor="#7b3294")
    axerr.grid(True, alpha=0.25)
    axdir = axerr.twinx()
    axdir.plot(t, case["direction_error"], color="#008837", linewidth=1.7)
    axdir.set_ylabel("accumulated forward-direction gap [deg]", color="#008837")
    axdir.tick_params(axis="y", labelcolor="#008837")
    axerr.set_title("Accumulated pose divergence after identical frame 0")

    axlocal = fig.add_subplot(2, 2, 4)
    transition_t = np.arange(len(case["local_translation"])) / 20.0
    axlocal.plot(
        transition_t, 1000.0 * case["local_translation"], color="#c51b7d", linewidth=1.6
    )
    axlocal.set_xlabel("time [s]")
    axlocal.set_ylabel("relative translation error [mm]", color="#c51b7d")
    axlocal.tick_params(axis="y", labelcolor="#c51b7d")
    axlocal.grid(True, alpha=0.25)
    axrot = axlocal.twinx()
    axrot.plot(transition_t, case["local_rotation"], color="#4d9221", linewidth=1.4)
    axrot.set_ylabel("relative rotation error [deg]", color="#4d9221")
    axrot.tick_params(axis="y", labelcolor="#4d9221")
    axlocal.set_title("Per-20-FPS-step discrepancy")

    title = (
        f"{case['uuid']}  start={case['start']}  ({(len(measured) - 1) / 20.0:.2f}s)\n"
        f"local={1000.0 * metrics['local_translation_mean_m']:.2f} mm / "
        f"{metrics['local_rotation_mean_deg']:.2f} deg; "
        f"endpoint={100.0 * metrics['position_error_endpoint_m']:.1f} cm / "
        f"{metrics['forward_direction_error_endpoint_deg']:.1f} deg; "
        f"median X error={metrics['head_camera_frame_rotation_deviation_median_deg']:.1f} deg; "
        f"Sim3 ATE={1000.0 * metrics['sim3_ate_rmse_m']:.2f} mm, "
        f"scale={metrics['sim3_scale_motion_to_measured']:.3f}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _animation(case: dict, path: Path, fps: int = 20) -> None:
    import imageio.v2 as imageio

    measured = case["measured_camera"]
    motion = case["motion_camera"]
    joints = case["joints"]
    all_points = np.concatenate(
        [joints.reshape(-1, 3), measured[:, :3, 3], motion[:, :3, 3]], axis=0
    )
    plot_points = all_points[:, [0, 2, 1]]
    minimum = plot_points.min(axis=0)
    maximum = plot_points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float(np.max(maximum - minimum)) * 0.58, 0.8)
    arrow_length = max(0.18, min(0.35, 0.18 * radius))
    time = np.arange(len(measured)) / float(fps)

    fig = plt.figure(figsize=(12, 8), dpi=110)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame in range(len(measured)):
            fig.clf()
            ax3d = fig.add_subplot(2, 2, 1, projection="3d")
            _plot_xyz(ax3d, measured[:, :3, 3], color=GT_COLOR, alpha=0.18, linewidth=1.0)
            _plot_xyz(ax3d, motion[:, :3, 3], color=MOTION_COLOR, alpha=0.18, linewidth=1.0)
            _plot_xyz(ax3d, measured[: frame + 1, :3, 3], color=GT_COLOR, linewidth=2.8)
            _plot_xyz(ax3d, motion[: frame + 1, :3, 3], color=MOTION_COLOR, linewidth=2.5)
            _draw_skeleton(
                ax3d, joints[frame], color=SKELETON_COLOR, alpha=0.95, linewidth=1.8
            )
            _draw_forward_arrows_3d(
                ax3d,
                measured,
                np.asarray([frame]),
                color=GT_COLOR,
                length=arrow_length,
            )
            _draw_forward_arrows_3d(
                ax3d,
                motion,
                np.asarray([frame]),
                color=MOTION_COLOR,
                length=arrow_length,
            )
            ax3d.set_xlim(center[0] - radius, center[0] + radius)
            ax3d.set_ylim(center[1] - radius, center[1] + radius)
            ax3d.set_zlim(max(-0.1, center[2] - radius), center[2] + radius)
            ax3d.set_box_aspect((1.0, 1.0, 1.0))
            ax3d.set_xlabel("x")
            ax3d.set_ylabel("z")
            ax3d.set_zlabel("y up")
            ax3d.view_init(elev=24, azim=-58)
            ax3d.set_title("GT skeleton with camera +Z directions")

            axtop = fig.add_subplot(2, 2, 2)
            axtop.plot(measured[:, 0, 3], measured[:, 2, 3], color=GT_COLOR, alpha=0.18)
            axtop.plot(motion[:, 0, 3], motion[:, 2, 3], color=MOTION_COLOR, alpha=0.18)
            axtop.plot(
                measured[: frame + 1, 0, 3], measured[: frame + 1, 2, 3],
                color=GT_COLOR, linewidth=2.8, label="Phase-1 GT camera",
            )
            axtop.plot(
                motion[: frame + 1, 0, 3], motion[: frame + 1, 2, 3],
                color=MOTION_COLOR, linewidth=2.5, label="GT motion -> camera",
            )
            _draw_forward_arrows_top(
                axtop, measured, np.asarray([frame]), color=GT_COLOR, length=arrow_length
            )
            _draw_forward_arrows_top(
                axtop, motion, np.asarray([frame]), color=MOTION_COLOR, length=arrow_length
            )
            axtop.set_aspect("equal", adjustable="datalim")
            axtop.grid(True, alpha=0.25)
            axtop.set_xlabel("x [m]")
            axtop.set_ylabel("z [m]")
            axtop.set_title("Top view")
            axtop.legend(fontsize=8)

            axpos = fig.add_subplot(2, 2, 3)
            axpos.plot(time, 100.0 * case["position_error"], color="#7b3294", linewidth=2.0)
            axpos.axvline(time[frame], color="black", linewidth=1.0, alpha=0.6)
            axpos.scatter(
                [time[frame]], [100.0 * case["position_error"][frame]],
                color="#7b3294", s=35, zorder=3,
            )
            axpos.set_xlim(time[0], time[-1])
            axpos.set_ylim(0.0, max(1.0, 105.0 * float(case["position_error"].max())))
            axpos.set_xlabel("time [s]")
            axpos.set_ylabel("position gap [cm]")
            axpos.set_title("Accumulated translation divergence")
            axpos.grid(True, alpha=0.25)

            axdir = fig.add_subplot(2, 2, 4)
            axdir.plot(
                time,
                case["direction_error"],
                color="#008837",
                linewidth=2.0,
                label="optical +Z forward",
            )
            axdir.plot(time, case["orientation_error"], color="#542788", linewidth=1.5,
                       label="full orientation")
            axdir.axvline(time[frame], color="black", linewidth=1.0, alpha=0.6)
            axdir.scatter(
                [time[frame]], [case["direction_error"][frame]],
                color="#008837", s=35, zorder=3,
            )
            axdir.set_xlim(time[0], time[-1])
            ymax = max(
                1.0,
                1.05 * float(max(case["direction_error"].max(), case["orientation_error"].max())),
            )
            axdir.set_ylim(0.0, ymax)
            axdir.set_xlabel("time [s]")
            axdir.set_ylabel("angle [deg]")
            axdir.set_title("Accumulated optical/full orientation divergence")
            axdir.legend(fontsize=8)
            axdir.grid(True, alpha=0.25)

            fig.suptitle(
                f"{case['uuid']} start={case['start']}  frame={frame}/{len(measured) - 1}  "
                f"blue=Phase-1 GT camera, orange=GT-motion-derived camera",
                fontsize=10,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(image)
    finally:
        writer.close()
        plt.close(fig)


def _aggregate(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _summary_plot(cases: list[dict], path: Path) -> None:
    local_mm = np.asarray([1000.0 * c["metrics"]["local_translation_mean_m"] for c in cases])
    endpoint_cm = np.asarray([100.0 * c["metrics"]["position_error_endpoint_m"] for c in cases])
    sim3_mm = np.asarray([1000.0 * c["metrics"]["sim3_ate_rmse_m"] for c in cases])
    frame_deviation = np.asarray(
        [c["metrics"]["head_camera_frame_rotation_deviation_median_deg"] for c in cases]
    )
    displacement = np.asarray(
        [c["metrics"]["measured_net_displacement_m"] for c in cases]
    )
    rank = np.argsort(endpoint_cm)[::-1]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=130)
    axes[0, 0].bar(np.arange(len(cases)), endpoint_cm[rank], color="#7b3294")
    axes[0, 0].set_xlabel("windows ranked by endpoint gap")
    axes[0, 0].set_ylabel("frame-96 position gap [cm]")
    axes[0, 0].grid(True, axis="y", alpha=0.25)

    axes[0, 1].scatter(local_mm, endpoint_cm, c=frame_deviation, cmap="viridis", s=42)
    axes[0, 1].set_xlabel("mean local translation error [mm/step]")
    axes[0, 1].set_ylabel("frame-96 position gap [cm]")
    axes[0, 1].grid(True, alpha=0.25)
    colorbar = fig.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    colorbar.set_label("median Head-to-camera frame rotation error [deg]")

    axes[0, 2].scatter(frame_deviation, endpoint_cm, c=displacement, cmap="plasma", s=42)
    axes[0, 2].set_xlabel("median Head-to-camera frame rotation error [deg]")
    axes[0, 2].set_ylabel("frame-96 position gap [cm]")
    axes[0, 2].grid(True, alpha=0.25)
    colorbar = fig.colorbar(axes[0, 2].collections[0], ax=axes[0, 2])
    colorbar.set_label("measured net displacement [m]")

    axes[1, 0].hist(endpoint_cm, bins=16, color="#7b3294", alpha=0.85)
    axes[1, 0].axvline(np.median(endpoint_cm), color="black", linestyle="--", label="median")
    axes[1, 0].set_xlabel("frame-96 position gap [cm]")
    axes[1, 0].set_ylabel("windows")
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    axes[1, 1].scatter(endpoint_cm, sim3_mm, color="#008837", s=42)
    axes[1, 1].set_xlabel("raw frame-96 position gap [cm]")
    axes[1, 1].set_ylabel("Sim(3)-aligned trajectory RMSE [mm]")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 2].scatter(displacement, endpoint_cm, c=frame_deviation, cmap="viridis", s=42)
    axes[1, 2].set_xlabel("measured net displacement [m]")
    axes[1, 2].set_ylabel("frame-96 position gap [cm]")
    axes[1, 2].grid(True, alpha=0.25)
    colorbar = fig.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    colorbar.set_label("median Head-to-camera frame rotation error [deg]")
    fig.suptitle(
        f"GT motion-derived camera versus Phase-1 GT camera, {len(cases)} held-out windows\n"
        "Both trajectories share exactly the same frame-0 pose; no model prediction is involved",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _selected_cases(cases: list[dict], static_count: int) -> tuple[list[dict], list[dict]]:
    endpoint_sorted = sorted(
        cases, key=lambda case: case["metrics"]["position_error_endpoint_m"], reverse=True
    )
    local_worst = max(cases, key=lambda case: case["metrics"]["local_translation_mean_m"])
    median = sorted(cases, key=lambda case: case["metrics"]["position_error_endpoint_m"])[
        len(cases) // 2
    ]
    best = min(cases, key=lambda case: case["metrics"]["position_error_endpoint_m"])

    static = endpoint_sorted[: max(1, static_count)] + [local_worst, median, best]
    animations = [endpoint_sorted[0], local_worst, median]

    def unique(items: list[dict]) -> list[dict]:
        seen = set()
        output = []
        for item in items:
            if item["name"] not in seen:
                output.append(item)
                seen.add(item["name"])
        return output

    return unique(static), unique(animations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--windows-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--head-camera-calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--manifest", default=C.NYMERIA_MANIFEST)
    parser.add_argument("--split-file", default=C.NYMERIA_SPLIT_FILE)
    parser.add_argument("--uniego-root", type=Path, default=DEFAULT_UNIEGO_ROOT)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--static-count", type=int, default=5)
    parser.add_argument(
        "--animations",
        type=int,
        default=3,
        help="number of worst/local-worst/median selected animations to render; 0 disables",
    )
    args = parser.parse_args()

    eval_root = args.eval_root.resolve()
    windows_json = (
        args.windows_json.resolve()
        if args.windows_json is not None
        else eval_root / "motion_clean71_windows.json"
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else eval_root / "gt_camera_vs_gt_motion_viz"
    )
    if not windows_json.is_file():
        raise FileNotFoundError(windows_json)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = build_full_index(
        args.manifest,
        args.split_file,
        "test",
        args.num_frames,
        latent_root="",
        uniego_root=str(args.uniego_root),
        require_latents=False,
        windows_json=str(windows_json),
    )
    requested = json.load(windows_json.open())
    if len(index) != len(requested):
        raise ValueError(f"matched {len(index)}/{len(requested)} requested windows")

    mean = np.load(C.MOTION_STATS_MEAN).astype(np.float64)
    std = np.load(C.MOTION_STATS_STD).astype(np.float64)
    calibration_rotation, calibration_lever, calibration_payload = (
        load_head_camera_calibration(args.head_camera_calibration)
    )
    cases = []
    for order, item in enumerate(index):
        case = _load_case(
            order,
            item,
            num_frames=args.num_frames,
            mean=mean,
            std=std,
            calibration_rotation=calibration_rotation,
            calibration_lever=calibration_lever,
        )
        cases.append(case)
        print(
            f"[{order + 1:02d}/{len(index)}] {case['name']}: "
            f"local={1000.0 * case['metrics']['local_translation_mean_m']:.2f}mm/"
            f"{case['metrics']['local_rotation_mean_deg']:.2f}deg "
            f"endpoint={100.0 * case['metrics']['position_error_endpoint_m']:.1f}cm",
            flush=True,
        )

    contract_max = {
        key: max(case["contracts"][key] for case in cases)
        for key in cases[0]["contracts"]
    }
    if contract_max["frame0_pose_max_abs"] > 1e-8:
        raise ValueError(f"frame-0 alignment contract failed: {contract_max}")
    if contract_max["stored_vs_recomputed_translation_max_abs_m"] > 5e-5:
        raise ValueError(f"stored/recomputed camera action mismatch: {contract_max}")
    if contract_max["measured_pose_vs_action_translation_max_abs_m"] > 5e-5:
        raise ValueError(f"absolute/relative camera pose mismatch: {contract_max}")
    if contract_max["motion_pose_vs_formula_translation_max_abs_m"] > 5e-5:
        raise ValueError(f"motion camera formula mismatch: {contract_max}")
    if max(
        contract_max["stored_vs_recomputed_rotation_max_deg"],
        contract_max["measured_pose_vs_action_rotation_max_deg"],
        contract_max["motion_pose_vs_formula_rotation_max_deg"],
    ) > 0.02:
        raise ValueError(f"rotation contract mismatch: {contract_max}")

    static_cases, animation_cases = _selected_cases(cases, args.static_count)
    _summary_plot(cases, out_dir / "all71_summary.png")
    for rank, case in enumerate(static_cases):
        _case_plot(case, out_dir / "cases" / f"static_{rank:02d}_{case['name']}.png")
    for rank, case in enumerate(animation_cases[: max(0, args.animations)]):
        print(f"[animation] {rank + 1}/{min(args.animations, len(animation_cases))}: {case['name']}")
        _animation(
            case,
            out_dir / "cases" / f"animation_{rank:02d}_{case['name']}.mp4",
            fps=args.fps,
        )

    metric_keys = list(cases[0]["metrics"])
    aggregate = {}
    for key in metric_keys:
        if key == "position_error_endpoint_xyz_m":
            continue
        aggregate[key] = _aggregate(np.asarray([case["metrics"][key] for case in cases]))
    endpoint = np.asarray(
        [case["metrics"]["position_error_endpoint_m"] for case in cases]
    )
    local_translation = np.asarray(
        [case["metrics"]["local_translation_mean_m"] for case in cases]
    )
    frame_rotation = np.asarray(
        [
            case["metrics"]["head_camera_frame_rotation_deviation_median_deg"]
            for case in cases
        ]
    )
    displacement = np.asarray(
        [case["metrics"]["measured_net_displacement_m"] for case in cases]
    )
    orientation_chord = np.asarray(
        [case["metrics"]["orientation_chord_times_displacement_m"] for case in cases]
    )
    correlations = {
        "endpoint_vs_local_translation": _pearson(endpoint, local_translation),
        "endpoint_vs_head_camera_frame_rotation": _pearson(endpoint, frame_rotation),
        "endpoint_vs_measured_net_displacement": _pearson(endpoint, displacement),
        "endpoint_vs_orientation_chord_times_displacement": _pearson(
            endpoint, orientation_chord
        ),
    }
    payload = {
        "kind": "gt_camera_vs_gt_motion_relative_alignment_visualization",
        "n": len(cases),
        "windows_json": str(windows_json),
        "split": "test",
        "num_frames": args.num_frames,
        "fps": args.fps,
        "duration_s": (args.num_frames - 1) / float(args.fps),
        "camera_source": (
            "Phase-1 cam_world_pos_upright/cam_world_rot_upright; per-sequence Aria "
            "device-to-RGB extrinsic plus upright optical rotation"
        ),
        "motion_source": (
            "floor-calibrated, frame-0-canonicalized GT 283-D UniEgo; decoded SOMA Head joint 6"
        ),
        "absolute_alignment": (
            "one constant left world transform per window aligns measured camera frame 0 to "
            "decoded Head@X frame 0; all subsequent poses are untouched"
        ),
        "camera_forward_axis": "+Z of the upright RGB optical camera",
        "head_camera_calibration": str(Path(args.head_camera_calibration).resolve()),
        "head_camera_calibration_split": calibration_payload.get("split"),
        "contract_check_maxima": contract_max,
        "aggregate": aggregate,
        "diagnostic_correlations": correlations,
        "selected_static": [case["name"] for case in static_cases],
        "selected_animations": [
            case["name"] for case in animation_cases[: max(0, args.animations)]
        ],
        "per_window": {
            case["name"]: {
                "uuid": case["uuid"],
                "start": case["start"],
                "caption": case["caption"],
                "metrics": case["metrics"],
                "contracts": case["contracts"],
            }
            for case in cases
        },
    }
    _write_json(out_dir / "summary.json", payload)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
