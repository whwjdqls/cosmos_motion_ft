#!/usr/bin/env python3
"""Render original-vs-camhead_v1 UniEgo motion with the standard SOMA skin.

This is deliberately a *visualization* skinning path.  The restored repository has
Kimodo's standard 77-joint SOMA surface, but not the subject-specific SOMA-X meshes.
The 30-joint UniEgo rotations are converted to local rotations, expanded to SOMA-77
with the standard relaxed-hand pose, forward-kinematized on the standard skeleton,
and passed through the skin's linear-blend-skinning weights.  This mirrors
``kimodo.viz.soma_skin.SOMASkin`` without importing the full Kimodo model package.

The comparison keeps both panels synchronized and uses the same standard body.  It
therefore isolates the visual effect of replacing the Head world rotation.  The
actual camhead_v1 corpus still preserves every decoded joint position and every
non-Head decoded world rotation; surface-vertex displacement is a derived rendering
effect, not a claim that position channels were changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np
import torch

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


matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

WEKA_ROOT = Path(os.environ.get("WEKA_ROOT", "/mnt/projects/ll/jungbinc/weka"))
DATA_ROOT = WEKA_ROOT / "nymeriaplus_kimodo_proportional"
RUN_ROOT = Path(os.environ.get("RUN_ROOT", WEKA_ROOT / "cosmos_motion_ft_runs"))
DEFAULT_OLD_ROOT = DATA_ROOT / "uniego_rep"
DEFAULT_NEW_ROOT = DATA_ROOT / "uniego_rep_camhead_v1"
DEFAULT_CAMERA_ROOT = DATA_ROOT / "camera_rgb"
DEFAULT_CALIBRATION = Path(__file__).with_name("head_camera_calibration_train.json")
DEFAULT_SOURCE_GALLERY = (
    RUN_ROOT
    / "nymeria_camera_head_recanonicalization_v1"
    / "qualitative"
    / "gallery_manifest.json"
)
DEFAULT_OUTPUT = (
    RUN_ROOT
    / "nymeria_camera_head_recanonicalization_v1"
    / "qualitative_soma_mesh"
)
DEFAULT_SOMA_ASSETS = (
    WEKA_ROOT
    / "shape_aware_motion_eval_c45_20260715"
    / "code"
    / "kimodo_open"
    / "kimodo"
    / "assets"
    / "skeletons"
)

SOMA30_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumbEnd",
    "LeftHandMiddleEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumbEnd",
    "RightHandMiddleEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)
SOMA30_PARENTS = np.asarray(
    (-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13,
     3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28),
    dtype=np.int64,
)
HEAD_SUBTREE_NAMES = frozenset(("Head", "HeadEnd", "Jaw", "LeftEye", "RightEye"))
BODY_RGB = np.asarray((0.69, 0.73, 0.79), dtype=np.float64)
OLD_RGB = np.asarray((0.93, 0.43, 0.16), dtype=np.float64)
NEW_RGB = np.asarray((0.12, 0.64, 0.34), dtype=np.float64)
CAMERA_RGB = np.asarray((0.05, 0.35, 0.86), dtype=np.float64)


@dataclass(frozen=True)
class StandardSomaSkin:
    """Compact standard SOMA visualization skin and exact rig data."""

    bind_vertices: np.ndarray
    faces: np.ndarray
    bind_rig_transform_inv: np.ndarray
    lbs_indices: np.ndarray
    lbs_weights: np.ndarray
    neutral_joints: np.ndarray
    relaxed_local_rotations: np.ndarray
    parents77: np.ndarray
    map30_to77: np.ndarray
    rig_joint_names: tuple[str, ...]
    head_influence: np.ndarray
    face_head_influence: np.ndarray
    head_face_indices: np.ndarray
    source_vertex_count: int
    source_face_count: int
    cluster_m: float


_SKIN_CACHE: dict[tuple[str, float], StandardSomaSkin] = {}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _rotation_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return rotation_angle_deg(np.swapaxes(predicted, -1, -2) @ target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load_array(path: Path) -> np.ndarray:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor).squeeze()


def _parents_from_connections(connections: np.ndarray, joint_count: int) -> np.ndarray:
    parents = np.full(joint_count, -1, dtype=np.int64)
    for parent, child in np.asarray(connections, dtype=np.int64):
        if child == 0 or parents[child] >= 0:
            raise ValueError(f"invalid or duplicate SOMA rig child {child}")
        parents[child] = parent
    if np.flatnonzero(parents < 0).tolist() != [0]:
        raise ValueError("SOMA rig must contain exactly one root at joint 0")
    if np.any(parents[1:] >= np.arange(1, joint_count)):
        raise ValueError("SOMA rig is not stored in parent-before-child order")
    return parents


def _cluster_skin(
    vertices: np.ndarray,
    faces: np.ndarray,
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
    joint_count: int,
    cluster_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vertex-cluster a skin while averaging and renormalizing LBS weights."""
    if cluster_m <= 0.0:
        return (
            vertices.astype(np.float64, copy=True),
            faces.astype(np.int64, copy=True),
            lbs_indices.astype(np.int64, copy=True),
            lbs_weights.astype(np.float64, copy=True),
        )

    quantized = np.rint((vertices - vertices.min(axis=0)) / cluster_m).astype(np.int64)
    _keys, inverse = np.unique(quantized, axis=0, return_inverse=True)
    cluster_count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=cluster_count).astype(np.float64)

    compact_vertices = np.zeros((cluster_count, 3), dtype=np.float64)
    np.add.at(compact_vertices, inverse, vertices)
    compact_vertices /= counts[:, None]

    dense_weights = np.zeros((len(vertices), joint_count), dtype=np.float64)
    rows = np.arange(len(vertices))
    for slot in range(lbs_indices.shape[1]):
        np.add.at(
            dense_weights,
            (rows, lbs_indices[:, slot]),
            lbs_weights[:, slot],
        )
    compact_dense = np.zeros((cluster_count, joint_count), dtype=np.float64)
    np.add.at(compact_dense, inverse, dense_weights)
    compact_dense /= counts[:, None]
    keep_weights = min(lbs_indices.shape[1], joint_count)
    top = np.argpartition(compact_dense, -keep_weights, axis=1)[:, -keep_weights:]
    top_values = np.take_along_axis(compact_dense, top, axis=1)
    order = np.argsort(top_values, axis=1)[:, ::-1]
    compact_indices = np.take_along_axis(top, order, axis=1).astype(np.int64)
    compact_weights = np.take_along_axis(top_values, order, axis=1)
    sums = compact_weights.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError("clustered SOMA skin contains a vertex with no LBS weight")
    compact_weights /= sums

    compact_faces = inverse[faces]
    nondegenerate = (
        (compact_faces[:, 0] != compact_faces[:, 1])
        & (compact_faces[:, 1] != compact_faces[:, 2])
        & (compact_faces[:, 0] != compact_faces[:, 2])
    )
    compact_faces = compact_faces[nondegenerate]
    signatures = np.sort(compact_faces, axis=1)
    _unique, first = np.unique(signatures, axis=0, return_index=True)
    compact_faces = compact_faces[np.sort(first)].astype(np.int64)
    return compact_vertices, compact_faces, compact_indices, compact_weights


def load_standard_soma_skin(assets_root: Path, cluster_m: float = 0.012) -> StandardSomaSkin:
    """Load and optionally compact Kimodo's standard SOMA-77 visualization skin."""
    key = (str(assets_root.resolve()), float(cluster_m))
    cached = _SKIN_CACHE.get(key)
    if cached is not None:
        return cached

    skin_path = assets_root / "somaskel77" / "skin_standard.npz"
    joints_path = assets_root / "somaskel77" / "joints.p"
    relaxed_path = assets_root / "somaskel77" / "relaxed_hands_rest_pose.npy"
    for path in (skin_path, joints_path, relaxed_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(skin_path, allow_pickle=False) as archive:
        source_vertices = archive["bind_vertices"].astype(np.float64)
        source_faces = archive["faces"].astype(np.int64)
        bind_rig = archive["bind_rig_transform"].astype(np.float64)
        rig_names = tuple(map(str, archive["rig_joint_names"].tolist()))
        source_indices = archive["lbs_indices"].astype(np.int64)
        source_weights = archive["lbs_weights"].astype(np.float64)
        connections = archive["rig_joint_connections"].astype(np.int64)
    if len(rig_names) != 77 or len(source_vertices) != len(source_indices):
        raise ValueError("unexpected standard SOMA skin shape")
    if not set(SOMA30_NAMES).issubset(rig_names):
        raise ValueError("standard skin is missing one or more SOMA-30 joints")
    parents77 = _parents_from_connections(connections, len(rig_names))
    map30 = np.asarray([rig_names.index(name) for name in SOMA30_NAMES], dtype=np.int64)
    neutral_joints = _torch_load_array(joints_path).astype(np.float64)
    relaxed = np.load(relaxed_path).astype(np.float64)
    if neutral_joints.shape != (77, 3) or relaxed.shape != (77, 3, 3):
        raise ValueError("unexpected standard SOMA skeleton asset shape")

    vertices, faces, indices, weights = _cluster_skin(
        source_vertices,
        source_faces,
        source_indices,
        source_weights,
        len(rig_names),
        cluster_m,
    )
    subtree = np.asarray(
        [index for index, name in enumerate(rig_names) if name in HEAD_SUBTREE_NAMES],
        dtype=np.int64,
    )
    head_influence = np.zeros(len(vertices), dtype=np.float64)
    for joint in subtree:
        head_influence += np.where(indices == joint, weights, 0.0).sum(axis=1)
    face_head_influence = head_influence[faces].mean(axis=1)
    head_face_indices = np.flatnonzero(np.any(head_influence[faces] > 1e-4, axis=1))
    if len(head_face_indices) == 0:
        raise ValueError("standard SOMA skin has no Head-influenced surface")

    skin = StandardSomaSkin(
        bind_vertices=vertices,
        faces=faces,
        bind_rig_transform_inv=np.linalg.inv(bind_rig),
        lbs_indices=indices,
        lbs_weights=weights,
        neutral_joints=neutral_joints,
        relaxed_local_rotations=relaxed,
        parents77=parents77,
        map30_to77=map30,
        rig_joint_names=rig_names,
        head_influence=head_influence,
        face_head_influence=face_head_influence,
        head_face_indices=head_face_indices,
        source_vertex_count=len(source_vertices),
        source_face_count=len(source_faces),
        cluster_m=float(cluster_m),
    )
    _SKIN_CACHE[key] = skin
    return skin


def world_to_local_rotations(
    world_rotations: np.ndarray,
    parents: np.ndarray = SOMA30_PARENTS,
) -> np.ndarray:
    """Convert parent-before-child global rotations to local rotations."""
    world_rotations = np.asarray(world_rotations, dtype=np.float64)
    if world_rotations.ndim != 4 or world_rotations.shape[-2:] != (3, 3):
        raise ValueError(f"expected [T,J,3,3] rotations, got {world_rotations.shape}")
    if world_rotations.shape[1] != len(parents):
        raise ValueError("rotation joint count does not match the parent array")
    local = np.empty_like(world_rotations)
    root = int(np.flatnonzero(parents < 0)[0])
    local[:, root] = world_rotations[:, root]
    for joint, parent in enumerate(parents):
        if parent < 0:
            continue
        local[:, joint] = (
            np.swapaxes(world_rotations[:, parent], -1, -2)
            @ world_rotations[:, joint]
        )
    return local


def _forward_kinematics(
    local_rotations: np.ndarray,
    root_positions: np.ndarray,
    neutral_joints: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frames, joints = local_rotations.shape[:2]
    global_rotations = np.empty_like(local_rotations)
    global_positions = np.empty((frames, joints, 3), dtype=np.float64)
    root = int(np.flatnonzero(parents < 0)[0])
    global_rotations[:, root] = local_rotations[:, root]
    global_positions[:, root] = root_positions
    for joint, parent in enumerate(parents):
        if parent < 0:
            continue
        global_rotations[:, joint] = (
            global_rotations[:, parent] @ local_rotations[:, joint]
        )
        offset = neutral_joints[joint] - neutral_joints[parent]
        global_positions[:, joint] = global_positions[:, parent] + np.einsum(
            "tij,j->ti", global_rotations[:, parent], offset, optimize=True
        )
    return global_rotations, global_positions


def standardize_soma30_transforms(
    world_rotations30: np.ndarray,
    root_positions: np.ndarray,
    skin: StandardSomaSkin,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror SOMASkin's 30->77 expansion and standard-skeleton FK."""
    local30 = world_to_local_rotations(world_rotations30)
    frames = len(local30)
    local77 = np.broadcast_to(
        skin.relaxed_local_rotations, (frames, 77, 3, 3)
    ).copy()
    local77[:, skin.map30_to77] = local30
    return _forward_kinematics(
        local77,
        np.asarray(root_positions, dtype=np.float64),
        skin.neutral_joints,
        skin.parents77,
    )


def skin_soma30_motion(
    world_rotations30: np.ndarray,
    root_positions: np.ndarray,
    skin: StandardSomaSkin,
    chunk_frames: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skin SOMA-30 motion onto the standard SOMA-77 surface in bounded chunks."""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    rotations77, positions77 = standardize_soma30_transforms(
        world_rotations30, root_positions, skin
    )
    frames = len(rotations77)
    vertices = np.empty((frames, len(skin.bind_vertices), 3), dtype=np.float32)
    homogeneous = np.concatenate(
        (skin.bind_vertices, np.ones((len(skin.bind_vertices), 1))), axis=1
    )
    for start in range(0, frames, chunk_frames):
        end = min(start + chunk_frames, frames)
        transforms = np.broadcast_to(
            np.eye(4, dtype=np.float64), (end - start, 77, 4, 4)
        ).copy()
        transforms[:, :, :3, :3] = rotations77[start:end]
        transforms[:, :, :3, 3] = positions77[start:end]
        affine = (
            transforms @ skin.bind_rig_transform_inv[None]
        )[:, :, :3, :]
        selected = affine[:, skin.lbs_indices]
        transformed = np.einsum(
            "tvwkl,vl->tvwk", selected, homogeneous, optimize=True
        )
        chunk_vertices = (
            transformed * skin.lbs_weights[None, :, :, None]
        ).sum(axis=2)
        vertices[start:end] = chunk_vertices.astype(np.float32)
    return vertices, rotations77, positions77


def _load_case_window(
    case: dict[str, Any],
    old_root: Path,
    new_root: Path,
    camera_root: Path,
) -> dict[str, Any]:
    relative = Path(str(case["uuid"]) + ".npz")
    start = int(case["start"])
    requested_end = int(case.get("end", start + int(case["frames"])))
    with np.load(old_root / relative, allow_pickle=False) as archive:
        old_features = archive["features"][:requested_end].copy()
    with np.load(new_root / relative, allow_pickle=False) as archive:
        new_features = archive["features"][:requested_end].copy()
    with np.load(camera_root / relative, allow_pickle=False) as archive:
        camera_frames = min(
            requested_end,
            len(old_features),
            len(new_features),
            len(archive["cam_world_rot_upright"]),
        )
        camera_rotation = camera_rotations_to_kimodo(
            archive["cam_world_rot_upright"][start:camera_frames]
        )
        camera_position = np.einsum(
            "ij,tj->ti",
            ARIA_Z_UP_TO_KIMODO_Y_UP,
            archive["cam_world_pos_upright"][start:camera_frames].astype(np.float64),
            optimize=True,
        )
        fps = float(np.asarray(archive["fps"]).item())
    if camera_frames <= start:
        raise ValueError(f"empty case window {case['uuid']}@{start}")
    old_decoded = decode_uniego(old_features[:camera_frames])
    new_decoded = decode_uniego(new_features[:camera_frames])
    selector = slice(start, camera_frames)
    return {
        "old_features": old_features[selector],
        "new_features": new_features[selector],
        "old_rotations": old_decoded.world_rotations[selector],
        "new_rotations": new_decoded.world_rotations[selector],
        "old_positions": old_decoded.world_positions[selector],
        "new_positions": new_decoded.world_positions[selector],
        "camera_rotation": camera_rotation,
        "camera_position": camera_position,
        "fps": fps,
        "end": camera_frames,
    }


def _prepare_case(
    case: dict[str, Any],
    old_root: Path,
    new_root: Path,
    camera_root: Path,
    skin: StandardSomaSkin,
    rotation_head_to_camera: np.ndarray,
    skin_chunk_frames: int,
) -> dict[str, Any]:
    sequence = _load_case_window(case, old_root, new_root, camera_root)
    old_vertices, old_rot77, old_pos77 = skin_soma30_motion(
        sequence["old_rotations"],
        sequence["old_positions"][:, 0],
        skin,
        chunk_frames=skin_chunk_frames,
    )
    new_vertices, new_rot77, new_pos77 = skin_soma30_motion(
        sequence["new_rotations"],
        sequence["new_positions"][:, 0],
        skin,
        chunk_frames=skin_chunk_frames,
    )

    # Calibrated floor is a world-y shift.  X/Z follow the root only in the mesh
    # panels; the world-trajectory panel below remains world-relative.
    floor_offset = float(case["floor_offset"])
    old_vertices[..., 1] -= floor_offset
    new_vertices[..., 1] -= floor_offset
    old_pos77[..., 1] -= floor_offset
    new_pos77[..., 1] -= floor_offset
    horizontal_center = old_pos77[:, 0][:, (0, 2)].copy()
    for values in (old_vertices, new_vertices, old_pos77, new_pos77):
        values[..., 0] -= horizontal_center[:, None, 0]
        values[..., 2] -= horizontal_center[:, None, 1]

    measured_rotation = sequence["camera_rotation"]
    old_head_rotation = sequence["old_rotations"][:, HEAD_JOINT_IDX]
    new_head_rotation = sequence["new_rotations"][:, HEAD_JOINT_IDX]
    old_implied_camera = old_head_rotation @ rotation_head_to_camera
    new_implied_camera = new_head_rotation @ rotation_head_to_camera
    orientation_old = _rotation_error(old_implied_camera, measured_rotation)
    orientation_new = _rotation_error(new_implied_camera, measured_rotation)
    head_change = _rotation_error(old_head_rotation, new_head_rotation)

    displacement = np.linalg.norm(
        new_vertices.astype(np.float64) - old_vertices.astype(np.float64), axis=-1
    )
    affected = skin.head_influence > 1e-6
    unaffected = ~affected
    displacement_mean = displacement.mean(axis=1)
    displacement_p95 = np.quantile(displacement[:, affected], 0.95, axis=1)
    displacement_max = displacement.max(axis=1)
    protected_body_max = (
        displacement[:, unaffected].max(axis=1)
        if np.any(unaffected)
        else np.zeros(len(displacement), dtype=np.float64)
    )

    old_positions = sequence["old_positions"]
    new_positions = sequence["new_positions"]
    position_delta = np.linalg.norm(new_positions - old_positions, axis=-1)
    nonhead = np.arange(old_positions.shape[1]) != HEAD_JOINT_IDX
    nonhead_rotation_delta = _rotation_error(
        sequence["old_rotations"][:, nonhead],
        sequence["new_rotations"][:, nonhead],
    )
    foot_indices = np.asarray(FOOT_JOINT_IDX, dtype=np.int64)
    old_feet = old_positions[:, foot_indices].copy()
    new_feet = new_positions[:, foot_indices].copy()
    old_feet[..., 1] -= floor_offset
    new_feet[..., 1] -= floor_offset
    foot_height_old = old_feet[..., 1].min(axis=1)
    foot_height_new = new_feet[..., 1].min(axis=1)
    fps = float(sequence["fps"])
    foot_speed_old = np.linalg.norm(np.diff(old_feet, axis=0), axis=-1) * fps
    foot_speed_new = np.linalg.norm(np.diff(new_feet, axis=0), axis=-1) * fps
    contacts = sequence["old_features"][:, DELTA_END:] > 0.5
    contact_count = int(contacts[:-1].sum())
    contact_skate_old = float(
        (foot_speed_old * contacts[:-1]).sum() / max(contact_count, 1)
    )
    contact_skate_new = float(
        (foot_speed_new * contacts[:-1]).sum() / max(contact_count, 1)
    )

    root_trajectory = old_positions[:, 0][:, (0, 2)]
    root_trajectory = root_trajectory - root_trajectory[:1]
    camera_trajectory = sequence["camera_position"][:, (0, 2)]
    camera_trajectory = camera_trajectory - camera_trajectory[:1]
    peak_frame = int(np.argmax(head_change))
    metrics = {
        "frames": len(old_vertices),
        "fps": fps,
        "old_camera_rotation_error_mean_deg": float(orientation_old.mean()),
        "old_camera_rotation_error_p90_deg": float(np.quantile(orientation_old, 0.90)),
        "old_camera_rotation_error_max_deg": float(orientation_old.max()),
        "new_camera_rotation_error_mean_deg": float(orientation_new.mean()),
        "new_camera_rotation_error_p90_deg": float(np.quantile(orientation_new, 0.90)),
        "new_camera_rotation_error_max_deg": float(orientation_new.max()),
        "head_rotation_change_mean_deg": float(head_change.mean()),
        "head_rotation_change_p95_deg": float(np.quantile(head_change, 0.95)),
        "head_rotation_change_max_deg": float(head_change.max()),
        "surface_displacement_mean_mm": float(displacement.mean() * 1000.0),
        "surface_displacement_affected_p95_mm": float(
            np.quantile(displacement[:, affected], 0.95) * 1000.0
        ),
        "surface_displacement_max_mm": float(displacement.max() * 1000.0),
        "protected_body_surface_displacement_max_mm": float(
            protected_body_max.max() * 1000.0
        ),
        "decoded_joint_position_delta_max_mm": float(position_delta.max() * 1000.0),
        "decoded_nonhead_rotation_delta_max_deg": float(nonhead_rotation_delta.max()),
        "foot_height_delta_max_mm": float(
            np.abs(foot_height_new - foot_height_old).max() * 1000.0
        ),
        "contact_foot_skate_original_cmps": contact_skate_old * 100.0,
        "contact_foot_skate_aligned_cmps": contact_skate_new * 100.0,
        "contact_foot_skate_delta_cmps": (contact_skate_new - contact_skate_old) * 100.0,
        "head_affected_compact_vertices": int(affected.sum()),
        "protected_compact_vertices": int(unaffected.sum()),
        "peak_impact_frame_relative": peak_frame,
        "peak_impact_frame_absolute": int(case["start"]) + peak_frame,
    }
    return {
        **sequence,
        "old_vertices": old_vertices,
        "new_vertices": new_vertices,
        "old_rot77": old_rot77,
        "new_rot77": new_rot77,
        "old_pos77": old_pos77,
        "new_pos77": new_pos77,
        "old_implied_camera": old_implied_camera,
        "new_implied_camera": new_implied_camera,
        "orientation_old": orientation_old,
        "orientation_new": orientation_new,
        "head_change": head_change,
        "displacement_mean": displacement_mean,
        "displacement_p95": displacement_p95,
        "displacement_max": displacement_max,
        "protected_body_max": protected_body_max,
        "foot_height_old": foot_height_old,
        "foot_height_new": foot_height_new,
        "root_trajectory": root_trajectory,
        "camera_trajectory": camera_trajectory,
        "metrics": metrics,
    }


def _face_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_influence: np.ndarray,
    accent: np.ndarray,
) -> np.ndarray:
    triangles = vertices[faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-12)
    light = np.asarray((-0.25, 0.82, 0.52), dtype=np.float64)
    light /= np.linalg.norm(light)
    intensity = 0.42 + 0.58 * np.clip(normals @ light, 0.0, 1.0)
    blend = np.clip(face_influence * 2.5, 0.0, 1.0)[:, None]
    base = BODY_RGB[None] * (1.0 - blend) + accent[None] * blend
    rgb = np.clip(base * intensity[:, None] + 0.045, 0.0, 1.0)
    return np.concatenate((rgb, np.ones((len(rgb), 1))), axis=1)


def _plot_triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertices[faces][:, :, (0, 2, 1)]


def _configure_mesh_axis(axis: Any, title: str, *, closeup: bool) -> None:
    axis.set_title(title, fontsize=12, fontweight="bold", pad=3)
    axis.view_init(elev=11 if not closeup else 7, azim=-64)
    axis.set_proj_type("persp", focal_length=0.9)
    axis.set_facecolor("#f5f7fa")
    axis.set_axis_off()


def _add_floor(axis: Any, x_half: float, z_half: float) -> None:
    for x in np.linspace(-x_half, x_half, 9):
        axis.plot([x, x], [-z_half, z_half], [0.0, 0.0], color="0.78", lw=0.55, zorder=0)
    for z in np.linspace(-z_half, z_half, 9):
        axis.plot([-x_half, x_half], [z, z], [0.0, 0.0], color="0.78", lw=0.55, zorder=0)


def _draw_orientation_pair(
    axis: Any,
    origin_xyz: np.ndarray,
    measured: np.ndarray,
    implied: np.ndarray,
    accent: np.ndarray,
    length: float,
) -> list[Any]:
    origin = origin_xyz[np.asarray((0, 2, 1))]
    artists: list[Any] = []
    for rotation, color, alpha, width in (
        (measured, CAMERA_RGB, 1.0, 2.8),
        (implied, accent, 0.95, 2.8),
    ):
        # Forward and up together expose full orientation, including camera roll.
        for local_axis, scale, axis_alpha in ((2, 1.0, alpha), (1, 0.72, alpha * 0.60)):
            direction = rotation[np.asarray((0, 2, 1)), local_axis] * length * scale
            artists.append(
                axis.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    direction[0],
                    direction[1],
                    direction[2],
                    color=color,
                    alpha=axis_alpha,
                    linewidth=width if local_axis == 2 else 1.7,
                    arrow_length_ratio=0.22,
                )
            )
    return artists


def _set_equal_trajectory_limits(axis: Any, arrays: list[np.ndarray]) -> None:
    points = np.concatenate(arrays, axis=0)
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    radius = max(float(np.ptp(points, axis=0).max()) * 0.58, 0.25)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_aspect("equal", adjustable="box")


def _render_case_figure(
    case: dict[str, Any],
    data: dict[str, Any],
    skin: StandardSomaSkin,
) -> tuple[Any, Any]:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.88),
        height_ratios=(1.16, 0.84),
        left=0.025,
        right=0.985,
        bottom=0.08,
        top=0.88,
        wspace=0.025,
        hspace=0.035,
    )
    old_full = fig.add_subplot(outer[0, 0], projection="3d")
    new_full = fig.add_subplot(outer[0, 1], projection="3d")
    old_head = fig.add_subplot(outer[1, 0], projection="3d")
    new_head = fig.add_subplot(outer[1, 1], projection="3d")
    right = outer[:, 2].subgridspec(4, 1, hspace=0.52)
    error_axis = fig.add_subplot(right[0, 0])
    trajectory_axis = fig.add_subplot(right[1, 0])
    displacement_axis = fig.add_subplot(right[2, 0])
    foot_axis = fig.add_subplot(right[3, 0])

    _configure_mesh_axis(old_full, "ORIGINAL UniEgo · full body", closeup=False)
    _configure_mesh_axis(new_full, "CAMHEAD_V1 · full body", closeup=False)
    _configure_mesh_axis(old_head, "ORIGINAL · head/neck close-up", closeup=True)
    _configure_mesh_axis(new_head, "CAMHEAD_V1 · head/neck close-up", closeup=True)

    all_vertices = np.concatenate((data["old_vertices"], data["new_vertices"]), axis=0)
    x_half = max(0.92, float(np.quantile(np.abs(all_vertices[..., 0]), 0.998)) * 1.08)
    z_half = max(0.72, float(np.quantile(np.abs(all_vertices[..., 2]), 0.998)) * 1.08)
    y_min = min(-0.12, float(np.quantile(all_vertices[..., 1], 0.001)) - 0.06)
    y_max = max(1.86, float(np.quantile(all_vertices[..., 1], 0.999)) + 0.08)
    for axis in (old_full, new_full):
        axis.set_xlim(-x_half, x_half)
        axis.set_ylim(-z_half, z_half)
        axis.set_zlim(y_min, y_max)
        axis.set_box_aspect((2.0 * x_half, 2.0 * z_half, y_max - y_min))
        _add_floor(axis, x_half, z_half)

    head_faces = skin.faces[skin.head_face_indices]
    first_old = data["old_vertices"][0]
    first_new = data["new_vertices"][0]
    old_colors = _face_colors(first_old, skin.faces, skin.face_head_influence, OLD_RGB)
    new_colors = _face_colors(first_new, skin.faces, skin.face_head_influence, NEW_RGB)
    old_full_mesh = Poly3DCollection(
        _plot_triangles(first_old, skin.faces), facecolors=old_colors, linewidths=0.0
    )
    new_full_mesh = Poly3DCollection(
        _plot_triangles(first_new, skin.faces), facecolors=new_colors, linewidths=0.0
    )
    old_head_mesh = Poly3DCollection(
        _plot_triangles(first_old, head_faces),
        facecolors=old_colors[skin.head_face_indices],
        linewidths=0.0,
    )
    new_head_mesh = Poly3DCollection(
        _plot_triangles(first_new, head_faces),
        facecolors=new_colors[skin.head_face_indices],
        linewidths=0.0,
    )
    for axis, mesh in (
        (old_full, old_full_mesh),
        (new_full, new_full_mesh),
        (old_head, old_head_mesh),
        (new_head, new_head_mesh),
    ):
        mesh.set_zsort("average")
        axis.add_collection3d(mesh)

    frames = int(data["metrics"]["frames"])
    fps = float(data["metrics"]["fps"])
    time_axis = np.arange(frames) / fps
    error_axis.plot(time_axis, data["orientation_old"], color=OLD_RGB, lw=2.0, label="original")
    error_axis.plot(time_axis, data["orientation_new"], color=NEW_RGB, lw=2.0, label="camhead_v1")
    error_cursor = error_axis.axvline(0.0, color="0.1", lw=1.1)
    error_axis.set_ylabel("SO(3) error [deg]")
    error_axis.set_title("Measured camera vs Head-implied camera", fontsize=10)
    error_axis.grid(alpha=0.22)
    error_axis.legend(fontsize=8, ncol=2, loc="upper right")

    root_traj = data["root_trajectory"]
    camera_traj = data["camera_trajectory"]
    trajectory_axis.plot(root_traj[:, 0], root_traj[:, 1], color="0.3", alpha=0.20, lw=1.4)
    trajectory_axis.plot(camera_traj[:, 0], camera_traj[:, 1], color=CAMERA_RGB, alpha=0.20, lw=1.4)
    root_progress, = trajectory_axis.plot([], [], color="0.25", lw=2.2, label="pelvis")
    camera_progress, = trajectory_axis.plot([], [], color=CAMERA_RGB, lw=2.2, label="measured camera")
    root_marker, = trajectory_axis.plot([], [], "o", color="0.15", ms=5)
    camera_marker, = trajectory_axis.plot([], [], "o", color=CAMERA_RGB, ms=5)
    _set_equal_trajectory_limits(trajectory_axis, [root_traj, camera_traj])
    trajectory_axis.set_xlabel("world x from frame 0 [m]")
    trajectory_axis.set_ylabel("world z from frame 0 [m]")
    trajectory_axis.set_title("World trajectory (unchanged)", fontsize=10)
    trajectory_axis.grid(alpha=0.22)
    trajectory_axis.legend(fontsize=8, ncol=2, loc="best")

    displacement_axis.plot(
        time_axis,
        data["displacement_max"] * 100.0,
        color="tab:red",
        lw=1.7,
        label="max surface",
    )
    displacement_axis.plot(
        time_axis,
        data["displacement_p95"] * 100.0,
        color="tab:purple",
        lw=1.7,
        label="p95 Head-affected surface",
    )
    displacement_axis.plot(
        time_axis,
        data["displacement_mean"] * 100.0,
        color="0.25",
        lw=1.2,
        label="whole-surface mean",
    )
    displacement_cursor = displacement_axis.axvline(0.0, color="0.1", lw=1.1)
    displacement_axis.set_ylabel("old↔new [cm]")
    displacement_axis.set_title("Derived standard-skin displacement", fontsize=10)
    displacement_axis.grid(alpha=0.22)
    displacement_axis.legend(fontsize=7.5, loc="upper right")

    foot_axis.plot(
        time_axis,
        data["foot_height_old"] * 100.0,
        color=OLD_RGB,
        lw=2.1,
        label="original min foot height",
    )
    foot_axis.plot(
        time_axis,
        data["foot_height_new"] * 100.0,
        color=NEW_RGB,
        lw=1.5,
        ls="--",
        label="camhead_v1 (overlap expected)",
    )
    foot_axis.axhline(0.0, color="0.35", lw=0.9)
    foot_cursor = foot_axis.axvline(0.0, color="0.1", lw=1.1)
    foot_axis.set_xlabel("time [s]")
    foot_axis.set_ylabel("height [cm]")
    foot_axis.set_title("Foot/floor preservation", fontsize=10)
    foot_axis.grid(alpha=0.22)
    foot_axis.legend(fontsize=7.5, loc="upper right")

    metrics = data["metrics"]
    header = fig.suptitle("", fontsize=15, fontweight="bold", y=0.975)
    caption = str(case.get("caption", "")).replace("\n", " ")
    if len(caption) > 220:
        caption = caption[:217] + "..."
    fig.text(
        0.5,
        0.925,
        caption,
        ha="center",
        va="top",
        fontsize=9.2,
        color="0.22",
        wrap=True,
    )
    status = fig.text(
        0.5,
        0.035,
        "",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="0.18",
    )

    state: dict[str, Any] = {"orientation_artists": []}

    def update(frame: int) -> None:
        old_vertices = data["old_vertices"][frame]
        new_vertices = data["new_vertices"][frame]
        old_face_colors = _face_colors(
            old_vertices, skin.faces, skin.face_head_influence, OLD_RGB
        )
        new_face_colors = _face_colors(
            new_vertices, skin.faces, skin.face_head_influence, NEW_RGB
        )
        old_full_mesh.set_verts(_plot_triangles(old_vertices, skin.faces))
        old_full_mesh.set_facecolors(old_face_colors)
        new_full_mesh.set_verts(_plot_triangles(new_vertices, skin.faces))
        new_full_mesh.set_facecolors(new_face_colors)
        old_head_mesh.set_verts(_plot_triangles(old_vertices, head_faces))
        old_head_mesh.set_facecolors(old_face_colors[skin.head_face_indices])
        new_head_mesh.set_verts(_plot_triangles(new_vertices, head_faces))
        new_head_mesh.set_facecolors(new_face_colors[skin.head_face_indices])

        old_head_origin = data["old_pos77"][frame, HEAD_JOINT_IDX]
        new_head_origin = data["new_pos77"][frame, HEAD_JOINT_IDX]
        for axis, center in ((old_head, old_head_origin), (new_head, new_head_origin)):
            axis.set_xlim(center[0] - 0.27, center[0] + 0.27)
            axis.set_ylim(center[2] - 0.24, center[2] + 0.24)
            axis.set_zlim(center[1] - 0.24, center[1] + 0.31)
            axis.set_box_aspect((0.54, 0.48, 0.55))

        for artist in state["orientation_artists"]:
            artist.remove()
        state["orientation_artists"] = []
        for axis, origin, implied, accent, length in (
            (old_full, old_head_origin, data["old_implied_camera"][frame], OLD_RGB, 0.25),
            (new_full, new_head_origin, data["new_implied_camera"][frame], NEW_RGB, 0.25),
            (old_head, old_head_origin, data["old_implied_camera"][frame], OLD_RGB, 0.18),
            (new_head, new_head_origin, data["new_implied_camera"][frame], NEW_RGB, 0.18),
        ):
            state["orientation_artists"].extend(
                _draw_orientation_pair(
                    axis,
                    origin,
                    data["camera_rotation"][frame],
                    implied,
                    accent,
                    length,
                )
            )

        current_time = time_axis[frame]
        for cursor in (error_cursor, displacement_cursor, foot_cursor):
            cursor.set_xdata([current_time, current_time])
        root_progress.set_data(root_traj[: frame + 1, 0], root_traj[: frame + 1, 1])
        camera_progress.set_data(camera_traj[: frame + 1, 0], camera_traj[: frame + 1, 1])
        root_marker.set_data([root_traj[frame, 0]], [root_traj[frame, 1]])
        camera_marker.set_data([camera_traj[frame, 0]], [camera_traj[frame, 1]])
        header.set_text(
            f"{case['category']}  |  {case['uuid']}@{int(case['start'])}  |  "
            f"t={current_time:.2f}s  (frame {frame + 1}/{frames})"
        )
        status.set_text(
            "Blue = measured camera; orange/green = Head-implied camera; thick = +Z forward, thin = +Y up   ·   "
            f"SO(3) error {data['orientation_old'][frame]:.2f}° → {data['orientation_new'][frame]:.4f}°   ·   "
            f"Head ΔR {data['head_change'][frame]:.2f}°   ·   "
            f"decoded max Δposition {metrics['decoded_joint_position_delta_max_mm']:.4f} mm   ·   "
            f"contact skate {metrics['contact_foot_skate_original_cmps']:.2f} → "
            f"{metrics['contact_foot_skate_aligned_cmps']:.2f} cm/s"
        )

    return fig, update


def _process_case(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        case,
        old_root_text,
        new_root_text,
        camera_root_text,
        assets_root_text,
        output_text,
        rotation_values,
        cluster_m,
        skin_chunk_frames,
        frame_stride,
        render_video,
    ) = payload
    started = time.time()
    output_root = Path(output_text)
    filename = (
        f"{_safe_name(str(case['category']))}__{_safe_name(str(case['uuid']))}__"
        f"{int(case['start']):06d}"
    )
    video_path = output_root / "videos" / f"{filename}.mp4"
    poster_path = output_root / "posters" / f"{filename}.png"
    try:
        skin = load_standard_soma_skin(Path(assets_root_text), float(cluster_m))
        data = _prepare_case(
            case,
            Path(old_root_text),
            Path(new_root_text),
            Path(camera_root_text),
            skin,
            np.asarray(rotation_values, dtype=np.float64),
            int(skin_chunk_frames),
        )
        fig, update = _render_case_figure(case, data, skin)
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        peak = int(data["metrics"]["peak_impact_frame_relative"])
        update(peak)
        fig.savefig(poster_path, dpi=100, facecolor="white")

        rendered_frames = 0
        if render_video:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = video_path.with_suffix(".tmp.mp4")
            writer = FFMpegWriter(
                fps=float(data["fps"]) / int(frame_stride),
                codec="libx264",
                bitrate=-1,
                extra_args=["-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart"],
                metadata={
                    "title": f"SOMA mesh original vs camhead_v1: {case['category']}",
                    "artist": "cosmos_motion_ft",
                },
            )
            render_frames = list(range(0, int(data["metrics"]["frames"]), int(frame_stride)))
            if render_frames[-1] != int(data["metrics"]["frames"]) - 1:
                render_frames.append(int(data["metrics"]["frames"]) - 1)
            with writer.saving(fig, str(temporary), dpi=100):
                for frame in render_frames:
                    update(frame)
                    writer.grab_frame()
            os.replace(temporary, video_path)
            rendered_frames = len(render_frames)
        plt.close(fig)
        return {
            "status": "ok",
            "category": case["category"],
            "uuid": case["uuid"],
            "start": int(case["start"]),
            "video": str(video_path) if render_video else None,
            "poster": str(poster_path),
            "rendered_frames": rendered_frames,
            "playback_fps": float(data["fps"]) / int(frame_stride) if render_video else None,
            "metrics": data["metrics"],
            "seconds": time.time() - started,
        }
    except Exception as error:  # noqa: BLE001
        return {
            "status": "error",
            "category": case.get("category"),
            "uuid": case.get("uuid"),
            "start": int(case.get("start", -1)),
            "video": str(video_path),
            "poster": str(poster_path),
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.time() - started,
        }


def _render_metric_summary(records: list[dict[str, Any]], output: Path) -> None:
    successful = [record for record in records if record["status"] == "ok"]
    labels = [str(record["category"]).replace("known_", "k_") for record in successful]
    x = np.arange(len(successful))
    old_error = np.asarray(
        [record["metrics"]["old_camera_rotation_error_mean_deg"] for record in successful]
    )
    new_error = np.asarray(
        [record["metrics"]["new_camera_rotation_error_mean_deg"] for record in successful]
    )
    surface_p95 = np.asarray(
        [record["metrics"]["surface_displacement_affected_p95_mm"] for record in successful]
    )
    surface_max = np.asarray(
        [record["metrics"]["surface_displacement_max_mm"] for record in successful]
    )
    position = np.asarray(
        [record["metrics"]["decoded_joint_position_delta_max_mm"] for record in successful]
    )
    protected = np.asarray(
        [record["metrics"]["protected_body_surface_displacement_max_mm"] for record in successful]
    )
    skate_delta = np.asarray(
        [abs(record["metrics"]["contact_foot_skate_delta_cmps"]) for record in successful]
    )
    head_change = np.asarray(
        [record["metrics"]["head_rotation_change_mean_deg"] for record in successful]
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    width = 0.38
    axes[0, 0].bar(x - width / 2, old_error, width, color=OLD_RGB, label="original")
    axes[0, 0].bar(x + width / 2, new_error, width, color=NEW_RGB, label="camhead_v1")
    axes[0, 0].set_ylabel("mean SO(3) error [deg]")
    axes[0, 0].set_title("Measured vs Head-implied camera orientation")
    axes[0, 0].legend()

    axes[0, 1].bar(x - width / 2, surface_p95, width, color="tab:purple", label="affected p95")
    axes[0, 1].bar(x + width / 2, surface_max, width, color="tab:red", label="maximum")
    axes[0, 1].set_ylabel("derived surface displacement [mm]")
    axes[0, 1].set_title("Visible impact on the standard SOMA skin")
    axes[0, 1].legend()

    epsilon = 1e-9
    axes[1, 0].bar(x - width, np.maximum(position, epsilon), width, color="tab:blue", label="decoded joint positions")
    axes[1, 0].bar(x, np.maximum(protected, epsilon), width, color="0.45", label="non-Head-weighted surface")
    axes[1, 0].bar(x + width, np.maximum(skate_delta, epsilon), width, color="tab:cyan", label="contact-skate |Δ| [cm/s]")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("preservation residual (log scale)")
    axes[1, 0].set_title("Body-motion / foot preservation")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].scatter(head_change, surface_p95, c=old_error, cmap="magma", s=65)
    for label, x_value, y_value in zip(labels, head_change, surface_p95):
        axes[1, 1].annotate(label, (x_value, y_value), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axes[1, 1].set_xlabel("mean Head rotation replacement [deg]")
    axes[1, 1].set_ylabel("affected-surface p95 displacement [mm]")
    axes[1, 1].set_title("Rotation correction vs visible skin response")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.22)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(x, labels, rotation=38, ha="right", fontsize=8)
    fig.suptitle(
        "camhead_v1 impact across diverse windows · standard SOMA visualization skin",
        fontsize=16,
    )
    fig.savefig(output / "soma_mesh_impact_summary.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["status"] == "ok"]
    if not successful:
        return {}
    metrics = [record["metrics"] for record in successful]
    fields = (
        "old_camera_rotation_error_mean_deg",
        "new_camera_rotation_error_mean_deg",
        "head_rotation_change_mean_deg",
        "surface_displacement_mean_mm",
        "surface_displacement_affected_p95_mm",
        "surface_displacement_max_mm",
        "protected_body_surface_displacement_max_mm",
        "decoded_joint_position_delta_max_mm",
        "decoded_nonhead_rotation_delta_max_deg",
        "foot_height_delta_max_mm",
        "contact_foot_skate_delta_cmps",
    )
    summary: dict[str, Any] = {}
    for field in fields:
        values = np.asarray([row[field] for row in metrics], dtype=np.float64)
        summary[field] = {
            "case_mean": float(values.mean()),
            "case_median": float(np.median(values)),
            "case_min": float(values.min()),
            "case_max": float(values.max()),
        }
    frame_weights = np.asarray([row["frames"] for row in metrics], dtype=np.float64)
    summary["frame_weighted"] = {
        field: float(
            np.average(
                [row[field] for row in metrics],
                weights=frame_weights,
            )
        )
        for field in (
            "old_camera_rotation_error_mean_deg",
            "new_camera_rotation_error_mean_deg",
            "head_rotation_change_mean_deg",
        )
    }
    summary["total_source_frames"] = int(frame_weights.sum())
    return summary


def _build_poster_contact_sheet(records: list[dict[str, Any]], output: Path) -> None:
    from PIL import Image, ImageDraw

    successful = [record for record in records if record["status"] == "ok"]
    columns = 3
    thumb_width, thumb_height = 640, 360
    label_height = 34
    rows = (len(successful) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(successful):
        image = Image.open(record["poster"]).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        cell = Image.new("RGB", (thumb_width, thumb_height), "white")
        cell.paste(image, ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2))
        sheet.paste(cell, (x, y))
        draw.text(
            (x + 8, y + thumb_height + 7),
            f"{record['category']} · {record['uuid']}@{record['start']}",
            fill=(25, 25, 25),
        )
    sheet.save(output / "peak_impact_poster_contact_sheet.jpg", quality=92)


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
    parser.add_argument("--source-gallery", type=Path, default=DEFAULT_SOURCE_GALLERY)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--camera-root", type=Path, default=DEFAULT_CAMERA_ROOT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--soma-assets-root", type=Path, default=DEFAULT_SOMA_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mesh-cluster-m", type=float, default=0.012)
    parser.add_argument("--skin-chunk-frames", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--case-limit", type=int, default=0, help="0 renders every gallery case")
    parser.add_argument("--category", action="append", default=[], help="render only named categories")
    parser.add_argument("--skip-videos", action="store_true", help="analyze and write posters only")
    args = parser.parse_args()
    if args.mesh_cluster_m < 0.0:
        parser.error("--mesh-cluster-m cannot be negative")
    if args.skin_chunk_frames <= 0 or args.frame_stride <= 0 or args.render_workers <= 0:
        parser.error("chunk size, frame stride, and worker count must be positive")
    if args.case_limit < 0:
        parser.error("--case-limit cannot be negative")

    source = json.loads(args.source_gallery.read_text())
    cases = list(source["cases"])
    if args.category:
        requested = set(args.category)
        cases = [case for case in cases if case["category"] in requested]
        missing = requested - {case["category"] for case in cases}
        if missing:
            parser.error(f"categories not present in source gallery: {sorted(missing)}")
    if args.case_limit:
        cases = cases[: args.case_limit]
    if not cases:
        parser.error("no cases selected")
    args.output.mkdir(parents=True, exist_ok=True)

    rotation, calibration = load_rotation_head_to_camera(args.calibration)
    skin = load_standard_soma_skin(args.soma_assets_root, args.mesh_cluster_m)
    skin_path = args.soma_assets_root / "somaskel77" / "skin_standard.npz"
    payloads = [
        (
            case,
            str(args.old_root),
            str(args.new_root),
            str(args.camera_root),
            str(args.soma_assets_root),
            str(args.output),
            rotation.tolist(),
            args.mesh_cluster_m,
            args.skin_chunk_frames,
            args.frame_stride,
            not args.skip_videos,
        )
        for case in cases
    ]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.render_workers) as executor:
        for index, record in enumerate(executor.map(_process_case, payloads), 1):
            records.append(record)
            print(
                f"[soma-mesh-viz] {index}/{len(payloads)} {record['category']} "
                f"status={record['status']} seconds={record['seconds']:.1f}",
                flush=True,
            )

    errors = [record for record in records if record["status"] != "ok"]
    if records and not errors:
        _render_metric_summary(records, args.output)
        _build_poster_contact_sheet(records, args.output)

    report = {
        "schema_version": 1,
        "kind": "nymeria_camhead_v1_standard_soma_mesh_gallery",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_gallery": str(args.source_gallery),
        "data_contract": {
            "original_uniego_root": str(args.old_root),
            "camera_aligned_uniego_root": str(args.new_root),
            "camera_root": str(args.camera_root),
            "calibration": str(args.calibration),
            "calibration_fit_split": calibration.get(
                "fit_split", calibration.get("split")
            ),
            "correction_description": (
                "camhead_v1 replaces each frame's Head world rotation using the measured "
                "upright RGB-camera rotation and one train-fit global Head-to-camera rotation; "
                "it is algorithmically aligned, not manually hand-edited frame by frame"
            ),
        },
        "mesh_contract": {
            "purpose": "standard SOMA visualization skin; not subject-identity SOMA-X geometry",
            "skin_asset": str(skin_path),
            "skin_asset_sha256": _sha256(skin_path),
            "source_vertices": skin.source_vertex_count,
            "source_faces": skin.source_face_count,
            "render_vertices": len(skin.bind_vertices),
            "render_faces": len(skin.faces),
            "head_closeup_faces": len(skin.head_face_indices),
            "vertex_cluster_m": skin.cluster_m,
            "skinning": (
                "SOMA30 global->local; name-based expansion to SOMA77 with relaxed hands; "
                "standard-skeleton FK; standard-skin linear blend skinning"
            ),
            "interpretation": (
                "surface displacement is a derived response of the common visualization skin; "
                "decoded joint-position preservation is reported separately"
            ),
        },
        "render_contract": {
            "cases": len(cases),
            "categories": [case["category"] for case in cases],
            "frame_stride": args.frame_stride,
            "video_resolution": [1920, 1080],
            "panels": [
                "original full-body standard SOMA mesh",
                "camhead_v1 full-body standard SOMA mesh",
                "original synchronized head/neck close-up",
                "camhead_v1 synchronized head/neck close-up",
                "measured-vs-Head-implied camera SO(3) error",
                "world pelvis/measured-camera trajectory",
                "derived surface displacement",
                "foot/floor preservation",
            ],
            "axis_legend": (
                "blue measured camera; orange original Head-implied camera; green camhead_v1 "
                "Head-implied camera; thick arrows +Z forward; thin arrows +Y up"
            ),
        },
        "cases": [
            {
                **case,
                "render": record,
            }
            for case, record in zip(cases, records)
        ],
        "aggregate_metrics": _aggregate_metrics(records),
        "artifacts": {
            "videos": str(args.output / "videos"),
            "peak_impact_posters": str(args.output / "posters"),
            "poster_contact_sheet": str(args.output / "peak_impact_poster_contact_sheet.jpg"),
            "metric_summary": str(args.output / "soma_mesh_impact_summary.png"),
        },
        "errors": errors,
    }
    _write_json(args.output / "gallery_manifest.json", report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(cases),
                "videos_ok": sum(
                    record["status"] == "ok" and record.get("video") is not None
                    for record in records
                ),
                "posters_ok": sum(record["status"] == "ok" for record in records),
                "errors": errors,
            },
            indent=2,
        ),
        flush=True,
    )
    if errors:
        raise SystemExit(f"{len(errors)} SOMA mesh render(s) failed")


if __name__ == "__main__":
    main()
