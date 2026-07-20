"""Relative head-motion to upright-RGB camera-action geometry.

The proportional UniEgo stream and the Aria camera stream do not have a reliable
shared absolute origin for every sequence. Their frame-to-frame motion is synchronized,
however, and a train-split calibration supplies the approximately rigid transform from
the SOMA ``Head`` joint frame to the upright RGB-camera frame.

For ``T_world_camera = T_world_head @ T_head_camera`` and a head-frame relative
transform ``(R_h, t_h)``, the corresponding camera action is

    R_c = R_x.T @ R_h @ R_x
    t_c = R_x.T @ (t_h + (R_h - I) @ r_x)

where ``R_x`` and ``r_x`` are the rotation and camera-origin lever arm of
``T_head_camera``. This is invariant to a common world transform and therefore remains
valid after the motion loader grounds and frame-0-canonicalizes a training window.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from decode_uniego_torch import cont6d_to_matrix, decode_transforms


HEAD_JOINT_IDX = 6
DEFAULT_CALIBRATION = str(Path(__file__).with_name("head_camera_calibration_train.json"))
HEAD_CAMERA_GLOBAL_METRIC_KEYS = (
    "translation_m",
    "rotation_deg",
    "gt_calibration_translation_m",
    "gt_calibration_rotation_deg",
)
HEAD_CAMERA_ORACLE_ACTOR_METRIC_KEYS = (
    "oracle_actor_translation_m",
    "oracle_actor_rotation_deg",
    "gt_oracle_actor_translation_m",
    "gt_oracle_actor_rotation_deg",
)


def _validate_calibration_tensors(
    rotation_value,
    lever_value,
    *,
    source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = torch.tensor(rotation_value, dtype=torch.float32)
    lever = torch.tensor(lever_value, dtype=torch.float32)
    if rotation.shape != (3, 3) or lever.shape != (3,):
        raise ValueError(
            f"bad head-camera calibration shapes in {source}: R={tuple(rotation.shape)} "
            f"lever={tuple(lever.shape)}"
        )
    if not torch.isfinite(rotation).all() or not torch.isfinite(lever).all():
        raise ValueError(f"non-finite head-camera calibration in {source}")
    eye = torch.eye(3, dtype=rotation.dtype)
    ortho_error = float((rotation.T @ rotation - eye).abs().max())
    determinant = float(torch.det(rotation))
    if ortho_error > 1e-4 or abs(determinant - 1.0) > 1e-4:
        raise ValueError(
            f"head-camera calibration is not SO(3): ortho_error={ortho_error:.3e} "
            f"det={determinant:.6f} ({source})"
        )
    return rotation, lever


def load_head_camera_calibration(path: str = DEFAULT_CALIBRATION) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load train-split ``(R_head_camera, camera_origin_in_head, metadata)``."""
    with open(path) as f:
        payload = json.load(f)
    rotation, lever = _validate_calibration_tensors(
        payload["rotation_head_to_upright_camera"],
        payload["camera_origin_in_head_m"],
        source=path,
    )
    return rotation, lever, payload


def actor_id_from_uuid(uuid: str) -> str:
    """Return the Nymeria actor prefix (for example ``S07``) from a sequence UUID."""
    actor = str(uuid).split("/", 1)[0]
    if len(actor) != 3 or not actor.startswith("S") or not actor[1:].isdigit():
        raise ValueError(f"cannot derive Nymeria actor id from UUID {uuid!r}")
    return actor


def load_oracle_actor_head_camera_calibrations(
    path: str,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Load test-derived per-actor calibrations used only for oracle diagnostics.

    The artifact deliberately contains test-label information. Requiring its explicit kind and
    leakage flags prevents it from being mistaken for the train calibration used by Phase 3.
    """
    with open(path) as f:
        payload = json.load(f)
    if payload.get("kind") != "oracle_test_actor_head_camera_calibration":
        raise ValueError(f"{path}: not an oracle test-actor calibration artifact")
    if payload.get("split") != "test":
        raise ValueError(f"{path}: oracle actor calibration must declare split='test'")
    leakage = payload.get("leakage_contract", {})
    if not (
        leakage.get("uses_test_gt_motion") is True
        and leakage.get("uses_test_gt_camera") is True
        and leakage.get("diagnostic_only") is True
    ):
        raise ValueError(f"{path}: incomplete oracle leakage contract")
    actors = payload.get("actors")
    if not isinstance(actors, dict) or not actors:
        raise ValueError(f"{path}: no per-actor calibrations")
    calibrations = {}
    for actor, entry in sorted(actors.items()):
        if actor_id_from_uuid(actor) != actor:
            raise ValueError(f"{path}: invalid actor key {actor!r}")
        calibrations[actor] = _validate_calibration_tensors(
            entry["rotation_head_to_upright_camera"],
            entry["camera_origin_in_head_m"],
            source=f"{path}:{actor}",
        )
    return calibrations, payload


def matrix_to_cont6d(rotation: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[...,3,3]`` -> Cosmos column-convention 6D."""
    return torch.cat([rotation[..., :, 0], rotation[..., :, 1]], dim=-1)


def motion_to_camera_action(
    motion_unnormalized: torch.Tensor,
    rotation_head_to_camera: torch.Tensor,
    camera_origin_in_head: torch.Tensor,
    *,
    head_idx: int = HEAD_JOINT_IDX,
) -> torch.Tensor:
    """Convert UniEgo motion ``[B,T,283]`` to derived camera actions ``[B,T-1,9]``.

    Only relative transforms are used. Gradients flow through the complete UniEgo
    decoder, so the result can supervise V2M x0 predictions.
    """
    if motion_unnormalized.dim() != 3:
        raise ValueError(
            f"motion must be [B,T,D], got {tuple(motion_unnormalized.shape)}"
        )
    if motion_unnormalized.shape[1] < 2:
        return motion_unnormalized.new_zeros(
            motion_unnormalized.shape[0], 0, 9
        )

    head = decode_transforms(motion_unnormalized)[:, :, head_idx]
    rotation = head[..., :3, :3]
    position = head[..., :3, 3]
    rotation_t = rotation[:, :-1]
    relative_rotation = rotation_t.transpose(-1, -2) @ rotation[:, 1:]
    relative_translation = (
        rotation_t.transpose(-1, -2)
        @ (position[:, 1:] - position[:, :-1]).unsqueeze(-1)
    ).squeeze(-1)

    x_rotation = rotation_head_to_camera.to(
        device=motion_unnormalized.device, dtype=motion_unnormalized.dtype
    )
    lever = camera_origin_in_head.to(
        device=motion_unnormalized.device, dtype=motion_unnormalized.dtype
    )
    eye = torch.eye(3, device=motion_unnormalized.device, dtype=motion_unnormalized.dtype)
    camera_rotation = x_rotation.T @ relative_rotation @ x_rotation
    lever_velocity = ((relative_rotation - eye) @ lever.view(3, 1)).squeeze(-1)
    camera_translation = (
        x_rotation.T @ (relative_translation + lever_velocity).unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat([camera_translation, matrix_to_cont6d(camera_rotation)], dim=-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.dim() < values.dim():
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    denom = expanded.sum().clamp_min(1)
    return (values * expanded.to(values.dtype)).sum() / denom


def head_camera_alignment_losses(
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    transition_mask: torch.Tensor,
    *,
    translation_scale_m: float = 0.02,
    rotation_scale_deg: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust dimensionless translation and rotation losses over valid transitions."""
    if predicted_action.shape != target_action.shape:
        raise ValueError(
            f"head-camera action shape mismatch: pred={tuple(predicted_action.shape)} "
            f"target={tuple(target_action.shape)}"
        )
    if transition_mask.shape != predicted_action.shape[:2]:
        raise ValueError(
            f"head-camera mask shape mismatch: mask={tuple(transition_mask.shape)} "
            f"action={tuple(predicted_action.shape)}"
        )
    if translation_scale_m <= 0.0 or rotation_scale_deg <= 0.0:
        raise ValueError("head-camera loss scales must be positive")

    translation_error = (
        predicted_action[..., :3] - target_action[..., :3]
    ) / float(translation_scale_m)
    translation_loss = _masked_mean(
        F.smooth_l1_loss(
            translation_error,
            torch.zeros_like(translation_error),
            beta=1.0,
            reduction="none",
        ),
        transition_mask,
    )

    predicted_rotation = cont6d_to_matrix(predicted_action[..., 3:9])
    target_rotation = cont6d_to_matrix(target_action[..., 3:9])
    # The Frobenius chord between two rotation matrices is
    # ``2*sqrt(2)*sin(theta/2)``. It is linear in small angular error, bounded for bad
    # predictions, and avoids the singular derivative of acos at zero. Normalize so an
    # error of ``rotation_scale_deg`` is one unit before applying the robust penalty.
    chord = torch.linalg.matrix_norm(
        predicted_rotation - target_rotation,
        ord="fro",
        dim=(-2, -1),
    )
    scale_chord = 2.0 * math.sqrt(2.0) * math.sin(
        0.5 * math.radians(float(rotation_scale_deg))
    )
    rotation_error = chord / scale_chord
    rotation_loss = _masked_mean(
        F.smooth_l1_loss(
            rotation_error,
            torch.zeros_like(rotation_error),
            beta=1.0,
            reduction="none",
        ),
        transition_mask,
    )
    return translation_loss, rotation_loss


def head_camera_errors(
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    transition_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean translation error in metres and rotation error in degrees."""
    translation = (predicted_action[..., :3] - target_action[..., :3]).norm(dim=-1)
    predicted_rotation = cont6d_to_matrix(predicted_action[..., 3:9])
    target_rotation = cont6d_to_matrix(target_action[..., 3:9])
    relative = predicted_rotation.transpose(-1, -2) @ target_rotation
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation_deg = torch.acos(cosine) * (180.0 / math.pi)
    return _masked_mean(translation, transition_mask), _masked_mean(rotation_deg, transition_mask)


__all__ = [
    "DEFAULT_CALIBRATION",
    "HEAD_CAMERA_GLOBAL_METRIC_KEYS",
    "HEAD_CAMERA_ORACLE_ACTOR_METRIC_KEYS",
    "HEAD_JOINT_IDX",
    "actor_id_from_uuid",
    "head_camera_alignment_losses",
    "head_camera_errors",
    "load_head_camera_calibration",
    "load_oracle_actor_head_camera_calibrations",
    "matrix_to_cont6d",
    "motion_to_camera_action",
]
