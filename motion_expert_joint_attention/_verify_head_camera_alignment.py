#!/usr/bin/env python3
"""CPU contracts for relative head-camera geometry and task packing."""
from __future__ import annotations

import math

import torch

import task_plan as TP
from head_camera_alignment import (
    head_camera_alignment_losses,
    matrix_to_cont6d,
    motion_to_camera_action,
)
from nymeria_joint_dataset import collate_joint
from uniego_layout import FEAT_DIM, IDENTITY_DELTA9


def _rotation_y(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _se3(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    out = torch.eye(4, dtype=rotation.dtype)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def _inverse(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return _se3(rotation.T, -(rotation.T @ translation))


def _motion_from_head_transforms(head: torch.Tensor) -> torch.Tensor:
    n_frames = len(head)
    motion = torch.zeros(1, n_frames, FEAT_DIM)
    identity6 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    for joint in range(30):
        motion[0, :, joint * 9:joint * 9 + 6] = identity6
    delta = []
    for frame in range(n_frames):
        delta.append(head[frame] if frame == 0 else _inverse(head[frame - 1]) @ head[frame])
    delta = torch.stack(delta)
    motion[0, :, 270:276] = matrix_to_cont6d(delta[:, :3, :3])
    motion[0, :, 276:279] = delta[:, :3, 3]
    return motion


def verify_geometry() -> None:
    head = torch.stack([
        _se3(_rotation_y(0.00), torch.tensor([0.00, 0.0, 0.00])),
        _se3(_rotation_y(0.10), torch.tensor([0.03, 0.0, 0.01])),
        _se3(_rotation_y(0.18), torch.tensor([0.07, 0.0, 0.03])),
    ])
    x_rotation = _rotation_y(-0.35)
    lever = torch.tensor([0.02, 0.10, 0.08])
    x_transform = _se3(x_rotation, lever)
    camera = head @ x_transform
    target = []
    for frame in range(len(camera) - 1):
        relative = _inverse(camera[frame]) @ camera[frame + 1]
        target.append(torch.cat([relative[:3, 3], matrix_to_cont6d(relative[:3, :3])]))
    target = torch.stack(target).unsqueeze(0)

    motion = _motion_from_head_transforms(head).requires_grad_(True)
    predicted = motion_to_camera_action(motion, x_rotation, lever)
    assert torch.allclose(predicted, target, atol=2e-5), (predicted - target).abs().max()
    mask = torch.ones(1, len(target[0]), dtype=torch.bool)
    translation_loss, rotation_loss = head_camera_alignment_losses(predicted, target, mask)
    total = translation_loss + rotation_loss + predicted.square().mean()
    total.backward()
    assert torch.isfinite(total)
    assert motion.grad is not None and torch.isfinite(motion.grad).all()
    print(
        f"[contract] exact rigid geometry: max_error={(predicted-target).abs().max():.3e} "
        f"losses={translation_loss:.3e}/{rotation_loss:.3e}"
    )


def verify_task_resolution() -> None:
    resolved = TP.resolve_sample(
        "motimg2video",
        t_lat=25,
        n_camera=96,
        motion_valid_mask=[True] * 97,
        derived_camera_condition=True,
    )
    camera = resolved.modalities["camera"]
    assert camera.n_tokens == 96
    assert all(camera.condition_mask)
    assert not camera.supervised and camera.loss_weight == 0.0
    try:
        TP.resolve_sample(
            "video2motion",
            t_lat=25,
            n_camera=96,
            motion_valid_mask=[True] * 97,
            derived_camera_condition=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("derived GT camera must never be accepted as a V2M condition")
    print("[contract] M2V derived camera is clean-only; V2M rejects it")


def verify_collate() -> None:
    action = torch.randn(2, 9)
    sample = {
        "mode": "video2motion",
        "source": "nymeria",
        "caption": "",
        "domain_id": torch.tensor(2),
        "motion": torch.zeros(3, FEAT_DIM),
        "motion_pad_mask": torch.zeros(3, dtype=torch.bool),
        "neutral_joints": torch.zeros(30, 3),
        "camera_action": None,
        "camera_alignment_action": action,
        "video_latents": None,
        "video_frames": None,
        "reasoner_image": None,
    }
    batch = collate_joint([sample])
    assert batch["camera_action"] is None
    assert torch.equal(batch["camera_alignment_action"][0], action)
    assert not batch["camera_alignment_pad_mask"].any()
    print("[contract] auxiliary camera target remains separate from task camera input")


if __name__ == "__main__":
    assert torch.equal(torch.from_numpy(IDENTITY_DELTA9), torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ))
    verify_geometry()
    verify_task_resolution()
    verify_collate()
    print("PASS: head-camera alignment contracts")
