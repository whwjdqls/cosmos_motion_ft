#!/usr/bin/env python3
"""CPU contracts for relative head-camera geometry and task packing."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

import torch

import task_plan as TP
from estimate_head_camera_calibration import (
    ARIA_Z_UP_TO_KIMODO_Y_UP,
    calibration_sample_from_arrays,
    fit_head_camera_calibration,
    optimize_head_camera_transform_from_relative_actions,
)
from head_camera_alignment import (
    actor_id_from_uuid,
    head_camera_alignment_losses,
    load_oracle_actor_head_camera_calibrations,
    matrix_to_cont6d,
    motion_to_camera_action,
)
from nymeria_joint_dataset import collate_joint
from uniego_layout import FEAT_DIM, IDENTITY_DELTA9


def _rotation_y(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_x(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_z(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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


def verify_oracle_actor_loader() -> None:
    payload = {
        "kind": "oracle_test_actor_head_camera_calibration",
        "split": "test",
        "leakage_contract": {
            "uses_test_gt_motion": True,
            "uses_test_gt_camera": True,
            "diagnostic_only": True,
        },
        "actors": {
            "S07": {
                "rotation_head_to_upright_camera": torch.eye(3).tolist(),
                "camera_origin_in_head_m": [0.01, 0.02, 0.03],
            }
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "oracle.json"
        path.write_text(json.dumps(payload))
        calibrations, metadata = load_oracle_actor_head_camera_calibrations(str(path))
        rotation, lever = calibrations["S07"]
        assert torch.equal(rotation, torch.eye(3))
        assert torch.allclose(lever, torch.tensor([0.01, 0.02, 0.03]))
        assert metadata["split"] == "test"

        payload["leakage_contract"]["diagnostic_only"] = False
        path.write_text(json.dumps(payload))
        try:
            load_oracle_actor_head_camera_calibrations(str(path))
        except ValueError:
            pass
        else:
            raise AssertionError("oracle loader accepted missing diagnostic-only guard")
    assert actor_id_from_uuid("S07/example") == "S07"
    print("[contract] oracle test-actor calibration is explicit and leakage-guarded")


def verify_calibration_fit() -> None:
    x_rotation = (_rotation_y(-0.35) @ _rotation_x(0.12)).double()
    lever = torch.tensor([0.02, 0.10, 0.08], dtype=torch.float64)
    head_rotation = torch.stack([
        _rotation_x(0.08),
        _rotation_y(-0.11),
        _rotation_z(0.07),
        _rotation_x(-0.05) @ _rotation_y(0.09),
        _rotation_z(-0.06) @ _rotation_x(0.04),
    ]).double()
    head_translation = torch.tensor([
        [0.010, 0.002, -0.004],
        [0.003, -0.001, 0.008],
        [-0.006, 0.004, 0.002],
        [0.004, 0.005, 0.006],
        [0.002, -0.003, -0.005],
    ], dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)
    camera_rotation = x_rotation.T @ head_rotation @ x_rotation
    camera_translation = (
        x_rotation.T
        @ (head_translation + ((head_rotation - eye) @ lever[:, None]).squeeze(-1))[:, :, None]
    ).squeeze(-1)
    sample = {
        "frame_rotations": x_rotation.repeat(20, 1, 1),
        "head_relative_rotations": head_rotation,
        "head_relative_translations": head_translation,
        "camera_relative_rotations": camera_rotation,
        "camera_relative_translations": camera_translation,
    }
    fitted = fit_head_camera_calibration([sample])
    assert torch.allclose(fitted["rotation"], x_rotation, atol=2e-6)
    assert torch.allclose(fitted["lever"], lever, atol=2e-6)
    initial = x_rotation @ _rotation_z(0.04).double()
    optimized_rotation, optimized_lever, optimizer = (
        optimize_head_camera_transform_from_relative_actions(
            [sample], initial, lever + torch.tensor([0.01, -0.01, 0.02]), max_samples=100
        )
    )
    assert optimizer["final"]["loss"] < optimizer["initial"]["loss"]
    assert torch.allclose(optimized_rotation, x_rotation, atol=2e-5)
    assert torch.allclose(optimized_lever, lever, atol=2e-5)
    print("[contract] robust calibration fitter recovers exact synthetic rigid transform")


def verify_nonzero_calibration_window() -> None:
    head = torch.stack([
        _se3(
            _rotation_y(0.04 * frame) @ _rotation_x(-0.015 * frame),
            torch.tensor([0.025 * frame, 0.003 * frame, -0.012 * frame]),
        )
        for frame in range(9)
    ])
    x_rotation = (_rotation_y(-0.31) @ _rotation_x(0.09)).double()
    lever = torch.tensor([0.02, 0.10, 0.08], dtype=torch.float64)
    x_transform = _se3(x_rotation, lever)
    camera_kimodo = head.double() @ x_transform
    basis = ARIA_Z_UP_TO_KIMODO_Y_UP
    camera_aria = camera_kimodo.clone()
    camera_aria[:, :3, :3] = basis.T @ camera_kimodo[:, :3, :3]
    camera_aria[:, :3, 3] = (basis.T @ camera_kimodo[:, :3, 3, None]).squeeze(-1)
    camera_actions = []
    for frame in range(len(camera_aria) - 1):
        relative = _inverse(camera_aria[frame]) @ camera_aria[frame + 1]
        camera_actions.append(
            torch.cat([relative[:3, 3], matrix_to_cont6d(relative[:3, :3])])
        )
    motion = _motion_from_head_transforms(head)[0].double().numpy()
    sample = calibration_sample_from_arrays(
        motion,
        camera_aria[:, :3, :3].numpy(),
        torch.stack(camera_actions).numpy(),
        start=3,
        window_frames=5,
        orientation_stride=1,
    )
    expected = x_rotation.repeat(5, 1, 1)
    assert torch.allclose(sample["frame_rotations"], expected, atol=3e-5), (
        sample["frame_rotations"] - expected
    ).abs().max()
    print("[contract] nonzero calibration windows preserve sequence-absolute head orientation")


if __name__ == "__main__":
    assert torch.equal(torch.from_numpy(IDENTITY_DELTA9), torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ))
    verify_geometry()
    verify_task_resolution()
    verify_collate()
    verify_oracle_actor_loader()
    verify_calibration_fit()
    verify_nonzero_calibration_window()
    print("PASS: head-camera alignment contracts")
