"""Focused contracts for the Phase-3 generator-motion multitask extension."""
from __future__ import annotations

from types import SimpleNamespace

import torch

import flow
import sample as sample_module
import task_plan as TP
from modality_bridge import BridgeMeta, LocalModalityBridge
from sample import _make_joint_gen_motion_closure, _validate_joint_native_contract


def _meta(*, gen_clean, starts, ends, motion_clean, mode="test") -> BridgeMeta:
    ng = len(gen_clean)
    nm = len(motion_clean)
    return BridgeMeta(
        mode=mode,
        gen_frame=torch.full((ng,), -1, dtype=torch.long),
        gen_clean=torch.tensor(gen_clean, dtype=torch.bool),
        motion_frame=torch.tensor([-1] + list(range(nm - 1)), dtype=torch.long),
        motion_clean=torch.tensor(motion_clean, dtype=torch.bool),
        gen_source_start=torch.tensor(starts, dtype=torch.long),
        gen_source_end=torch.tensor(ends, dtype=torch.long),
    )


def verify_task_plans() -> None:
    inv = TP.resolve_sample(
        "video2camera_motion",
        t_lat=3,
        n_camera=8,
        motion_valid_mask=[True] * 9,
    )
    assert inv.text_is_empty
    assert all(inv.modalities["video"].condition_mask)
    assert not any(inv.modalities["camera"].condition_mask)
    assert inv.modalities["motion"].condition_mask == [True] + [False] * 9
    assert inv.modalities["camera"].loss_weight == 0.5 * TP.ACTION_LOSS_WEIGHT
    assert inv.modalities["motion"].loss_weight == 0.5 * TP.W_MOTION

    fwd = TP.resolve_sample(
        "camimg2video_motion",
        t_lat=3,
        n_camera=8,
        motion_valid_mask=[True] * 9,
    )
    assert fwd.text_is_empty
    assert fwd.modalities["video"].condition_mask == [True, False, False]
    assert all(fwd.modalities["camera"].condition_mask)
    assert fwd.modalities["video"].loss_weight == 0.5 * TP.W_VISION
    assert fwd.modalities["motion"].loss_weight == 0.5 * TP.W_MOTION


def verify_role_driven_masks() -> None:
    bridge = LocalModalityBridge(hidden=8, num_heads=2, head_dim=4)

    # G rows: latent0->[0], latent1->[1..4], camera0->[0..1].
    starts = [0, 1, 0]
    ends = [0, 4, 1]

    # V2M corner: only noisy motion queries cross into clean video.
    v2m = _meta(
        mode="video2motion",
        gen_clean=[True, True, True],
        starts=starts,
        ends=ends,
        motion_clean=[True, False, False, False, False, False],
    )
    mask = bridge._attention_mask(v2m)
    ng = 3
    assert mask[ng + 2, 1]  # motion frame1 reads latent1
    assert not mask[1, ng + 2]  # clean gen row cannot query motion

    # M2V corner: only noisy generator rows query clean motion (plus shape).
    m2v = _meta(
        mode="motimg2video",
        gen_clean=[True, False, True],
        starts=starts,
        ends=ends,
        motion_clean=[True] * 6,
    )
    mask = bridge._attention_mask(m2v)
    assert mask[1, ng]  # noisy latent1 reads shape
    assert mask[1, ng + 2] and mask[1, ng + 5]  # local frames1..4
    assert not mask[ng + 2, 1]  # clean motion row cannot query gen target

    # Joint inverse: noisy camera and noisy motion communicate both ways, while the clean
    # video query row remains cross-modal read-only.
    joint_inv = _meta(
        mode="video2camera_motion",
        gen_clean=[True, True, False],
        starts=starts,
        ends=ends,
        motion_clean=[True, False, False, False, False, False],
    )
    mask = bridge._attention_mask(joint_inv)
    camera = 2
    motion0, motion1, motion2 = ng + 1, ng + 2, ng + 3
    assert mask[camera, motion0] and mask[camera, motion1]
    assert mask[motion0, camera] and mask[motion1, camera]
    assert not mask[camera, motion2] and not mask[motion2, camera]
    assert not mask[1, motion1]  # clean latent query cannot read motion target

    # Joint forward: noisy video and noisy motion are bidirectional; clean camera remains a key.
    joint_fwd = _meta(
        mode="camimg2video_motion",
        gen_clean=[True, False, True],
        starts=starts,
        ends=ends,
        motion_clean=[True, False, False, False, False, False],
    )
    mask = bridge._attention_mask(joint_fwd)
    assert mask[1, motion1] and mask[motion1, 1]
    assert mask[motion0, camera] and mask[motion1, camera]
    assert not mask[camera, motion0]


def verify_independent_native_sigmas() -> None:
    gm = torch.Generator().manual_seed(11)
    gg = torch.Generator().manual_seed(29)
    motion = flow.sample_sigma_native_logitnormal(128, "cpu", shift=3.0, generator=gm)
    gen = flow.sample_sigma_native_waver(128, "cpu", shift=3.0, generator=gg)
    assert bool(((motion >= 0) & (motion <= 1)).all())
    assert bool(((gen >= 0) & (gen <= 1)).all())
    assert not torch.equal(motion, gen)
    assert float((motion - gen).abs().mean()) > 0.05


class _FakeJointModel:
    motion_dim = 2

    def forward(self, **kwargs):
        self.kwargs = kwargs
        motion = kwargs["x_t"]
        camera = kwargs["camera_action"][0]
        return {
            "motion_pred": torch.zeros_like(motion),
            "camera_pred": [torch.ones_like(camera)],
            "video_pred": [None],
        }


def verify_coupled_closure() -> None:
    model = _FakeJointModel()
    predict, meta = _make_joint_gen_motion_closure(
        model,
        mode="video2camera_motion",
        input_ids=torch.zeros(1, 1, dtype=torch.long),
        gen_target="camera",
        base_video=torch.zeros(1, 2, 1, 1),
        base_camera=torch.zeros(1, TP.CAMERA_RAW_DIM),
        gen_noised_frames=torch.tensor([0]),
        neutral_joints=torch.zeros(1, 30, 3),
        motion_T=2,
        device="cpu",
    )
    n = meta["gen_scalar_count"] + meta["motion_scalar_count"]
    state = torch.arange(n, dtype=torch.float32).view(1, n, 1)
    velocity = predict(
        state, torch.tensor([0.5]), scheduler_sigma_b=torch.tensor([0.25])
    ).reshape(1, -1)
    torch.testing.assert_close(
        velocity[:, :TP.CAMERA_RAW_DIM], torch.ones(1, TP.CAMERA_RAW_DIM)
    )
    torch.testing.assert_close(
        velocity[:, TP.CAMERA_RAW_DIM:], 4.0 * state.reshape(1, -1)[:, TP.CAMERA_RAW_DIM:]
    )
    torch.testing.assert_close(model.kwargs["motion_t_or_sigma"], torch.tensor([0.5]))
    torch.testing.assert_close(model.kwargs["gen_t_or_sigma"], torch.tensor([0.5]))


def verify_joint_sampler_guard() -> None:
    valid = SimpleNamespace(
        objective="x0",
        motion_schedule="native",
        gen_schedule="native",
        motion_native_solver="unipc",
        gen_native_solver="unipc",
        motion_num_train_timesteps=1000,
        gen_num_train_timesteps=1000,
        motion_shift=3.0,
        gen_shift=3.0,
    )
    _validate_joint_native_contract(valid)
    valid.gen_shift = 2.0
    try:
        _validate_joint_native_contract(valid)
    except ValueError as exc:
        assert "shift mismatch" in str(exc)
    else:
        raise AssertionError("joint sampler accepted incompatible ladders")


def verify_joint_camera_state_precision() -> None:
    captured = {}

    def fake_sampler(model, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    original = sample_module._sample_joint_gen_motion
    sample_module._sample_joint_gen_motion = fake_sampler
    try:
        result = sample_module.sample_video2camera_motion(
            object(),
            video_latents=torch.zeros(2, 3, 1, 1, dtype=torch.bfloat16),
            neutral_joints=torch.zeros(1, 30, 3),
            T=9,
        )
    finally:
        sample_module._sample_joint_gen_motion = original
    assert result == {"ok": True}
    assert captured["clean_camera_action"].dtype == torch.float32


def main() -> None:
    verify_task_plans()
    verify_role_driven_masks()
    verify_independent_native_sigmas()
    verify_coupled_closure()
    verify_joint_sampler_guard()
    verify_joint_camera_state_precision()
    print("phase3 multitask contracts: PASS")


if __name__ == "__main__":
    main()
