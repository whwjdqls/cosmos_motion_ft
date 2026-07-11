"""Native-Cosmos sigma schedule with an x0-prediction motion sampler.

This module is the schedule-only BONES-SEED Phase-2 POC. It preserves the
existing MotionExpert clean-x0 objective while matching Cosmos-3's native
logit-normal training-time sampler, rational flow shift, 1000-step timestep
scale, and shifted inference sigma ladder.

The formulas are intentionally dependency-free so this remains runnable in the
``kimodo`` environment, which does not install ``cosmos_framework`` or
``diffusers``. They mirror NVIDIA Cosmos Framework commit
``3d9c0878fd0dde76eac98161aed0493d85a036fd``:

* ``TrainTimeSampler('logitnormal')`` in
  ``model/generator/diffusion/rectified_flow.py``;
* ``FlowUniPCMultistepScheduler.set_timesteps`` in
  ``model/generator/diffusion/samplers/fm_solvers_unipc.py``.

This is not a local UniPC reimplementation. ``sample_x0`` uses the proven x0
DDIM/straight-path update over the exact native shifted sigma ladder. An x0
prediction induces the rectified-flow velocity
``v = (x_sigma - x0_hat) / sigma``, and this update is the corresponding
first-order step. A later solver ablation can wrap the same model for UniPC
without retraining.
"""
from __future__ import annotations

import numpy as np
import torch


DEFAULT_SHIFT = 3.0
DEFAULT_NUM_TRAIN_TIMESTEPS = 1000


def shift_sigma(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply Cosmos's rational flow shift to values in ``[0, 1]``."""
    shift = float(shift)
    if shift <= 0.0:
        raise ValueError(f"shift must be positive, got {shift}")
    return shift * t / (1.0 + (shift - 1.0) * t)


@torch.no_grad()
def sample_train_sigma(
    batch_size: int,
    device: torch.device | str,
    *,
    shift: float = DEFAULT_SHIFT,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample native shifted logit-normal training sigmas.

    NVIDIA samples the standard normal on CPU and then transfers it to the
    target device. Keeping that order also makes deterministic checks against
    the framework straightforward.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    raw = torch.sigmoid(torch.randn((batch_size,), generator=generator))
    raw = raw.to(device=device, dtype=dtype)
    return shift_sigma(raw, shift)


def inference_schedule(
    num_steps: int,
    *,
    shift: float = DEFAULT_SHIFT,
    num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the native shifted sigma ladder and quantized model timesteps.

    Returns:
        sigmas: float32 ``[num_steps + 1]``. The first value is shifted
            ``(N-1)/N`` and the final value is exactly zero.
        timesteps: int64 ``[num_steps]`` equal to the framework's
            ``(sigmas[:-1] * N).to(int64)`` values. The motion model receives
            ``timesteps / N`` so its existing ``sigma * 1000`` embedding sees
            the same integer timestep as native Cosmos inference.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if num_train_timesteps <= 1:
        raise ValueError(
            f"num_train_timesteps must be greater than one, got {num_train_timesteps}"
        )
    shift = float(shift)
    if shift <= 0.0:
        raise ValueError(f"shift must be positive, got {shift}")

    # FlowUniPCMultistepScheduler initializes sigma_max=(N-1)/N and then uses
    # np.linspace(sigma_max, 0, num_steps+1)[:-1] before applying the shift.
    sigma_max = (num_train_timesteps - 1.0) / num_train_timesteps
    base = np.linspace(sigma_max, 0.0, num_steps + 1).copy()[:-1]
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    timesteps = (shifted * num_train_timesteps).astype(np.int64)
    sigmas = np.concatenate([shifted, np.array([0.0])]).astype(np.float32)
    return (
        torch.from_numpy(sigmas).to(device=device),
        torch.from_numpy(timesteps).to(device=device),
    )


@torch.no_grad()
def sample_x0(
    model,
    H_R: torch.Tensor,
    h_pad_mask: torch.Tensor | None,
    neutral_joints: torch.Tensor,
    T: int,
    motion_dim: int,
    steps: int = 50,
    guidance: float = 1.0,
    H_null=None,
    null_pad_mask=None,
    device="cuda",
    dtype=torch.float32,
    generator=None,
    sigma_eps: float = 1e-6,
    native_shift: float = DEFAULT_SHIFT,
    native_num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
):
    """Sample clean motion with x0 prediction on the native Cosmos ladder.

    The state update uses the scheduler's full-precision sigma while the model
    receives the native quantized timestep normalized back into ``[0,1]``.
    CFG is applied to x0, which is algebraically equivalent to applying CFG to
    ``v = (x - x0) / sigma`` because both branches share the same current state.
    """
    if H_R is None:
        raise ValueError("H_R/text embedding is required")
    batch = int(H_R.shape[0])
    x = torch.randn(
        batch,
        T,
        motion_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    sigmas, timesteps = inference_schedule(
        steps,
        shift=native_shift,
        num_train_timesteps=native_num_train_timesteps,
        device=device,
    )

    for i in range(steps):
        sigma = sigmas[i].float().clamp(min=sigma_eps)
        model_sigma = (
            timesteps[i].to(dtype=torch.float32) / float(native_num_train_timesteps)
        ).expand(batch)
        x0_hat = model(
            x,
            model_sigma,
            H_R,
            h_pad_mask,
            neutral_joints,
        ).float()
        if guidance != 1.0 and H_null is not None:
            x0_null = model(
                x,
                model_sigma,
                H_null,
                null_pad_mask,
                neutral_joints,
            ).float()
            x0_hat = x0_null + guidance * (x0_hat - x0_null)

        velocity = (x.float() - x0_hat) / sigma
        next_sigma = sigmas[i + 1].float()
        x = (x.float() + (next_sigma - sigma) * velocity).to(dtype)

    return x


__all__ = [
    "DEFAULT_SHIFT",
    "DEFAULT_NUM_TRAIN_TIMESTEPS",
    "shift_sigma",
    "sample_train_sigma",
    "inference_schedule",
    "sample_x0",
]
