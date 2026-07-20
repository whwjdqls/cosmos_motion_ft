"""Native-Cosmos schedule and samplers for the x0-prediction motion model.

This module is the schedule-only BONES-SEED Phase-2 POC. It preserves the
existing MotionExpert clean-x0 objective while matching Cosmos-3's native
logit-normal training-time sampler, rational flow shift, 1000-step timestep
scale, and shifted inference sigma ladder.

The training sampler and diagnostic Euler/Heun samplers are dependency-free.
They mirror NVIDIA Cosmos Framework commit
``3d9c0878fd0dde76eac98161aed0493d85a036fd``:

* ``TrainTimeSampler('logitnormal')`` in
  ``model/generator/diffusion/rectified_flow.py``;
* the one-shift sigma formula used by
  ``FlowUniPCMultistepScheduler.set_timesteps`` in
  ``model/generator/diffusion/samplers/fm_solvers_unipc.py``.

``sample_x0_unipc`` imports and calls NVIDIA's real
``FlowUniPCMultistepScheduler``. The only model adapter is the exact conversion
from this model's guided clean prediction to the flow velocity expected by the
official scheduler: ``v = (x_sigma - x0_hat) / sigma``. ``sample_x0`` and
``sample_x0_heun`` remain historical one-shift diagnostic alternatives.
"""
from __future__ import annotations

import numpy as np
import torch


DEFAULT_SHIFT = 3.0
DEFAULT_NUM_TRAIN_TIMESTEPS = 1000


def create_unipc_scheduler(
    num_steps: int,
    *,
    shift: float = DEFAULT_SHIFT,
    num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
    device: torch.device | str = "cuda",
):
    """Construct the unmodified native Cosmos-3 UniPC scheduler.

    These are the exact constructor and ``set_timesteps`` calls made by
    ``cosmos_framework.model.generator.diffusion.samplers.unipc.UniPCSampler``.
    Solver order, solver type, prediction type, x0 conversion, corrector, and
    final-step behavior therefore remain the official scheduler defaults.
    """
    try:
        from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import (
            FlowUniPCMultistepScheduler,
        )
    except ImportError as exc:
        raise RuntimeError(
            "native UniPC requires NVIDIA cosmos-framework and diffusers in the "
            "active environment"
        ) from exc

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=int(num_train_timesteps),
        shift=float(shift),
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(
        int(num_steps),
        device=device,
        shift=float(shift),
    )
    return scheduler


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
    """Build the original POC's one-shift diagnostic sigma ladder.

    This remains the historical Euler/Heun comparison schedule. The real
    Cosmos ``UniPCSampler`` first shifts the scheduler's training sigma range
    in its constructor and then shifts the inference ladder in
    ``set_timesteps``. Use ``create_unipc_scheduler`` or ``sample_x0_unipc``
    when exact native Cosmos inference behavior is required.

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
    # The framework materializes its base training ladder as torch.float32
    # before reading ``sigma_max.item()`` for inference. Preserve that endpoint
    # rounding so this helper is bit-identical, not merely numerically close.
    sigma_max = np.float32((num_train_timesteps - 1.0) / num_train_timesteps).item()
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
    """Sample clean motion with Euler on the one-shift diagnostic ladder.

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


@torch.no_grad()
def sample_x0_unipc(
    model,
    H_R: torch.Tensor,
    h_pad_mask: torch.Tensor | None,
    neutral_joints: torch.Tensor,
    T: int,
    motion_dim: int,
    steps: int = 35,
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
    """Sample through NVIDIA Cosmos-3's official UniPC implementation.

    The motion model predicts clean ``x0`` while native Cosmos UniPC accepts a
    rectified-flow velocity. Applying classifier-free guidance to ``x0`` and
    returning ``(x - x0_cfg) / sigma`` makes the scheduler's own
    ``convert_model_output`` recover exactly ``x0_cfg``. All integration logic
    is performed by ``FlowUniPCMultistepScheduler.step``.
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
    scheduler = create_unipc_scheduler(
        steps,
        shift=native_shift,
        num_train_timesteps=native_num_train_timesteps,
        device=device,
    )

    for i, timestep in enumerate(scheduler.timesteps):
        model_sigma = (
            timestep.to(device=device, dtype=torch.float32)
            / float(native_num_train_timesteps)
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

        sigma = scheduler.sigmas[i].to(device=x.device, dtype=torch.float32)
        velocity = (x.float() - x0_hat) / sigma.clamp(min=sigma_eps)
        x = scheduler.step(
            model_output=velocity,
            timestep=timestep,
            sample=x.float(),
            return_dict=False,
            generator=generator,
        )[0].to(dtype)

    return x


@torch.no_grad()
def sample_x0_heun(
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
    """Sample with Heun RK2 over the one-shift diagnostic sigma ladder.

    Each non-final interval evaluates the x0-derived flow velocity at its
    beginning, predicts the state at the next sigma, evaluates velocity there,
    and applies the trapezoidal correction. The final interval uses Euler
    because its endpoint is sigma zero, where ``(x - x0) / sigma`` is
    undefined. Consequently ``steps`` intervals use ``2 * steps - 1`` model
    evaluations (before counting the conditional/unconditional CFG branches).
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

    def predict_x0(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        model_sigma = (
            timestep.to(dtype=torch.float32) / float(native_num_train_timesteps)
        ).expand(batch)
        x0_hat = model(
            state,
            model_sigma,
            H_R,
            h_pad_mask,
            neutral_joints,
        ).float()
        if guidance != 1.0 and H_null is not None:
            x0_null = model(
                state,
                model_sigma,
                H_null,
                null_pad_mask,
                neutral_joints,
            ).float()
            x0_hat = x0_null + guidance * (x0_hat - x0_null)
        return x0_hat

    for i in range(steps):
        sigma = sigmas[i].float().clamp(min=sigma_eps)
        next_sigma = sigmas[i + 1].float()
        delta = next_sigma - sigma

        x0_hat = predict_x0(x, timesteps[i])
        velocity = (x.float() - x0_hat) / sigma
        euler_state = x.float() + delta * velocity

        if i == steps - 1:
            x = euler_state.to(dtype)
            continue

        next_sigma_safe = next_sigma.clamp(min=sigma_eps)
        x0_next = predict_x0(euler_state.to(dtype), timesteps[i + 1])
        velocity_next = (euler_state - x0_next) / next_sigma_safe
        x = (x.float() + 0.5 * delta * (velocity + velocity_next)).to(dtype)

    return x


__all__ = [
    "DEFAULT_SHIFT",
    "DEFAULT_NUM_TRAIN_TIMESTEPS",
    "shift_sigma",
    "sample_train_sigma",
    "inference_schedule",
    "create_unipc_scheduler",
    "sample_x0",
    "sample_x0_unipc",
    "sample_x0_heun",
]
