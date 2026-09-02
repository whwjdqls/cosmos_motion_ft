"""Native shifted x0 noising and sampling for Edge Phase-2 motion."""
from __future__ import annotations

import numpy as np
import torch


NATIVE_SHIFT = 3.0
NUM_TRAIN_TIMESTEPS = 1000


def shift_sigma(sigma: torch.Tensor, shift: float = NATIVE_SHIFT) -> torch.Tensor:
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


@torch.no_grad()
def sample_training_sigma(
    batch: int,
    device: torch.device | str,
    *,
    shift: float = NATIVE_SHIFT,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Match Cosmos action-style shifted logit-normal noise sampling."""

    raw = torch.sigmoid(torch.randn((batch,), dtype=torch.float32, device="cpu"))
    return shift_sigma(1.0 - raw.to(device=device, dtype=dtype), shift)


def add_noise_x0_masked(
    x0: torch.Tensor,
    condition_mask: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = sigma.to(device=x0.device, dtype=x0.dtype)
    noise = torch.randn_like(x0)
    noised_mask = ~condition_mask.bool()
    gate = noised_mask.to(x0.dtype)
    while gate.dim() < x0.dim():
        gate = gate.unsqueeze(-1)
    sigma_eff = sigma.view(-1, *([1] * (x0.dim() - 1))) * gate
    x_sigma = (1.0 - sigma_eff) * x0 + sigma_eff * noise
    return x_sigma, sigma, x0, noised_mask


def native_inference_schedule(
    steps: int,
    *,
    shift: float = NATIVE_SHIFT,
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    if steps <= 0 or num_train_timesteps <= 1:
        raise ValueError("steps must be positive and num_train_timesteps must exceed one")
    sigma_max = np.float32((num_train_timesteps - 1.0) / num_train_timesteps).item()
    base = np.linspace(sigma_max, 0.0, steps + 1).copy()[:-1]
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    sigmas = torch.tensor(np.append(shifted, 0.0), dtype=torch.float32, device=device)
    timesteps = (sigmas[:-1] * num_train_timesteps).to(torch.int64)
    return sigmas, timesteps


@torch.no_grad()
def sample_x0_unipc(
    predict,
    *,
    T: int,
    motion_dim: int,
    steps: int = 35,
    guidance: float = 2.0,
    predict_null=None,
    batch: int = 1,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    shift: float = NATIVE_SHIFT,
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
) -> torch.Tensor:
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import (
        FlowUniPCMultistepScheduler,
    )

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(steps, device=device, shift=float(shift))
    if initial_noise is None:
        x = torch.randn(batch, T, motion_dim, device=device, dtype=dtype, generator=generator)
    else:
        expected = (batch, T, motion_dim)
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"initial_noise must be {expected}, got {tuple(initial_noise.shape)}")
        x = initial_noise.to(device=device, dtype=dtype).clone()
    for index, timestep in enumerate(scheduler.timesteps):
        model_sigma = (timestep.float() / float(num_train_timesteps)).expand(batch)
        x0_hat = predict(x, model_sigma).float()
        if guidance != 1.0 and predict_null is not None:
            x0_null = predict_null(x, model_sigma).float()
            x0_hat = x0_null + guidance * (x0_hat - x0_null)
        sigma = scheduler.sigmas[index].to(device=device).float().clamp(min=1e-6)
        velocity = (x.float() - x0_hat) / sigma
        x = scheduler.step(
            model_output=velocity,
            timestep=timestep,
            sample=x.float(),
            return_dict=False,
            generator=generator,
        )[0].to(dtype)
    return x


__all__ = [
    "NATIVE_SHIFT",
    "NUM_TRAIN_TIMESTEPS",
    "add_noise_x0_masked",
    "native_inference_schedule",
    "sample_training_sigma",
    "sample_x0_unipc",
    "shift_sigma",
]

