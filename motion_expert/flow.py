"""v2: rectified-flow noising + x0-prediction sampler (cosmos env).

Training noising stays rectified-flow: x_σ = σ·ε + (1−σ)·x0.
The model now predicts **x0** (clean motion) — errors don't compound through the residual
canon-frame as badly as velocity. Sampler is DDIM-style in σ (1→0): each step predicts x0_hat,
derives ε_hat = (x_σ − (1−σ)x0_hat)/σ, and re-noises to the next σ. CFG is applied on x0_hat.
"""
from __future__ import annotations

import torch


def sample_sigma_logitnormal(batch: int, device, m: float = 0.0, s: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(m + s * torch.randn(batch, device=device))


def add_noise(x0: torch.Tensor, sigma: torch.Tensor):
    """x0 [B,T,D], sigma [B] → (x_σ, eps). (x0 itself is the prediction target.)"""
    eps = torch.randn_like(x0)
    s = sigma.view(-1, *([1] * (x0.dim() - 1)))
    x_sigma = s * eps + (1.0 - s) * x0
    return x_sigma, eps


@torch.no_grad()
def sample_x0(
    model,
    H_R: torch.Tensor, h_pad_mask: torch.Tensor, neutral_joints: torch.Tensor,
    T: int, motion_dim: int,
    steps: int = 50, guidance: float = 1.0,
    H_null=None, null_pad_mask=None,
    device="cuda", dtype=torch.float32, generator=None, sigma_eps: float = 1e-3,
):
    """DDIM-style sampling from an x0-prediction model. Returns clean x0 [B,T,motion_dim]."""
    B = H_R.shape[0]
    x = torch.randn(B, T, motion_dim, device=device, dtype=dtype, generator=generator)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        s = sigmas[i].clamp(min=sigma_eps).expand(B)
        x0_hat = model(x, s, H_R, h_pad_mask, neutral_joints).float()
        if guidance != 1.0 and H_null is not None:
            x0_u = model(x, s, H_null, null_pad_mask, neutral_joints).float()
            x0_hat = x0_u + guidance * (x0_hat - x0_u)
        si = sigmas[i].clamp(min=sigma_eps)
        eps_hat = (x - (1.0 - si) * x0_hat) / si
        snext = sigmas[i + 1]
        x = (1.0 - snext) * x0_hat + snext * eps_hat
    return x
