"""Masked reconstruction and contact-aware losses for BONES motion training."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from uniego_layout import FOOT_JOINT_IDX, FOOT_SLICE, FOOT_Y_IDX


def masked_mse(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Elementwise MSE averaged over valid frames and all trailing dimensions."""
    while valid.dim() < a.dim():
        valid = valid.unsqueeze(-1)
    squared_error = (a - b).square() * valid
    return squared_error.sum() / valid.expand_as(a).sum().clamp(min=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp(min=1)


def contact_aware_losses(
    x0_hat: torch.Tensor,
    x0: torch.Tensor,
    joints_hat: torch.Tensor,
    valid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    fps: float = 20.0,
    contact_logit_scale: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contact BCE, planted-foot horizontal velocity, and height losses.

    Contact values are reconstructed in the raw UniEgo domain. The BCE logit is
    centered at the evaluation threshold of 0.5, and per-channel positive
    weights are derived from the training-set contact means so positive and
    negative frames contribute equally. Physical losses use GT contact masks,
    which prevents the model from avoiding them by predicting no contact.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if contact_logit_scale <= 0.0:
        raise ValueError(
            f"contact_logit_scale must be positive, got {contact_logit_scale}"
        )

    contact_mean = mean[FOOT_SLICE]
    contact_std = std[FOOT_SLICE]
    contacts_hat = x0_hat[..., FOOT_SLICE] * contact_std + contact_mean
    contacts_gt = (x0[..., FOOT_SLICE] * contact_std + contact_mean).clamp(0.0, 1.0)
    contacts_gt_binary = (contacts_gt > 0.5).to(x0_hat.dtype)

    logits = contact_logit_scale * (contacts_hat - 0.5)
    eps = torch.finfo(contact_mean.dtype).eps
    positive_weight = ((1.0 - contact_mean) / contact_mean.clamp(min=eps)).to(
        device=x0_hat.device,
        dtype=x0_hat.dtype,
    )
    contact_bce = F.binary_cross_entropy_with_logits(
        logits,
        contacts_gt_binary,
        pos_weight=positive_weight,
        reduction="none",
    )
    loss_contact = _masked_mean(contact_bce, valid)

    feet_hat = joints_hat[..., FOOT_JOINT_IDX, :]
    contact_mask = contacts_gt_binary.bool() & valid.unsqueeze(-1)

    pair_valid = valid[:, 1:] & valid[:, :-1]
    velocity_contact_mask = contacts_gt_binary[:, :-1].bool() & pair_valid.unsqueeze(-1)
    horizontal_velocity = (
        feet_hat[:, 1:, :, (0, 2)] - feet_hat[:, :-1, :, (0, 2)]
    ) * float(fps)
    loss_foot_velocity = _masked_mean(
        horizontal_velocity.square(),
        velocity_contact_mask,
    )

    features_hat = x0_hat * std + mean
    features_gt = x0 * std + mean
    height_error = (
        features_hat[..., FOOT_Y_IDX] - features_gt[..., FOOT_Y_IDX]
    ).square()
    loss_foot_height = _masked_mean(height_error, contact_mask)
    return loss_contact, loss_foot_velocity, loss_foot_height


__all__ = ["masked_mse", "contact_aware_losses"]
