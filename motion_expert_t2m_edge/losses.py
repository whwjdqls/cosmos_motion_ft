"""Original Phase-2 motion reconstruction losses."""
from __future__ import annotations

import torch

from decode_uniego_torch import decode_joints
from motion_losses import contact_aware_losses


def masked_mse(left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid
    while mask.ndim < left.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(left)
    return ((left - right).square() * mask).sum() / mask.sum().clamp(min=1)


def t2m_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    w_feat: float = 1.0,
    w_joint: float = 10.0,
    w_smooth: float = 50.0,
    w_contact: float = 0.0,
    w_foot_vel: float = 0.0,
    w_foot_height: float = 0.0,
    fps: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    feature = masked_mse(prediction, target, valid)
    zero = feature.new_zeros(())
    joint = smooth = contact = foot_vel = foot_height = zero
    need_joints = any(
        weight > 0.0
        for weight in (w_joint, w_smooth, w_contact, w_foot_vel, w_foot_height)
    )
    if need_joints:
        joints_hat = decode_joints(prediction * std + mean)
        with torch.no_grad():
            joints_gt = decode_joints(target * std + mean)
        if not torch.isfinite(joints_hat).all():
            raise FloatingPointError("non-finite decoded joints in T2M loss")
        relative_hat = joints_hat - joints_hat.mean(dim=2, keepdim=True)
        relative_gt = joints_gt - joints_gt.mean(dim=2, keepdim=True)
        if w_joint > 0.0:
            joint = masked_mse(relative_hat, relative_gt, valid)
        if w_smooth > 0.0:
            pair_valid = valid[:, 1:] & valid[:, :-1]
            smooth = masked_mse(
                joints_hat[:, 1:] - joints_hat[:, :-1],
                joints_gt[:, 1:] - joints_gt[:, :-1],
                pair_valid,
            )
        if any(weight > 0.0 for weight in (w_contact, w_foot_vel, w_foot_height)):
            contact, foot_vel, foot_height = contact_aware_losses(
                prediction, target, joints_hat, valid, mean, std, fps=fps
            )
    terms = {
        "feature": feature,
        "joint": joint,
        "smooth": smooth,
        "contact": contact,
        "foot_vel": foot_vel,
        "foot_height": foot_height,
    }
    total = (
        w_feat * feature
        + w_joint * joint
        + w_smooth * smooth
        + w_contact * contact
        + w_foot_vel * foot_vel
        + w_foot_height * foot_height
    )
    return total, terms


__all__ = ["masked_mse", "t2m_reconstruction_loss"]
