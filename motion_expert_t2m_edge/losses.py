"""Phase-2 losses with explicit legacy-pooled and Nano per-task reductions."""
from __future__ import annotations

from collections.abc import Sequence

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
    w_global_rot: float = 0.0,
    w_global_trans: float = 0.0,
    w_joint: float = 10.0,
    joint_loss_mode: str = "centered",
    w_smooth: float = 50.0,
    w_contact: float = 0.0,
    w_foot_vel: float = 0.0,
    w_foot_height: float = 0.0,
    fps: float = 20.0,
    modes: Sequence[str] | None = None,
    reduction: str = "pooled",
    decoded_nonfinite: str = "raise",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decode once, reduce each active task, then weight by its sample count.

    Per-task matches the checked-in Nano trainer: each task's masked means
    contribute count(task)/local_batch. DDP averages these rank-local losses,
    just as Nano's manual gradient all-reduce did. It is not a per-sequence
    mean or a globally pooled-frame mean. Defaults preserve old Edge runs.
    """
    if joint_loss_mode not in {"centered", "absolute"}:
        raise ValueError(f"unsupported joint_loss_mode={joint_loss_mode!r}")
    if reduction not in {"pooled", "per_task"}:
        raise ValueError(f"unsupported loss reduction={reduction!r}")
    if decoded_nonfinite not in {"raise", "skip"}:
        raise ValueError(f"unsupported decoded_nonfinite={decoded_nonfinite!r}")
    batch = prediction.shape[0]
    if reduction == "per_task":
        if modes is None or len(modes) != batch or batch == 0:
            raise ValueError("per_task reduction requires one mode per nonempty batch row")
        if set(modes) - {"text2motion", "textimg2motion"}:
            raise ValueError("Phase-2 reconstruction supports only T2M and TI2M")
        groups = [
            (
                valid & torch.tensor(
                    [mode == name for mode in modes], device=valid.device, dtype=torch.bool
                )[:, None],
                sum(mode == name for mode in modes) / float(batch),
            )
            for name in sorted(set(modes))
        ]
    else:
        groups = [(valid, 1.0)]

    weights = {
        "feature": w_feat,
        "global_rot": w_global_rot,
        "global_trans": w_global_trans,
        "joint": w_joint,
        "smooth": w_smooth,
        "contact": w_contact,
        "foot_vel": w_foot_vel,
        "foot_height": w_foot_height,
    }
    if (w_global_rot > 0.0 or w_global_trans > 0.0) and prediction.shape[-1] != 283:
        raise ValueError("global feature weighting requires 283-D UniEgo")
    need_joints = any(
        weight > 0.0
        for weight in (w_joint, w_smooth, w_contact, w_foot_vel, w_foot_height)
    )
    joints_hat = joints_gt = None
    if need_joints:
        joints_hat = decode_joints(prediction * std + mean)
        with torch.no_grad():
            joints_gt = decode_joints(target * std + mean)
        if not torch.isfinite(joints_hat).all():
            if decoded_nonfinite == "raise":
                raise FloatingPointError("non-finite decoded joints in T2M loss")
            # Nano keeps feature supervision but omits all decoded/contact
            # terms when the shared decoded prediction is non-finite.
            joints_hat = joints_gt = None

    terms = {key: prediction.new_zeros(()) for key in weights}
    for mask, fraction in groups:
        local = {key: prediction.new_zeros(()) for key in weights}
        local["feature"] = masked_mse(prediction, target, mask)
        # Retain the original ablation denominator: +10 is 11x per global
        # coordinate with w_feat=1, not 10x a separately averaged feature group.
        if w_global_rot > 0.0:
            local["global_rot"] = masked_mse(
                prediction[..., 270:276], target[..., 270:276], mask
            ) * (6.0 / 283.0)
        if w_global_trans > 0.0:
            local["global_trans"] = masked_mse(
                prediction[..., 276:279], target[..., 276:279], mask
            ) * (3.0 / 283.0)
        if joints_hat is not None:
            if w_joint > 0.0:
                if joint_loss_mode == "absolute":
                    local["joint"] = masked_mse(joints_hat, joints_gt, mask)
                else:
                    relative_hat = joints_hat - joints_hat.mean(dim=2, keepdim=True)
                    relative_gt = joints_gt - joints_gt.mean(dim=2, keepdim=True)
                    local["joint"] = masked_mse(relative_hat, relative_gt, mask)
            if w_smooth > 0.0:
                local["smooth"] = masked_mse(
                    joints_hat[:, 1:] - joints_hat[:, :-1],
                    joints_gt[:, 1:] - joints_gt[:, :-1],
                    mask[:, 1:] & mask[:, :-1],
                )
            if any(w > 0.0 for w in (w_contact, w_foot_vel, w_foot_height)):
                local["contact"], local["foot_vel"], local["foot_height"] = contact_aware_losses(
                    prediction, target, joints_hat, mask, mean, std, fps=fps
                )
        for key, value in local.items():
            terms[key] = terms[key] + fraction * value
    total = sum(weights[key] * terms[key] for key in weights)
    return total, terms


__all__ = ["masked_mse", "t2m_reconstruction_loss"]
