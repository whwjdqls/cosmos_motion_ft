"""Use the Phase-2/3 clean-motion objective inside the native action stream."""
from pathlib import Path
import sys

import torch

# The specialist's loss modules retain historical flat sibling imports. Append
# their directory without changing precedence for the native framework/config.
_shared = str(Path(__file__).resolve().parents[1] / "motion_expert_joint_attention")
if _shared not in sys.path:
    sys.path.append(_shared)
from motion_expert_t2m_edge.losses import t2m_reconstruction_loss
from motion_expert_phase3_edge.settings import RECIPE

MOTION_WEIGHTS = dict(RECIPE["motion_weights"])
OBJECTIVE = "x0_reconstruction_shared_waver_v1"


def reconstruction_loss(predictions, targets, masks, mean, std):
    """Remove the observed shape prefix and padding before specialist decoding.

Motion2video has no motion supervision. The three motion-generation tasks use
the exact Phase-3 reconstruction call, including its task reduction and guard.
"""
    active = []
    for i, (prediction, target, mask) in enumerate(zip(predictions, targets, masks, strict=True)):
        if prediction.shape != (98, 288) or target.shape != prediction.shape or mask.shape != (98, 1):
            raise ValueError("Expected native motion stream [98,288] with frame mask [98,1]")
        if not bool(mask[0].all()):
            raise ValueError("Neutral skeleton must always be observed")
        noisy = mask[1:, 0] == 0
        if bool(noisy.any()):
            if not bool(noisy.all()):
                raise ValueError("The current four-task recipe supervises all97 motion frames or none")
            active.append(i)
    zero = sum(p.float().sum() * 0. for p in predictions)
    if not active:
        return zero, {name: zero for name in ("feature", "joint", "smooth", "contact", "foot_vel", "foot_height")}
    pred = torch.stack([predictions[i][1:, :283].float() for i in active])
    target = torch.stack([targets[i][1:, :283].detach().float() for i in active])
    valid = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    # Explicit FP32 matches the specialist's decoded-joint and physical losses.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        loss, terms = t2m_reconstruction_loss(
            pred, target, valid, mean.to(pred.device), std.to(pred.device),
            **MOTION_WEIGHTS, joint_loss_mode="centered", fps=20.,
            modes=["text2motion"] * len(active), reduction="per_task", decoded_nonfinite="skip")
    fraction = len(active) / len(predictions)
    return loss * fraction, {name: value * fraction for name, value in terms.items()}


def x0_segment_to_velocity(native_output, state, vision_numel, action_shape, observed, sigma):
    """Convert only the action tail, preserving native visual velocity exactly."""
    if action_shape != (98, 288) or state.numel() != vision_numel + 98 * 288:
        raise ValueError("Unexpected inference state layout")
    x = state[vision_numel:].reshape(action_shape).float()
    x0 = native_output[vision_numel:].reshape(action_shape).float()
    sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32)
    if not bool(sigma > 0):
        raise ValueError("x0-to-velocity conversion requires positive sampler sigma")
    velocity = (x - x0) / sigma
    velocity[list(observed)] = 0
    velocity[:, 283:] = 0
    return torch.cat((native_output[:vision_numel], velocity.flatten()))
