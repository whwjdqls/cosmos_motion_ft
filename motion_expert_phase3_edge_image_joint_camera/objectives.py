"""No camera target or camera loss; existing motion/video noising stays exact."""
import torch

from motion_expert_phase3_edge_image_joint.objectives import noise_batch as base_noise
from motion_expert_phase3_edge_image_joint.objectives import specialist_loss


def camera_clock(modes, motion_sigma, video_sigma):
    use_motion = torch.tensor([m == "video2motion" for m in modes], device=motion_sigma.device)
    return torch.where(use_motion, motion_sigma, video_sigma)


def noise_batch(batch, device):
    inputs, target = base_noise(batch, device)
    sigma = camera_clock(inputs["modes"], inputs["motion_sigma"], inputs["video_sigma"])
    camera = batch["camera_action"].to(device=device, dtype=torch.float32)
    if camera.shape != (len(inputs["modes"]), 96, 64) or len(batch["draws"]) != len(camera):
        raise ValueError("Need aligned native camera actions and global draw IDs")
    if not torch.isfinite(camera).all():
        raise ValueError("Nonfinite camera target input")
    noise = []
    for draw in batch["draws"]:
        generator = torch.Generator(device=device).manual_seed(104729 + 1000003 * (int(draw) + 1))
        noise.append(torch.randn((96, 64), generator=generator, device=device, dtype=torch.float32))
    state = torch.lerp(camera, torch.stack(noise), sigma[:, None, None])
    # Deliberately keep target keys identical to the baseline: motion/velocity.
    return {**inputs, "camera_state": state, "camera_sigma": sigma}, target
