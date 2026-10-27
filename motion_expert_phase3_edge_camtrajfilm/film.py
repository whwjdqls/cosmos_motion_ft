"""Framewise FP32 trajectory FiLM, with a direct motion diffusion-time path."""
import torch
from torch import nn

from .settings import has_camera


class CameraMotionFiLM(nn.Module):
    def __init__(self, hidden, context_dim=256, mlp_dim=256):
        super().__init__()
        self.motion_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.camera_projection = nn.Sequential(nn.Linear(18, context_dim), nn.SiLU())
        self.noise_projection = nn.Sequential(nn.Linear(hidden, context_dim), nn.SiLU())
        self.camera_norm = nn.LayerNorm(context_dim, elementwise_affine=False)
        self.noise_norm = nn.LayerNorm(context_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden + 2 * context_dim, mlp_dim),
                                 nn.SiLU(), nn.Linear(mlp_dim, 2 * hidden))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def delta(self, motion, camera, noise):
        if camera.shape != (motion.shape[0], 18) or noise.shape != (motion.shape[-1],):
            raise ValueError("FiLM expects frame-aligned [T,18] camera and [hidden] motion-time embedding")
        with torch.autocast(device_type=motion.device.type, enabled=False):
            normalized = self.motion_norm(motion.float())
            camera_context = self.camera_norm(self.camera_projection(camera.float()))
            noise_context = self.noise_norm(self.noise_projection(noise.float()))
            noise_context = noise_context.unsqueeze(0).expand(motion.shape[0], -1)
            gamma, beta = self.mlp(torch.cat((normalized, camera_context, noise_context), -1)).chunk(2, -1)
            return gamma * normalized + beta

    def forward(self, motion, camera, noise):
        return (motion.float() + self.delta(motion, camera, noise)).to(motion.dtype)


class CameraFiLMBridge(nn.Module):
    def __init__(self, attention, context_dim=256, mlp_dim=256):
        super().__init__()
        self.attention = attention
        weight = attention.q.weight
        self.film = CameraMotionFiLM(weight.shape[1], context_dim, mlp_dim).to(
            device=weight.device, dtype=torch.float32)
        self.film_enabled = True

    def forward(self, generator, motion, meta, camera_features=None, motion_time=None):
        generator, motion = self.attention(generator, motion, meta)
        if not self.film_enabled or not has_camera(meta.mode):
            anchor = sum(p.flatten()[0] * 0.0 for p in self.film.parameters())
            return generator, motion + anchor.to(motion.dtype)
        if camera_features is None or motion_time is None:
            raise ValueError("Camera tasks require trajectory features and motion-noise embedding")
        frames = self.film(motion[1:], camera_features, motion_time)
        return generator, torch.cat((motion[:1], frames))
