"""Time-aligned, motion-context-aware residual camera modulation."""
import torch
from torch import nn


class CameraMotionFiLM(nn.Module):
    def __init__(self, hidden, context_dim=256, mlp_dim=256):
        super().__init__()
        self.motion_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.camera_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.context_norm = nn.LayerNorm(context_dim, elementwise_affine=False)
        self.camera_projection = nn.Sequential(nn.Linear(2 * hidden + 2, context_dim), nn.SiLU())
        self.mlp = nn.Sequential(nn.Linear(hidden + context_dim, mlp_dim), nn.SiLU(),
                                 nn.Linear(mlp_dim, 2 * hidden))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def align_transitions(camera):
        zero = camera.new_zeros((1, camera.shape[-1]))
        incoming = torch.cat((zero, camera), dim=0)
        outgoing = torch.cat((camera, zero), dim=0)
        frame = torch.arange(camera.shape[0] + 1, device=camera.device)
        validity = torch.stack((frame > 0, frame < camera.shape[0]), dim=-1).to(camera.dtype)
        return incoming, outgoing, validity

    def delta(self, motion, camera):
        if camera.shape[0] + 1 != motion.shape[0]:
            raise ValueError("FiLM requires T-1 ordered camera transitions for T motion frames")
        normalized_motion = self.motion_norm(motion.float())
        incoming, outgoing, valid = self.align_transitions(self.camera_norm(camera.float()))
        context = self.camera_projection(torch.cat((incoming, outgoing, valid), dim=-1))
        gamma, beta = self.mlp(torch.cat((normalized_motion, self.context_norm(context)), dim=-1)).chunk(2, dim=-1)
        return gamma * normalized_motion + beta

    def forward(self, motion, camera):
        return (motion.float() + self.delta(motion, camera)).to(motion.dtype)


class CameraFiLMBridge(nn.Module):
    """Preserve attention, then modulate only camera-conditioned target frames."""
    def __init__(self, attention, context_dim=256, mlp_dim=256):
        super().__init__()
        self.attention = attention
        self.film = CameraMotionFiLM(attention.hidden, context_dim, mlp_dim).to(
            device=attention.q.weight.device, dtype=torch.float32)
        self.film_enabled = True

    def forward(self, gen, motion, meta):
        gen, motion = self.attention(gen, motion, meta)
        if not self.film_enabled or meta.mode not in {"camera2motion", "cameravideo2motion"}:
            # Keep DDP parameters represented even in batches without camera tasks.
            anchor = sum(parameter.flatten()[0] * 0.0 for parameter in self.film.parameters())
            return gen, motion + anchor.to(motion.dtype)
        camera_rows = ((meta.gen_frame < 0) & (meta.gen_source_start >= 0)
                       & (meta.gen_source_end == meta.gen_source_start + 1))
        camera = gen[camera_rows]
        if camera.shape[0] != motion.shape[0] - 2:
            raise ValueError("Camera/motion routing must contain T-1 actions and shape+T motion rows")
        frames = self.film(motion[1:], camera)
        return gen, torch.cat((motion[:1], frames), dim=0)
