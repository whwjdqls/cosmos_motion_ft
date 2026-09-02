"""Trainable UniEgo motion I/O heads for the Edge T2M expert."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import config


class MotionHeads(nn.Module):
    def __init__(self, hidden_size: int = config.HIDDEN_SIZE) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.motion2llm = nn.Linear(config.MOTION_DIM, hidden_size)
        self.shape2llm = nn.Sequential(
            nn.Linear(config.NUM_JOINTS * 3, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.llm2motion = nn.Linear(hidden_size, config.MOTION_DIM)
        self.motion_modality_embed = nn.Parameter(torch.zeros(hidden_size))
        self.shape_type_embed = nn.Parameter(torch.zeros(hidden_size))

        from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder

        self.time_embedder = TimestepEmbedder(hidden_size)
        self.time_embedder._init_weights(buffer_device=None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.trunc_normal_(self.motion2llm.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.motion2llm.bias)
        for module in self.shape2llm:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.llm2motion.weight)
        nn.init.zeros_(self.llm2motion.bias)
        nn.init.trunc_normal_(self.motion_modality_embed, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.shape_type_embed, std=std, a=-3 * std, b=3 * std)

    def encode_shape(self, neutral_joints: torch.Tensor) -> torch.Tensor:
        batch = neutral_joints.shape[0]
        dtype = self.shape2llm[0].weight.dtype
        hidden = self.shape2llm(neutral_joints.reshape(batch, -1).to(dtype))
        hidden = hidden + self.motion_modality_embed + self.shape_type_embed
        return hidden.unsqueeze(1)

    def encode_motion(
        self,
        x_sigma: torch.Tensor,
        normalized_sigma: torch.Tensor,
        noisy_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode frames with the same normalized flow-time signal Edge sees.

        Edge's native network receives scheduler timesteps and multiplies them
        by `1/1000` before `TimestepEmbedder`. This Phase-2 API already receives
        normalized sigma, so it is embedded exactly once with no hidden rescale.
        """

        dtype = self.motion2llm.weight.dtype
        hidden = self.motion2llm(x_sigma.to(dtype)) + self.motion_modality_embed
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            time_hidden = self.time_embedder(normalized_sigma.float()).float()
        hidden = hidden + noisy_frame_mask.to(hidden.dtype).unsqueeze(-1) * time_hidden.to(
            hidden.dtype
        ).unsqueeze(1)
        return hidden

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.llm2motion(hidden.to(self.llm2motion.weight.dtype)).float()


__all__ = ["MotionHeads"]

