"""Trainable generator-motion bridge modules.

The bridge is deliberately separate from the native Cosmos generator attention. It runs after the
native generator/reasoner and motion-expert updates, and its attention mask controls target-to-
condition information flow so clean conditioning tokens do not read noisy target tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BridgeMeta:
    """Per-sample metadata for local gen-motion bridge masks."""

    mode: str
    gen_frame: torch.Tensor       # [N_gen], -1 for non-video/image tokens
    gen_clean: torch.Tensor       # [N_gen] bool, True == clean condition
    motion_frame: torch.Tensor    # [N_mot], -1 for shape/non-frame token


class LocalModalityBridge(nn.Module):
    """Small gated self-attention bridge over ``[G | M]`` tokens.

    Cross-modal attention is directional:
      * video2motion: noised motion rows may attend local clean video rows.
      * motimg2video: noised video rows may attend local clean motion rows.

    Same-modality attention is allowed within the bridge input, but the residual gates are
    zero-initialized so the bridge is an exact no-op at initialization.
    """

    def __init__(self, hidden: int, num_heads: int, head_dim: int):
        super().__init__()
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        qkv_dim = self.num_heads * self.head_dim
        self.norm = nn.LayerNorm(self.hidden)
        self.q = nn.Linear(self.hidden, qkv_dim, bias=False)
        self.k = nn.Linear(self.hidden, qkv_dim, bias=False)
        self.v = nn.Linear(self.hidden, qkv_dim, bias=False)
        self.o = nn.Linear(qkv_dim, self.hidden, bias=False)
        self.gate_g = nn.Parameter(torch.zeros(()))
        self.gate_m = nn.Parameter(torch.zeros(()))

    def _local_pair_mask(self, gen_frame: torch.Tensor, motion_frame: torch.Tensor) -> torch.Tensor:
        """Return [N_gen, N_mot] True where video latent frame and motion frame are aligned."""
        if gen_frame.numel() == 0 or motion_frame.numel() == 0:
            return torch.zeros((gen_frame.numel(), motion_frame.numel()), device=gen_frame.device, dtype=torch.bool)
        gf = gen_frame.view(-1, 1)
        mf = motion_frame.view(1, -1)
        valid = (gf >= 0) & (mf >= 0)
        return valid & (mf >= gf * 4) & (mf < gf * 4 + 4)

    def _attention_mask(self, meta: BridgeMeta) -> torch.Tensor:
        device = meta.gen_frame.device
        ng = int(meta.gen_frame.numel())
        nm = int(meta.motion_frame.numel())
        n = ng + nm
        mask = torch.eye(n, dtype=torch.bool, device=device)
        if ng:
            mask[:ng, :ng] = True
        if nm:
            mask[ng:, ng:] = True

        local = self._local_pair_mask(meta.gen_frame, meta.motion_frame)
        if meta.mode == "video2motion":
            # Motion frame rows are noisy targets; allow them to read local clean video.
            motion_rows = meta.motion_frame >= 0
            if motion_rows.any() and local.any():
                mask[ng:, :ng] |= local.T & motion_rows.view(-1, 1)
        elif meta.mode == "motimg2video":
            # Video future rows are noisy targets; allow only those rows to read local clean motion.
            gen_noisy_rows = (~meta.gen_clean.bool()) & (meta.gen_frame >= 0)
            motion_frame_rows = meta.motion_frame >= 0
            if gen_noisy_rows.any() and motion_frame_rows.any() and local.any():
                mask[:ng, ng:] |= local & gen_noisy_rows.view(-1, 1) & motion_frame_rows.view(1, -1)
            # The motion shape token is clean conditioning and may be read by noisy video rows.
            shape_cols = meta.motion_frame < 0
            if gen_noisy_rows.any() and shape_cols.any():
                mask[:ng, ng:] |= gen_noisy_rows.view(-1, 1) & shape_cols.view(1, -1)
        return mask

    def forward(
        self,
        gen: torch.Tensor,
        motion: torch.Tensor,
        meta: BridgeMeta,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gen.numel() == 0 or motion.numel() == 0:
            return gen, motion
        x = torch.cat([gen, motion], dim=0)
        h = self.norm(x.float())
        q = self.q(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        allowed = self._attention_mask(meta).to(x.device)
        attn_bias = torch.zeros_like(allowed, dtype=torch.float32)
        attn_bias.masked_fill_(~allowed, torch.finfo(torch.float32).min)
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn + attn_bias.unsqueeze(0), dim=-1)
        y = torch.matmul(attn, v).transpose(0, 1).reshape(x.shape[0], -1)
        y = self.o(y).to(x.dtype)

        ng = gen.shape[0]
        yg, ym = y[:ng], y[ng:]
        return gen + self.gate_g.to(gen.dtype) * yg, motion + self.gate_m.to(motion.dtype) * ym
