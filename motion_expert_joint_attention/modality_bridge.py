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
    motion_clean: torch.Tensor    # [N_mot] bool, True == clean condition/shape
    gen_source_start: torch.Tensor  # [N_gen], inclusive source-frame interval, -1 if absent
    gen_source_end: torch.Tensor    # [N_gen], inclusive source-frame interval, -1 if absent


class LocalModalityBridge(nn.Module):
    """Small gated self-attention bridge over ``[G | M]`` tokens.

    Cross-modal attention is role-driven rather than task-name-driven. A noised target row may
    attend aligned rows from the other modality; clean cross-modal query rows never read targets.
    This gives the two historical directions as one-sided masks and joint video/motion targets as
    a bidirectional mask without changing bridge parameters.

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
        """Return pairs aligned by the causal Wan-VAE 4x temporal layout.

        For a 97-frame clip the 25 latent frames represent source-frame groups
        ``{0}, {1..4}, {5..8}, ..., {93..96}``.  Equivalently, source motion
        frame ``m`` belongs to latent frame ``ceil(m / 4)``.  The historical
        ``[4*g, 4*g+3]`` rule was shifted by three frames, assigned frames 1..3
        to latent 0, and left source frames 94..96 without a bridge edge.
        """
        if gen_frame.numel() == 0 or motion_frame.numel() == 0:
            return torch.zeros((gen_frame.numel(), motion_frame.numel()), device=gen_frame.device, dtype=torch.bool)
        gf = gen_frame.view(-1, 1)
        mf = motion_frame.view(1, -1)
        valid = (gf >= 0) & (mf >= 0)
        return valid & (torch.div(mf + 3, 4, rounding_mode="floor") == gf)

    def _source_interval_pair_mask(self, meta: BridgeMeta) -> torch.Tensor:
        """Pair generator rows with motion frames in their physical source-frame interval.

        Video latent intervals are ``{0}, {1..4}, ...``. Camera action ``i`` spans source
        transition ``i -> i+1`` and therefore uses interval ``[i, i+1]``. The union preserves
        the historical video locality while adding direct local action-motion edges.
        """
        starts = meta.gen_source_start.view(-1, 1)
        ends = meta.gen_source_end.view(-1, 1)
        motion = meta.motion_frame.view(1, -1)
        valid = (starts >= 0) & (ends >= starts) & (motion >= 0)
        return valid & (motion >= starts) & (motion <= ends)

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

        local = self._source_interval_pair_mask(meta)
        gen_noisy_rows = ~meta.gen_clean.bool()
        motion_noisy_rows = (~meta.motion_clean.bool()) & (meta.motion_frame >= 0)
        motion_frame_rows = meta.motion_frame >= 0

        # Motion targets read local generator rows, whether those rows are clean conditions or
        # sibling noisy targets. Clean motion rows never receive a cross-modal residual update.
        if motion_noisy_rows.any() and local.any():
            mask[ng:, :ng] |= local.T & motion_noisy_rows.view(-1, 1)

        # Generator targets read local motion frames. In a joint task those are sibling targets;
        # in M2V they are clean conditions. Shape is globally available to every noisy gen row.
        if gen_noisy_rows.any() and motion_frame_rows.any() and local.any():
            mask[:ng, ng:] |= (
                local & gen_noisy_rows.view(-1, 1) & motion_frame_rows.view(1, -1)
            )
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
