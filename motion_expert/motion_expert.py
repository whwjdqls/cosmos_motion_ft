"""Phase 1: MotionExpert (trainable) + ShapeEncoder.

A small motion-native transformer that cross-attends ONE-WAY to the frozen reasoner
hidden states H_R, conditioned in-context on the actor skeleton, with DiT-style AdaLN-zero
time conditioning (per-block FiLM from the flow-time embedding — required so the denoiser
can perform σ-dependent denoising; a one-time additive time token is too weak).

Sequence fed to self-attention: [shape_tok, motion_1 .. motion_T]
  - motion tokens   : motion_in(x_σ:283→d) + sinusoidal FRAME pos-emb
  - shape_tok       : ShapeEncoder(neutral_joints:90→d) + learned type-emb (no frame pos)
  - flow-time       : AdaLN-zero modulation (shift/scale/gate per block), NOT an additive token
Each of N blocks (pre-norm, AdaLN-zero on self-attn + FFN):
  self-attn (full, over [shape,motion])  →  cross-attn (Q=seq, K=V=H_R; kdim=vdim=4096)  →  FFN
Output: motion_out(d→283) read on the T motion positions only (shape_tok dropped).

Attention contract (holds by construction — reasoner is a separate forward):
  R→R causal (in reasoner), M→M full, M→R cross-attn, no R→M, no generator.
No separate Hproj: the cross-attn's own K/V linears (kdim=vdim=4096) adapt the reasoner dim.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from uniego_layout import FEAT_DIM, N_JOINTS  # 283, 30


def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """values [N] (float) → [N, dim] sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=values.device, dtype=torch.float32) / half
    )
    args = values.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=-1)
    return emb


def modulate(x, shift, scale):
    """FiLM: x [B,L,d], shift/scale [B,d] → x*(1+scale) + shift (broadcast over L)."""
    return x * (1 + scale[:, None]) + shift[:, None]


class ShapeEncoder(nn.Module):
    """neutral_joints (B,30,3) → shape_tok (B,1,d). Encodes per-actor bone sizes."""

    def __init__(self, d: int, n_joints: int = N_JOINTS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_joints * 3, d), nn.GELU(),
            nn.LayerNorm(d), nn.Linear(d, d),
        )

    def forward(self, neutral_joints: torch.Tensor) -> torch.Tensor:
        B = neutral_joints.shape[0]
        return self.mlp(neutral_joints.reshape(B, -1)).unsqueeze(1)  # [B,1,d]


class Block(nn.Module):
    """AdaLN-zero block: self-attn (full) → cross-attn (KV=H_R) → FFN.

    Self-attn and FFN are AdaLN-modulated + gated by the flow-time embedding (DiT).
    Cross-attn uses plain pre-norm (its conditioning is H_R, not σ).
    """

    def __init__(self, d: int, heads: int, ffn: int, kv_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True, kdim=kv_dim, vdim=kv_dim
        )
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        # AdaLN-zero: produce (shift,scale,gate) for self-attn and FFN from t_emb
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)  # start as identity (gates=0 → residual passthrough)

    def forward(self, x, H_R, t_emb, self_pad_mask=None, kv_pad_mask=None):
        sa_shift, sa_scale, sa_gate, ff_shift, ff_scale, ff_gate = self.ada(t_emb).chunk(6, dim=-1)
        h = modulate(self.n1(x), sa_shift, sa_scale)
        x = x + sa_gate[:, None] * self.self_attn(h, h, h, key_padding_mask=self_pad_mask, need_weights=False)[0]
        x = x + self.cross_attn(self.n2(x), H_R, H_R, key_padding_mask=kv_pad_mask, need_weights=False)[0]
        h = modulate(self.n3(x), ff_shift, ff_scale)
        x = x + ff_gate[:, None] * self.ffn(h)
        return x


class MotionExpert(nn.Module):
    def __init__(self, d: int = 512, n_layers: int = 8, heads: int = 8, ffn: int = 2048,
                 kv_dim: int = 4096, motion_dim: int = FEAT_DIM, n_joints: int = N_JOINTS,
                 dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.motion_dim = motion_dim
        self.motion_in = nn.Linear(motion_dim, d)
        self.shape_enc = ShapeEncoder(d, n_joints)
        self.shape_type = nn.Parameter(torch.zeros(d))          # distinguishes shape_tok
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, ffn, kv_dim, dropout) for _ in range(n_layers)])
        # final AdaLN + zero-init output head (DiT)
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada_out[1].weight); nn.init.zeros_(self.ada_out[1].bias)
        self.motion_out = nn.Linear(d, motion_dim)
        nn.init.zeros_(self.motion_out.weight); nn.init.zeros_(self.motion_out.bias)

    def forward(
        self,
        x_sigma: torch.Tensor,            # [B,T,283] noised motion
        sigma: torch.Tensor,             # [B] flow time in [0,1]
        H_R: torch.Tensor,               # [B,Ttext,4096] frozen reasoner hidden states
        h_pad_mask: torch.Tensor | None,  # [B,Ttext] True=pad (reasoner) — KV mask
        neutral_joints: torch.Tensor,    # [B,30,3]
        motion_pad_mask: torch.Tensor | None = None,  # [B,T] True=pad (short windows)
    ) -> torch.Tensor:
        B, T, _ = x_sigma.shape
        d = self.d

        m = self.motion_in(x_sigma)                                  # [B,T,d]
        pos = sinusoidal_embedding(torch.arange(T, device=x_sigma.device), d).to(m.dtype)
        m = m + pos[None]                                           # frame pos (motion only)

        shape_tok = self.shape_enc(neutral_joints) + self.shape_type.to(m.dtype)  # [B,1,d]
        seq = torch.cat([shape_tok, m], dim=1)                       # [B,1+T,d]

        # flow-time embedding → AdaLN modulation signal (NOT added to tokens)
        t_emb = self.time_mlp(sinusoidal_embedding(sigma * 1000.0, d).to(m.dtype))  # [B,d]

        self_pad = None
        if motion_pad_mask is not None:
            shape_valid = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
            self_pad = torch.cat([shape_valid, motion_pad_mask], dim=1)  # [B,1+T]

        for blk in self.blocks:
            seq = blk(seq, H_R, t_emb, self_pad_mask=self_pad, kv_pad_mask=h_pad_mask)

        out_shift, out_scale = self.ada_out(t_emb).chunk(2, dim=-1)
        seq = modulate(self.norm_out(seq), out_shift, out_scale)
        v = self.motion_out(seq[:, 1:])                            # drop shape_tok → [B,T,283]
        return v
