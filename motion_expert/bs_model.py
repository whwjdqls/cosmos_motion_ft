"""MotionExpert with in-context text conditioning (BONES-SEED POC).

Identical backbone to ``motion_expert.py:MotionExpert`` (d=512 / 8 layers / 8 heads / ffn 2048,
AdaLN-zero DiT time conditioning, x0 prediction, zero-init output head, the same ShapeEncoder and
sinusoidal frame positions). Two changes only:

  1. Text is a **prepended in-context token** (not cross-attention). The pooled llm2vec vector
     [B,1,4096] is projected to d and prepended next to the shape token:
         seq = [text_tok, shape_tok, motion_1..T]
  2. **No cross-attention** — each block is self-attn (AdaLN-zero) -> FFN (AdaLN-zero).

Dropping / CFG: text is dropped to the cache's "" null embedding (handled by the caller); the
**shape token is never dropped**. The output is read on the T motion positions only (``seq[:, 2:]``).

The forward signature ``(x_sigma, sigma, text_emb, text_pad_mask, neutral_joints, motion_pad_mask)``
mirrors the cross-attn model's positional order (text_emb in place of H_R, text_pad_mask ignored —
the text is a single token) so ``flow.py:sample_x0`` is reused verbatim for sampling.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Reuse the *identical* helpers + ShapeEncoder from the original MotionExpert.
from motion_expert import ShapeEncoder, modulate, sinusoidal_embedding
from uniego_layout import FEAT_DIM, N_JOINTS  # 283, 30


class Block(nn.Module):
    """AdaLN-zero DiT block: self-attn (full) -> FFN. (No cross-attn.)

    Self-attn and FFN are AdaLN-modulated + gated by the flow-time embedding, exactly as in
    ``motion_expert.py:Block`` with the cross-attention sublayer removed.
    """

    def __init__(self, d: int, heads: int, ffn: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)  # start as identity (gates=0 -> residual passthrough)

    def forward(self, x, t_emb, self_pad_mask=None):
        sa_shift, sa_scale, sa_gate, ff_shift, ff_scale, ff_gate = self.ada(t_emb).chunk(6, dim=-1)
        h = modulate(self.n1(x), sa_shift, sa_scale)
        x = x + sa_gate[:, None] * self.self_attn(
            h, h, h, key_padding_mask=self_pad_mask, need_weights=False,
        )[0]
        h = modulate(self.n3(x), ff_shift, ff_scale)
        x = x + ff_gate[:, None] * self.ffn(h)
        return x


class MotionExpertInContext(nn.Module):
    def __init__(self, d: int = 512, n_layers: int = 8, heads: int = 8, ffn: int = 2048,
                 text_dim: int = 4096, motion_dim: int = FEAT_DIM, n_joints: int = N_JOINTS,
                 dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.motion_dim = motion_dim
        self.motion_in = nn.Linear(motion_dim, d)
        self.text_proj = nn.Linear(text_dim, d)                 # pooled llm2vec 4096 -> d
        self.text_type = nn.Parameter(torch.zeros(d))           # distinguishes text_tok
        self.shape_enc = ShapeEncoder(d, n_joints)
        self.shape_type = nn.Parameter(torch.zeros(d))          # distinguishes shape_tok
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, ffn, dropout) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada_out[1].weight); nn.init.zeros_(self.ada_out[1].bias)
        self.motion_out = nn.Linear(d, motion_dim)
        nn.init.zeros_(self.motion_out.weight); nn.init.zeros_(self.motion_out.bias)

    def forward(
        self,
        x_sigma: torch.Tensor,                       # [B,T,283] noised motion
        sigma: torch.Tensor,                         # [B] flow time in [0,1]
        text_emb: torch.Tensor,                      # [B,1,4096] pooled llm2vec (or null "")
        text_pad_mask: torch.Tensor | None,          # ignored (single text token; for flow.py parity)
        neutral_joints: torch.Tensor,                # [B,30,3]
        motion_pad_mask: torch.Tensor | None = None,  # [B,T] True=pad
    ) -> torch.Tensor:
        B, T, _ = x_sigma.shape
        d = self.d

        m = self.motion_in(x_sigma)                                  # [B,T,d]
        pos = sinusoidal_embedding(torch.arange(T, device=x_sigma.device), d).to(m.dtype)
        m = m + pos[None]                                           # frame pos (motion only)

        text_tok = self.text_proj(text_emb) + self.text_type.to(m.dtype)         # [B,1,d]
        shape_tok = self.shape_enc(neutral_joints) + self.shape_type.to(m.dtype)  # [B,1,d]
        seq = torch.cat([text_tok, shape_tok, m], dim=1)            # [B, 2+T, d]

        t_emb = self.time_mlp(sinusoidal_embedding(sigma * 1000.0, d).to(m.dtype))  # [B,d]

        self_pad = None
        if motion_pad_mask is not None:
            cond_valid = torch.zeros(B, 2, dtype=torch.bool, device=seq.device)  # text+shape always valid
            self_pad = torch.cat([cond_valid, motion_pad_mask], dim=1)           # [B, 2+T]

        for blk in self.blocks:
            seq = blk(seq, t_emb, self_pad_mask=self_pad)

        out_shift, out_scale = self.ada_out(t_emb).chunk(2, dim=-1)
        seq = modulate(self.norm_out(seq), out_shift, out_scale)
        return self.motion_out(seq[:, 2:])                          # drop text+shape -> [B,T,283]
