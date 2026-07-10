"""Shape-aware TMR v3 — SnapMoGen's RELEASED evaluator architecture (EncoderV2 style).

The architectural shift vs v1/v2 (per SnapMoGen model/evaluator/modules.py:EncoderV2, exp
`evalv2_rec1_cst0.1_ld256`): **non-VAE, single learnable token, and SPLIT projection heads**
off that token — `cst_linear` (the contrastive/retrieval embedding) and `rec_linear` (the
latent the decoder reconstructs from). Reconstruction pressure therefore cannot fight the
retrieval embedding inside one vector, and there is no KL/sampling noise.

Shape-awareness and inputs are unchanged:
  - motion tower: sequence [CLS_tok, SHAPE_TOK, proj(frames)...] over the SAME 186-d
    TMRMotionRep (raw joint positions) — rep untouched.
  - text tower: same EncoderV2 form over the cached pooled llm2vec vector (B,1,4096).
  - decoder: shape-aware (memory [rec_z, shape_tok]), reconstructs the 186-d features.

encode(x_dict) -> (fid_emb (B,256), cst (B,256), rec (B,256)).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import repeat

from kimodo.model.tmr import PositionalEncoding

from st_model import ShapeEncoder, ShapeAwareDecoder


class EncoderV3(nn.Module):
    """SnapMoGen EncoderV2 + optional shape token. Non-VAE, single CLS token, split heads."""

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        output_dim: int = 256,          # cst head dim (shared contrastive space)
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        shape_aware: bool = False,
        n_joints: int = 30,
    ) -> None:
        super().__init__()
        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)
        self.tokens = nn.Parameter(torch.randn(1, latent_dim))
        self.shape_aware = shape_aware
        if shape_aware:
            self.shape_enc = ShapeEncoder(latent_dim, n_joints)
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation, batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                     enable_nested_tensor=False)
        self.rec_linear = nn.Linear(latent_dim, latent_dim)
        self.cst_linear = nn.Linear(latent_dim, output_dim)

    def encode(self, x_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.projection(x_dict["x"])
        mask = x_dict["mask"]                                    # (B,T) True = valid
        bs = len(x)
        device = x.device

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        parts, n_cond = [tokens], 1
        if self.shape_aware:
            parts.append(self.shape_enc(x_dict["neutral_joints"]))
            n_cond = 2
        xseq = torch.cat(parts + [x], dim=1)
        cond_mask = torch.ones((bs, n_cond), dtype=torch.bool, device=device)
        aug_mask = torch.cat((cond_mask, mask), 1)

        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        h = final[:, 0]                                          # CLS token
        return h, self.cst_linear(h), self.rec_linear(h)


def build_shape_tmr_v3(
    nfeats: int = 186,
    text_dim: int = 4096,
    latent_dim: int = 256,
    output_dim: int = 256,
    ff_size: int = 1024,
    enc_layers: int = 6,
    dec_layers: int = 6,
    num_heads: int = 4,
    dropout: float = 0.1,
    device: str = "cuda",
) -> Tuple[EncoderV3, EncoderV3, ShapeAwareDecoder]:
    """(motion_encoder [shape-aware], text_encoder, shape-aware decoder)."""
    motion_encoder = EncoderV3(
        nfeats=nfeats, latent_dim=latent_dim, output_dim=output_dim, ff_size=ff_size,
        num_layers=enc_layers, num_heads=num_heads, dropout=dropout, shape_aware=True,
    ).to(device)
    text_encoder = EncoderV3(
        nfeats=text_dim, latent_dim=latent_dim, output_dim=output_dim, ff_size=ff_size,
        num_layers=enc_layers, num_heads=num_heads, dropout=dropout, shape_aware=False,
    ).to(device)
    motion_decoder = ShapeAwareDecoder(
        nfeats=nfeats, latent_dim=latent_dim, ff_size=ff_size,
        num_layers=dec_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    return motion_encoder, text_encoder, motion_decoder
