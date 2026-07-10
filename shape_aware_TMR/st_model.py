"""Shape-aware TMR model.

Dual-encoder VAE TMR (ACTOR-style) whose MOTION encoder and decoder are conditioned on the
actor's skeleton (centered ``neutral_joints (30,3)`` — the identical conditioning the
shape-aware generation model uses), so the shared text<->motion latent captures motion
*semantics* rather than body size (the text encoder is shape-free; InfoNCE pushes shape
information out of z; the decoder gets shape back so reconstruction doesn't need z to
store skeleton geometry).

- ``ShapeAwareMotionEncoder`` — copy of kimodo ``ACTORStyleEncoder``
  (kimodo/model/tmr.py:58-129) with one addition: sequence
  ``[mu_tok, logvar_tok, SHAPE_TOK, proj(motion)_1..T]`` (shape token always valid).
- text encoder — kimodo ``ACTORStyleEncoder(llm_shape=(1,4096), vae=True)`` imported
  UNMODIFIED, trained from the frozen llm2vec cache (TAP's arrangement).
- ``ShapeAwareDecoder`` — copy of TAP ``ACTORStyleDecoder`` (tmr_g1/model/tmr_model.py:25-70)
  with memory ``[z, shape_tok]`` (2 memory tokens instead of 1).
- ``ShapeEncoder`` — the generation model's shape MLP (motion_expert/motion_expert.py:48-60).

Dims per the TAP run config: latent 256, encoder layers 6, decoder layers 4, heads 4,
ff 1024, dropout 0.1, gelu, VAE. The shape token is NEVER dropped.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import repeat

from kimodo.model.tmr import ACTORStyleEncoder, PositionalEncoding


class ShapeEncoder(nn.Module):
    """(B,30,3) centered neutral_joints -> (B,1,d). Encodes per-actor bone sizes.

    Same architecture as the generation model's ShapeEncoder so both consume the
    identical shape conditioning.
    """

    def __init__(self, d: int = 256, n_joints: int = 30):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_joints * 3, d), nn.GELU(),
            nn.LayerNorm(d), nn.Linear(d, d),
        )

    def forward(self, neutral_joints: torch.Tensor) -> torch.Tensor:
        B = neutral_joints.shape[0]
        return self.mlp(neutral_joints.reshape(B, -1)).unsqueeze(1)  # (B,1,d)


class ShapeAwareMotionEncoder(nn.Module):
    """ACTOR-style motion encoder with an in-context shape token.

    forward(x_dict) where x_dict = {"x": (B,T,nfeats), "mask": (B,T) True=VALID,
    "neutral_joints": (B,30,3)}. Returns (B,2,latent) = (mu, logvar) tokens.
    """

    def __init__(
        self,
        nfeats: int,
        vae: bool = True,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        n_joints: int = 30,
    ) -> None:
        super().__init__()
        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)
        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
        self.shape_enc = ShapeEncoder(latent_dim, n_joints)
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation, batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                     enable_nested_tensor=False)

    def forward(self, x_dict: Dict) -> torch.Tensor:
        x = x_dict["x"]
        mask = x_dict["mask"]                     # (B,T) True = valid
        nj = x_dict["neutral_joints"]             # (B,30,3) centered

        x = self.projection(x)
        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        shape_tok = self.shape_enc(nj)                          # (B,1,latent)
        xseq = torch.cat((tokens, shape_tok, x), 1)             # (B, nbtokens+1+T, latent)

        cond_mask = torch.ones((bs, self.nbtokens + 1), dtype=torch.bool, device=device)
        aug_mask = torch.cat((cond_mask, mask), 1)

        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, : self.nbtokens]                        # (B, 2, latent) = (mu, logvar)


class ShapeAwareDecoder(nn.Module):
    """ACTOR-style transformer decoder conditioned on (z, shape).

    memory = [z, shape_tok] (2 tokens); query = positional encoding per output frame.
    forward(z (B,latent), neutral_joints (B,30,3), T) -> (B,T,nfeats).
    """

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        n_joints: int = 30,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.nfeats = nfeats
        self.shape_enc = ShapeEncoder(latent_dim, n_joints)     # decoder's own instance
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout, batch_first=True)
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation, batch_first=True,
        )
        self.seqTransDecoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.final_layer = nn.Linear(latent_dim, nfeats)

    def forward(self, z: torch.Tensor, neutral_joints: torch.Tensor, T: int) -> torch.Tensor:
        B = z.shape[0]
        shape_tok = self.shape_enc(neutral_joints)              # (B,1,latent)
        memory = torch.cat([z.unsqueeze(1), shape_tok], dim=1)  # (B,2,latent)
        tgt = torch.zeros(B, T, self.latent_dim, device=z.device, dtype=z.dtype)
        tgt = self.sequence_pos_encoding(tgt)
        out = self.seqTransDecoder(tgt=tgt, memory=memory)      # (B,T,latent)
        return self.final_layer(out)


def build_shape_tmr(
    nfeats: int = 186,
    llm_dim: int = 4096,
    latent_dim: int = 256,
    ff_size: int = 1024,
    enc_layers: int = 6,
    dec_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
    device: str = "cuda",
) -> Tuple[ShapeAwareMotionEncoder, ACTORStyleEncoder, ShapeAwareDecoder]:
    """(motion_encoder, text_encoder, motion_decoder) — all trained from scratch."""
    motion_encoder = ShapeAwareMotionEncoder(
        nfeats=nfeats, vae=True, latent_dim=latent_dim, ff_size=ff_size,
        num_layers=enc_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    text_encoder = ACTORStyleEncoder(
        motion_rep=None, llm_shape=(1, llm_dim), vae=True, latent_dim=latent_dim,
        ff_size=ff_size, num_layers=enc_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    motion_decoder = ShapeAwareDecoder(
        nfeats=nfeats, latent_dim=latent_dim, ff_size=ff_size,
        num_layers=dec_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    return motion_encoder, text_encoder, motion_decoder
