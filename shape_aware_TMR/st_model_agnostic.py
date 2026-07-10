"""Shape-agnostic TMR model.

This is the same dual-encoder VAE recipe as ``st_model.py`` but removes all
neutral-joint conditioning:

- motion encoder sequence is ``[mu_tok, logvar_tok, motion_1..T]``
- text encoder is unchanged
- decoder memory is ``[z]`` only
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import repeat

from kimodo.model.tmr import ACTORStyleEncoder, PositionalEncoding


class AgnosticMotionEncoder(nn.Module):
    """ACTOR-style motion encoder without a shape token."""

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
    ) -> None:
        super().__init__()
        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)
        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation, batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                     enable_nested_tensor=False)

    def forward(self, x_dict: Dict) -> torch.Tensor:
        x = self.projection(x_dict["x"])
        mask = x_dict["mask"]
        bs = len(x)
        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)
        token_mask = torch.ones((bs, self.nbtokens), dtype=torch.bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, : self.nbtokens]


class AgnosticDecoder(nn.Module):
    """ACTOR-style transformer decoder with memory = [z]."""

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.nfeats = nfeats
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout, batch_first=True)
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation, batch_first=True,
        )
        self.seqTransDecoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.final_layer = nn.Linear(latent_dim, nfeats)

    def forward(self, z: torch.Tensor, T: int) -> torch.Tensor:
        B = z.shape[0]
        memory = z.unsqueeze(1)
        tgt = torch.zeros(B, T, self.latent_dim, device=z.device, dtype=z.dtype)
        tgt = self.sequence_pos_encoding(tgt)
        out = self.seqTransDecoder(tgt=tgt, memory=memory)
        return self.final_layer(out)


def build_agnostic_tmr(
    nfeats: int = 186,
    llm_dim: int = 4096,
    latent_dim: int = 256,
    ff_size: int = 1024,
    enc_layers: int = 6,
    dec_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
    device: str = "cuda",
) -> Tuple[AgnosticMotionEncoder, ACTORStyleEncoder, AgnosticDecoder]:
    motion_encoder = AgnosticMotionEncoder(
        nfeats=nfeats, vae=True, latent_dim=latent_dim, ff_size=ff_size,
        num_layers=enc_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    text_encoder = ACTORStyleEncoder(
        motion_rep=None, llm_shape=(1, llm_dim), vae=True, latent_dim=latent_dim,
        ff_size=ff_size, num_layers=enc_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    motion_decoder = AgnosticDecoder(
        nfeats=nfeats, latent_dim=latent_dim, ff_size=ff_size,
        num_layers=dec_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    return motion_encoder, text_encoder, motion_decoder
