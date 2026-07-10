"""Shape-aware TMR v2 — SnapMoGen-evaluator composition (cached llm2vec text).

Same transformer family as v1, composed per SnapMoGen `config/evaluator.yaml`:
  - motion encoder: our ShapeAwareMotionEncoder UNCHANGED (186-d TMRMotionRep in,
    [mu, logvar, SHAPE_TOK, motion...] sequence, 6 layers, latent 256) — the input motion
    representation (raw joint positions -> 186-d) does NOT change.
  - text encoder: kimodo ACTORStyleEncoder over the frozen pooled llm2vec vector
    (B,1,4096) — the SAME tower/input as v1 (per user: use the cache, no live T5).
    The encoder is sequence-generic, so token-level input also works if ever needed.
  - motion decoder: shape-aware (memory [z, shape_tok]) — 6 layers per SnapMoGen's
    latent_decoder (v1 used 4).

The shape token is NEVER dropped, on either side it appears (encoder + decoder).
"""
from __future__ import annotations

from typing import Tuple

from kimodo.model.tmr import ACTORStyleEncoder

from st_model import ShapeAwareMotionEncoder, ShapeAwareDecoder


def build_shape_tmr_v2(
    nfeats: int = 186,
    text_dim: int = 4096,
    latent_dim: int = 256,
    ff_size: int = 1024,
    enc_layers: int = 6,
    dec_layers: int = 6,
    num_heads: int = 4,
    dropout: float = 0.1,
    device: str = "cuda",
) -> Tuple[ShapeAwareMotionEncoder, ACTORStyleEncoder, ShapeAwareDecoder]:
    """(motion_encoder, text_encoder, motion_decoder) — all trained from scratch."""
    motion_encoder = ShapeAwareMotionEncoder(
        nfeats=nfeats, vae=True, latent_dim=latent_dim, ff_size=ff_size,
        num_layers=enc_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    # token-level text: ACTORStyleEncoder handles (B, L, text_dim) + per-token mask;
    # llm_shape only fixes nfeats.
    text_encoder = ACTORStyleEncoder(
        motion_rep=None, llm_shape=(1, text_dim), vae=True, latent_dim=latent_dim,
        ff_size=ff_size, num_layers=enc_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    motion_decoder = ShapeAwareDecoder(
        nfeats=nfeats, latent_dim=latent_dim, ff_size=ff_size,
        num_layers=dec_layers, num_heads=num_heads, dropout=dropout,
    ).to(device)
    return motion_encoder, text_encoder, motion_decoder
