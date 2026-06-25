"""Phase 1: frozen Cosmos-3 Nano reasoner → per-token hidden states H_R.

Loads the Nano network exactly as `train_motion_ft.py` does (build_network + materialize +
load_gen_weights — which loads BOTH the und/reasoner tower AND the gen pathway from the
`transformer/` shards), freezes everything, and exposes `encode_text` returning the
POST-transformer, post-norm reasoner hidden states (the semantic features), NOT the
pre-transformer `embed_tokens` lookup.

H_R = net.language_model.model.reasoner_forward(input_ids, cache=ReasonerKVCache.empty(L))
    → [B, T_text, 4096]   (run per-prompt to avoid padding-in-causal-attn correctness issues,
                           then padded to a batch with a key-padding mask)

Run the built-in sanity check (cosmos env, 1 GPU):
  ssh <node>; cd motion_expert; PYTHONPATH=/home/jungbin_cho/cosmos-framework \
    python reasoner.py
"""
from __future__ import annotations

import os
import sys

import torch

# import the trainer's load/build/tokenize/freeze patterns (reuse, no edits)
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft")
from train_motion_ft import (  # noqa: E402
    SPECIAL_TOKENS,
    build_network,
    build_text_processor,
    freeze_all,
    load_gen_weights,
    materialize,
)
from cosmos_framework.model.vfm.mot.unified_mot import ReasonerKVCache  # noqa: E402

D_REASONER = 4096


class FrozenReasoner:
    """Frozen Nano reasoner; text → H_R. All base params frozen, eval, no_grad."""

    def __init__(self, dtype=torch.bfloat16, device="cuda", verbose=True):
        self.dtype = dtype
        self.device = device
        net, _ = build_network(tiny=False, dtype=dtype)
        net = materialize(net, dtype)
        load_gen_weights(net, verbose=verbose)
        freeze_all(net)
        net.eval()
        self.net = net
        self.model = net.language_model.model           # the *TextModel (reasoner_forward lives here)
        self.n_layers = len(self.model.layers)
        self.proc = build_text_processor()
        self._null_H: torch.Tensor | None = None
        if verbose:
            n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"[reasoner] layers={self.n_layers} D={D_REASONER} trainable_base_params={n_train} (expect 0)")

    def _ids(self, text: str) -> list[int]:
        ids = self.proc.tokenize_text(text)
        if len(ids) == 0:
            ids = [SPECIAL_TOKENS["eos_token_id"]]
        return ids

    @torch.no_grad()
    def _one(self, text: str) -> torch.Tensor:
        """Single prompt → H_R [T_text, 4096] (post-transformer, post-norm)."""
        input_ids = torch.tensor([self._ids(text)], device=self.device)   # [1, T]
        cache = ReasonerKVCache.empty(self.n_layers)
        H = self.model.reasoner_forward(input_ids, cache=cache)            # [1, T, 4096]
        return H[0]

    @torch.no_grad()
    def encode_text(self, prompts: list[str]):
        """list[str] → (H_R [B,Tmax,4096], key_padding_mask [B,Tmax] with True=PAD).

        Per-prompt forward (no padding inside the causal reasoner), then right-pad the
        resulting variable-length H_R into a batch tensor + key-padding mask for the
        MotionExpert cross-attention.
        """
        Hs = [self._one(p) for p in prompts]
        Tmax = max(h.shape[0] for h in Hs)
        B = len(Hs)
        H = torch.zeros(B, Tmax, D_REASONER, device=self.device, dtype=Hs[0].dtype)
        pad = torch.ones(B, Tmax, dtype=torch.bool, device=self.device)   # True = pad
        for i, h in enumerate(Hs):
            t = h.shape[0]
            H[i, :t] = h
            pad[i, :t] = False
        return H, pad

    @torch.no_grad()
    def null_H(self) -> torch.Tensor:
        """Cached empty-prompt H_R [T0, 4096] for classifier-free guidance."""
        if self._null_H is None:
            self._null_H = self._one("")
        return self._null_H


# --------------------------------------------------------------------------------------
# Sanity check: distinct prompts must give distinct (non-zero, finite) H_R.
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    r = FrozenReasoner()
    prompts = [
        "a person walks forward",
        "a person sits down on a chair",
        "a person walks forward",          # repeat of #0 → should match #0
    ]
    H, pad = r.encode_text(prompts)
    print(f"[sanity] H_R {tuple(H.shape)} dtype={H.dtype} finite={bool(torch.isfinite(H).all())}")

    # mean-pool over valid tokens, compare cosine
    def pooled(i):
        m = (~pad[i]).float().unsqueeze(-1)
        return (H[i].float() * m).sum(0) / m.sum().clamp(min=1)

    v0, v1, v2 = pooled(0), pooled(1), pooled(2)
    c_diff = F.cosine_similarity(v0, v1, dim=0).item()
    c_same = F.cosine_similarity(v0, v2, dim=0).item()
    print(f"[sanity] cos(walk, sit)   = {c_diff:+.4f}  (distinct prompts → expect < 0.99)")
    print(f"[sanity] cos(walk, walk)  = {c_same:+.4f}  (identical prompt → expect ~1.00)")
    print(f"[sanity] H_R abs-mean={H.float().abs().mean():.4f} (expect >0, i.e. weights loaded)")
    null = r.null_H()
    print(f"[sanity] null H_R {tuple(null.shape)} cached for CFG")
    ok = bool(torch.isfinite(H).all()) and c_diff < 0.99 and c_same > 0.99 and H.float().abs().mean() > 1e-3
    print(f"[sanity] PASS={ok}")
