"""llm2vec text-embedding cache (BONES-SEED POC) — caption -> pooled vector lookup.

A frozen lookup table over the precomputed cache
``/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt`` =
``{captions: [str]*N, features: [N, 4096] f32, meta}``. One pooled vector per caption;
the empty string ``""`` is row 0 (the null / unconditional prompt).

This is the in-context text source — it replaces the Cosmos reasoner H_R of the original
MotionExpert POC. No model, no GPU encode: ``batch()`` just gathers rows. Mirrors
``kimodo/model/cached_text/cached_encoder.py:CachedTextEncoder`` (KeyError on a miss; no live
encoder fallback by design).
"""
from __future__ import annotations

import os

import torch

DEFAULT_CACHE = "/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt"


class LLM2VecCache:
    def __init__(self, cache_path: str = DEFAULT_CACHE, device: str = "cuda"):
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"llm2vec cache not found: {cache_path}")
        try:
            blob = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, TypeError):
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)

        self.captions = list(blob["captions"])
        feats = blob["features"]
        if feats.dtype != torch.float32:  # forces materialization; cache is already f32
            feats = feats.to(torch.float32)
        if feats.dim() != 2 or feats.shape[0] != len(self.captions):
            raise ValueError(
                f"cache shape mismatch: features={tuple(feats.shape)} vs "
                f"#captions={len(self.captions)}"
            )
        self.features = feats                                   # [N, 4096] CPU (mmap)
        self.cap2idx = {c: i for i, c in enumerate(self.captions)}
        if "" not in self.cap2idx:
            raise ValueError("llm2vec cache has no '' (null) caption row")
        self.device = torch.device(device)
        self.dim = int(feats.shape[1])

    def __len__(self) -> int:
        return len(self.captions)

    def __contains__(self, caption: str) -> bool:
        return caption in self.cap2idx

    @torch.no_grad()
    def batch(self, captions) -> torch.Tensor:
        """list[str] (len B) -> pooled text tokens [B, 1, dim] on device."""
        idxs, missing = [], []
        for c in captions:
            i = self.cap2idx.get(c)
            if i is None:
                missing.append(c)
            else:
                idxs.append(i)
        if missing:
            raise KeyError(
                f"{len(missing)} caption(s) not in llm2vec cache (e.g. {missing[:3]}). "
                f"Re-run precompute_text_embeddings with the current caption set."
            )
        feats = self.features[torch.as_tensor(idxs, dtype=torch.long)]      # [B, dim]
        return feats.unsqueeze(1).to(self.device, non_blocking=True)         # [B, 1, dim]

    @torch.no_grad()
    def null(self, n: int = 1) -> torch.Tensor:
        """The '' (null) embedding as [n, 1, dim] — CFG-unconditional / dropped-text branch."""
        row = self.features[self.cap2idx[""]]                               # [dim]
        return row.view(1, 1, -1).expand(n, 1, -1).to(self.device)          # [n, 1, dim]
