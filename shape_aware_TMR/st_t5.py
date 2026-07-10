"""Frozen T5 text encoder (SnapMoGen evaluator's text embedder).

`google/t5-v1_1-base` encoder (frozen, safetensors), TOKEN-LEVEL output — replaces the
pooled-llm2vec cache of the v1 model. Live encoding kills the cache-coverage problem:
any caption (train, benchmark, generated-eval prompts) embeds natively.

encode(texts)   -> (emb (B,L,768) float32, mask (B,L) bool True=valid)   [dynamic padding]
sent_emb(...)   -> (B,768) masked-mean pooled sentence embedding (for InfoNCE false-negative
                   filtering, SnapMoGen threshold_selfsim=0.8).
"""
from __future__ import annotations

import torch

T5_NAME = "google/t5-v1_1-base"
T5_DIM = 768
MAX_TOKENS = 120  # SnapMoGen max_text_length


class FrozenT5:
    def __init__(self, device: str = "cuda", name: str = T5_NAME, max_tokens: int = MAX_TOKENS):
        from transformers import AutoTokenizer, T5EncoderModel
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = T5EncoderModel.from_pretrained(name, use_safetensors=True).to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.device = device
        self.max_tokens = max_tokens
        self.dim = self.model.config.d_model

    @torch.no_grad()
    def encode(self, texts):
        t = self.tok(list(texts), return_tensors="pt", padding=True,
                     truncation=True, max_length=self.max_tokens)
        ids = t["input_ids"].to(self.device)
        am = t["attention_mask"].to(self.device)
        emb = self.model(input_ids=ids, attention_mask=am).last_hidden_state.detach()  # (B,L,768)
        return emb.float(), am.bool()

    @torch.no_grad()
    def sent_emb(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pooling -> (B,768)."""
        m = mask.float().unsqueeze(-1)
        return (emb * m).sum(1) / m.sum(1).clamp(min=1.0)
