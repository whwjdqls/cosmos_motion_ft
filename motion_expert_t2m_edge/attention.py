"""Two-role Edge attention: causal reasoner plus full motion queries."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cosmos_framework.model.attention import attention as attention_kernel
from cosmos_framework.model.attention.masks import CausalType


@dataclass(frozen=True)
class PackedLayout:
    text_idx: Tensor
    motion_idx: Tensor
    text_offsets: Tensor
    motion_offsets: Tensor
    sample_offsets: Tensor
    max_text_len: int
    max_motion_len: int
    max_sample_len: int


def _offsets(lengths: list[int], device: torch.device | str) -> Tensor:
    values = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    return torch.tensor(values, dtype=torch.int32, device=device)


def build_layout(
    text_lengths: list[int],
    motion_lengths: list[int],
    device: torch.device | str,
) -> PackedLayout:
    if len(text_lengths) != len(motion_lengths) or not text_lengths:
        raise ValueError("text_lengths and motion_lengths must be non-empty and have equal length")
    text_parts, motion_parts = [], []
    sample_lengths = []
    cursor = 0
    for nt, nm in zip(text_lengths, motion_lengths, strict=True):
        if nt <= 0 or nm <= 0:
            raise ValueError(f"each sample needs text and motion tokens, got text={nt} motion={nm}")
        text_parts.append(torch.arange(cursor, cursor + nt, device=device, dtype=torch.long))
        motion_parts.append(torch.arange(cursor + nt, cursor + nt + nm, device=device, dtype=torch.long))
        sample_lengths.append(nt + nm)
        cursor += nt + nm
    return PackedLayout(
        text_idx=torch.cat(text_parts),
        motion_idx=torch.cat(motion_parts),
        text_offsets=_offsets(text_lengths, device),
        motion_offsets=_offsets(motion_lengths, device),
        sample_offsets=_offsets(sample_lengths, device),
        max_text_len=max(text_lengths),
        max_motion_len=max(motion_lengths),
        max_sample_len=max(sample_lengths),
    )


def text_motion_attention(
    q_text: Tensor,
    k_text_causal: Tensor,
    v_text: Tensor,
    q_motion: Tensor,
    k_full_for_motion: Tensor,
    v_full: Tensor,
    layout: PackedLayout,
) -> tuple[Tensor, Tensor]:
    """Run the two calls required by the asymmetric Phase-2 mask.

    `k_text_causal` is the raw Edge reasoner K after its text-path norm and
    RoPE. `k_full_for_motion` contains the separately RMS-normalized reasoner K
    plus motion K. Keeping these tensors distinct is required by Edge Nemotron.
    """

    text_out = attention_kernel(
        q_text[None],
        k_text_causal[None],
        v_text[None],
        cumulative_seqlen_Q=layout.text_offsets,
        cumulative_seqlen_KV=layout.text_offsets,
        max_seqlen_Q=layout.max_text_len,
        max_seqlen_KV=layout.max_text_len,
        is_causal=True,
        causal_type=CausalType.DontCare,
    ).squeeze(0)
    motion_out = attention_kernel(
        q_motion[None],
        k_full_for_motion[None],
        v_full[None],
        cumulative_seqlen_Q=layout.motion_offsets,
        cumulative_seqlen_KV=layout.sample_offsets,
        max_seqlen_Q=layout.max_motion_len,
        max_seqlen_KV=layout.max_sample_len,
    ).squeeze(0)
    return text_out.flatten(-2, -1), motion_out.flatten(-2, -1)


def reasoner_causal_attention(q: Tensor, k: Tensor, v: Tensor, layout: PackedLayout) -> Tensor:
    return attention_kernel(
        q[None],
        k[None],
        v[None],
        cumulative_seqlen_Q=layout.text_offsets,
        cumulative_seqlen_KV=layout.text_offsets,
        max_seqlen_Q=layout.max_text_len,
        max_seqlen_KV=layout.max_text_len,
        is_causal=True,
        causal_type=CausalType.DontCare,
    ).squeeze(0).flatten(-2, -1)


__all__ = ["PackedLayout", "build_layout", "reasoner_causal_attention", "text_motion_attention"]
