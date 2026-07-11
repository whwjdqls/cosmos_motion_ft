# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""THE MASK — the two-call joint (shared) attention primitive for the 3-way motion expert.

This is the single source of truth for the joint-attention contract used at every layer of
the MoT decoder stack. It reproduces Cosmos's ``two_way_attention``
(``cosmos_framework/model/vfm/mot/attention.py``) EXACTLY, with motion packed as an additional
``"full"`` segment alongside the generator.

Three token groups live in ONE packed per-sample sequence, in this order:

    [ und(sample0) | gen(sample0) | mot(sample0) | und(sample1) | gen(sample1) | mot(sample1) | ... ]

      - REASONER ("und",  causal mode) : attends ONLY to reasoner, lower-triangular, per-sample.
      - GENERATOR ("gen", full   mode) : attends densely over {reasoner, generator, motion}.
      - MOTION    ("mot", full   mode) : attends densely over {reasoner, generator, motion}.

The mask is realized as TWO varlen attention calls (NOT a dense [N,N] mask):

  (1) CAUSAL block:  Q = K = V = und rows                  -> reasoner causal-self-only.
  (2) FULL   block:  Q = full(=gen UNION mot), K = V = ALL  -> gen/mot attend over everything.

Then scatter both results back into a single packed output of shape [N_total].

CRITICAL INVARIANTS (this is *why* the mask is correct):
  * Reasoner rows are NEVER placed in ``full_idx``. Hence:
      - reasoner is unreachable as a QUERY in the full block, and
      - reasoner as a query exists ONLY in the causal block (where K/V = und only).
    => no R->G and no R->M. The reasoner stays a pure causal context encoder.
  * In the full block, K/V = ``k_all``/``v_all`` include the und rows
    => generator and motion can READ the reasoner (conditioning), but the reasoner
       never reads them back (asymmetric, exactly as intended).
  * ``full_idx`` contains BOTH gen and mot rows over a SHARED KV of all rows
    => generator<->motion is BIDIRECTIONAL (gen queries see mot keys; mot queries see gen keys).

The mask is keyed purely on attn_mode (causal vs full). The weight pathway (reasoner / _moe_gen /
_moe_motion) is keyed on SEPARATE role-index tensors handled by ``mot_joint_layer.py``. Keep the
two independent: motion shares the "full" attention behavior with the generator but flows through
its own trainable weights. This file does not know about weights at all.
"""
from __future__ import annotations

import torch
from torch import Tensor

from cosmos_framework.model.attention import attention
from cosmos_framework.model.attention.masks import CausalType


def joint_attention(
    q_all: Tensor,  # [N_total, n_heads, head_dim]
    k_all: Tensor,  # [N_total, n_kv_heads, head_dim]
    v_all: Tensor,  # [N_total, n_kv_heads, head_dim]
    und_idx: Tensor,  # LongTensor [N_und_total]   -> reasoner rows (causal queries)
    full_idx: Tensor,  # LongTensor [N_full_total]  -> (gen UNION mot) rows (full queries), packed order
    sample_offsets: Tensor,  # int32 [B+1] cu_seqlens over ALL rows (und+gen+mot) per sample
    max_und_len: int,  # max per-sample und length
    max_full_len: int,  # max per-sample full(=gen+mot) length
    max_sample_len: int,  # max per-sample total length
    und_offsets: Tensor,  # int32 [B+1] cu_seqlens over und rows per sample
    full_offsets: Tensor,  # int32 [B+1] cu_seqlens over full rows per sample
) -> Tensor:  # [N_total, n_heads*head_dim]
    """Two-call joint attention. Mirrors ``two_way_attention`` exactly.

    ``q_all``/``k_all``/``v_all`` are the packed projected states for the WHOLE sequence
    (all roles already routed through their respective weights upstream). ``k_all``/``v_all``
    may have fewer heads than ``q_all`` (GQA / MQA); the underlying kernel broadcasts.

    Returns the packed per-row attention output, flattened to ``[N_total, n_heads*head_dim]``.
    """
    n_total = q_all.shape[0]
    n_heads = q_all.shape[1]
    head_dim = q_all.shape[2]
    out = q_all.new_zeros(n_total, n_heads * head_dim)

    # ------------------------------------------------------------------ #
    # (1) CAUSAL block: reasoner attends ONLY to reasoner, per-sample,
    #     lower-triangular. Q == K == V == und rows, so seqlen_Q == seqlen_KV
    #     and we use CausalType.DontCare (top-left == bottom-right here).
    #     Skipped when there are no reasoner queries (empty und).
    # ------------------------------------------------------------------ #
    if und_idx.numel() > 0:
        cq = q_all[und_idx]  # [N_und_total, n_heads,    head_dim]
        ck = k_all[und_idx]  # [N_und_total, n_kv_heads, head_dim]
        cv = v_all[und_idx]  # [N_und_total, n_kv_heads, head_dim]

        causal_res = attention(
            cq[None],  # [1, N_und, n_heads,    head_dim]
            ck[None],  # [1, N_und, n_kv_heads, head_dim]
            cv[None],  # [1, N_und, n_kv_heads, head_dim]
            cumulative_seqlen_Q=und_offsets,
            cumulative_seqlen_KV=und_offsets,
            max_seqlen_Q=max_und_len,
            max_seqlen_KV=max_und_len,
            is_causal=True,
            causal_type=CausalType.DontCare,
        )  # [1, N_und, n_heads, head_dim]
        out[und_idx] = causal_res.squeeze(0).flatten(-2, -1)  # [N_und, n_heads*head_dim]

    # ------------------------------------------------------------------ #
    # (2) FULL block: every gen/motion query attends DENSELY over ALL rows
    #     (und+gen+mot) within its own sample. NO is_causal.
    #     Q = full(=gen UNION mot) rows; K = V = ALL rows.
    #     KV cu_seqlens = sample_offsets (the full sample, including und).
    #     Skipped when there are no full queries -- e.g. a sparse-depth PLAIN
    #     layer in text->motion has no gen tokens, so full_idx is empty and the
    #     real varlen kernel must NOT be called with max_seqlen_Q == 0.
    # ------------------------------------------------------------------ #
    if full_idx.numel() > 0:
        fq = q_all[full_idx]  # [N_full_total, n_heads, head_dim]

        full_res = attention(
            fq[None],  # [1, N_full, n_heads,    head_dim]  queries = gen UNION mot
            k_all[None],  # [1, N_all,  n_kv_heads, head_dim]  keys    = ALL rows (incl. und)
            v_all[None],  # [1, N_all,  n_kv_heads, head_dim]  values  = ALL rows (incl. und)
            cumulative_seqlen_Q=full_offsets,
            cumulative_seqlen_KV=sample_offsets,
            max_seqlen_Q=max_full_len,
            max_seqlen_KV=max_sample_len,
        )  # [1, N_full, n_heads, head_dim]
        out[full_idx] = full_res.squeeze(0).flatten(-2, -1)  # [N_full, n_heads*head_dim]

    # ------------------------------------------------------------------ #
    # (3) Output is the packed [N_total] tensor; und rows <- causal, full rows <- full,
    #     any row in neither set (none in practice) stays zero.
    # ------------------------------------------------------------------ #
    return out


def build_offsets(
    per_sample_und_lens: list[int],
    per_sample_full_lens: list[int],
    per_sample_total_lens: list[int],
    device: torch.device | str,
):
    """Compute the absolute row indices and varlen cu_seqlens for the joint attention.

    Given per-sample token counts for the packed ordering
        [und(s0), gen(s0), mot(s0), und(s1), gen(s1), mot(s1), ...]
    where ``full = gen UNION mot`` (in packed order, i.e. gen rows then mot rows of a sample),
    this returns the absolute row indices selecting und rows and full rows, plus the int32
    cumulative offsets the varlen kernel expects.

    NOTE: ``per_sample_full_lens[i]`` must equal (#gen + #mot) for sample i, and
    ``per_sample_total_lens[i]`` must equal (#und + #gen + #mot). The within-sample packed
    layout is assumed to be und block first, then the full block (gen then mot). This matches
    how ``joint_motion_model.py`` packs the sequence.

    Returns:
        und_idx       LongTensor [sum(und_lens)]  absolute row indices of reasoner rows
        full_idx      LongTensor [sum(full_lens)] absolute row indices of gen UNION mot rows
        und_offsets   int32 [B+1] cu_seqlens over und rows
        full_offsets  int32 [B+1] cu_seqlens over full rows
        sample_offsets int32 [B+1] cu_seqlens over ALL rows
        maxlens       (max_und_len, max_full_len, max_sample_len)
    """
    assert len(per_sample_und_lens) == len(per_sample_full_lens) == len(per_sample_total_lens), (
        "per-sample length lists must have equal length (one entry per sample)"
    )
    for u, f, t in zip(per_sample_und_lens, per_sample_full_lens, per_sample_total_lens):
        assert u + f == t, (
            f"und({u}) + full({f}) != total({t}); the full segment must be exactly the "
            "non-reasoner (gen+mot) rows of the sample."
        )

    und_idx_list: list[int] = []
    full_idx_list: list[int] = []
    cursor = 0  # running absolute row pointer into the packed sequence
    for u, f, t in zip(per_sample_und_lens, per_sample_full_lens, per_sample_total_lens):
        # und block: [cursor, cursor + u)
        und_idx_list.extend(range(cursor, cursor + u))
        # full block (gen then mot): [cursor + u, cursor + u + f) == [cursor + u, cursor + t)
        full_idx_list.extend(range(cursor + u, cursor + t))
        cursor += t

    def _cu(lens: list[int]) -> Tensor:
        out = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
        if lens:
            out[1:] = torch.tensor(lens, dtype=torch.int32, device=device).cumsum(0)
        return out

    und_idx = torch.tensor(und_idx_list, dtype=torch.long, device=device)
    full_idx = torch.tensor(full_idx_list, dtype=torch.long, device=device)
    und_offsets = _cu(per_sample_und_lens)
    full_offsets = _cu(per_sample_full_lens)
    sample_offsets = _cu(per_sample_total_lens)

    max_und_len = max(per_sample_und_lens) if per_sample_und_lens else 0
    max_full_len = max(per_sample_full_lens) if per_sample_full_lens else 0
    max_sample_len = max(per_sample_total_lens) if per_sample_total_lens else 0

    return (
        und_idx,
        full_idx,
        und_offsets,
        full_offsets,
        sample_offsets,
        (max_und_len, max_full_len, max_sample_len),
    )


# ====================================================================== #
# Self-test: CPU, tiny dims, pure-pytorch SDPA reference for `attention`. #
# ====================================================================== #
if __name__ == "__main__":
    import math
    import sys
    import types

    # ------------------------------------------------------------------ #
    # Replace cosmos_framework.model.attention.attention with a pure
    # pytorch varlen SDPA reference, so this test runs on CPU with no
    # framework attention kernels. We patch the symbol used in THIS module.
    # ------------------------------------------------------------------ #
    def _ref_attention(
        query,  # [1, S_Q, H, D]
        key,  # [1, S_KV, H_KV, D]
        value,  # [1, S_KV, H_KV, D]
        is_causal: bool = False,
        causal_type=None,
        scale: float | None = None,
        cumulative_seqlen_Q=None,
        cumulative_seqlen_KV=None,
        max_seqlen_Q=None,
        max_seqlen_KV=None,
        **kwargs,
    ):
        q = query[0]  # [S_Q, H, D]
        k = key[0]  # [S_KV, H_KV, D]
        v = value[0]  # [S_KV, H_KV, D]
        s_q, h, d = q.shape
        s_kv, h_kv, _ = k.shape
        if scale is None:
            scale = 1.0 / math.sqrt(d)
        # GQA broadcast
        if h_kv != h:
            assert h % h_kv == 0
            rep = h // h_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = q.new_zeros(s_q, h, d)
        cq = cumulative_seqlen_Q.tolist()
        ckv = cumulative_seqlen_KV.tolist()
        b = len(cq) - 1
        for i in range(b):
            qs, qe = cq[i], cq[i + 1]
            ks, ke = ckv[i], ckv[i + 1]
            if qe == qs:
                continue
            qb = q[qs:qe].transpose(0, 1)  # [H, lq, D]
            kb = k[ks:ke].transpose(0, 1)  # [H, lk, D]
            vb = v[ks:ke].transpose(0, 1)  # [H, lk, D]
            scores = torch.matmul(qb, kb.transpose(-1, -2)) * scale  # [H, lq, lk]
            if is_causal:
                lq = qe - qs
                lk = ke - ks
                # DontCare / TopLeft for lq==lk: mask q_idx < kv_idx
                qi = torch.arange(lq).view(lq, 1)
                ki = torch.arange(lk).view(1, lk)
                mask = qi < ki  # True -> masked
                scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            ob = torch.matmul(attn, vb)  # [H, lq, D]
            out[qs:qe] = ob.transpose(0, 1)
        return out.unsqueeze(0)  # [1, S_Q, H, D]

    # Patch the module-level `attention` reference used by joint_attention.
    _this = sys.modules[__name__]
    _this.attention = _ref_attention

    # ------------------------------------------------------------------ #
    # 1-sample sequence: 2 und + 1 gen + 2 mot = 5 rows.
    # Packed order: [und0, und1, gen0, mot0, mot1].
    # Expected dense mask (rows = query, cols = key), True = allowed:
    #   und0 : und0                 (lower-tri within und, no gen/mot)
    #   und1 : und0, und1
    #   gen0 : ALL 5
    #   mot0 : ALL 5
    #   mot1 : ALL 5
    # ------------------------------------------------------------------ #
    torch.manual_seed(0)
    N, H, D = 5, 2, 4
    und_lens = [2]
    full_lens = [3]  # 1 gen + 2 mot
    total_lens = [5]
    dev = "cpu"

    (und_idx, full_idx, und_offsets, full_offsets, sample_offsets, maxlens) = build_offsets(
        und_lens, full_lens, total_lens, dev
    )
    max_und_len, max_full_len, max_sample_len = maxlens

    assert und_idx.tolist() == [0, 1], und_idx.tolist()
    assert full_idx.tolist() == [2, 3, 4], full_idx.tolist()
    assert und_offsets.tolist() == [0, 2]
    assert full_offsets.tolist() == [0, 3]
    assert sample_offsets.tolist() == [0, 5]

    # CRITICAL invariant: reasoner rows are never queries in the full block.
    assert set(und_idx.tolist()).isdisjoint(set(full_idx.tolist())), "reasoner must not be in full_idx"

    q = torch.randn(N, H, D)
    k = torch.randn(N, H, D)
    v = torch.randn(N, H, D)

    out = joint_attention(
        q, k, v,
        und_idx=und_idx,
        full_idx=full_idx,
        sample_offsets=sample_offsets,
        max_und_len=max_und_len,
        max_full_len=max_full_len,
        max_sample_len=max_sample_len,
        und_offsets=und_offsets,
        full_offsets=full_offsets,
    )  # [N, H*D]
    assert out.shape == (N, H * D), out.shape

    # ------------------------------------------------------------------ #
    # Reference: brute-force dense masked attention over the SAME q/k/v
    # using the expected mask, and compare row-by-row.
    # ------------------------------------------------------------------ #
    expected_mask = torch.zeros(N, N, dtype=torch.bool)
    expected_mask[0, 0] = True               # und0 -> und0
    expected_mask[1, 0] = True               # und1 -> und0
    expected_mask[1, 1] = True               # und1 -> und1
    expected_mask[2, :] = True               # gen0 -> all
    expected_mask[3, :] = True               # mot0 -> all
    expected_mask[4, :] = True               # mot1 -> all

    scale = 1.0 / math.sqrt(D)
    ref = torch.zeros(N, H * D)
    qh = q.transpose(0, 1)  # [H, N, D]
    kh = k.transpose(0, 1)
    vh = v.transpose(0, 1)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # [H, N, N]
    big_mask = ~expected_mask  # True -> masked
    scores = scores.masked_fill(big_mask.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    oh = torch.matmul(attn, vh)  # [H, N, D]
    ref = oh.transpose(0, 1).reshape(N, H * D)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), (out - ref).abs().max()

    # Explicit mask-semantics checks (independent of the value comparison):
    #   und block must be lower-triangular and must NOT see gen/mot.
    #   We verify by perturbing forbidden keys and confirming und outputs are unchanged.
    v2 = v.clone()
    v2[2:] += 100.0  # blow up gen/mot value rows
    out2 = joint_attention(
        q, k, v2,
        und_idx=und_idx, full_idx=full_idx, sample_offsets=sample_offsets,
        max_und_len=max_und_len, max_full_len=max_full_len, max_sample_len=max_sample_len,
        und_offsets=und_offsets, full_offsets=full_offsets,
    )
    # und rows (0,1) must be INVARIANT to gen/mot value changes -> they never read gen/mot.
    assert torch.allclose(out2[und_idx], out[und_idx], atol=1e-5), "reasoner leaked into gen/mot!"
    # full rows (gen/mot) MUST change -> they do read gen/mot.
    assert not torch.allclose(out2[full_idx], out[full_idx], atol=1e-3), "gen/mot did not read gen/mot!"

    # und1 must read und0 (lower-tri, not diagonal-only): perturb und0 value, und1 output changes.
    v3 = v.clone()
    v3[0] += 100.0
    out3 = joint_attention(
        q, k, v3,
        und_idx=und_idx, full_idx=full_idx, sample_offsets=sample_offsets,
        max_und_len=max_und_len, max_full_len=max_full_len, max_sample_len=max_sample_len,
        und_offsets=und_offsets, full_offsets=full_offsets,
    )
    assert not torch.allclose(out3[1], out[1], atol=1e-3), "und1 did not attend to und0 (causal broken)"
    # und0 must NOT read und1 (causal): perturb und1 value, und0 output unchanged.
    v4 = v.clone()
    v4[1] += 100.0
    out4 = joint_attention(
        q, k, v4,
        und_idx=und_idx, full_idx=full_idx, sample_offsets=sample_offsets,
        max_und_len=max_und_len, max_full_len=max_full_len, max_sample_len=max_sample_len,
        und_offsets=und_offsets, full_offsets=full_offsets,
    )
    assert torch.allclose(out4[0], out[0], atol=1e-5), "und0 attended to future und1 (not causal!)"

    print("mot_joint_attention self-test PASSED: "
          "reasoner causal-self-only; gen/mot full over {und,gen,mot}; gen<->mot bidirectional.")
