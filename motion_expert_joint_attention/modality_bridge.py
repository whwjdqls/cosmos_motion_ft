"""Trainable generator-motion bridge modules.

The bridge is deliberately separate from the native Cosmos generator attention. It runs after the
native generator/reasoner and motion-expert updates, and its attention mask controls target-to-
condition information flow so clean conditioning tokens do not read noisy target tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


BRIDGE_PARAMETERIZATIONS = ("shared", "separate_norm", "separate_norm_out", "mmdit")


@dataclass
class BridgeMeta:
    """Per-sample metadata for local gen-motion bridge masks."""

    mode: str
    gen_frame: torch.Tensor       # [N_gen], -1 for non-video/image tokens
    gen_clean: torch.Tensor       # [N_gen] bool, True == clean condition
    motion_frame: torch.Tensor    # [N_mot], -1 for shape/non-frame token
    motion_clean: torch.Tensor    # [N_mot] bool, True == clean condition/shape
    gen_source_start: torch.Tensor  # [N_gen], inclusive source-frame interval, -1 if absent
    gen_source_end: torch.Tensor    # [N_gen], inclusive source-frame interval, -1 if absent


def temporally_local_attention_mask(meta: BridgeMeta, conditioning_attention: str) -> torch.Tensor:
    """One joint attention mask with causal Wan groups on *both* axes.

    Video group t contains its spatial tokens and motion frames {0} at t=0,
    otherwise {4*t-3, ..., 4*t}. Shape rows are removed before bridge
    normalization/projection/attention. If full metadata is inspected directly,
    non-frame rows have isolated self placeholders, never temporal edges.
    """
    if conditioning_attention not in {"directional", "bidirectional"}:
        raise ValueError("Unknown bridge conditioning attention")
    gen_group = meta.gen_frame
    motion_frame = meta.motion_frame
    motion_group = torch.div(motion_frame + 3, 4, rounding_mode="floor")
    gen_temporal = gen_group >= 0
    motion_temporal = motion_frame >= 0
    ng, nm = gen_group.numel(), motion_frame.numel()
    allowed = torch.eye(ng + nm, dtype=torch.bool, device=gen_group.device)

    # Same-modality keys are local even in bidirectional conditioning mode.
    allowed[:ng, :ng] |= ((gen_group[:, None] == gen_group[None, :])
                          & gen_temporal[:, None] & gen_temporal[None, :])
    allowed[ng:, ng:] |= ((motion_group[:, None] == motion_group[None, :])
                          & motion_temporal[:, None] & motion_temporal[None, :])

    aligned = ((gen_group[:, None] == motion_group[None, :])
               & gen_temporal[:, None] & motion_temporal[None, :])
    if conditioning_attention == "bidirectional":
        allowed[:ng, ng:] |= aligned
        allowed[ng:, :ng] |= aligned.T
    else:
        # Preserve role-based cross-modal conditioning. Both-noisy joint
        # targets may read each other; clean queries cannot read the other stream.
        allowed[:ng, ng:] |= (~meta.gen_clean.bool())[:, None] & aligned
        allowed[ng:, :ng] |= ((~meta.motion_clean.bool()) & motion_temporal)[:, None] & aligned.T
    return allowed


@dataclass
class GroupedAttentionPlan:
    query_indexes: torch.Tensor
    key_indexes: torch.Tensor
    restore_indexes: torch.Tensor
    bias: torch.Tensor


def grouped_attention_plan(meta: BridgeMeta, device, conditioning_attention: str) -> GroupedAttentionPlan:
    """Cache small rectangular groups on this forward's immutable metadata.

    Frame groups share one joint softmax, without shape keys. Unidentified
    rows in standalone metadata get isolated singleton placeholders. The
    single metadata-to-CPU copy happens on first use, not at every bridge or
    checkpoint recomputation. Plans contain no model parameters/state buffers.
    """
    if conditioning_attention not in {"directional", "bidirectional"}:
        raise ValueError("Unknown bridge conditioning attention")
    device = torch.device(device)
    cache_key = "_phase3_grouped_attention_no_shape_v2_" + conditioning_attention
    saved = getattr(meta, cache_key, None)
    if saved is not None:
        if saved.query_indexes.device != device:
            raise RuntimeError("Bridge metadata moved devices after grouping")
        return saved
    ng = meta.gen_frame.numel()
    frames, clean = torch.stack((torch.cat((meta.gen_frame, meta.motion_frame)),
        torch.cat((meta.gen_clean, meta.motion_clean)).long())).detach().cpu().tolist()
    plan = _cached_grouped_attention_plan(tuple(frames), tuple(clean), ng, device, conditioning_attention)
    setattr(meta, cache_key, plan)
    return plan


@lru_cache(maxsize=32)
def _cached_grouped_attention_plan(frames, clean, ng, device, conditioning_attention):
    """Reuse verified layouts across samples; cache is bounded and content keyed."""
    temporal, singletons = {}, []
    for i, frame in enumerate(frames):
        if frame < 0:
            singletons.append(i)
        else:
            group = frame if i < ng else (frame + 3) // 4
            temporal.setdefault(group, []).append(i)
    blocks = [(rows, rows) for _, rows in sorted(temporal.items())]
    for i in singletons:
        blocks.append(([i], [i]))
    nq, nk = max(len(q) for q, _ in blocks), max(len(k) for _, k in blocks)
    query_indexes = torch.zeros(len(blocks), nq, dtype=torch.long)
    key_indexes = torch.zeros(len(blocks), nk, dtype=torch.long)
    restore = torch.empty(len(frames), dtype=torch.long)
    allowed = torch.zeros(len(blocks), nq, nk, dtype=torch.bool)
    for block, (queries, keys) in enumerate(blocks):
        qi, ki = torch.tensor(queries), torch.tensor(keys)
        query_indexes[block, :len(queries)] = qi
        key_indexes[block, :len(keys)] = ki
        restore[qi] = block * nq + torch.arange(len(queries))
        same_modality = (qi[:, None] < ng) == (ki[None, :] < ng)
        cross_queries = torch.tensor([conditioning_attention == "bidirectional" or not clean[q] for q in queries])
        allowed[block, :len(queries), :len(keys)] = same_modality | cross_queries[:, None]
        # Padding query outputs are discarded. Keep their softmax finite.
        allowed[block, len(queries):, 0] = True
    bias = torch.zeros(len(blocks), 1, nq, nk, dtype=torch.float32)
    bias.masked_fill_(~allowed[:, None], torch.finfo(torch.float32).min)
    return GroupedAttentionPlan(query_indexes.to(device), key_indexes.to(device),
                                restore.to(device), bias.to(device))


def grouped_joint_attention(q, k, v, meta, conditioning_attention, *, compute_dtype=torch.float32):
    """Grouped local attention with FP32 softmax and optional BF16 matmuls."""
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Grouped attention compute dtype must be FP32 or BF16")
    plan = grouped_attention_plan(meta, q.device, conditioning_attention)
    heads, _, dim = q.shape
    blocks, nq = plan.query_indexes.shape
    nk = plan.key_indexes.shape[1]
    qi, ki = plan.query_indexes.flatten(), plan.key_indexes.flatten()
    with torch.autocast(device_type=q.device.type, enabled=False):
        qg = q.to(compute_dtype).index_select(1, qi).reshape(heads, blocks, nq, dim).transpose(0, 1)
        kg = k.to(compute_dtype).index_select(1, ki).reshape(heads, blocks, nk, dim).transpose(0, 1)
        vg = v.to(compute_dtype).index_select(1, ki).reshape(heads, blocks, nk, v.shape[-1]).transpose(0, 1)
        scores = torch.matmul(qg, kg.transpose(-2, -1)).float() * (dim ** -0.5)
        probabilities = F.softmax(scores + plan.bias, dim=-1)
        result = torch.matmul(probabilities.to(compute_dtype), vg).float()
        return result.transpose(0, 1).reshape(heads, blocks * nq, v.shape[-1]).index_select(1, plan.restore_indexes)


class LocalModalityBridge(nn.Module):
    """Small gated self-attention bridge over ``[G | M]`` tokens.

    Cross-modal attention is role-driven rather than task-name-driven. A noised target row may
    attend aligned rows from the other modality; clean cross-modal query rows never read targets.
    This gives the two historical directions as one-sided masks and joint video/motion targets as
    a bidirectional mask without changing bridge parameters.

    Same-modality and cross-modal temporal attention are restricted to causal
    Wan groups. Shape bypasses the bridge unchanged and is absent from Q/K/V.
    Zero-initialized residual gates make the bridge an exact no-op initially.

    ``parameterization`` selects the historical shared weights, independent
    modality norms, norms plus output projections, or ``mmdit`` (independent
    norms and Q/K/V/output projections). All variants use the same joint mask.
    """

    def __init__(self, hidden: int, num_heads: int, head_dim: int,
                 conditioning_attention: str = "directional", parameterization: str = "shared"):
        super().__init__()
        if conditioning_attention not in {"directional", "bidirectional"}:
            raise ValueError("Unknown bridge conditioning attention")
        self.conditioning_attention = conditioning_attention
        if parameterization not in BRIDGE_PARAMETERIZATIONS:
            raise ValueError("Unknown bridge parameterization")
        self.parameterization = parameterization
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        qkv_dim = self.num_heads * self.head_dim
        if parameterization == "shared":
            # Preserve historical initialization order and state-dict keys.
            self.norm = nn.LayerNorm(self.hidden)
        else:
            self.norm_g = nn.LayerNorm(self.hidden)
            self.norm_m = nn.LayerNorm(self.hidden)
        if parameterization == "mmdit":
            for role in ("g", "m"):
                for projection in ("q", "k", "v"):
                    setattr(self, f"{projection}_{role}", nn.Linear(self.hidden, qkv_dim, bias=False))
        else:
            self.q = nn.Linear(self.hidden, qkv_dim, bias=False)
            self.k = nn.Linear(self.hidden, qkv_dim, bias=False)
            self.v = nn.Linear(self.hidden, qkv_dim, bias=False)
        if parameterization in {"separate_norm_out", "mmdit"}:
            self.o_g = nn.Linear(qkv_dim, self.hidden, bias=False)
            self.o_m = nn.Linear(qkv_dim, self.hidden, bias=False)
        else:
            self.o = nn.Linear(qkv_dim, self.hidden, bias=False)
        self.gate_g = nn.Parameter(torch.zeros(()))
        self.gate_m = nn.Parameter(torch.zeros(()))

    def _local_pair_mask(self, gen_frame: torch.Tensor, motion_frame: torch.Tensor) -> torch.Tensor:
        """Return pairs aligned by the causal Wan-VAE 4x temporal layout.

        For a 97-frame clip the 25 latent frames represent source-frame groups
        ``{0}, {1..4}, {5..8}, ..., {93..96}``.  Equivalently, source motion
        frame ``m`` belongs to latent frame ``ceil(m / 4)``.  The historical
        ``[4*g, 4*g+3]`` rule was shifted by three frames, assigned frames 1..3
        to latent 0, and left source frames 94..96 without a bridge edge.
        """
        if gen_frame.numel() == 0 or motion_frame.numel() == 0:
            return torch.zeros((gen_frame.numel(), motion_frame.numel()), device=gen_frame.device, dtype=torch.bool)
        gf = gen_frame.view(-1, 1)
        mf = motion_frame.view(1, -1)
        valid = (gf >= 0) & (mf >= 0)
        return valid & (torch.div(mf + 3, 4, rounding_mode="floor") == gf)

    def _source_interval_pair_mask(self, meta: BridgeMeta) -> torch.Tensor:
        """Pair generator rows with motion frames in their physical source-frame interval.

        Video latent intervals are ``{0}, {1..4}, ...``. Camera action ``i`` spans source
        transition ``i -> i+1`` and therefore uses interval ``[i, i+1]``. The union preserves
        the historical video locality while adding direct local action-motion edges.
        """
        starts = meta.gen_source_start.view(-1, 1)
        ends = meta.gen_source_end.view(-1, 1)
        motion = meta.motion_frame.view(1, -1)
        valid = (starts >= 0) & (ends >= starts) & (motion >= 0)
        return valid & (motion >= starts) & (motion <= ends)

    def _attention_mask(self, meta: BridgeMeta) -> torch.Tensor:
        return temporally_local_attention_mask(meta, self.conditioning_attention)

    def _joint_attention(self, q, k, v, meta, *, attn_bias=None):
        """Dense FP32 reference; the execution policy can select grouped math."""
        if attn_bias is None:
            allowed = self._attention_mask(meta).to(q.device)
            attn_bias = torch.zeros_like(allowed, dtype=torch.float32)
            attn_bias.masked_fill_(~allowed, torch.finfo(torch.float32).min)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (self.head_dim ** -0.5)
        probabilities = F.softmax(scores + attn_bias.unsqueeze(0), dim=-1)
        return torch.matmul(probabilities, v)

    def forward(
        self,
        gen: torch.Tensor,
        motion: torch.Tensor,
        meta: BridgeMeta,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_without_shape(gen, motion, meta, self._forward_frames)

    def _forward_without_shape(self, gen, motion, meta, forward):
        """Remove non-frame motion rows before any norm/QKV, then restore verbatim.

        Cache only immutable metadata/indices, not activations. The same compact
        metadata (and its attention caches) is reused by all seven bridges and
        activation-checkpoint recomputations in this model forward.
        """
        if gen.numel() == 0 or motion.numel() == 0:
            return gen, motion
        cache_key = '_phase3_frame_only_motion_v1'
        cached = getattr(meta, cache_key, None)
        if cached is None:
            indexes = torch.nonzero(meta.motion_frame >= 0, as_tuple=True)[0]
            frame_meta = BridgeMeta(meta.mode, meta.gen_frame, meta.gen_clean,
                meta.motion_frame.index_select(0, indexes),
                meta.motion_clean.index_select(0, indexes),
                meta.gen_source_start, meta.gen_source_end)
            cached = (indexes, frame_meta)
            setattr(meta, cache_key, cached)
        indexes, frame_meta = cached
        if indexes.device != motion.device:
            raise RuntimeError('Bridge metadata moved devices after shape exclusion')
        if indexes.numel() == motion.shape[0]:
            return forward(gen, motion, meta)
        if indexes.numel() == 0:
            return gen, motion
        gout, mout = forward(gen, motion.index_select(0, indexes), frame_meta)
        return gout, motion.index_copy(0, indexes, mout)

    def _forward_frames(self, gen, motion, meta):
        """Attention/residual implementation for frame-only bridge inputs."""
        if gen.numel() == 0 or motion.numel() == 0:
            return gen, motion
        if self.parameterization != "shared":
            return self.forward_per_modality(gen, motion, meta)
        x = torch.cat([gen, motion], dim=0)
        h = self.norm(x.float())
        q = self.q(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        y = self._joint_attention(q, k, v, meta).transpose(0, 1).reshape(x.shape[0], -1)
        y = self.o(y).to(x.dtype)

        ng = gen.shape[0]
        yg, ym = y[:ng], y[ng:]
        return gen + self.gate_g.to(gen.dtype) * yg, motion + self.gate_m.to(motion.dtype) * ym

    def forward_per_modality(self, gen, motion, meta, *, attn_bias=None):
        """Private modality parameters, with one joint attention over [G | M].

        The same helper is used by the cached-mask execution policy. Only the
        source of the bias changes; projection routing and gradients do not.
        """
        ng, nm = gen.shape[0], motion.shape[0]
        hg, hm = self.norm_g(gen.float()), self.norm_m(motion.float())
        if self.parameterization == "mmdit":
            q = torch.cat((self.q_g(hg), self.q_m(hm)))
            k = torch.cat((self.k_g(hg), self.k_m(hm)))
            v = torch.cat((self.v_g(hg), self.v_m(hm)))
        else:
            h = torch.cat((hg, hm))
            q, k, v = self.q(h), self.k(h), self.v(h)
        q = q.view(ng + nm, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(ng + nm, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(ng + nm, self.num_heads, self.head_dim).transpose(0, 1)
        y = self._joint_attention(q, k, v, meta, attn_bias=attn_bias).transpose(0, 1).reshape(ng + nm, -1)
        if self.parameterization in {"separate_norm_out", "mmdit"}:
            yg, ym = self.o_g(y[:ng]), self.o_m(y[ng:])
        else:
            y = self.o(y)
            yg, ym = y[:ng], y[ng:]
        return (gen + self.gate_g.to(gen.dtype) * yg.to(gen.dtype),
                motion + self.gate_m.to(motion.dtype) * ym.to(motion.dtype))
