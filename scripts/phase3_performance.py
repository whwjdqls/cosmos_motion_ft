"""Opt-in execution policies for Phase3 bridges; FP32 remains the default.

Parameters, attention edges and checkpoint keys do not change. Optional BF16
compute modes keep FP32 parameter/optimizer storage and FP32 normalization.
Masks belong to a single forward's BridgeMeta and are reused across its seven
bridges and activation-checkpoint recomputations, never across data batches.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys

import torch
import torch.nn.functional as F


def attention_bias(meta, device, conditioning_attention="directional"):
    from modality_bridge import temporally_local_attention_mask

    if conditioning_attention not in {"directional", "bidirectional"}:
        raise ValueError("Unknown bridge conditioning attention")
    cache_key = "_phase3_fp32_bias_temporally_local_no_shape_v3_" + conditioning_attention
    saved = getattr(meta, cache_key, None)
    if saved is not None:
        if saved.device != device:
            raise RuntimeError("Bridge metadata moved devices after its mask was cached")
        return saved
    allowed = temporally_local_attention_mask(meta, conditioning_attention).to(device)
    bias = torch.zeros_like(allowed, dtype=torch.float32)
    bias.masked_fill_(~allowed, torch.finfo(torch.float32).min)
    setattr(meta, cache_key, bias)
    return bias


def bridge_forward(self, gen, motion, meta):
    return self._forward_without_shape(gen, motion, meta,
        lambda g, m, frame_meta: _cached_bridge_forward(self, g, m, frame_meta))


def _cached_bridge_forward(self, gen, motion, meta):
    if gen.numel() == 0 or motion.numel() == 0:
        return gen, motion
    # A trainer may enter the cached policy inside an outer grouped policy.
    # In that case do not build a dense bias that the grouped kernel ignores.
    grouped = getattr(self._joint_attention, "__func__", None) in (grouped_attention_output, grouped_bf16_attention_output)
    bias = None if grouped else attention_bias(meta, gen.device, self.conditioning_attention)
    if getattr(self, "parameterization", "shared") != "shared":
        return self.forward_per_modality(gen, motion, meta, attn_bias=bias)
    x = torch.cat([gen, motion], dim=0)
    h = self.norm(x.float())
    q = self.q(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
    k = self.k(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
    v = self.v(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
    y = self._joint_attention(q, k, v, meta, attn_bias=bias).transpose(0, 1).reshape(x.shape[0], -1)
    y = self.o(y).to(x.dtype)
    ng = gen.shape[0]
    return (gen + self.gate_g.to(gen.dtype) * y[:ng],
            motion + self.gate_m.to(motion.dtype) * y[ng:])


def grouped_attention_output(self, q, k, v, meta, *, attn_bias=None):
    from modality_bridge import grouped_joint_attention
    if attn_bias is not None:
        raise ValueError("Grouped execution uses the canonical local mask; custom attention bias needs dense execution")
    return grouped_joint_attention(q, k, v, meta, self.conditioning_attention)


def grouped_bf16_attention_output(self, q, k, v, meta, *, attn_bias=None):
    from modality_bridge import grouped_joint_attention
    if attn_bias is not None:
        raise ValueError("Grouped execution uses the canonical local mask; custom attention bias needs dense execution")
    return grouped_joint_attention(q, k, v, meta, self.conditioning_attention, compute_dtype=torch.bfloat16)


@contextmanager
def execution_policy(*, cache_bridge_masks=True, checkpoint_motion=True, grouped_attention=False, precision=None):
    """Temporarily install a policy without changing historical source files."""
    from modality_bridge import LocalModalityBridge
    old_forward = LocalModalityBridge._forward_frames
    old_attention = LocalModalityBridge._joint_attention
    old_precision = getattr(LocalModalityBridge, "_execution_precision", None)
    precision = precision or old_precision or "fp32"
    if precision not in {"fp32", "bf16_attention", "bf16_bridge"}:
        raise ValueError("Unknown bridge compute precision")
    active_grouped = grouped_attention or old_attention in (grouped_attention_output, grouped_bf16_attention_output)
    if precision != "fp32" and not active_grouped:
        raise ValueError("BF16 compute modes require grouped_attention=True")
    changed = []
    if active_grouped:
        LocalModalityBridge._joint_attention = (grouped_attention_output if precision == "fp32"
                                                 else grouped_bf16_attention_output)
    forward = _cached_bridge_forward if cache_bridge_masks else getattr(old_forward, "_bridge_unwrapped_frames", old_forward)
    if precision == "bf16_bridge":
        def bf16_forward(self, gen, motion, meta):
            # Only the compact frame bridge is autocast; frozen specialists and
            # shape bypass are outside this scope. FP32-input LayerNorm stays FP32.
            with torch.autocast(device_type=gen.device.type, dtype=torch.bfloat16):
                return forward(self, gen, motion, meta)
        bf16_forward._bridge_unwrapped_frames = forward
        LocalModalityBridge._forward_frames = bf16_forward
    else:
        LocalModalityBridge._forward_frames = forward
    LocalModalityBridge._execution_precision = precision
    try:
        if not checkpoint_motion:
            # These modules own the forward closures used by the three current
            # experiments. Native video and bridge checkpointing remain enabled.
            for name in ("motion_expert_phase3_edge.model",
                         "motion_expert_phase3_edge_fourtask.model",
                         "motion_expert_phase3_edge_camfilm.backbone",
                         "motion_expert_phase3_edge_camtrajfilm.backbone"):
                module = sys.modules.get(name)
                if module is None:
                    continue
                original = module.checkpoint
                def selective(function, *args, _original=original, **kwargs):
                    if function.__name__ == "motion_step":
                        return function(*args)
                    return _original(function, *args, **kwargs)
                module.checkpoint = selective
                changed.append((module, original))
        yield
    finally:
        LocalModalityBridge._forward_frames = old_forward
        LocalModalityBridge._joint_attention = old_attention
        if old_precision is None:
            del LocalModalityBridge._execution_precision
        else:
            LocalModalityBridge._execution_precision = old_precision
        for module, original in changed:
            module.checkpoint = original
