"""Opt-in execution-only optimizations for the existing FP32 Phase-3 bridge.

No parameters, tensor precision, attention edges, or checkpoint keys change.
Masks belong to a single forward's BridgeMeta and are reused across its seven
bridges and activation-checkpoint recomputations, never across data batches.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys

import torch
import torch.nn.functional as F


def attention_bias(meta, device):
    saved = getattr(meta, "_phase3_fp32_bias_v1", None)
    if saved is not None:
        if saved.device != device:
            raise RuntimeError("Bridge metadata moved devices after its mask was cached")
        return saved
    ng, nm = meta.gen_frame.numel(), meta.motion_frame.numel()
    allowed = torch.eye(ng + nm, dtype=torch.bool, device=device)
    allowed[:ng, :ng] = True
    allowed[ng:, ng:] = True
    starts, ends = meta.gen_source_start[:, None], meta.gen_source_end[:, None]
    frames = meta.motion_frame[None, :]
    local = (starts >= 0) & (ends >= starts) & (frames >= 0) & (frames >= starts) & (frames <= ends)
    gen_noisy = ~meta.gen_clean.bool()
    motion_noisy = (~meta.motion_clean.bool()) & (meta.motion_frame >= 0)
    # Unconditional unions equal the original conditional unions, including
    # empty/false rows, and avoid CUDA-to-host any()/bool() synchronization.
    allowed[ng:, :ng] |= local.T & motion_noisy[:, None]
    allowed[:ng, ng:] |= local & gen_noisy[:, None] & (meta.motion_frame >= 0)[None, :]
    allowed[:ng, ng:] |= gen_noisy[:, None] & (meta.motion_frame < 0)[None, :]
    bias = torch.zeros_like(allowed, dtype=torch.float32)
    bias.masked_fill_(~allowed, torch.finfo(torch.float32).min)
    meta._phase3_fp32_bias_v1 = bias
    return bias


def bridge_forward(self, gen, motion, meta):
    if gen.numel() == 0 or motion.numel() == 0:
        return gen, motion
    x = torch.cat([gen, motion], dim=0)
    h = self.norm(x.float())
    q = self.q(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
    k = self.k(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
    v = self.v(h).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
    bias = attention_bias(meta, x.device)
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (self.head_dim ** -0.5)
    attn = F.softmax(attn + bias.unsqueeze(0), dim=-1)
    y = torch.matmul(attn, v).transpose(0, 1).reshape(x.shape[0], -1)
    y = self.o(y).to(x.dtype)
    ng = gen.shape[0]
    return (gen + self.gate_g.to(gen.dtype) * y[:ng],
            motion + self.gate_m.to(motion.dtype) * y[ng:])


@contextmanager
def execution_policy(*, cache_bridge_masks=True, checkpoint_motion=True):
    """Temporarily install a policy without changing historical source files."""
    from modality_bridge import LocalModalityBridge
    old_forward = LocalModalityBridge.forward
    changed = []
    if cache_bridge_masks:
        LocalModalityBridge.forward = bridge_forward
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
        LocalModalityBridge.forward = old_forward
        for module, original in changed:
            module.checkpoint = original
