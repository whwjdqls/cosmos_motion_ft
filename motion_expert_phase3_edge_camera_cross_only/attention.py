"""Remove same-modality bridge keys; preserve parameters and Wan groups.

Rows with no valid cross keys produce exact zero updates. A temporary softmax
placeholder is numerical bookkeeping only, never a self-attention edge.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
import modality_bridge as bridge

_joint_mask = bridge.temporally_local_attention_mask


def cross_attention_mask(meta, conditioning_attention):
    allowed = _joint_mask(meta, conditioning_attention)
    ng = meta.gen_frame.numel()
    allowed[:ng, :ng] = False
    allowed[ng:, ng:] = False
    return allowed


def safe_attention(q, k, v, allowed, compute_dtype):
    active = allowed.any(dim=-1, keepdim=True)
    # Avoid -inf-only rows without enabling a model-visible attention edge.
    safe = allowed.clone()
    safe[..., 0] |= ~active.squeeze(-1)
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = torch.matmul(q.to(compute_dtype), k.to(compute_dtype).transpose(-2, -1)).float() * q.shape[-1] ** -.5
        probabilities = F.softmax(scores.masked_fill(~safe, float('-inf')), dim=-1)
        probabilities = probabilities * active
        return torch.matmul(probabilities.to(compute_dtype), v.to(compute_dtype)).float()


def dense_attention(self, q, k, v, meta, *, attn_bias=None):
    allowed = cross_attention_mask(meta, self.conditioning_attention).to(q.device)
    if attn_bias is not None:
        allowed = allowed & (attn_bias == 0)
    return safe_attention(q, k, v, allowed.unsqueeze(0), torch.float32)


def grouped_cross_attention(q, k, v, meta, conditioning_attention, *, compute_dtype=torch.float32):
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Unsupported attention dtype')
    plan = bridge.grouped_attention_plan(meta, q.device, conditioning_attention)
    heads, _, dim = q.shape
    blocks, nq = plan.query_indexes.shape
    nk = plan.key_indexes.shape[1]
    qi, ki = plan.query_indexes.flatten(), plan.key_indexes.flatten()
    cache_key = '_cross_only_allowed_v1_' + conditioning_attention
    allowed = getattr(meta, cache_key, None)
    if allowed is None:
        ng = meta.gen_frame.numel()
        cross = (plan.query_indexes[:, :, None] < ng) != (plan.key_indexes[:, None, :] < ng)
        # The canonical grouped plan already enforces locality, clean-query
        # directionality and padding. Only same-modality keys are removed.
        allowed = (plan.bias == 0) & cross[:, None]
        setattr(meta, cache_key, allowed)
    qg = q.index_select(1, qi).reshape(heads, blocks, nq, dim).transpose(0, 1)
    kg = k.index_select(1, ki).reshape(heads, blocks, nk, dim).transpose(0, 1)
    vg = v.index_select(1, ki).reshape(heads, blocks, nk, v.shape[-1]).transpose(0, 1)
    result = safe_attention(qg, kg, vg, allowed, compute_dtype)
    return result.transpose(0, 1).reshape(heads, blocks * nq, v.shape[-1]).index_select(1, plan.restore_indexes)


@contextmanager
def cross_only_attention():
    old_mask, old_grouped = bridge.temporally_local_attention_mask, bridge.grouped_joint_attention
    old_dense = bridge.LocalModalityBridge._joint_attention
    bridge.temporally_local_attention_mask = cross_attention_mask
    bridge.grouped_joint_attention = grouped_cross_attention
    bridge.LocalModalityBridge._joint_attention = dense_attention
    try:
        yield
    finally:
        bridge.temporally_local_attention_mask = old_mask
        bridge.grouped_joint_attention = old_grouped
        bridge.LocalModalityBridge._joint_attention = old_dense
