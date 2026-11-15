"""Process-scoped global bridge topology; native specialist attention is unchanged.

Use the same BF16 QKV/output compute and FP32 norm/softmax as the local run.
Only the allowed temporal edges change. Never install this on a live old run.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

def global_attention_mask(meta, conditioning_attention):
    if conditioning_attention not in {'directional','bidirectional'}:
        raise ValueError('Unknown conditioning policy')
    ng = meta.gen_frame.numel()
    gt, mt = meta.gen_frame >= 0, meta.motion_frame >= 0
    mask = torch.eye(ng+len(mt), dtype=torch.bool, device=meta.gen_frame.device)
    mask[:ng,:ng] |= gt[:,None] & gt[None,:]
    mask[ng:,ng:] |= mt[:,None] & mt[None,:]
    pair = gt[:,None] & mt[None,:]
    if conditioning_attention == 'bidirectional':
        mask[:ng,ng:] |= pair
        mask[ng:,:ng] |= pair.T
    else:
        mask[:ng,ng:] |= (~meta.gen_clean.bool())[:,None] & pair
        mask[ng:,:ng] |= (~meta.motion_clean.bool())[:,None] & pair.T
    return mask

def global_attention_bias(meta, device, conditioning_attention='directional'):
    key = '_phase3_camera_global_bias_no_shape_v1_'+conditioning_attention
    bias = getattr(meta, key, None)
    if bias is None:
        allowed = global_attention_mask(meta, conditioning_attention).to(device)
        bias = torch.zeros_like(allowed, dtype=torch.float32)
        bias.masked_fill_(~allowed, torch.finfo(torch.float32).min)
        setattr(meta, key, bias)
    if bias.device != torch.device(device):
        raise RuntimeError('Global bridge metadata moved devices')
    return bias

def global_joint_attention(q, k, v, meta, conditioning_attention, *, compute_dtype=torch.float32):
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Unsupported attention dtype')
    bias = global_attention_bias(meta, q.device, conditioning_attention)
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = torch.matmul(q.to(compute_dtype), k.to(compute_dtype).transpose(-2,-1)).float() * q.shape[-1]**-.5
        probabilities = F.softmax(scores+bias.unsqueeze(0), dim=-1)
        return torch.matmul(probabilities.to(compute_dtype), v.to(compute_dtype)).float()

@contextmanager
def attention_scope(scope):
    if scope == 'temporal_local':
        yield
        return
    if scope != 'global':
        raise ValueError('Unknown bridge attention scope')
    import modality_bridge as bridge
    import scripts.phase3_performance as performance
    old_mask, old_grouped, old_bias = bridge.temporally_local_attention_mask, bridge.grouped_joint_attention, performance.attention_bias
    bridge.temporally_local_attention_mask = global_attention_mask
    bridge.grouped_joint_attention = global_joint_attention
    performance.attention_bias = global_attention_bias
    try:
        yield
    finally:
        bridge.temporally_local_attention_mask = old_mask
        bridge.grouped_joint_attention = old_grouped
        performance.attention_bias = old_bias
