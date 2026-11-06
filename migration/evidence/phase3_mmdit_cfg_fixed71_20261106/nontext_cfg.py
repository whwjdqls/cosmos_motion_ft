"""Inference-only directional bridge-key removal; conditioned model is unchanged."""
from contextlib import contextmanager
import torch


@contextmanager
def reference_pass(mode, counters):
    import modality_bridge as mb
    if mode not in ('video2motion', 'motimg2video'):
        raise ValueError(mode)
    original = mb.grouped_attention_plan

    def plan(meta, device, attention):
        base = original(meta, device, attention)
        ng = meta.gen_frame.numel()
        qvideo = base.query_indexes < ng
        kvideo = base.key_indexes < ng
        removed = ((~qvideo[:, :, None]) & kvideo[:, None, :] if mode == 'video2motion'
                   else qvideo[:, :, None] & (~kvideo[:, None, :]))
        bias = base.bias.clone().masked_fill(removed[:, None], torch.finfo(torch.float32).min)
        counters['reference_attention_calls'] = counters.get('reference_attention_calls', 0) + 1
        return mb.GroupedAttentionPlan(base.query_indexes, base.key_indexes, base.restore_indexes, bias)

    mb.grouped_attention_plan = plan
    try:
        yield
    finally:
        mb.grouped_attention_plan = original


class Guided:
    def __init__(self, model, mode, scale):
        self.model, self.mode, self.scale = model, mode, scale
        self.counters = dict(conditioned_calls=0, reference_calls=0, reference_attention_calls=0)

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, **inputs):
        assert inputs['modes'] == [self.mode] * len(inputs['modes'])
        assert not any(inputs['captions'])
        conditioned = self.model(**inputs)
        self.counters['conditioned_calls'] += 1
        if self.scale == 1:
            return conditioned
        with reference_pass(self.mode, self.counters):
            reference = self.model(**inputs)
        self.counters['reference_calls'] += 1
        key = 'motion' if self.mode == 'video2motion' else 'video'
        output = dict(conditioned)
        output[key] = reference[key].float() + self.scale * (conditioned[key].float() - reference[key].float())
        return output


def check_masks():
    """Check all temporal groups and both policies, including unchanged reverse edges."""
    import modality_bridge as mb
    receipts = []
    for attention in ('directional', 'bidirectional'):
        for mode in ('video2motion', 'motimg2video'):
            g = torch.arange(25).repeat_interleave(64)
            m = torch.arange(97)
            meta = mb.BridgeMeta(mode, g, torch.ones_like(g, dtype=torch.bool) if mode == 'video2motion' else g == 0,
                                m, torch.full_like(m, mode == 'motimg2video', dtype=torch.bool),
                                torch.where(g == 0, g, g * 4 - 3), g * 4)
            base = mb.grouped_attention_plan(meta, 'cpu', attention)
            counters = {}
            with reference_pass(mode, counters):
                ref = mb.grouped_attention_plan(meta, 'cpu', attention)
            qv, kv = base.query_indexes < len(g), base.key_indexes < len(g)
            blocked = ((~qv[:, :, None]) & kv[:, None, :] if mode == 'video2motion'
                       else qv[:, :, None] & (~kv[:, None, :]))[:, None]
            assert torch.equal(ref.bias.masked_select(~blocked), base.bias.masked_select(~blocked))
            assert (ref.bias.masked_select(blocked) == torch.finfo(torch.float32).min).all()
            assert (ref.bias > -1e20).any(-1).all()
            assert torch.equal(base.query_indexes, ref.query_indexes)
            receipts.append(dict(attention=attention, mode=mode, passed=True))
    return receipts
