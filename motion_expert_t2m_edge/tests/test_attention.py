from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

import attention as attention_module
from attention import build_layout, text_motion_attention


def reference_attention(query, key, value, **kwargs):
    q, k, v = query[0], key[0], value[0]
    if k.shape[1] != q.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    output = torch.zeros_like(q)
    q_offsets = kwargs["cumulative_seqlen_Q"].tolist()
    kv_offsets = kwargs["cumulative_seqlen_KV"].tolist()
    for index in range(len(q_offsets) - 1):
        qs, qe = q_offsets[index : index + 2]
        ks, ke = kv_offsets[index : index + 2]
        qb = q[qs:qe].transpose(0, 1)
        kb = k[ks:ke].transpose(0, 1)
        vb = v[ks:ke].transpose(0, 1)
        scores = qb @ kb.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if kwargs.get("is_causal", False):
            causal = torch.arange(qe - qs)[:, None] < torch.arange(ke - ks)[None, :]
            scores = scores.masked_fill(causal, float("-inf"))
        output[qs:qe] = (scores.softmax(-1) @ vb).transpose(0, 1)
    return output.unsqueeze(0)


class SharedAttentionTest(unittest.TestCase):
    def test_reasoner_is_causal_and_motion_reads_full_sample(self):
        torch.manual_seed(0)
        layout = build_layout([2], [2], "cpu")
        q_text = torch.randn(2, 2, 4)
        k_text = torch.randn(2, 1, 4)
        v_text = torch.randn(2, 1, 4)
        q_motion = torch.randn(2, 2, 4)
        k_full = torch.randn(4, 1, 4)
        v_full = torch.randn(4, 1, 4)
        with mock.patch.object(attention_module, "attention_kernel", reference_attention):
            text_a, motion_a = text_motion_attention(
                q_text, k_text, v_text, q_motion, k_full, v_full, layout
            )
            changed = v_full.clone()
            changed[layout.motion_idx] += 10.0
            text_b, motion_b = text_motion_attention(
                q_text, k_text, v_text, q_motion, k_full, changed, layout
            )
        torch.testing.assert_close(text_a, text_b)
        self.assertFalse(torch.allclose(motion_a, motion_b))


if __name__ == "__main__":
    unittest.main()
