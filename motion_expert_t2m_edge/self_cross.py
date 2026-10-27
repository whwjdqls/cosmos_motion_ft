"""Separate motion self-attention and frozen-memory cross-attention."""
from __future__ import annotations

import torch
import torch.nn as nn

import config
from attention import PackedLayout, motion_self_attention, reasoner_causal_attention
from cosmos_framework.model.attention import attention as attention_kernel
from layer import EdgeTextMotionLayer, _fresh_norm_like, _run_mlp


def motion_cross_attention(q, k, v, layout: PackedLayout):
    """Motion queries attend only to the same sample's reasoner memory."""
    if layout.max_text_len == 1:
        # Caption-only CFG can leave one reasoner token per sample. Its
        # cross-attention softmax is exactly [1], but NATTEN rejects a
        # maximum KV length of one. Preserve the prompt instead of padding
        # it, and broadcast only each sample's own V to its motion queries.
        query_heads, value_heads = q.shape[-2], v.shape[-2]
        if query_heads % value_heads:
            raise ValueError("cross query heads must be divisible by value heads")
        values = v.repeat_interleave(query_heads // value_heads, dim=-2)
        motion_lengths = layout.motion_offsets[1:] - layout.motion_offsets[:-1]
        output = values.repeat_interleave(
            motion_lengths, dim=0, output_size=q.shape[0]
        )
        # Q/K have mathematically zero derivatives for a one-key softmax.
        # Keep their graph edges so DDP does not see unused projection/norm
        # parameters when every caption in a packed batch is dropped.
        zero_qk = (q.sum(dtype=torch.float32) + k.sum(dtype=torch.float32)) * 0.0
        return (output + zero_qk.to(output.dtype)).flatten(-2, -1)
    return attention_kernel(
        q[None], k[None], v[None],
        cumulative_seqlen_Q=layout.motion_offsets,
        cumulative_seqlen_KV=layout.text_offsets,
        max_seqlen_Q=layout.max_motion_len,
        max_seqlen_KV=layout.max_text_len,
        is_causal=False,
    ).squeeze(0).flatten(-2, -1)


@torch.no_grad()
def encode_reasoner_memory(layers, packed, layout, cos, sin, final_norm, mode):
    """Execute the native causal branch once; never expose motion to it."""
    text = packed.index_select(0, layout.text_idx).detach()
    text_cos = cos.index_select(0, layout.text_idx)
    text_sin = sin.index_select(0, layout.text_idx)
    memories = {}
    for index, layer in enumerate(layers):
        f = layer._frozen
        normalized = f.input_norm(text)
        q = f.q_norm(f.q_proj(normalized).view(-1, layer.num_heads, layer.head_dim))
        k = f.k_norm(f.k_proj(normalized).view(-1, layer.num_kv_heads, layer.head_dim))
        v = f.v_proj(normalized).view(-1, layer.num_kv_heads, layer.head_dim)
        q, k = layer._rotary(q, k, text_cos, text_sin)
        if mode == "self_cross_layer_kv" and layer.has_motion:
            memories[index] = (k, v)
        update = reasoner_causal_attention(q, k, v, layout)
        text = text + f.o_proj(update)
        text = (text + _run_mlp(f.mlp, f.post_norm(text))).to(packed.dtype)
    if mode == "self_cross_final_hidden":
        memories = final_norm(text).detach()
    return text, memories


class EdgeSelfCrossMotionLayer(EdgeTextMotionLayer):
    """Seven-block ablation: pre-norm SA, pre-norm CA, one pre-norm FFN."""

    def __init__(self, base_layer, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.attention_mode = config.MOTION_ATTENTION_MODE
        self.cross_attention_k_norm = config.CROSS_ATTENTION_K_NORM
        self.cross_attention_v_norm = config.CROSS_ATTENTION_V_NORM
        self.cross_attention_reasoner_rope = config.CROSS_ATTENTION_REASONER_ROPE
        if not self.has_motion:
            return
        attention = base_layer.self_attn
        bias = attention.q_proj.bias is not None
        # Extra modules do not advance the initialization stream used by the
        # shared SA/FFN/heads. Common CA Q/O weights also match both variants.
        with torch.random.fork_rng(devices=[]):
            self.input_norm_cross_motion = _fresh_norm_like(
                base_layer.input_layernorm, self.hidden_size
            )
            self.q_norm_cross_motion = _fresh_norm_like(
                attention.k_norm_und_for_gen, self.head_dim
            )
            self.q_proj_cross_motion = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=bias
            )
            self.o_proj_cross_motion = nn.Linear(
                self.num_heads * self.head_dim, self.hidden_size, bias=bias
            )
            for module in (self.q_proj_cross_motion, self.o_proj_cross_motion):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if self.attention_mode == "self_cross_final_hidden":
                self.k_proj_cross_motion = nn.Linear(
                    self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
                )
                self.v_proj_cross_motion = nn.Linear(
                    self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
                )
                for module in (self.k_proj_cross_motion, self.v_proj_cross_motion):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                if self.cross_attention_k_norm == "rmsnorm":
                    # Fresh motion-owned affine gains, matching cross-Q's
                    # native RMSNorm class/epsilon. Normalize each 128-D
                    # key head, not the flattened 1024-D multihead tensor.
                    self.k_norm_cross_motion = _fresh_norm_like(
                        attention.k_norm_und_for_gen, self.head_dim
                    )
                if self.cross_attention_v_norm == "rmsnorm":
                    # Independent motion-owned affine gains for each block;
                    # normalize each projected 128-D value head, never memory.
                    self.v_norm_cross_motion = _fresh_norm_like(
                        attention.k_norm_und_for_gen, self.head_dim
                    )

    def forward_motion(self, motion, memory, layout, motion_cos, motion_sin, text_cos, text_sin):
        normalized = self.input_norm_motion(motion)
        q = self.q_norm_motion(
            self.q_proj_motion(normalized).view(-1, self.num_heads, self.head_dim)
        )
        k = self.k_norm_motion(
            self.k_proj_motion(normalized).view(-1, self.num_kv_heads, self.head_dim)
        )
        v = self.v_proj_motion(normalized).view(-1, self.num_kv_heads, self.head_dim)
        q, k = self._rotary(q, k, motion_cos, motion_sin)
        motion = motion + self.o_proj_motion(motion_self_attention(q, k, v, layout))

        normalized = self.input_norm_cross_motion(motion)
        q = self.q_norm_cross_motion(
            self.q_proj_cross_motion(normalized).view(-1, self.num_heads, self.head_dim)
        )
        # Rotate the new cross query once. The paired helper's second result
        # is discarded; native reasoner K is not passed through it again.
        q, _ = self._rotary(q, q, motion_cos, motion_sin)
        if self.attention_mode == "self_cross_final_hidden":
            k = self.k_proj_cross_motion(memory).view(-1, self.num_kv_heads, self.head_dim)
            v = self.v_proj_cross_motion(memory).view(-1, self.num_kv_heads, self.head_dim)
            if self.cross_attention_k_norm == "rmsnorm":
                k = self.k_norm_cross_motion(k)
            if self.cross_attention_v_norm == "rmsnorm":
                v = self.v_norm_cross_motion(v)
            if self.cross_attention_reasoner_rope:
                _, k = self._rotary(k, k, text_cos, text_sin)
        else:
            k, v = memory
        motion = motion + self.o_proj_cross_motion(motion_cross_attention(q, k, v, layout))
        return motion + self.mlp_motion(self.post_norm_motion(motion))


def run_self_cross_layers(layers, packed, layout, cos, sin, final_norm, mode):
    text, memories = encode_reasoner_memory(
        layers, packed, layout, cos, sin, final_norm, mode
    )
    motion = packed.index_select(0, layout.motion_idx)
    motion_cos, motion_sin = cos.index_select(0, layout.motion_idx), sin.index_select(0, layout.motion_idx)
    text_cos, text_sin = cos.index_select(0, layout.text_idx), sin.index_select(0, layout.text_idx)
    for index, layer in enumerate(layers):
        if layer.has_motion:
            memory = memories if mode == "self_cross_final_hidden" else memories[index]
            motion = layer.forward_motion(
                motion, memory, layout, motion_cos, motion_sin, text_cos, text_sin
            )
    packed = packed.index_copy(0, layout.text_idx, text.to(packed.dtype))
    return packed.index_copy(0, layout.motion_idx, motion.to(packed.dtype))
