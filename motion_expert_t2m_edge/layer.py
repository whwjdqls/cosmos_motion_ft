"""One frozen Edge reasoner layer with an optional trainable motion pathway."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from attention import PackedLayout, reasoner_causal_attention, text_motion_attention


def _fresh_norm_like(reference: nn.Module, width: int) -> nn.Module:
    eps = getattr(reference, "variance_epsilon", getattr(reference, "eps", 1e-5))
    if isinstance(reference, nn.Identity):
        return nn.Identity()
    try:
        return type(reference)(width, eps=eps)
    except TypeError:
        return type(reference)(width)


def _run_mlp(module: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    output = module(hidden)
    return output[0] if isinstance(output, tuple) else output


class EdgeMotionMLP(nn.Module):
    """Original Phase-2 motion FFN: fresh three-linear SwiGLU.

    This deliberately follows the Nano motion expert instead of cloning the
    Edge reasoner FFN.  Shared attention constrains the residual and projected
    head geometry, not the private motion FFN activation or topology.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, *, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))


class EdgeTextMotionLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        motion_intermediate_size: int,
        has_motion: bool,
    ) -> None:
        super().__init__()
        attention = base_layer.self_attn
        if attention.k_norm_und_for_gen is None:
            raise ValueError("Edge text-motion attention requires k_norm_und_for_gen")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.has_motion = bool(has_motion)
        self._apply_rotary = attention._apply_rotary_pos_emb
        frozen = SimpleNamespace(
            input_norm=base_layer.input_layernorm,
            post_norm=base_layer.post_attention_layernorm,
            q_proj=attention.q_proj,
            k_proj=attention.k_proj,
            v_proj=attention.v_proj,
            o_proj=attention.o_proj,
            q_norm=attention.q_norm,
            k_norm=attention.k_norm,
            k_norm_for_motion=attention.k_norm_und_for_gen,
            mlp=base_layer.mlp,
        )
        object.__setattr__(self, "_frozen", frozen)

        if has_motion:
            bias = attention.q_proj.bias is not None
            self.input_norm_motion = _fresh_norm_like(base_layer.input_layernorm, hidden_size)
            self.post_norm_motion = _fresh_norm_like(base_layer.post_attention_layernorm, hidden_size)
            self.q_proj_motion = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
            self.k_proj_motion = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
            self.v_proj_motion = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
            self.o_proj_motion = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
            self.q_norm_motion = _fresh_norm_like(attention.k_norm_und_for_gen, head_dim)
            self.k_norm_motion = _fresh_norm_like(attention.k_norm_und_for_gen, head_dim)
            self.mlp_motion = EdgeMotionMLP(
                hidden_size, motion_intermediate_size, bias=base_layer.mlp.up_proj.bias is not None
            )
            self.reset_motion_parameters()

    @torch.no_grad()
    def reset_motion_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _rotary(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_rotary(q, k, cos, sin, unsqueeze_dim=1)

    def _update_reasoner_only(
        self,
        packed: torch.Tensor,
        layout: PackedLayout,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        f = self._frozen
        # T2M has an asymmetric mask: the frozen reasoner never reads motion.
        # Detaching this disjoint branch is exact and avoids retaining a 28-layer
        # autograd graph through frozen Edge weights.
        text = packed.index_select(0, layout.text_idx).detach()
        normalized = f.input_norm(text)
        q = f.q_norm(f.q_proj(normalized).view(-1, self.num_heads, self.head_dim))
        k = f.k_norm(f.k_proj(normalized).view(-1, self.num_kv_heads, self.head_dim))
        v = f.v_proj(normalized).view(-1, self.num_kv_heads, self.head_dim)
        text_cos = cos.index_select(0, layout.text_idx)
        text_sin = sin.index_select(0, layout.text_idx)
        q, k = self._rotary(q, k, text_cos, text_sin)
        attention_out = reasoner_causal_attention(q, k, v, layout)
        text = text + f.o_proj(attention_out)
        text = text + _run_mlp(f.mlp, f.post_norm(text))
        return packed.index_copy(0, layout.text_idx, text.to(packed.dtype))

    def forward(
        self,
        packed: torch.Tensor,
        layout: PackedLayout,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_motion:
            return self._update_reasoner_only(packed, layout, cos, sin)

        f = self._frozen
        text = packed.index_select(0, layout.text_idx).detach()
        motion = packed.index_select(0, layout.motion_idx)
        text_norm = f.input_norm(text)
        motion_norm = self.input_norm_motion(motion)

        q_text = f.q_norm(f.q_proj(text_norm).view(-1, self.num_heads, self.head_dim))
        k_text_projected = f.k_proj(text_norm).view(
            -1, self.num_kv_heads, self.head_dim
        )
        k_text_raw = f.k_norm(k_text_projected)
        v_text = f.v_proj(text_norm).view(-1, self.num_kv_heads, self.head_dim)
        # Edge's cross-path norm is applied before RoPE and only for the K copy
        # consumed by full-attention motion queries.
        k_text_for_motion = f.k_norm_for_motion(k_text_projected)

        q_motion = self.q_norm_motion(
            self.q_proj_motion(motion_norm).view(-1, self.num_heads, self.head_dim)
        )
        k_motion = self.k_norm_motion(
            self.k_proj_motion(motion_norm).view(-1, self.num_kv_heads, self.head_dim)
        )
        v_motion = self.v_proj_motion(motion_norm).view(-1, self.num_kv_heads, self.head_dim)

        text_cos = cos.index_select(0, layout.text_idx)
        text_sin = sin.index_select(0, layout.text_idx)
        motion_cos = cos.index_select(0, layout.motion_idx)
        motion_sin = sin.index_select(0, layout.motion_idx)
        q_text, k_text_raw = self._rotary(q_text, k_text_raw, text_cos, text_sin)
        # The q result is discarded; passing q_text only supplies a compatible
        # tensor to the backbone's paired rotary helper.
        _, k_text_for_motion = self._rotary(
            q_text, k_text_for_motion, text_cos, text_sin
        )
        q_motion, k_motion = self._rotary(
            q_motion, k_motion, motion_cos, motion_sin
        )

        k_full = packed.new_empty(
            packed.shape[0], self.num_kv_heads, self.head_dim
        )
        v_full = torch.empty_like(k_full)
        k_full.index_copy_(0, layout.text_idx, k_text_for_motion.to(k_full.dtype))
        k_full.index_copy_(0, layout.motion_idx, k_motion.to(k_full.dtype))
        v_full.index_copy_(0, layout.text_idx, v_text.to(v_full.dtype))
        v_full.index_copy_(0, layout.motion_idx, v_motion.to(v_full.dtype))

        text_attn, motion_attn = text_motion_attention(
            q_text,
            k_text_raw,
            v_text,
            q_motion,
            k_full,
            v_full,
            layout,
        )
        text = text + f.o_proj(text_attn)
        motion = motion + self.o_proj_motion(motion_attn)
        text = text + _run_mlp(f.mlp, f.post_norm(text))
        motion = motion + self.mlp_motion(self.post_norm_motion(motion))

        packed = packed.index_copy(0, layout.text_idx, text.to(packed.dtype))
        return packed.index_copy(0, layout.motion_idx, motion.to(packed.dtype))


__all__ = ["EdgeMotionMLP", "EdgeTextMotionLayer"]
