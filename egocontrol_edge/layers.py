"""EgoControl's global pose modulation and frame-token CA on Edge's GEN stream.

Edge keeps its pretrained RMSNorm, MoT attention, native RoPE, and SwiGLU.
New pose CA uses the upstream 16x128 layout, Q/K RMSNorm, unnormalized V,
and sinusoidal pose positions, without extra cross-attention RoPE.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed.tensor import DTensor, distribute_tensor

from cosmos_framework.model.generator.mot.unified_mot import MoTDecoderLayer, _run_mlp, _real_token_mask
from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits, get_gen_seq, get_und_seq, get_num_real_tokens,
)


def initialize_linear(module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.trunc_normal_(child.weight, std=child.in_features ** -.5,
                                  a=-3 * child.in_features ** -.5, b=3 * child.in_features ** -.5)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, (nn.LayerNorm, nn.RMSNorm)):
            child.reset_parameters()


class Mlp(nn.Sequential):
    def __init__(self, inputs, hidden, outputs):
        super().__init__(nn.Linear(inputs, hidden), nn.GELU(approximate='tanh'), nn.Linear(hidden, outputs))


class PoseEmbedding(nn.Module):
    def __init__(self, hidden=2048, pose_channels=1024, final_modulation=True):
        super().__init__()
        self.hidden, self.pose_channels = hidden, pose_channels
        self.pose_scaler = nn.Parameter(torch.empty(96 * 138))
        self.pose_embedder_B_D = Mlp(96 * 138, hidden * 4, hidden)
        self.pose_embedder_B_3D = Mlp(96 * 138, hidden * 4, hidden * 3)
        self.pose_to_token = nn.Sequential(nn.Linear(138, pose_channels), nn.GELU(), nn.LayerNorm(pose_channels))
        self.crossattn_proj = nn.Linear(pose_channels, pose_channels)
        self.final_modulation = (nn.Sequential(nn.SiLU(), nn.Linear(hidden, 256, bias=False),
                                               nn.Linear(256, hidden * 2, bias=False))
                                 if final_modulation else None)
        self.reset_parameters()

    def reset_parameters(self):
        initialize_linear(self)
        with torch.no_grad():
            scale = torch.ones((96, 23, 6), device=self.pose_scaler.device, dtype=self.pose_scaler.dtype)
            scale[:, :2, :3] *= 20
            scale[:, :2, 3:] *= 10
            scale = scale.flatten()
            if isinstance(self.pose_scaler, DTensor):
                scale = distribute_tensor(scale, self.pose_scaler.device_mesh,
                                          self.pose_scaler.placements)
            self.pose_scaler.copy_(scale)
        # Edge has no pretrained AdaLN gates: start the additive modulation at
        # zero, preserving its trained SA/FFN residuals rather than replacing them.
        nn.init.zeros_(self.pose_embedder_B_3D[-1].weight)
        nn.init.zeros_(self.pose_embedder_B_3D[-1].bias)
        if self.final_modulation is not None:
            nn.init.zeros_(self.final_modulation[-1].weight)

    def forward(self, pose):
        pose = pose.to(device=self.pose_scaler.device, dtype=self.pose_scaler.dtype)
        if pose.ndim != 3 or tuple(pose.shape[1:]) != (96, 138):
            raise ValueError(f'Expected [B,96,138] pose, got {pose.shape}')
        scaled = pose.flatten(1) * self.pose_scaler
        global_d = self.pose_embedder_B_D(scaled)
        global_3d = self.pose_embedder_B_3D(scaled)
        tokens = self.pose_to_token(pose)
        position = torch.arange(96, device=pose.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, self.pose_channels, 2, device=pose.device,
                                          dtype=torch.float32) * (-math.log(10000.) / self.pose_channels))
        phase = position * frequency
        pe = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        tokens = self.crossattn_proj(tokens + pe.to(tokens.dtype)[None])
        return {'tokens': tokens, 'global_d': global_d, 'global_3d': global_3d}

    def final(self, gen, context):
        batch = context['global_d'].shape[0]
        count = batch * 1600
        x = gen[:count].reshape(batch, 1600, self.hidden)
        shift, scale = (self.final_modulation(context['global_d'])
                        + context['global_3d'][:, :2 * self.hidden]).chunk(2, -1)
        real = (x * (1 + scale[:, None]) + shift[:, None]).flatten(0, 1)
        return torch.cat((real, gen[count:]), dim=0)


class PoseBlock(nn.Module):
    def __init__(self, hidden=2048, pose_channels=1024, heads=16):
        super().__init__()
        self.hidden, self.heads, self.head_dim = hidden, heads, hidden // heads
        self.modulations = nn.ModuleDict({name: nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, 256, bias=False), nn.Linear(256, hidden * 3, bias=False))
            for name in ('self', 'cross', 'mlp')})
        self.cross_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(pose_channels, hidden, bias=False)
        self.v_proj = nn.Linear(pose_channels, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.output_proj = nn.Linear(hidden, hidden, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        initialize_linear(self)
        for mod in self.modulations.values():
            nn.init.zeros_(mod[-1].weight)

    def modulate(self, value, context, branch):
        batch = context['global_d'].shape[0]
        count = batch * 1600
        if value.shape[0] < count:
            raise RuntimeError(f'Missing video tokens: got {value.shape[0]}/{batch}')
        x = value[:count].reshape(batch, 1600, self.hidden)
        shift, scale, gate = (self.modulations[branch](context['global_d'])
                              + context['global_3d']).chunk(3, -1)
        real = (x * (1 + scale[:, None]) + shift[:, None]).flatten(0, 1)
        gates = gate[:, None].expand(batch, 1600, self.hidden).flatten(0, 1)
        return (torch.cat((real, value[count:]), dim=0),
                torch.cat((gates, torch.zeros_like(value[count:])), dim=0))

    def cross(self, value, context):
        batch = context['tokens'].shape[0]
        count = batch * 1600
        query, gate = self.modulate(self.cross_norm(value), context, 'cross')
        q = self.q_norm(self.q_proj(query[:count]).reshape(batch, 1600, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context['tokens']).reshape(batch, 96, self.heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(context['tokens']).reshape(batch, 96, self.heads, self.head_dim).transpose(1, 2)
        update = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=False)
        update = self.output_proj(update.transpose(1, 2).reshape(batch, 1600, self.hidden))
        real = value[:count] + gate[:count] * update.flatten(0, 1)
        return torch.cat((real, value[count:]), dim=0)


class EgoControlDecoderLayer(MoTDecoderLayer):
    def forward(self, input, attention_mask, packed_position_embeddings,
                natten_metadata=None, memory_value=None, gen_only=False):
        if memory_value is not None or gen_only:
            raise NotImplementedError('This baseline uses full-window native packing, not KV-cache inference')
        context = attention_mask.egocontrol_context
        if get_num_real_tokens(input)[1] != context['tokens'].shape[0] * 1600:
            raise RuntimeError('Real generation tokens do not match T97 video windows')
        und, gen = get_und_seq(input), get_gen_seq(input)
        gen_norm, gate_sa = self.egocontrol.modulate(self.input_layernorm_moe_gen(gen), context, 'self')
        norm_pack = from_und_gen_splits(self.input_layernorm(und), gen_norm, input)
        attended, cache = self.self_attn(norm_pack, attention_mask, packed_position_embeddings,
                                       natten_metadata=natten_metadata, memory_value=None)
        batch = context['tokens'].shape[0]
        residual_und = und + get_und_seq(attended)
        residual_gen = gen + (1 + gate_sa) * get_gen_seq(attended)
        residual_gen = self.egocontrol.cross(residual_gen, context)
        gen_norm, gate_mlp = self.egocontrol.modulate(
            self.post_attention_layernorm_moe_gen(residual_gen), context, 'mlp')
        und_len, gen_len = get_num_real_tokens(attended)
        und_mlp, _ = _run_mlp(self.mlp, self.post_attention_layernorm(residual_und),
                             _real_token_mask(residual_und.shape[0], und_len, residual_und.device))
        gen_mlp, _ = _run_mlp(self.mlp_moe_gen, gen_norm,
                             _real_token_mask(gen_norm.shape[0], gen_len, gen_norm.device))
        und_output = residual_und + und_mlp
        gen_output = residual_gen + (1 + gate_mlp) * gen_mlp
        packed = from_und_gen_splits(und_output, gen_output, input)
        if getattr(self, '_acceptance_initial_noop', False):
            # Keep the same grad mode: attention dispatch may select different
            # training/inference kernels when entered under torch.no_grad().
            reference, _, _ = MoTDecoderLayer.forward(
                self, input, attention_mask, packed_position_embeddings,
                natten_metadata=natten_metadata, memory_value=None)
            if not (torch.equal(get_und_seq(packed), get_und_seq(reference))
                    and torch.equal(get_gen_seq(packed), get_gen_seq(reference))):
                differences = {key: float((access(packed).detach().float() - access(reference).detach().float()).abs().max())
                               for key, access in [('und', get_und_seq), ('gen', get_gen_seq)]}
                differences.update({key + '_real': float((access(packed)[:count].detach().float()
                                     - access(reference)[:count].detach().float()).abs().max())
                                    for key, access, count in [('und', get_und_seq, und_len),
                                                               ('gen', get_gen_seq, gen_len)]})
                raise RuntimeError('Initial adapted block differs from native: ' + str(differences))
            print('[egocontrol/acceptance] native_block_noop_bitwise=True', flush=True)
        return packed, {}, cache
