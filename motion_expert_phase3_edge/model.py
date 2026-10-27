"""Interleave the native Edge generator, frozen motion blocks and seven bridges."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from attention import build_layout
from model import EdgePhase2MotionExpert, _motion_positions
from self_cross import encode_reasoner_memory
from modality_bridge import BridgeMeta, LocalModalityBridge
from motion_expert_phase3_edge.native import FullFrozenEdge
from motion_expert_phase3_edge.settings import RECIPE


def bridge_meta(mode, device):
    frames = torch.arange(25, device=device).repeat_interleave(64)
    motion_frames = torch.arange(-1, 97, device=device)
    return BridgeMeta(
        mode=mode, gen_frame=frames,
        gen_clean=torch.ones_like(frames, dtype=torch.bool) if mode == "video2motion" else frames == 0,
        motion_frame=motion_frames,
        motion_clean=motion_frames < 0 if mode == "video2motion" else torch.ones_like(motion_frames, dtype=torch.bool),
        gen_source_start=torch.where(frames == 0, frames, 4 * frames - 3),
        gen_source_end=4 * frames,
    )


class EdgePhase3(nn.Module):
    def __init__(self, phase1, phase2, device="cuda", activation_checkpointing=True):
        super().__init__()
        edge = FullFrozenEdge(phase1, phase2, device)
        object.__setattr__(self, "edge", edge)
        object.__setattr__(self, "native", edge.net)
        self.motion = EdgePhase2MotionExpert(device=device, frozen_edge=edge)
        payload = torch.load(phase2, map_location="cpu", mmap=True, weights_only=False)
        if int(payload["step"]) != 200000:
            raise RuntimeError("This recipe requires the selected phase-2 200k checkpoint")
        saved_args = payload["args"]
        for name, weight in RECIPE["motion_weights"].items():
            if float(saved_args[name]) != weight:
                raise RuntimeError(f"Source phase-2 loss mismatch: {name}")
        if saved_args.get("joint_loss_mode", "centered") != "centered":
            raise RuntimeError("Expected the centered decoded-joint specialist")
        self.motion.load_motion_state_dict(payload["motion_state"], strict=True)
        self.motion.requires_grad_(False).eval()
        self.bridges = nn.ModuleDict({
            str(index): LocalModalityBridge(2048, 16, 128).to(device=device, dtype=torch.float32)
            for index in RECIPE["bridge_layers"]
        })
        self.activation_checkpointing = bool(activation_checkpointing)
        self.provenance = {**edge.provenance, "phase2_step": 200000,
                           "motion_tensors": len(payload["motion_state"])}
        print(f"[phase3/model] motion tensors={len(payload['motion_state'])}; "
              f"trainable bridges={sum(p.numel() for p in self.bridges.parameters()):,}", flush=True)

    def train(self, mode=True):
        super().train(mode)
        self.motion.eval()
        self.native.eval()
        return self

    def prepare_motion(self, motion, neutral, sigma, captions, modes):
        device, batch = motion.device, len(captions)
        condition = self.motion.prepare_conditions([
            "" if mode == "video2motion" else caption
            for caption, mode in zip(captions, modes, strict=True)
        ])
        noisy = torch.tensor([mode == "video2motion" for mode in modes], device=device)[:, None].expand(batch, 97)
        shape = self.motion.heads.encode_shape(neutral)
        frames = self.motion.heads.encode_motion(motion, sigma, noisy)
        parts, positions, lengths = [], [], []
        for i, text in enumerate(condition):
            emb = text["inputs_embeds"].to(torch.bfloat16)
            parts.extend((emb, torch.cat((shape[i], frames[i])).to(torch.bfloat16)))
            positions.extend((text["position_ids"], _motion_positions(97, int(text["next_position_id"]), device)))
            lengths.append(len(emb))
        packed = torch.cat(parts)
        layout = build_layout(lengths, [98] * batch, device)
        cos, sin = self.edge.rope(torch.cat(positions, dim=1))
        _, memory = encode_reasoner_memory(self.motion.layers, packed, layout, cos, sin,
                                            self.motion.backbone.norm, "self_cross_final_hidden")
        return (
            packed.index_select(0, layout.motion_idx), memory, layout,
            cos.index_select(0, layout.motion_idx), sin.index_select(0, layout.motion_idx),
            cos.index_select(0, layout.text_idx), sin.index_select(0, layout.text_idx),
        )

    def forward(self, *, motion, neutral, video, motion_sigma, video_sigma, captions,
                modes, bridges_enabled=True):
        from cosmos_framework.model.generator.mot.attention import build_packed_sequence
        from cosmos_framework.data.generator.sequence_packing.runtime import (
            from_all_seq, from_und_gen_splits, get_und_seq, get_gen_seq, get_all_seq,
        )

        if tuple(motion.shape[1:]) != (97, 283) or tuple(video.shape[1:]) != (48, 25, 16, 16):
            raise ValueError("Phase-3 requires T97 motion and [48,25,16,16] cached latents")
        batch = motion.shape[0]
        if len(modes) != batch or set(modes) - {"video2motion", "motimg2video"}:
            raise ValueError("Unsupported phase-3 task batch")
        motion_state, memory, layout, mc, ms, tc, ts = self.prepare_motion(
            motion, neutral, motion_sigma, captions, modes)
        packed = self.edge.pack_video(captions, modes, [v.unsqueeze(0) for v in video], video_sigma)
        sequence, dtype = self.native._encode_text(packed)
        shapes = self.native._encode_vision(packed, sequence, dtype)
        state, attention, natten = build_packed_sequence(
            "two_way", packed_sequence=sequence, attn_modes=packed.attn_modes,
            split_lens=packed.split_lens, sample_lens=packed.sample_lens,
            packed_und_token_indexes=packed.text_indexes,
            packed_gen_token_indexes=packed.vision.sequence_indexes,
            num_heads=16, is_image_batch=False, head_dim=128, num_layers=28,
            token_shapes=packed.vision.token_shapes, natten_parameter_list=None,
            cp_world_size=1, video_temporal_causal=False,
            vision_token_shapes=packed.vision.token_shapes,
            action_token_shapes=None, num_action_tokens_per_supertoken=0,
            null_action_supertokens=False, pad_for_cuda_graphs=False,
            prepared_metadata=packed.get_sequence_pack_metadata(),
        )
        # Native generator positions remain floating point for FPS modulation.
        cos, sin = self.native.language_model.model.rotary_emb(
            get_gen_seq(state), position_ids=packed.position_ids.unsqueeze(1))
        native_rope = (from_all_seq(cos.squeeze(0), state), from_all_seq(sin.squeeze(0), state))
        metadata = [bridge_meta(mode, motion.device) for mode in modes]
        use_checkpoint = self.training and torch.is_grad_enabled() and self.activation_checkpointing

        for index, native_layer in enumerate(self.native.language_model.model.layers):
            # Capture loop values in defaults: checkpoint recomputation happens
            # after this loop has advanced to another layer.
            def native_step(und, gen, layer=native_layer, template=state):
                output, _, _ = layer(from_und_gen_splits(und, gen, template), attention, native_rope)
                return get_und_seq(output), get_gen_seq(output)

            und, gen = get_und_seq(state), get_gen_seq(state)
            if use_checkpoint:
                und, gen = checkpoint(native_step, und, gen, use_reentrant=False)
            else:
                und, gen = native_step(und, gen)
            state = from_und_gen_splits(und, gen, state)
            if str(index) not in self.bridges:
                continue
            layer = self.motion.layers[index]

            def motion_step(value, block=layer):
                return block.forward_motion(value, memory, layout, mc, ms, tc, ts)

            motion_state = (checkpoint(motion_step, motion_state, use_reentrant=False)
                            if use_checkpoint else motion_step(motion_state))
            if bridges_enabled:
                bridge = self.bridges[str(index)]

                def bridge_step(gen_value, motion_value, block=bridge):
                    gen_parts, motion_parts = [], []
                    for sample, meta in enumerate(metadata):
                        g, m = block(gen_value[sample * 1600:(sample + 1) * 1600],
                                     motion_value[sample * 98:(sample + 1) * 98], meta)
                        gen_parts.append(g)
                        motion_parts.append(m)
                    # Preserve any packer-owned padding, never bridge samples together.
                    gen_parts.append(gen_value[batch * 1600:])
                    return torch.cat(gen_parts), torch.cat(motion_parts)

                gen, motion_state = (checkpoint(bridge_step, gen, motion_state, use_reentrant=False)
                                     if use_checkpoint else bridge_step(gen, motion_state))
                state = from_und_gen_splits(und, gen, state)

        backbone = self.native.language_model.model
        state = from_und_gen_splits(backbone.norm(get_und_seq(state)),
                                    backbone.norm_moe_gen(get_gen_seq(state)), state)
        output = {}
        self.native._decode_vision(packed, get_all_seq(state), output, shapes)
        motion_hidden = self.motion.norm_moe_motion(motion_state).reshape(batch, 98, 2048)
        # Preserve the specialist's per-sequence FP32 projection shape. A
        # batched GEMM changes accumulation order despite identical inputs,
        # breaking bit-exact zero-bridge parity with Phase 2.
        output["motion"] = torch.stack([
            self.motion.heads.decode(rows[1:]) for rows in motion_hidden
        ])
        output["video"] = torch.stack([item.reshape(48, 25, 16, 16) for item in output["preds_vision"]])
        # Homogeneous task batches can leave the final opposite-direction gate
        # unused. Explicit zero edges keep DDP's parameter set stable.
        output["anchor"] = sum(p.flatten()[0] * 0.0 for p in self.bridges.parameters())
        output["packed"] = packed
        return output
