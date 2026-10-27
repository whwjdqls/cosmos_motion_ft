"""The original frozen specialists and bridges, including native camera rows."""
import torch
from torch.utils.checkpoint import checkpoint

from attention import build_layout
from model import _motion_positions
from self_cross import encode_reasoner_memory
from modality_bridge import BridgeMeta
from motion_expert_phase3_edge.model import EdgePhase3 as BaseEdgePhase3
from .native import pack_conditions
from .settings import RECIPE, effective_caption, has_camera, has_video


def bridge_meta(mode, device):
    frames, starts, ends, clean = [], [], [], []
    if has_video(mode):
        video_frames = torch.arange(25, device=device).repeat_interleave(64)
        frames.append(video_frames)
        starts.append(torch.where(video_frames == 0, video_frames, 4 * video_frames - 3))
        ends.append(4 * video_frames)
        clean.append(video_frames == 0 if mode == "motimg2video" else torch.ones_like(video_frames, dtype=torch.bool))
    if has_camera(mode):
        action_frames = torch.arange(96, device=device)
        frames.append(torch.full_like(action_frames, -1))
        starts.append(action_frames)
        ends.append(action_frames + 1)
        clean.append(torch.ones_like(action_frames, dtype=torch.bool))
    motion_frames = torch.arange(-1, 97, device=device)
    return BridgeMeta(mode=mode, gen_frame=torch.cat(frames), gen_clean=torch.cat(clean),
                      motion_frame=motion_frames,
                      motion_clean=(torch.ones_like(motion_frames, dtype=torch.bool)
                                    if mode == "motimg2video" else motion_frames < 0),
                      gen_source_start=torch.cat(starts), gen_source_end=torch.cat(ends))


class EdgePhase3(BaseEdgePhase3):
    def prepare_motion(self, motion, neutral, sigma, captions, modes):
        device, batch = motion.device, len(captions)
        conditions = self.motion.prepare_conditions([
            effective_caption(mode, caption) for mode, caption in zip(modes, captions, strict=True)])
        noisy = torch.tensor([mode != "motimg2video" for mode in modes], device=device)[:, None].expand(batch, 97)
        shape = self.motion.heads.encode_shape(neutral)
        frames = self.motion.heads.encode_motion(motion, sigma, noisy)
        parts, positions, lengths = [], [], []
        for i, text in enumerate(conditions):
            emb = text["inputs_embeds"].to(torch.bfloat16)
            parts.extend((emb, torch.cat((shape[i], frames[i])).to(torch.bfloat16)))
            positions.extend((text["position_ids"], _motion_positions(97, int(text["next_position_id"]), device)))
            lengths.append(len(emb))
        packed = torch.cat(parts)
        layout = build_layout(lengths, [98] * batch, device)
        cos, sin = self.edge.rope(torch.cat(positions, dim=1))
        _, memory = encode_reasoner_memory(self.motion.layers, packed, layout, cos, sin,
                                            self.motion.backbone.norm, "self_cross_final_hidden")
        return (packed.index_select(0, layout.motion_idx), memory, layout,
                cos.index_select(0, layout.motion_idx), sin.index_select(0, layout.motion_idx),
                cos.index_select(0, layout.text_idx), sin.index_select(0, layout.text_idx))

    def forward(self, *, motion, neutral, video, motion_sigma, video_sigma, captions,
                modes, camera_action=None, camera_features=None, bridges_enabled=True):
        from cosmos_framework.model.generator.mot.attention import build_packed_sequence
        from cosmos_framework.data.generator.sequence_packing.runtime import (
            from_all_seq, from_und_gen_splits, get_und_seq, get_gen_seq, get_all_seq,
        )
        if tuple(motion.shape[1:]) != (97, 283) or tuple(video.shape[1:]) != (48, 25, 16, 16):
            raise ValueError("Expected T97 motion and native [48,25,16,16] video latents")
        batch = motion.shape[0]
        if len(modes) != batch or set(modes) - set(RECIPE["tasks"]):
            raise ValueError("Unsupported camera-conditioning task batch")
        if any(has_camera(mode) for mode in modes):
            if camera_features is None or tuple(camera_features.shape) != (batch, 97, 18):
                raise ValueError("Camera tasks require [B,97,18] floor-aware trajectory features")
        with torch.no_grad(), torch.autocast(device_type=motion.device.type, enabled=False):
            motion_time = self.motion.heads.time_embedder(motion_sigma.float()).float()
        motion_state, memory, layout, mc, ms, tc, ts = self.prepare_motion(
            motion, neutral, motion_sigma, captions, modes)
        packed, video_samples, camera_samples = pack_conditions(
            self.edge, captions, modes, video, video_sigma, camera_action)
        sequence, dtype = self.native._encode_text(packed)
        shapes = self.native._encode_vision(packed, sequence, dtype) if video_samples else None
        if camera_samples:
            self.native._encode_action(packed, sequence, dtype)
        indexes = []
        if video_samples:
            indexes.append(packed.vision.sequence_indexes)
        if camera_samples:
            indexes.append(packed.action.sequence_indexes)
        state, attention, _ = build_packed_sequence(
            "two_way", packed_sequence=sequence, attn_modes=packed.attn_modes,
            split_lens=packed.split_lens, sample_lens=packed.sample_lens,
            packed_und_token_indexes=packed.text_indexes, packed_gen_token_indexes=torch.cat(indexes),
            num_heads=16, is_image_batch=False, head_dim=128, num_layers=28,
            token_shapes=packed.vision.token_shapes if video_samples else None,
            natten_parameter_list=None, cp_world_size=1, video_temporal_causal=False,
            vision_token_shapes=packed.vision.token_shapes if video_samples else None,
            action_token_shapes=packed.action.token_shapes if camera_samples else None,
            num_action_tokens_per_supertoken=0, null_action_supertokens=False,
            pad_for_cuda_graphs=False, prepared_metadata=packed.get_sequence_pack_metadata(),
        )
        cos, sin = self.native.language_model.model.rotary_emb(
            get_gen_seq(state), position_ids=packed.position_ids.unsqueeze(1))
        native_rope = (from_all_seq(cos.squeeze(0), state), from_all_seq(sin.squeeze(0), state))
        labels = torch.arange(sequence.shape[0], device=sequence.device)[:, None]
        gen_labels = get_gen_seq(from_all_seq(labels, state)).flatten().long()
        inverse = torch.full((sequence.shape[0],), -1, device=sequence.device, dtype=torch.long)
        inverse[gen_labels] = torch.arange(len(gen_labels), device=sequence.device)
        per_sample = [[] for _ in modes]
        for i, sample in enumerate(video_samples):
            per_sample[sample].append(inverse[packed.vision.sequence_indexes[i * 1600:(i + 1) * 1600]])
        for i, sample in enumerate(camera_samples):
            per_sample[sample].append(inverse[packed.action.sequence_indexes[i * 96:(i + 1) * 96]])
        bridge_indexes = [torch.cat(items) for items in per_sample]
        metadata = [bridge_meta(mode, motion.device) for mode in modes]
        for indices, meta in zip(bridge_indexes, metadata, strict=True):
            if indices.numel() != meta.gen_frame.numel() or (indices < 0).any():
                raise RuntimeError("Camera/video bridge routing disagrees with native packing")
        all_bridge_indexes = torch.cat(bridge_indexes)
        if all_bridge_indexes.unique().numel() != all_bridge_indexes.numel():
            raise RuntimeError("A generator token was assigned to multiple samples")
        use_checkpoint = self.training and torch.is_grad_enabled() and self.activation_checkpointing
        for index, native_layer in enumerate(self.native.language_model.model.layers):
            def native_step(und, gen, layer=native_layer, template=state):
                output, _, _ = layer(from_und_gen_splits(und, gen, template), attention, native_rope)
                return get_und_seq(output), get_gen_seq(output)
            und, gen = get_und_seq(state), get_gen_seq(state)
            und, gen = (checkpoint(native_step, und, gen, use_reentrant=False)
                        if use_checkpoint else native_step(und, gen))
            state = from_und_gen_splits(und, gen, state)
            if str(index) not in self.bridges:
                continue
            def motion_step(value, block=self.motion.layers[index]):
                return block.forward_motion(value, memory, layout, mc, ms, tc, ts)
            motion_state = (checkpoint(motion_step, motion_state, use_reentrant=False)
                            if use_checkpoint else motion_step(motion_state))
            if bridges_enabled:
                def bridge_step(gen_value, motion_value, block=self.bridges[str(index)]):
                    gen_parts, motion_parts = [], []
                    for sample, (indices, meta) in enumerate(zip(bridge_indexes, metadata, strict=True)):
                        g, m = block(gen_value.index_select(0, indices),
                                     motion_value[sample * 98:(sample + 1) * 98], meta,
                                     camera_features[sample] if camera_features is not None else None,
                                     motion_time[sample])
                        gen_parts.append(g)
                        motion_parts.append(m)
                    return gen_value.index_copy(0, all_bridge_indexes, torch.cat(gen_parts)), torch.cat(motion_parts)
                gen, motion_state = (checkpoint(bridge_step, gen, motion_state, use_reentrant=False)
                                     if use_checkpoint else bridge_step(gen, motion_state))
                state = from_und_gen_splits(und, gen, state)
        backbone = self.native.language_model.model
        state = from_und_gen_splits(backbone.norm(get_und_seq(state)), backbone.norm_moe_gen(get_gen_seq(state)), state)
        output = {}
        if video_samples:
            self.native._decode_vision(packed, get_all_seq(state), output, shapes)
        else:
            output["preds_vision"] = []
        decoded = {i: value.reshape(48, 25, 16, 16) for i, value in zip(video_samples, output["preds_vision"], strict=True)}
        output["video"] = torch.stack([decoded[i] if i in decoded else video[i].new_zeros(video[i].shape) for i in range(batch)])
        hidden = self.motion.norm_moe_motion(motion_state).reshape(batch, 98, 2048)
        output["motion"] = torch.stack([self.motion.heads.decode(rows[1:]) for rows in hidden])
        output["anchor"] = sum(p.flatten()[0] * 0.0 for p in self.bridges.parameters())
        output["packed"] = packed
        output["vision_sample_indices"] = video_samples
        output["camera_sample_indices"] = camera_samples
        return output
