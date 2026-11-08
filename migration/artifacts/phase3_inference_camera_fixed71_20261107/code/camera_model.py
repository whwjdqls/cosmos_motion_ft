"""Unchanged image-joint specialists/bridges plus native camera target tokens."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from motion_expert_phase3_edge_image_joint.model import EdgePhase3 as BaseModel
from motion_expert_phase3_edge_image_joint.policy import captions_for_modes
from motion_expert_phase3_edge_joint.model import bridge_meta
from motion_expert_phase3_edge_joint.settings import RECIPE
from camera_native import pack_predictions
CAMERA_POLICY = {'tokens': 96, 'dim': 64, 'raw_dim': 9, 'domain': 2, 'initialization': 'per_case_Gaussian_seed131071_plus_case_seed_no_GT_camera', 'clock': 'motion_shift3_for_V2M_video_shift10_for_M2V', 'guidance': 'native_conditioned_velocity_only_no_extra_camera_CFG', 'bridge_edges': 'video1600_motion98_unchanged_camera_native_GEN_only'}
from motion_expert_phase3_edge_joint.adapters import trainable_parameters


class EdgePhase3(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provenance["inference_camera_prediction"] = CAMERA_POLICY.copy()

    def forward(self, *, motion, neutral, video, motion_sigma, video_sigma, captions,
                modes, camera_state=None, camera_sigma=None, camera_enabled=True, bridges_enabled=True):
        from cosmos_framework.model.generator.mot.attention import build_packed_sequence
        from cosmos_framework.data.generator.sequence_packing.runtime import (
            from_all_seq, from_und_gen_splits, get_und_seq, get_gen_seq, get_all_seq,
        )

        if not camera_enabled:
            return super().forward(motion=motion, neutral=neutral, video=video,
                motion_sigma=motion_sigma, video_sigma=video_sigma,
                captions=captions, modes=modes, bridges_enabled=bridges_enabled)
        captions = captions_for_modes(captions, modes)
        if camera_state is None or camera_sigma is None:
            raise ValueError("Camera prediction needs camera_state and camera_sigma")
        if tuple(motion.shape[1:]) != (97, 283) or tuple(video.shape[1:]) != (48, 25, 16, 16):
            raise ValueError("Phase-3 requires T97 motion and [48,25,16,16] cached latents")
        batch = motion.shape[0]
        if len(modes) != batch or set(modes) - set(RECIPE["tasks"]):
            raise ValueError("Unsupported phase-3 task batch")
        motion_state, memory, layout, mc, ms, tc, ts = self.prepare_motion(
            motion, neutral, motion_sigma, captions, modes)
        packed = pack_predictions(self.edge, captions, modes, [v.unsqueeze(0) for v in video], video_sigma, camera_state, camera_sigma)
        sequence, dtype = self.native._encode_text(packed)
        shapes = self.native._encode_vision(packed, sequence, dtype)
        self.native._encode_action(packed, sequence, dtype)
        state, attention, natten = build_packed_sequence(
            "two_way", packed_sequence=sequence, attn_modes=packed.attn_modes,
            split_lens=packed.split_lens, sample_lens=packed.sample_lens,
            packed_und_token_indexes=packed.text_indexes,
            packed_gen_token_indexes=torch.cat((packed.vision.sequence_indexes, packed.action.sequence_indexes)).sort().values,
            num_heads=16, is_image_batch=False, head_dim=128, num_layers=28,
            token_shapes=packed.vision.token_shapes, natten_parameter_list=None,
            cp_world_size=1, video_temporal_causal=False,
            vision_token_shapes=packed.vision.token_shapes,
            action_token_shapes=packed.action.token_shapes, num_action_tokens_per_supertoken=0,
            null_action_supertokens=False, pad_for_cuda_graphs=False,
            prepared_metadata=packed.get_sequence_pack_metadata(),
        )
        # Native generator positions remain floating point for FPS modulation.
        cos, sin = self.native.language_model.model.rotary_emb(
            get_gen_seq(state), position_ids=packed.position_ids.unsqueeze(1))
        native_rope = (from_all_seq(cos.squeeze(0), state), from_all_seq(sin.squeeze(0), state))
        # Camera tokens participate in native GEN attention only. The existing
        # bridges still read/write precisely the same 1600 video and98 motion
        # rows per sample; do not create direct camera-motion bridge edges.
        labels = torch.arange(sequence.shape[0], device=sequence.device)[:, None]
        gen_labels = get_gen_seq(from_all_seq(labels, state)).flatten().long()
        inverse = torch.full((sequence.shape[0],), -1, device=sequence.device, dtype=torch.long)
        inverse[gen_labels] = torch.arange(len(gen_labels), device=sequence.device)
        vision_indices = inverse[packed.vision.sequence_indexes].reshape(batch, 1600)
        flat_vision = vision_indices.flatten()
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
                        g, m = block(gen_value.index_select(0, vision_indices[sample]),
                                     motion_value[sample * 98:(sample + 1) * 98], meta)
                        gen_parts.append(g)
                        motion_parts.append(m)
                    return gen_value.index_copy(0, flat_vision, torch.cat(gen_parts)), torch.cat(motion_parts)

                gen, motion_state = (checkpoint(bridge_step, gen, motion_state, use_reentrant=False)
                                     if use_checkpoint else bridge_step(gen, motion_state))
                state = from_und_gen_splits(und, gen, state)

        backbone = self.native.language_model.model
        state = from_und_gen_splits(backbone.norm(get_und_seq(state)),
                                    backbone.norm_moe_gen(get_gen_seq(state)), state)
        output = {}
        self.native._decode_vision(packed, get_all_seq(state), output, shapes)
        # The camera head is frozen and is decoded only for prediction. Native
        # attention remains differentiable through camera hidden states, while
        # this final readout has no loss edge or parameter gradients.
        with torch.no_grad():
            self.native._decode_action(packed, get_all_seq(state).detach(), output)
            output["camera_velocity"] = torch.stack(output["preds_action"]).float().detach()
        if output["camera_velocity"].shape != (batch, 96, 64):
            raise RuntimeError("Native camera output must cover96 transitions per sample")
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
        output["anchor"] = sum(p.flatten()[0] * 0.0 for p in trainable_parameters(self))
        output["packed"] = packed
        return output

