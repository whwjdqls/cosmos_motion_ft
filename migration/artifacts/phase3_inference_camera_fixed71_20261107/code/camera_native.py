"""Native noisy camera tokens, preserving baseline video inputs and prompts."""
import torch


def pack_predictions(edge, captions, modes, latents, video_sigma, camera_state, camera_sigma):
    from cosmos_framework.data.generator.sequence_packing import pack_input_sequence
    from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
    from native_phase_training.latent_nymeria_dataset import _format_prompt_for_mode

    assert not any(captions), "Native generator captions must remain empty"
    batch = len(modes)
    if camera_state.shape != (batch, 96, 64) or camera_sigma.shape != (batch,):
        raise ValueError("Expected Bx96x64 camera tokens and B camera times")
    text, plans = [], []
    for caption, mode in zip(captions, modes, strict=True):
        if mode == "video2motion":
            caption, clean = "", list(range(25))
        elif mode in {"motimg2video", "joint"}:
            sample = {"ai_caption": caption, "conditioning_fps": 20.0,
                      "video": torch.empty(3, 97, 1, 1), "image_size": (256, 256)}
            caption = _format_prompt_for_mode(sample, "image2video", None)["ai_caption"]
            clean = [0]
        else:
            raise ValueError(mode)
        text.append(list(edge.tokenizer(caption, add_special_tokens=False)["input_ids"]))
        plans.append(SequencePlan(has_text=True, has_vision=True, has_action=True,
                                  condition_frame_indexes_vision=clean,
                                  condition_frame_indexes_action=[]))
    domains = [torch.tensor(2, device=camera_state.device) for _ in modes]
    data = GenerationDataClean(
        batch_size=batch, is_image_batch=False, x0_tokens_vision=list(latents),
        fps_vision=torch.full((batch,), 20.0), x0_tokens_action=list(camera_state),
        fps_action=torch.full((batch,), 20.0), action_domain_id=domains,
        raw_action_dim=[torch.tensor(9, device=camera_state.device) for _ in modes],
    )
    packed = pack_input_sequence(
        sequence_plans=plans, input_text_indexes=text, gen_data_clean=data,
        input_timesteps=video_sigma.detach().float().cpu() * 1000.0,
        special_tokens=edge.special_tokens, latent_patch_size=2,
        include_end_of_generation_token=False, unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000, enable_fps_modulation=True,
        base_fps=24.0, temporal_compression_factor=4,
        vision_temporal_position_mode="latent_index", action_dim=64,
    )
    packed.action.domain_id = domains
    # The native packer accepts one time per sample. Preserve the baseline's
    # video time exactly, then set only action's independent modality clock.
    packed.action.timesteps = (camera_sigma.detach().float().cpu() * 1000.0).repeat_interleave(96)
    packed.to_cuda()
    if len(packed.action.tokens) != batch or any(bool(mask.any()) for mask in packed.action.condition_mask):
        raise RuntimeError("Camera prediction must have no clean/conditioned action tokens")
    return packed
