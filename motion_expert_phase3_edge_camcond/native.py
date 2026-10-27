"""Native clean camera tokens with explicit per-sample modality omission."""
import torch

from .settings import has_camera, has_video, effective_caption


def pack_conditions(edge, captions, modes, video, sigma, camera_action):
    from cosmos_framework.data.generator.sequence_packing import pack_input_sequence
    from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
    from native_phase_training.latent_nymeria_dataset import _format_prompt_for_mode

    text, plans, latents, actions, video_samples, camera_samples = [], [], [], [], [], []
    for i, (caption, mode) in enumerate(zip(captions, modes, strict=True)):
        caption = effective_caption(mode, caption)
        if mode == "motimg2video":
            sample = {"ai_caption": caption, "conditioning_fps": 20.0,
                      "video": torch.empty(3, 97, 1, 1), "image_size": (256, 256)}
            caption = _format_prompt_for_mode(sample, "image2video", None)["ai_caption"]
        text.append(list(edge.tokenizer(caption, add_special_tokens=False)["input_ids"]))
        vision, camera = has_video(mode), has_camera(mode)
        plans.append(SequencePlan(
            has_text=True, has_vision=vision, has_action=camera,
            condition_frame_indexes_vision=([0] if mode == "motimg2video" else list(range(25))) if vision else [],
            condition_frame_indexes_action=list(range(96)) if camera else [],
        ))
        if vision:
            video_samples.append(i)
            latents.append(video[i].unsqueeze(0))
        if camera:
            if camera_action is None or camera_action.shape != (len(modes), 96, 64):
                raise ValueError("Camera conditions must be [B,96,64], raw9 padded with zeros")
            camera_samples.append(i)
            actions.append(camera_action[i])
    domains = [torch.tensor(2, device=video.device) for _ in actions]
    data = GenerationDataClean(
        batch_size=len(modes), is_image_batch=False,
        x0_tokens_vision=latents or None, fps_vision=torch.full((len(modes),), 20.0),
        x0_tokens_action=actions or None, fps_action=torch.full((len(actions),), 20.0),
        action_domain_id=domains or None,
        raw_action_dim=[torch.tensor(9, device=video.device) for _ in actions] or None,
    )
    packed = pack_input_sequence(
        sequence_plans=plans, input_text_indexes=text, gen_data_clean=data,
        input_timesteps=sigma.detach().float().cpu() * 1000.0,
        special_tokens=edge.special_tokens, latent_patch_size=2,
        include_end_of_generation_token=False,
        unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000,
        enable_fps_modulation=True, base_fps=24.0, temporal_compression_factor=4,
        vision_temporal_position_mode="latent_index", action_dim=64,
    )
    if actions:
        packed.action.domain_id = domains
    packed.to_cuda()
    return packed, video_samples, camera_samples
