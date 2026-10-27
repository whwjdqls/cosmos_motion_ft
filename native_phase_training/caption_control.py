"""Explicit, config-persisted semantic-caption policy for Phase-1 datasets."""


def get_dataset_with_caption_policy(*, caption_conditioning: bool = True, **kwargs):
    from native_phase_training.latent_nymeria_dataset import get_nymeria_camera_latent_sft_dataset

    if not isinstance(caption_conditioning, bool):
        raise TypeError("caption_conditioning must be a boolean")
    dataset = get_nymeria_camera_latent_sft_dataset(**kwargs)
    if not hasattr(dataset, "_action_prompt"):
        raise TypeError("Caption policy requires the native Nymeria latent dataset")
    dataset.caption_conditioning = caption_conditioning
    print(f"[nativep1] mode={kwargs.get('mode')} caption_conditioning={caption_conditioning}", flush=True)
    return dataset
