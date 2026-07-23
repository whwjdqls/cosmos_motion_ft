"""Local official-inference extension for multi-frame action-video prefixes."""

from __future__ import annotations

from typing import Any


def install_action_prefix_support() -> None:
    """Make official inference honor and validate explicit latent prefixes."""
    import cosmos_framework.inference.inference as inference_module

    if getattr(inference_module, "_native_phase_prefix_patch_installed", False):
        return

    original_get_sample_data = inference_module.get_sample_data

    def get_sample_data_with_action_prefix(sample_args: Any, model: Any, *, device: Any = "cuda") -> dict:
        data_batch = original_get_sample_data(sample_args, model, device=device)
        mode = sample_args.model_mode.value
        if mode not in {"forward_dynamics", "policy", "image2video"}:
            return data_batch

        requested = list(sample_args.condition_frame_indexes_vision or [])
        if not requested:
            raise ValueError(f"{mode} requires at least one clean vision latent frame")
        if requested != list(range(len(requested))):
            raise ValueError(f"{mode} requires a contiguous causal prefix, got {requested}")
        plans = data_batch.get("sequence_plan")
        if not isinstance(plans, list) or not plans:
            raise ValueError(f"{mode} inference returned no sequence plans")
        for plan in plans:
            plan.condition_frame_indexes_vision = requested.copy()
        return data_batch

    inference_module.get_sample_data = get_sample_data_with_action_prefix
    inference_module._native_phase_prefix_patch_installed = True
