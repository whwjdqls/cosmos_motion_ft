"""Inference-only camera CFG; no training or checkpoint changes.

In this isolated configuration sample_args.guidance denotes CAMERA guidance.
Text guidance stays at 1. Both branches retain exactly the same text and image.
"""
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
import inspect
import math

from native_phase_training.inference_config import make_config as native_make_config

_request = ContextVar("camera_cfg_request", default=None)


def install_camera_cfg():
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    if getattr(OmniMoTModel, "_isolated_camera_cfg_installed", False):
        return
    original_generate = OmniMoTModel.generate_samples_from_batch
    original_velocity = OmniMoTModel._get_velocity
    original_reuse = OmniMoTModel._can_reuse_inference_text_kv
    signature = inspect.signature(original_generate)

    @wraps(original_generate)
    def generate(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        scale = float(bound.arguments.get("guidance", 1.5))
        if not math.isfinite(scale) or scale < 1:
            raise ValueError("This camera-CFG experiment requires a finite scale >= 1")
        if bound.arguments.get("velocity_postprocess_builder") is not None:
            raise ValueError("Camera CFG cannot be combined with another intervention")
        if bound.arguments.get("normalize_cfg", False):
            raise ValueError("CFG rescaling is disabled for this controlled comparison")
        if bound.arguments.get("guidance_interval") is not None:
            raise ValueError("This experiment guides at every denoising step")
        if self.parallel_dims is not None and self.parallel_dims.cfgp_enabled:
            raise ValueError("Use cfgp-size=1 for the sequential camera-only branches")
        bound.arguments["guidance"] = 1.0
        state = {"scale": scale, "conditional": 0, "camera_free": 0}
        token = _request.set(state)
        print(f"CAMERA_CFG_BEGIN scale={scale} text_guidance=1 same_image_text=True", flush=True)
        try:
            output = original_generate(*bound.args, **bound.kwargs)
            if state["conditional"] == 0:
                raise RuntimeError("Camera-CFG velocity hook was not invoked")
            print(f"CAMERA_CFG_END {state}", flush=True)
            return output
        finally:
            _request.reset(token)

    @wraps(original_reuse)
    def reuse_text_kv(self, *args, **kwargs):
        # Do not install a memory-aware attention dispatcher across different
        # sequence layouts. Disable it for baseline as well as guided samples.
        if _request.get() is not None:
            return False
        return original_reuse(self, *args, **kwargs)

    @wraps(original_velocity)
    def velocity(self, **kwargs):
        state = _request.get()
        if state is None:
            return original_velocity(self, **kwargs)
        plans = kwargs["sequence_plans"]
        if kwargs["has_noisy_actions"] or not all(p.has_action and p.has_vision for p in plans):
            raise ValueError("Camera CFG is restricted to camera-conditioned video generation")
        conditional = original_velocity(self, **kwargs)
        state["conditional"] += 1
        scale = state["scale"]
        if scale == 1.0:
            return conditional
        camera_free_args = dict(kwargs)
        camera_free_args.update(
            sequence_plans=[replace(p, has_action=False, condition_frame_indexes_action=[]) for p in plans],
            packed_sequence_template=None,
            memory=None,
        )
        # _get_velocity rebuilds GenerationDataClean without action fields when
        # has_action=False. It retains the same noisy vision, text and prefix.
        # No zero trajectory and no in-place modifications of shared tensors.
        camera_free = original_velocity(self, **camera_free_args)
        state["camera_free"] += 1
        return [c + (scale - 1.0) * (c - u) for c, u in zip(conditional, camera_free, strict=True)]

    OmniMoTModel.generate_samples_from_batch = generate
    OmniMoTModel._get_velocity = velocity
    OmniMoTModel._can_reuse_inference_text_kv = reuse_text_kv
    OmniMoTModel._isolated_camera_cfg_installed = True


def make_config():
    config = native_make_config()
    install_camera_cfg()
    return config
