"""Direct x0 motion head; native video/noising/trainer remain intact."""
from dataclasses import replace

import numpy as np
import torch

from native_phase_training.latent_omni_model_edge import LatentOmniMoTModelEdge
from native_motion_phase1.objectives import reconstruction_loss, x0_segment_to_velocity
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler


def sampling_sigmas(cfg, steps, shift):
    """Native UniPC quantizes model timesteps; retain its actual integration sigma."""
    schedule = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps, shift=cfg.shift,
        use_dynamic_shifting=cfg.use_dynamic_shifting)
    schedule.set_timesteps(steps, device="cpu", shift=float(shift))
    values = {int(t): float(s) for t, s in zip(schedule.timesteps, schedule.sigmas[:-1], strict=True)}
    if len(values) != steps:
        raise ValueError("This sampler adapter requires unique native timestep IDs")
    return values


class MotionX0Model(LatentOmniMoTModelEdge):
    def __init__(self, config, motion_mean_path, motion_std_path,
                 adaptation_mode="global_lora", model_family="edge"):
        super().__init__(config, adaptation_mode=adaptation_mode, model_family=model_family)
        rf = self.config.rectified_flow_training_config
        if rf.action_loss_weight != 0 or rf.independent_action_schedule:
            raise ValueError("x0 reconstruction replaces native action loss and retains the shared schedule")
        if rf.train_time_video_distribution != "waver" or rf.sample_level_loss_averaging:
            raise ValueError("Expected unchanged native WAVER scheduling and batch reduction")
        self.motion_mean = torch.from_numpy(np.load(motion_mean_path).astype(np.float32))
        self.motion_std = torch.from_numpy(np.load(motion_std_path).astype(np.float32))
        if self.motion_mean.shape != (283,) or self.motion_std.shape != (283,) or not bool((self.motion_std > 0).all()):
            raise ValueError("Invalid motion normalization")
        self._motion_targets = None
        self._sampling_sigmas = None

    def _add_noise_to_input(self, gen_data_clean, *args, **kwargs):
        # Capture exact clean labels before native packing replaces its payloads.
        # This does not draw RNG or alter the parent's interpolation/schedule.
        self._motion_targets = tuple(t.detach() for t in gen_data_clean.x0_tokens_action)
        return super()._add_noise_to_input(gen_data_clean, *args, **kwargs)

    def _compute_losses(self, out_net, data_batch_packed, gen_data_noised,
                        timesteps, is_image_batch, **kwargs):
        targets, self._motion_targets = self._motion_targets, None
        if targets is None:
            raise RuntimeError("Clean motion labels were not captured for this training batch")
        # Keep the parent's exact video loss and graph connectivity. Native
        # action velocity loss is explicitly disabled; its diagnostic is zero.
        native_out = dict(out_net, preds_action=[p * 0. for p in out_net["preds_action"]])
        native_noise = replace(gen_data_noised, vt_target_action=[torch.zeros_like(p) for p in out_net["preds_action"]])
        total, metrics = super()._compute_losses(native_out, data_batch_packed, native_noise,
                                                 timesteps, is_image_batch, **kwargs)
        motion, terms = reconstruction_loss(out_net["preds_action"], targets,
            data_batch_packed.action.condition_mask, self.motion_mean, self.motion_std)
        metrics["motion_reconstruction_loss"] = motion
        metrics.update({f"motion_{name}_loss": value for name, value in terms.items()})
        return total + motion, metrics

    def generate_samples_from_batch(self, data_batch, *, sampler=None, num_steps=35, shift=5., **kwargs):
        sampler = sampler or self.sampler
        if not isinstance(sampler, UniPCSampler):
            raise ValueError("The x0-motion adapter currently supports the native UniPC sampler")
        previous = self._sampling_sigmas
        self._sampling_sigmas = sampling_sigmas(sampler.cfg, num_steps, shift)
        try:
            return super().generate_samples_from_batch(data_batch, sampler=sampler,
                                                       num_steps=num_steps, shift=shift, **kwargs)
        finally:
            self._sampling_sigmas = previous

    def _get_velocity(self, **kwargs):
        # Native packing/network/visual masking run unchanged. The raw action
        # output now means x0, so convert that segment before native CFG/UniPC.
        output = super()._get_velocity(**kwargs)
        if not kwargs["has_noisy_actions"]:
            return output
        if self._sampling_sigmas is None:
            raise RuntimeError("Missing actual UniPC sigma for motion conversion")
        clean = kwargs["gen_data_clean"]
        if clean.num_vision_items_per_sample is not None or clean.x0_tokens_sound is not None or clean.x0_tokens_lidar is not None:
            raise ValueError("Expected the single-video, motion-only ablation layout")
        result = []
        for i, (value, state, plan) in enumerate(zip(output, kwargs["noise_x"], kwargs["sequence_plans"], strict=True)):
            timestep = int(kwargs["timestep"][i].reshape(-1)[0])
            result.append(x0_segment_to_velocity(value, state, clean.x0_tokens_vision[i].numel(),
                tuple(clean.x0_tokens_action[i].shape), plan.condition_frame_indexes_action,
                self._sampling_sigmas[timestep]))
        return result
