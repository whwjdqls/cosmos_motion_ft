"""Cached-latent native Cosmos3-Edge model adapter.

This mirrors :mod:`latent_omni_model` while targeting the Edge-capable
``model.generator`` API.  Only VAE encoding is bypassed; native noising,
packing, losses, action heads, LoRA, EMA, and inference remain unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.utils import log


class LatentOmniMoTModelEdge(OmniMoTModel):
    """Cosmos3-Edge with optional precomputed Wan2.2 video latents."""

    _ACTION_MODULES = ("action2llm", "llm2action", "action_modality_embed")

    def __init__(
        self,
        config,
        adaptation_mode: str = "global_lora",
        model_family: str = "edge",
    ) -> None:
        self.adaptation_mode = str(adaptation_mode)
        self.model_family = str(model_family).lower()
        if self.model_family != "edge":
            raise ValueError(f"LatentOmniMoTModelEdge requires model_family='edge', got {model_family!r}")
        if self.adaptation_mode not in {"global_lora", "action_only", "camera_kv_lora"}:
            raise ValueError(f"unsupported Phase-1 adaptation mode: {self.adaptation_mode!r}")
        super().__init__(config)

    def add_lora(
        self,
        network: torch.nn.Module,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: str,
    ) -> torch.nn.Module:
        network = super().add_lora(
            network,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
        )
        if self.adaptation_mode == "camera_kv_lora":
            from native_phase_training.camera_token_lora import install_camera_token_lora

            network = install_camera_token_lora(network)
        return network

    def install_attention_dispatch(self, net: torch.nn.Module) -> None:
        super().install_attention_dispatch(net)
        for name, parameter in net.named_parameters():
            is_action = any(module in name for module in self._ACTION_MODULES)
            is_lora = "lora_" in name
            parameter.requires_grad_(is_action or (self.adaptation_mode != "action_only" and is_lora))

    def set_up_model(self) -> None:
        super().set_up_model()
        self._validate_and_log_trainable_parameters()

    def _validate_and_log_trainable_parameters(self) -> None:
        named = list(self.net.named_parameters())
        trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
        trainable_names = [name for name, _ in trainable]
        lora_names = [name for name, _ in named if "lora_" in name]

        if self.adaptation_mode == "action_only":
            if lora_names:
                raise RuntimeError(f"action_only unexpectedly instantiated LoRA parameters: {lora_names[:8]}")
            unexpected = [
                name for name in trainable_names if not any(module in name for module in self._ACTION_MODULES)
            ]
        else:
            if not lora_names:
                raise RuntimeError(f"{self.adaptation_mode} instantiated no LoRA parameters")
            unexpected = [
                name
                for name in trainable_names
                if "lora_" not in name and not any(module in name for module in self._ACTION_MODULES)
            ]
        if unexpected:
            raise RuntimeError(f"unexpected trainable Phase-1 parameters: {unexpected}")

        expected_targets = {
            "global_lora": ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"),
            "camera_kv_lora": ("k_proj_moe_gen", "v_proj_moe_gen"),
        }
        if self.adaptation_mode in expected_targets:
            missing = [
                target for target in expected_targets[self.adaptation_mode] if not any(target in name for name in lora_names)
            ]
            if missing:
                raise RuntimeError(f"{self.adaptation_mode} is missing adapters for {missing}")
        if self.adaptation_mode == "camera_kv_lora":
            invalid = [
                name for name in lora_names if "k_proj_moe_gen" not in name and "v_proj_moe_gen" not in name
            ]
            if invalid:
                raise RuntimeError(f"camera_kv_lora has non-K/V adapters: {invalid[:8]}")

        total_count = sum(parameter.numel() for _, parameter in named)
        trainable_count = sum(parameter.numel() for _, parameter in trainable)
        log.info(
            f"Edge Phase-1 adaptation: mode={self.adaptation_mode}, total={total_count:,}, "
            f"trainable={trainable_count:,} ({100.0 * trainable_count / max(total_count, 1):.4f}%), "
            f"lora_tensors={len(lora_names)}"
        )
        log.info("Edge Phase-1 trainable parameter names:\n" + "\n".join(trainable_names))

    @staticmethod
    def _prepare_raw_video_metadata(raw: Any) -> tuple[list[torch.Tensor], list[int] | None]:
        if not isinstance(raw, list):
            raise TypeError(f"video metadata must be a list, got {type(raw)!r}")
        counts: list[int] = []
        flat: list[torch.Tensor] = []
        for item in raw:
            if isinstance(item, (list, tuple)):
                if not item:
                    raise ValueError("video metadata item list is empty")
                counts.append(len(item))
                flat.extend(item)
            else:
                counts.append(1)
                flat.append(item)
        for item in flat:
            if not isinstance(item, torch.Tensor) or item.dim() not in (4, 5):
                raise ValueError(f"invalid dummy video metadata item: {type(item)!r} {getattr(item, 'shape', None)}")
        return flat, counts if any(count > 1 for count in counts) else None

    def _prepare_precomputed_video_latents(self, raw: Any) -> list[torch.Tensor]:
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 4:
                items = [raw]
            elif raw.dim() == 5:
                items = [raw[index : index + 1] for index in range(raw.shape[0])]
            else:
                raise ValueError(f"video_latents tensor must be 4D/5D, got {tuple(raw.shape)}")
        elif isinstance(raw, list):
            items = []
            for item in raw:
                if isinstance(item, list):
                    if len(item) != 1:
                        raise ValueError("video_latents nested list must contain one tensor per sample")
                    item = item[0]
                items.append(item)
        else:
            raise TypeError(f"unsupported video_latents type: {type(raw)!r}")

        output: list[torch.Tensor] = []
        for item in items:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"video_latents item is not a tensor: {type(item)!r}")
            if item.dim() == 4:
                item = item.unsqueeze(0)
            if item.dim() != 5 or item.shape[0] != 1:
                raise ValueError(f"video_latents item must be [1,C,T,H,W], got {tuple(item.shape)}")
            output.append(item.to(**self.tensor_kwargs_fp32).contiguous())
        return output

    def get_data_and_condition(
        self,
        data_batch: dict[str, torch.Tensor],
        vision_condition_indexes: list[list[int]] | None = None,
        retain_raw_state_vision: bool = True,
        balance_vae_encode: bool = False,
    ) -> GenerationDataClean:
        if "video_latents" not in data_batch:
            return super().get_data_and_condition(
                data_batch,
                vision_condition_indexes=vision_condition_indexes,
                retain_raw_state_vision=retain_raw_state_vision,
                balance_vae_encode=balance_vae_encode,
            )
        del vision_condition_indexes, balance_vae_encode
        if self.is_image_batch(data_batch):
            raise ValueError("precomputed video_latents only support video batches")
        if self.input_video_key not in data_batch:
            raise ValueError(f"video_latents require {self.input_video_key!r} dummy metadata")
        temporal_mode = self.config.diffusion_expert_config.vision_temporal_position_mode
        if temporal_mode != "latent_index":
            raise ValueError(
                "cached Edge latents require vision_temporal_position_mode='latent_index', "
                f"got {temporal_mode!r}"
            )

        sample_vision_list, detected_counts = self._prepare_raw_video_metadata(data_batch[self.input_video_key])
        data_batch[self.input_video_key] = sample_vision_list
        if "num_vision_items_per_sample" not in data_batch:
            data_batch["num_vision_items_per_sample"] = detected_counts
        counts = data_batch["num_vision_items_per_sample"]
        batch_size = len(sample_vision_list) if counts is None else len(counts)

        x0_tokens_vision = self._prepare_precomputed_video_latents(data_batch["video_latents"])
        if len(x0_tokens_vision) != batch_size:
            raise ValueError(f"got {len(x0_tokens_vision)} latent items for batch_size={batch_size}")
        frame_size = data_batch.get("image_size")
        if frame_size is not None:
            x0_tokens_vision = self._remove_padding_from_latent(x0_tokens_vision, frame_size)

        raw_state_action, action_domain_id, action_family = self._normalize_action_databatch(data_batch)
        self._normalize_sound_databatch_inplace(data_batch)
        raw_state_sound = data_batch.get("sound")
        x0_tokens_sound = (
            [self.encode_sound(sound).contiguous().float() for sound in raw_state_sound]
            if raw_state_sound is not None and self.tokenizer_sound_gen is not None
            else None
        )

        fps_raw = data_batch.get("conditioning_fps")
        if isinstance(fps_raw, list):
            fps_raw = torch.stack([value.reshape(-1)[0] for value in fps_raw]).flatten()
        fps_vision = fps_raw.to(**self.tensor_kwargs) if fps_raw is not None else None
        fps_action = fps_vision
        if "conditioning_fps_action" in data_batch:
            fps_action_raw = data_batch["conditioning_fps_action"]
            if isinstance(fps_action_raw, list):
                fps_action_raw = torch.stack(fps_action_raw).flatten()
            fps_action = fps_action_raw.to(**self.tensor_kwargs)
        fps_sound = None
        if x0_tokens_sound is not None:
            fps_sound = torch.full(
                (len(x0_tokens_sound),), self._get_sound_fps_for_rope(), dtype=torch.float32
            ).to(**self.tensor_kwargs)

        return GenerationDataClean(
            batch_size=batch_size,
            is_image_batch=False,
            raw_state_vision=sample_vision_list if retain_raw_state_vision else None,
            x0_tokens_vision=x0_tokens_vision,
            fps_vision=fps_vision,
            temporal_positions_vision=None,
            num_vision_items_per_sample=counts,
            num_views_per_vision_item=None,
            raw_state_lidar=None,
            x0_tokens_lidar=None,
            fps_lidar=None,
            num_lidar_items_per_sample=None,
            raw_state_sound=raw_state_sound,
            x0_tokens_sound=x0_tokens_sound,
            fps_sound=fps_sound,
            raw_state_action=raw_state_action,
            x0_tokens_action=raw_state_action,
            fps_action=fps_action,
            action_domain_id=action_domain_id,
            action_family=action_family,
            raw_action_dim=data_batch.get("raw_action_dim"),
            action_valid_mask=data_batch.get("action_valid_mask"),
            control_weights=data_batch.get("control_weights"),
        )
