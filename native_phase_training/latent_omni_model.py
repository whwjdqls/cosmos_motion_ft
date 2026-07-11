"""Latent-aware native ``OmniMoTModel``.

The only behavioral difference from NVIDIA's native model is in
``get_data_and_condition``: if a batch includes ``video_latents`` we use those
as clean vision tokens instead of normalizing pixels and calling the Wan VAE.
All native rectified-flow noising, packing, losses, LoRA modules, and sampling
contracts remain unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel
from cosmos_framework.model.vfm.utils.data_and_condition import GenerationDataClean


class LatentOmniMoTModel(OmniMoTModel):
    """Native Cosmos model with an optional precomputed-video-latent input."""

    def _prepare_raw_video_metadata(self, raw: Any) -> tuple[list[torch.Tensor], list[int] | None]:
        """Flatten native joint-loader video metadata without normalizing pixels."""
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
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"video metadata item is not a tensor: {type(item)!r}")
            if item.dim() not in (4, 5):
                raise ValueError(f"video metadata tensor must be [C,T,H,W] or [B,C,T,H,W], got {tuple(item.shape)}")

        return flat, (counts if any(c > 1 for c in counts) else None)

    def _prepare_precomputed_video_latents(self, raw: Any) -> list[torch.Tensor]:
        """Return native clean vision-token list: ``[1,C,T,H,W]`` per sample."""
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 4:
                items = [raw]
            elif raw.dim() == 5:
                items = [raw[i : i + 1] for i in range(raw.shape[0])]
            else:
                raise ValueError(f"video_latents tensor must be 4D/5D, got {tuple(raw.shape)}")
        elif isinstance(raw, list):
            # IterativeJointDataLoader gives list[Tensor(1,C,T,H,W)].
            items = []
            for item in raw:
                if isinstance(item, list):
                    if len(item) != 1:
                        raise ValueError("video_latents nested list must contain one tensor per sample")
                    item = item[0]
                items.append(item)
        else:
            raise TypeError(f"unsupported video_latents type: {type(raw)!r}")

        out: list[torch.Tensor] = []
        for item in items:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"video_latents item is not a tensor: {type(item)!r}")
            if item.dim() == 4:
                item = item.unsqueeze(0)
            if item.dim() != 5 or item.shape[0] != 1:
                raise ValueError(f"video_latents item must have shape [1,C,T,H,W], got {tuple(item.shape)}")
            out.append(item.to(device=self.tensor_kwargs["device"], dtype=torch.float32).contiguous())
        return out

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor], iteration: int = 1) -> GenerationDataClean:
        if "video_latents" not in data_batch:
            return super().get_data_and_condition(data_batch, iteration=iteration)

        if self.is_image_batch(data_batch):
            raise ValueError("precomputed video_latents path only supports video batches, not image batches")
        if self.input_video_key not in data_batch:
            raise ValueError(f"video_latents path still requires {self.input_video_key!r} dummy video metadata")

        sample_vision_list, detected_num_vision_items = self._prepare_raw_video_metadata(data_batch[self.input_video_key])
        data_batch[self.input_video_key] = sample_vision_list
        if "num_vision_items_per_sample" not in data_batch:
            data_batch["num_vision_items_per_sample"] = detected_num_vision_items
        num_vision_items_per_sample = data_batch["num_vision_items_per_sample"]

        batch_size = (
            len(sample_vision_list)
            if num_vision_items_per_sample is None
            else len(num_vision_items_per_sample)
        )

        x0_tokens_vision = self._prepare_precomputed_video_latents(data_batch["video_latents"])
        if len(x0_tokens_vision) != batch_size:
            raise ValueError(
                f"video_latents batch mismatch: got {len(x0_tokens_vision)} latent items for batch_size {batch_size}"
            )
        frame_size = data_batch.get("image_size", None)
        if frame_size is not None:
            x0_tokens_vision = self._remove_padding_from_latent(x0_tokens_vision, frame_size)

        raw_state_action, action_domain_id = self._normalize_action_databatch(data_batch)
        x0_tokens_action = raw_state_action
        raw_action_dim = data_batch.get("raw_action_dim", None)

        self._normalize_sound_databatch_inplace(data_batch)
        raw_state_sound = data_batch.get("sound", None)
        if raw_state_sound is not None and self.tokenizer_sound_gen is not None:
            x0_tokens_sound = [self.encode_sound(s).contiguous().float() for s in raw_state_sound]
        else:
            x0_tokens_sound = None

        fps_raw = data_batch.get("conditioning_fps", None)
        if isinstance(fps_raw, list):
            fps_raw = torch.stack([x.reshape(-1)[0] for x in fps_raw]).flatten()
        fps_vision = fps_raw.to(**self.tensor_kwargs) if fps_raw is not None else None
        fps_action = fps_raw.to(**self.tensor_kwargs) if fps_raw is not None else None

        if x0_tokens_sound is not None:
            sound_batch_size = len(x0_tokens_sound)
            fps_sound = torch.full(
                (sound_batch_size,),
                self._get_sound_fps_for_rope(),
                dtype=torch.float32,
            ).to(**self.tensor_kwargs)
        else:
            fps_sound = None

        return GenerationDataClean(
            batch_size=batch_size,
            is_image_batch=False,
            raw_state_vision=sample_vision_list,
            raw_state_action=raw_state_action,
            raw_state_sound=raw_state_sound,
            x0_tokens_vision=x0_tokens_vision,
            x0_tokens_action=x0_tokens_action,
            x0_tokens_sound=x0_tokens_sound,
            fps_vision=fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
            action_domain_id=action_domain_id,
            raw_action_dim=raw_action_dim,
            num_vision_items_per_sample=num_vision_items_per_sample,
        )
