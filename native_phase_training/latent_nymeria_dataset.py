"""Nymeria camera dataset that feeds cached Wan-VAE latents to native Cosmos.

Each sample keeps the native action-SFT contract: tokenized text, sequence plan,
camera action metadata, FPS, viewpoint, and image-size metadata.  The expensive
pixel video is replaced by:

* ``video``: a tiny dummy tensor shaped ``[3,T,1,1]`` for native metadata logic.
* ``video_latents``: cached Wan2.2-VAE latents shaped ``[C,T_lat,H_lat,W_lat]``.
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cosmos_framework.data.vfm.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.vfm.action.transforms import (
    build_sequence_plan_from_mode,
    pad_action_to_max_dim,
)
from cosmos_framework.data.vfm.augmentors.duration_fps_text_timestamps import DEFAULT_TEMPLATE as DURATION_TEMPLATE
from cosmos_framework.data.vfm.augmentors.resolution_text_info import DEFAULT_VIDEO_TEMPLATE as RESOLUTION_TEMPLATE
from cosmos_framework.data.vfm.augmentors.text_tokenizer import TextTokenizerTransform
from cosmos_framework.data.vfm.joint_dataloader import IterativeJointDataLoader

from nymeria_camera_rgb_dataset import MODE_WEIGHTS
from camera_to_action import DOMAIN_ID


_ACTION_MODES = frozenset({"forward_dynamics", "inverse_dynamics", "policy"})


def latent_path(uuid: str, start: int, root: str) -> str:
    uuid_safe = uuid.replace("/", "__")
    subj = uuid.split("/")[0] if "/" in uuid else "_misc"
    return os.path.join(root, subj, f"{uuid_safe}_{int(start)}.npz")


@lru_cache(maxsize=16)
def build_cached_index(
    manifest_path: str,
    split_file: str,
    split: str,
    num_frames: int,
    latent_root: str,
    require_usable: bool = True,
) -> list[dict[str, Any]]:
    keep_uuids = None
    if split not in ("all", None):
        sp = json.load(open(split_file))
        assert split in sp, f"split {split!r} not in {split_file}"
        keep_uuids = set(sp[split])

    index: list[dict[str, Any]] = []
    missing_latents = 0
    candidate_windows = 0
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            uuid = rec.get("uuid")
            nb = int(rec.get("nb_frames", 0))
            if keep_uuids is not None and uuid not in keep_uuids:
                continue
            if not rec.get("camera_path") or not rec.get("vision_path"):
                continue
            for w in rec.get("t2w_windows", []):
                if require_usable and not w.get("usable", False):
                    continue
                caption = w.get("caption")
                if not caption:
                    continue
                ws, we = int(w["start_frame"]), int(w["end_frame"])
                hi = min(we, nb)
                s = ws
                while s + num_frames <= hi:
                    candidate_windows += 1
                    lp = latent_path(uuid, s, latent_root)
                    if os.path.isfile(lp):
                        index.append({"uuid": uuid, "s": s, "cap": caption, "latent_path": lp})
                    else:
                        missing_latents += 1
                    s += num_frames
    print(
        f"[NymeriaCameraLatentDataset] split={split} T={num_frames} kept={len(index)} "
        f"missing_latents={missing_latents} candidates={candidate_windows} root={latent_root}",
        flush=True,
    )
    return index


class CyclingDataLoader:
    """Infinite view over a finite DataLoader for Cosmos joint-loader streams."""

    def __init__(self, *args, collate_fn=None, **kwargs) -> None:
        self.dataloader = DataLoader(*args, collate_fn=collate_fn, **kwargs)

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        epoch = 0
        while True:
            sampler = getattr(self.dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            yielded = False
            for batch in self.dataloader:
                yielded = True
                yield batch

            if not yielded:
                return
            epoch += 1


def _iter_cached_latents(value: Any):
    """Yield per-video ``[C,T,H,W]`` latent tensors from loader nesting."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 4:
            yield value
            return
        if value.ndim == 5 and value.shape[0] == 1:
            yield value[0]
            return
        raise ValueError(f"expected cached latents [C,T,H,W] or [1,C,T,H,W], got {tuple(value.shape)}")
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_cached_latents(item)
        return
    raise TypeError(f"unsupported video_latents container: {type(value).__name__}")


def _iter_raw_videos(value: Any):
    """Yield per-video ``[C,T,H,W]`` tensors from loader nesting."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 4:
            yield value
            return
        if value.ndim == 5 and value.shape[0] == 1:
            yield value[0]
            return
        raise ValueError(f"expected video [C,T,H,W] or [1,C,T,H,W], got {tuple(value.shape)}")
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_raw_videos(item)
        return
    raise TypeError(f"unsupported video container: {type(value).__name__}")


class LatentAwareIterativeJointDataLoader(IterativeJointDataLoader):
    """Count cached latent patches instead of the dataset's 1x1 dummy pixels.

    The parent loader already counts text, EOS, vision boundary markers, action,
    and optional sound tokens. Its vision patch count is zero for our 1x1 dummy
    video, so this override adds the real patchified latent token count.
    """

    def _compute_num_tokens_per_sample(self, data_batch: dict) -> int:
        num_tokens = super()._compute_num_tokens_per_sample(data_batch)
        if "video_latents" not in data_batch:
            return num_tokens

        videos = list(_iter_raw_videos(data_batch["video"]))
        latents = list(_iter_cached_latents(data_batch["video_latents"]))
        if len(videos) != len(latents):
            raise ValueError(f"video/video_latents item mismatch: {len(videos)} != {len(latents)}")

        for video, latent in zip(videos, latents, strict=True):
            if tuple(video.shape[-2:]) != (1, 1):
                raise ValueError(
                    "latent-aware packing expects the cached-latent path's 1x1 dummy video, "
                    f"got spatial shape {tuple(video.shape[-2:])}"
                )
            latent_t, latent_h, latent_w = map(int, latent.shape[-3:])
            if latent_h % self.patch_spatial or latent_w % self.patch_spatial:
                raise ValueError(
                    f"latent spatial shape {(latent_h, latent_w)} is not divisible by patch size {self.patch_spatial}"
                )
            num_tokens += latent_t * (latent_h // self.patch_spatial) * (latent_w // self.patch_spatial)
        return num_tokens


def _format_prompt_for_mode(
    sample: dict[str, Any],
    mode: str,
    action_formatter: ActionPromptJsonFormatter,
) -> dict[str, Any]:
    """Match the prompt representation used by official Cosmos inference."""
    if mode in _ACTION_MODES:
        return action_formatter(sample)

    if mode != "image2video":
        raise ValueError(f"unsupported mode for prompt formatting: {mode!r}")

    caption = sample["ai_caption"].strip()
    fps_value = sample["conditioning_fps"]
    fps = float(fps_value.item()) if isinstance(fps_value, torch.Tensor) else float(fps_value)
    num_frames = int(sample["video"].shape[1])
    image_size = sample["image_size"]
    height, width = int(image_size[0]), int(image_size[1])

    duration_text = DURATION_TEMPLATE.format(duration=num_frames / fps, fps=fps)
    caption = caption.rstrip(".") + ". " + duration_text
    resolution_text = RESOLUTION_TEMPLATE.format(height=height, width=width)
    sample["ai_caption"] = caption.strip().rstrip(".") + ". " + resolution_text
    return sample


class NymeriaCameraLatentDataset(Dataset):
    def __init__(
        self,
        manifest_path: str = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl",
        split_file: str = "/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json",
        latent_root: str = "/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97",
        num_frames: int = 97,
        fps: float = 20.0,
        mode: str = "forward_dynamics",
        max_action_dim: int = 64,
        tokenizer_config: dict | None = None,
        cfg_dropout_rate: float = 0.1,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if num_frames % 4 != 1:
            raise ValueError(f"num_frames must be 4N+1 for Wan-VAE, got {num_frames}")
        if mode not in MODE_WEIGHTS and mode != "mixture":
            raise ValueError(f"mode must be one of {sorted(MODE_WEIGHTS)} or 'mixture', got {mode!r}")
        if tokenizer_config is None:
            raise ValueError("tokenizer_config is required because native training expects text_token_ids")

        self.num_frames = int(num_frames)
        self.fps = float(fps)
        self.mode = mode
        self.max_action_dim = int(max_action_dim)
        self._rng = random.Random(seed)
        self._modes = list(MODE_WEIGHTS)
        self._mode_weights = [MODE_WEIGHTS[m] for m in self._modes]

        index = build_cached_index(manifest_path, split_file, split, self.num_frames, latent_root)
        if max_samples is not None:
            index = index[:max_samples]
        if not index:
            raise RuntimeError(f"no cached latent windows found under {latent_root} for split={split}")
        self._index = index

        self._action_prompt = ActionPromptJsonFormatter()
        self._tokenizer = TextTokenizerTransform(
            input_keys=["ai_caption"],
            output_keys=["text_token_ids"],
            args={"tokenizer_config": tokenizer_config, "cfg_dropout_rate": cfg_dropout_rate},
        )

    def __len__(self) -> int:
        return len(self._index)

    def _choose_mode(self) -> str:
        if self.mode != "mixture":
            return self.mode
        return self._rng.choices(self._modes, weights=self._mode_weights, k=1)[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self._index[idx % len(self._index)]
        mode = self._choose_mode()
        with np.load(it["latent_path"]) as d:
            latents = d["latents"]
            action = d["camera_action"].astype(np.float32)
            image_size = d["image_size"].astype(np.float32)

        if action.shape[0] != self.num_frames - 1:
            raise ValueError(f"bad cached action shape {action.shape} in {it['latent_path']}")

        caption = "" if mode == "inverse_dynamics" else it["cap"]
        sample: dict[str, Any] = {
            "video": torch.empty((3, self.num_frames, 1, 1), dtype=torch.uint8),
            "video_latents": torch.from_numpy(latents),
            "image_size": torch.from_numpy(image_size),
            "ai_caption": caption,
            "conditioning_fps": torch.tensor(int(round(self.fps)), dtype=torch.long),
            "mode": mode,
            "viewpoint": "ego_view",
            "sequence_plan": build_sequence_plan_from_mode(
                mode=mode,
                video_length=self.num_frames,
                action_length=self.num_frames - 1,
                video_temporal_downsample=4,
            ),
        }

        if mode != "image2video":
            sample["action"] = pad_action_to_max_dim(torch.from_numpy(action).float(), self.max_action_dim)
            sample["raw_action_dim"] = torch.tensor(action.shape[1], dtype=torch.long)
            sample["domain_id"] = torch.tensor(DOMAIN_ID, dtype=torch.long)

        sample = _format_prompt_for_mode(
            sample,
            mode,
            self._action_prompt,
        )
        sample = self._tokenizer(sample)
        return sample


def get_nymeria_camera_latent_sft_dataset(**kwargs) -> NymeriaCameraLatentDataset:
    return NymeriaCameraLatentDataset(**kwargs)
