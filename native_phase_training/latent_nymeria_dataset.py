"""Nymeria camera dataset that feeds cached Wan-VAE latents to native Cosmos.

Each sample keeps the native action-SFT contract: tokenized text, sequence plan,
camera action metadata, FPS, viewpoint, and image-size metadata.  The expensive
pixel video is replaced by:

* ``video``: a tiny dummy tensor shaped ``[3,T,1,1]`` for native metadata logic.
* ``video_latents``: cached Wan2.2-VAE latents shaped ``[C,T_lat,H_lat,W_lat]``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

try:  # Cosmos3 Edge-capable framework (generator namespace).
    from cosmos_framework.data.generator.action.utils.action_processing import pad_action_to_max_dim
    from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
    from cosmos_framework.data.generator.action.utils.transforms import build_sequence_plan_from_mode
    from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import (
        DEFAULT_TEMPLATE as DURATION_TEMPLATE,
    )
    from cosmos_framework.data.generator.augmentors.resolution_text_info import (
        DEFAULT_VIDEO_TEMPLATE as RESOLUTION_TEMPLATE,
    )
    from cosmos_framework.data.generator.augmentors.text_tokenizer import TextTokenizerTransform
    from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
    from cosmos_framework.data.generator.sequence_packing import SequencePlan
except ImportError:  # Pinned Nano framework used by the historical runs.
    from cosmos_framework.data.vfm.action.json_formatter import ActionPromptJsonFormatter
    from cosmos_framework.data.vfm.action.transforms import (
        build_sequence_plan_from_mode,
        pad_action_to_max_dim,
    )
    from cosmos_framework.data.vfm.augmentors.duration_fps_text_timestamps import (
        DEFAULT_TEMPLATE as DURATION_TEMPLATE,
    )
    from cosmos_framework.data.vfm.augmentors.resolution_text_info import (
        DEFAULT_VIDEO_TEMPLATE as RESOLUTION_TEMPLATE,
    )
    from cosmos_framework.data.vfm.augmentors.text_tokenizer import TextTokenizerTransform
    from cosmos_framework.data.vfm.joint_dataloader import IterativeJointDataLoader
    from cosmos_framework.data.vfm.sequence_packing import SequencePlan

# Keep this cached-latent dataset independent of the pixel dataset's framework
# imports.  The values are the native Cosmos action-SFT contract.
_BASE_MODE_WEIGHTS = {
    "forward_dynamics": 0.40,
    "inverse_dynamics": 0.25,
    "policy": 0.20,
    "image2video": 0.15,
}
_DROPPED_MODES = {mode.strip() for mode in os.environ.get("NYMERIA_DROP_MODES", "").split(",") if mode.strip()}
MODE_WEIGHTS = {mode: weight for mode, weight in _BASE_MODE_WEIGHTS.items() if mode not in _DROPPED_MODES}
DOMAIN_ID = 2  # Native Cosmos embodiment domain ``camera_pose``.
from native_phase_training.latent_cache_contract import (
    CACHE_CONTRACT_FILENAME,
    LatentCacheContract,
    load_latent_cache_contract,
    validate_cached_sample,
)
from runtime_paths import WEKA_ROOT


_ACTION_MODES = frozenset({"forward_dynamics", "inverse_dynamics", "policy"})
_VISUAL_GENERATION_MODES = frozenset({"forward_dynamics", "policy", "image2video"})
_QUALITY_FILTER_KIND = "nymeria_camera_motion_quality_filter"
_QUALITY_FILTER_VERSION = 1
_STANDALONE_C_PATTERN = re.compile(r"(?<!\w)C(?!\w)")
_STANDALONE_C_SUBJECTS = {
    "person": ("The person", "the person"),
    "camera_wearer": ("The camera wearer", "the camera wearer"),
}


def replace_standalone_c(caption: str, subject: str) -> str:
    """Expand Nymeria's whole-token subject marker without touching words."""
    if subject not in _STANDALONE_C_SUBJECTS:
        raise ValueError(
            f"standalone C subject must be one of {sorted(_STANDALONE_C_SUBJECTS)}, "
            f"got {subject!r}"
        )
    sentence_initial, sentence_internal = _STANDALONE_C_SUBJECTS[subject]

    def replacement(match: re.Match[str]) -> str:
        prefix = match.string[: match.start()].rstrip()
        return sentence_initial if not prefix or prefix[-1] in ".!?" else sentence_internal

    return _STANDALONE_C_PATTERN.sub(replacement, caption)


def replace_standalone_c_with_person(caption: str) -> str:
    """Historical standalone-C rewrite used by Nano and early Edge runs."""
    return replace_standalone_c(caption, "person")


def replace_standalone_c_with_camera_wearer(caption: str) -> str:
    """Rewrite the subject marker as the off-camera egocentric actor."""
    return replace_standalone_c(caption, "camera_wearer")


def rgb_prefix_to_latent_frames(prefix_length: int, num_frames: int) -> int:
    """Map an exact causal Wan-VAE RGB prefix to its clean latent count."""
    if not isinstance(prefix_length, int) or isinstance(prefix_length, bool):
        raise TypeError(f"prefix length must be an integer, got {prefix_length!r}")
    if prefix_length < 1 or prefix_length >= num_frames:
        raise ValueError(f"prefix length must be in [1,{num_frames - 1}], got {prefix_length}")
    if (prefix_length - 1) % 4:
        raise ValueError(
            f"RGB prefix {prefix_length} is not an exact causal Wan-VAE boundary; "
            "expected 1 + 4N frames"
        )
    return 1 + (prefix_length - 1) // 4


def validate_prefix_sampling(
    prefix_lengths: list[int] | tuple[int, ...],
    prefix_sampling_weights: list[float] | tuple[float, ...] | None,
    num_frames: int,
) -> tuple[tuple[int, ...], tuple[float, ...] | None]:
    """Validate and canonicalize the visual-prefix sampling contract."""
    if any(not isinstance(value, int) or isinstance(value, bool) for value in prefix_lengths):
        raise TypeError(f"prefix_lengths must contain only integers, got {prefix_lengths!r}")
    lengths = tuple(prefix_lengths)
    if not lengths:
        raise ValueError("prefix_lengths must contain at least one RGB prefix")
    if len(lengths) != len(set(lengths)):
        raise ValueError(f"prefix_lengths contains duplicates: {lengths}")
    for prefix_length in lengths:
        rgb_prefix_to_latent_frames(prefix_length, num_frames)

    if prefix_sampling_weights is None:
        return lengths, None
    weights = tuple(float(value) for value in prefix_sampling_weights)
    if len(weights) != len(lengths):
        raise ValueError(
            f"prefix_sampling_weights length {len(weights)} does not match prefix_lengths {len(lengths)}"
        )
    if any(not np.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError(f"prefix_sampling_weights must be finite and non-negative, got {weights}")
    if not any(value > 0.0 for value in weights):
        raise ValueError("prefix_sampling_weights must contain at least one positive value")
    return lengths, weights


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def load_quality_filter_exclusions(
    path: str,
    num_frames: int,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Load and strictly validate a versioned physical-window exclusion artifact."""
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"quality filter does not exist: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"quality filter is empty: {path}")

    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"quality filter root must be an object: {path}")
    if payload.get("kind") != _QUALITY_FILTER_KIND:
        raise ValueError(
            f"quality filter kind must be {_QUALITY_FILTER_KIND!r}, got {payload.get('kind')!r}"
        )
    if payload.get("version") != _QUALITY_FILTER_VERSION:
        raise ValueError(
            f"quality filter version must be {_QUALITY_FILTER_VERSION}, got {payload.get('version')!r}"
        )
    if payload.get("num_frames") != num_frames:
        raise ValueError(
            f"quality filter T mismatch: artifact={payload.get('num_frames')!r}, requested={num_frames}"
        )

    raw_exclusions = payload.get("excluded_windows")
    if not isinstance(raw_exclusions, list):
        raise ValueError("quality filter excluded_windows must be a list")
    exclusions: dict[tuple[str, int, int], dict[str, Any]] = {}
    split_counts: Counter[str] = Counter()
    for index, entry in enumerate(raw_exclusions):
        if not isinstance(entry, dict):
            raise ValueError(f"excluded_windows[{index}] must be an object")
        split = entry.get("split")
        uuid = entry.get("uuid")
        start = entry.get("start")
        end = entry.get("end")
        reasons = entry.get("reasons")
        if split not in {"train", "test"}:
            raise ValueError(f"excluded_windows[{index}] has invalid split {split!r}")
        if not isinstance(uuid, str) or not uuid:
            raise ValueError(f"excluded_windows[{index}] has invalid uuid {uuid!r}")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"excluded_windows[{index}] has invalid start {start!r}")
        if not isinstance(end, int) or isinstance(end, bool) or end - start != num_frames:
            raise ValueError(
                f"excluded_windows[{index}] must span exactly T={num_frames}, got [{start},{end})"
            )
        if not isinstance(reasons, list) or not reasons or not all(
            isinstance(reason, str) and reason for reason in reasons
        ):
            raise ValueError(f"excluded_windows[{index}] has invalid reasons {reasons!r}")
        key = (uuid, start, end)
        if key in exclusions:
            raise ValueError(f"quality filter contains duplicate physical window {key}")
        exclusions[key] = {"split": split, "reasons": tuple(reasons)}
        split_counts[split] += 1

    summaries = payload.get("summary_by_split")
    if not isinstance(summaries, dict):
        raise ValueError("quality filter summary_by_split must be an object")
    for split in ("train", "test"):
        summary = summaries.get(split)
        expected = summary.get("excluded_unique_physical_windows") if isinstance(summary, dict) else None
        if expected != split_counts[split]:
            raise ValueError(
                f"quality filter {split} exclusion count mismatch: summary={expected!r}, "
                f"entries={split_counts[split]}"
            )

    print(
        f"[NymeriaCameraLatentDataset] quality_filter={path} sha256={_sha256(path)} "
        f"train_unique={split_counts['train']} test_unique={split_counts['test']}",
        flush=True,
    )
    return exclusions


def latent_path(uuid: str, start: int, root: str) -> str:
    uuid_safe = uuid.replace("/", "__")
    subj = uuid.split("/")[0] if "/" in uuid else "_misc"
    return os.path.join(root, subj, f"{uuid_safe}_{int(start)}.npz")


def validate_training_cache_contract(
    *,
    latent_root: str,
    num_frames: int,
    fps: float,
    model_resolution_tier: str | None,
    expected_latent_hw: int | None,
    expected_image_hw: int | None,
    require_contract: bool,
) -> LatentCacheContract | None:
    """Validate cache-wide geometry before workers start reading samples."""

    contract_path = os.path.join(latent_root, CACHE_CONTRACT_FILENAME)
    if not os.path.isfile(contract_path):
        if require_contract:
            raise FileNotFoundError(
                f"required latent-cache contract is missing: {contract_path}"
            )
        return None

    contract = load_latent_cache_contract(latent_root)
    if contract.num_frames != num_frames:
        raise ValueError(
            f"latent-cache T mismatch: cache={contract.num_frames} requested={num_frames}"
        )
    if abs(contract.fps - fps) > 1e-6:
        raise ValueError(f"latent-cache FPS mismatch: cache={contract.fps} requested={fps}")
    if (
        model_resolution_tier is not None
        and contract.model_resolution_tier != str(model_resolution_tier)
    ):
        raise ValueError(
            "latent-cache model tier mismatch: "
            f"cache={contract.model_resolution_tier!r} requested={model_resolution_tier!r}"
        )
    if expected_latent_hw is not None and contract.expected_latent_shape[-2:] != (
        expected_latent_hw,
        expected_latent_hw,
    ):
        raise ValueError(
            "latent-cache spatial shape mismatch: "
            f"cache={contract.expected_latent_shape[-2:]} "
            f"requested={(expected_latent_hw, expected_latent_hw)}"
        )
    if expected_image_hw is not None and contract.expected_image_hw != (
        expected_image_hw,
        expected_image_hw,
    ):
        raise ValueError(
            "latent-cache image shape mismatch: "
            f"cache={contract.expected_image_hw} "
            f"requested={(expected_image_hw, expected_image_hw)}"
        )
    return contract


@lru_cache(maxsize=16)
def build_cached_index(
    manifest_path: str,
    split_file: str,
    split: str,
    num_frames: int,
    latent_root: str,
    quality_filter_path: str = "",
    require_usable: bool = True,
) -> list[dict[str, Any]]:
    keep_uuids = None
    if split not in ("all", None):
        with open(split_file) as handle:
            sp = json.load(handle)
        assert split in sp, f"split {split!r} not in {split_file}"
        keep_uuids = set(sp[split])

    exclusions = load_quality_filter_exclusions(quality_filter_path, num_frames)
    index: list[dict[str, Any]] = []
    missing_latents = 0
    candidate_windows = 0
    quality_filtered_rows = 0
    quality_reason_rows: Counter[str] = Counter()
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
                    exclusion = exclusions.get((uuid, s, s + num_frames))
                    if exclusion is not None:
                        if split not in ("all", None) and exclusion["split"] != split:
                            raise ValueError(
                                f"quality filter split mismatch for {(uuid, s, s + num_frames)}: "
                                f"artifact={exclusion['split']!r}, dataset={split!r}"
                            )
                        quality_filtered_rows += 1
                        quality_reason_rows.update(exclusion["reasons"])
                        s += num_frames
                        continue
                    lp = latent_path(uuid, s, latent_root)
                    if os.path.isfile(lp):
                        index.append({"uuid": uuid, "s": s, "cap": caption, "latent_path": lp})
                    else:
                        missing_latents += 1
                    s += num_frames
    print(
        f"[NymeriaCameraLatentDataset] split={split} T={num_frames} kept={len(index)} "
        f"quality_filtered={quality_filtered_rows} missing_latents={missing_latents} "
        f"candidates={candidate_windows} quality_reasons={dict(sorted(quality_reason_rows.items()))} "
        f"root={latent_root}",
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

    def _cached_vision_token_delta(self, data_batch: dict) -> int:
        if "video_latents" not in data_batch:
            return 0

        videos = list(_iter_raw_videos(data_batch["video"]))
        latents = list(_iter_cached_latents(data_batch["video_latents"]))
        if len(videos) != len(latents):
            raise ValueError(f"video/video_latents item mismatch: {len(videos)} != {len(latents)}")

        delta = 0
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
            delta += latent_t * (latent_h // self.patch_spatial) * (latent_w // self.patch_spatial)
        return delta

    def _compute_token_split_per_sample(self, data_batch: dict) -> tuple[int, int]:
        """Attribute cached tokens to the generation tower in current Cosmos."""
        parent = getattr(super(), "_compute_token_split_per_sample", None)
        if parent is None:
            raise AttributeError("parent IterativeJointDataLoader has no token-split API")
        und_tokens, gen_tokens = parent(data_batch)
        return und_tokens, gen_tokens + self._cached_vision_token_delta(data_batch)

    def _compute_num_tokens_per_sample(self, data_batch: dict) -> int:
        # Current Cosmos computes this through self._compute_token_split_per_sample,
        # so the dynamic dispatch above has already added the cached tokens.  The
        # pinned Nano framework has no split API and needs the historical delta here.
        if hasattr(IterativeJointDataLoader, "_compute_token_split_per_sample"):
            return super()._compute_num_tokens_per_sample(data_batch)
        return super()._compute_num_tokens_per_sample(data_batch) + self._cached_vision_token_delta(data_batch)


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
        manifest_path: str = str(
            WEKA_ROOT / "nymeriaplus_kimodo_proportional" / "video" / "manifest_video.jsonl"
        ),
        split_file: str = str(
            WEKA_ROOT / "nymeriaplus_kimodo_proportional" / "train_test_split.json"
        ),
        latent_root: str = str(
            WEKA_ROOT / "nymeriaplus_kimodo_proportional" / "joint_latents_T97"
        ),
        num_frames: int = 97,
        fps: float = 20.0,
        mode: str = "forward_dynamics",
        max_action_dim: int = 64,
        tokenizer_config: dict | None = None,
        cfg_dropout_rate: float = 0.1,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 0,
        quality_filter_path: str = "",
        replace_standalone_c: bool = False,
        standalone_c_subject: str = "person",
        prefix_lengths: list[int] | tuple[int, ...] = (1,),
        prefix_sampling_weights: list[float] | tuple[float, ...] | None = None,
        prefix_seed: int = 42,
        model_resolution_tier: str | None = None,
        expected_latent_hw: int | None = None,
        expected_image_hw: int | None = None,
        require_latent_cache_contract: bool = False,
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
        self.replace_standalone_c = bool(replace_standalone_c)
        if standalone_c_subject not in _STANDALONE_C_SUBJECTS:
            raise ValueError(
                f"standalone_c_subject must be one of {sorted(_STANDALONE_C_SUBJECTS)}, "
                f"got {standalone_c_subject!r}"
            )
        self.standalone_c_subject = standalone_c_subject
        self.prefix_lengths, self.prefix_sampling_weights = validate_prefix_sampling(
            prefix_lengths,
            prefix_sampling_weights,
            self.num_frames,
        )
        self._base_seed = int(seed)
        self._prefix_seed = int(prefix_seed)
        self._worker_rng: random.Random | None = None
        self._modes = list(MODE_WEIGHTS)
        self._mode_weights = [MODE_WEIGHTS[m] for m in self._modes]
        self._cache_contract = validate_training_cache_contract(
            latent_root=latent_root,
            num_frames=self.num_frames,
            fps=self.fps,
            model_resolution_tier=model_resolution_tier,
            expected_latent_hw=expected_latent_hw,
            expected_image_hw=expected_image_hw,
            require_contract=require_latent_cache_contract,
        )
        if self._cache_contract is not None:
            self._expected_latent_hw = self._cache_contract.expected_latent_shape[-1]
            self._expected_image_hw = self._cache_contract.expected_image_hw[-1]
        else:
            self._expected_latent_hw = expected_latent_hw
            self._expected_image_hw = expected_image_hw

        index = build_cached_index(
            manifest_path,
            split_file,
            split,
            self.num_frames,
            latent_root,
            quality_filter_path,
        )
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
        return self._get_rng().choices(self._modes, weights=self._mode_weights, k=1)[0]

    def _get_rng(self) -> random.Random:
        if self._worker_rng is None:
            worker = get_worker_info()
            worker_seed = worker.seed if worker is not None else torch.initial_seed()
            self._worker_rng = random.Random(self._base_seed + self._prefix_seed + int(worker_seed))
        return self._worker_rng

    def _choose_prefix(self, mode: str) -> tuple[int, int]:
        if mode not in _VISUAL_GENERATION_MODES:
            return self.num_frames, 1 + (self.num_frames - 1) // 4
        if len(self.prefix_lengths) == 1:
            rgb_frames = self.prefix_lengths[0]
        else:
            rgb_frames = self._get_rng().choices(
                self.prefix_lengths,
                weights=self.prefix_sampling_weights,
                k=1,
            )[0]
        return rgb_frames, rgb_prefix_to_latent_frames(rgb_frames, self.num_frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self._index[idx % len(self._index)]
        mode = self._choose_mode()
        with np.load(it["latent_path"]) as d:
            latents = d["latents"]
            action = d["camera_action"].astype(np.float32)
            image_size = d["image_size"].astype(np.float32)

        if self._cache_contract is not None:
            validate_cached_sample(
                self._cache_contract,
                latents=latents,
                camera_action=action,
                image_size=image_size,
                context=it["latent_path"],
            )
        if action.shape[0] != self.num_frames - 1:
            raise ValueError(f"bad cached action shape {action.shape} in {it['latent_path']}")

        expected_latent_frames = 1 + (self.num_frames - 1) // 4
        if latents.ndim != 4 or latents.shape[1] != expected_latent_frames:
            raise ValueError(
                f"bad cached latent temporal shape {latents.shape} in {it['latent_path']}; "
                f"expected [C,{expected_latent_frames},H,W]"
            )
        if self._expected_latent_hw is not None and tuple(latents.shape[-2:]) != (
            self._expected_latent_hw,
            self._expected_latent_hw,
        ):
            raise ValueError(
                f"bad cached latent spatial shape {latents.shape} in {it['latent_path']}; "
                f"expected H=W={self._expected_latent_hw}"
            )
        if self._expected_image_hw is not None:
            actual_image_hw = tuple(int(round(float(value))) for value in image_size[:2])
            if actual_image_hw != (self._expected_image_hw, self._expected_image_hw):
                raise ValueError(
                    f"bad cached image_size {image_size.tolist()} in {it['latent_path']}; "
                    f"expected transformed H=W={self._expected_image_hw}"
                )

        caption = "" if mode == "inverse_dynamics" else it["cap"]
        if caption and self.replace_standalone_c:
            caption = replace_standalone_c(caption, self.standalone_c_subject)
        rgb_prefix_length, latent_prefix_length = self._choose_prefix(mode)
        if mode == "image2video":
            sequence_plan = SequencePlan(
                has_text=True,
                has_vision=True,
                has_action=False,
                condition_frame_indexes_vision=[0],
            )
        else:
            sequence_plan = build_sequence_plan_from_mode(
                mode=mode,
                video_length=self.num_frames,
                action_length=self.num_frames - 1,
                video_temporal_downsample=4,
            )
        if mode in _VISUAL_GENERATION_MODES:
            sequence_plan.condition_frame_indexes_vision = list(range(latent_prefix_length))

        sample: dict[str, Any] = {
            "video": torch.empty((3, self.num_frames, 1, 1), dtype=torch.uint8),
            "video_latents": torch.from_numpy(latents),
            "image_size": torch.from_numpy(image_size),
            "ai_caption": caption,
            "conditioning_fps": torch.tensor(int(round(self.fps)), dtype=torch.long),
            "mode": mode,
            "viewpoint": "ego_view",
            "sequence_plan": sequence_plan,
            "rgb_prefix_length": torch.tensor(rgb_prefix_length, dtype=torch.long),
            "latent_prefix_length": torch.tensor(latent_prefix_length, dtype=torch.long),
            "predicted_rgb_start": torch.tensor(rgb_prefix_length, dtype=torch.long),
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
