"""Phase-2 T2M/TI2M adapter over Nymeria and BONES motion data."""
from __future__ import annotations

import os
from collections.abc import Mapping

import config
from nymeria_joint_dataset import NymeriaJointDataset, collate_joint


PHASE2_TASKS = frozenset(("text2motion", "textimg2motion"))


def normalize_task_weights(task_weights: Mapping[str, float] | None = None) -> dict[str, float]:
    weights = dict(config.TASK_WEIGHTS if task_weights is None else task_weights)
    unknown = sorted(set(weights) - PHASE2_TASKS)
    if unknown:
        raise ValueError(f"Edge Phase 2 supports only {sorted(PHASE2_TASKS)}, got {unknown}")
    negative = {key: value for key, value in weights.items() if float(value) < 0.0}
    if negative:
        raise ValueError(f"task weights must be non-negative, got {negative}")
    weights = {key: float(value) for key, value in weights.items() if float(value) > 0.0}
    if not weights:
        raise ValueError("at least one Phase-2 task needs positive weight")
    return weights


def build_phase2_dataset(
    *,
    split: str,
    train: bool,
    num_frames: int = config.DEFAULT_T,
    ti2m_frames: int = config.TI2M_FRAMES,
    task_weights: Mapping[str, float] | None = None,
    bones_frac: float = config.BONES_TEXT2M_FRAC,
    cfg_dropout: float = 0.1,
    reasoner_image_size: int = config.REASONER_IMAGE_SIZE,
    max_samples: int | None = None,
    seed: int = 0,
) -> NymeriaJointDataset:
    """Build T2M plus reasoner-image TI2M without generator-token data.

    T2M uses Nymeria native caption spans or BONES. TI2M stays Nymeria-only,
    decodes one synchronized frame for the frozen Edge reasoner, and masks its
    valid motion to ``ti2m_frames`` while retaining ``num_frames`` output
    capacity. BONES has no egocamera and is never routed to TI2M.
    """

    representation = os.environ.get(
        "NYMERIA_MOTION_REPRESENTATION", config.MOTION_REPRESENTATION
    )
    if representation != config.MOTION_REPRESENTATION:
        raise RuntimeError(
            f"Edge Phase 2 requires {config.MOTION_REPRESENTATION}, got {representation}"
        )
    weights = normalize_task_weights(task_weights)
    if not 0.0 <= float(bones_frac) <= 1.0:
        raise ValueError(f"bones_frac must be in [0,1], got {bones_frac}")
    aligned_frames = min(int(ti2m_frames), int(num_frames))
    if aligned_frames <= 0:
        raise ValueError(f"ti2m_frames must be positive, got {ti2m_frames}")
    if "textimg2motion" not in weights:
        aligned_frames = int(num_frames)

    dataset = NymeriaJointDataset(
        split=split,
        num_frames=num_frames,
        aligned_num_frames=aligned_frames,
        fps=config.FPS,
        task_weights=weights,
        bones_text2motion_frac=float(bones_frac),
        cfg_dropout=cfg_dropout,
        prefer_latents=False,
        force_on_the_fly=False,
        reasoner_image_for_textimg="textimg2motion" in weights,
        reasoner_image_size=reasoner_image_size,
        camera_head_alignment=False,
        train=train,
        require_rgb_cam=True,
        require_usable=True,
        uniego_root=str(config.NYMERIA_UNIEGO_ROOT),
        max_samples=max_samples,
        seed=seed,
    )
    if dataset._modes != list(weights):
        raise RuntimeError(
            f"Phase-2 task contract drifted: live={dataset._modes} expected={list(weights)}"
        )
    if bones_frac > 0.0 and "text2motion" in weights and not dataset.has_bones:
        raise RuntimeError(
            "BONES was requested but could not be loaded; refusing to silently change the mixture"
        )
    return dataset


# Compatibility name for callers that intentionally run a T2M-only ablation.
def build_t2m_dataset(**kwargs) -> NymeriaJointDataset:
    kwargs.setdefault("task_weights", {"text2motion": 1.0})
    return build_phase2_dataset(**kwargs)


__all__ = [
    "PHASE2_TASKS",
    "build_phase2_dataset",
    "build_t2m_dataset",
    "collate_joint",
    "normalize_task_weights",
]
