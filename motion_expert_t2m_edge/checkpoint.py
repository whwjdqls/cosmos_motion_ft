"""Strict, atomic motion-only checkpoints for Edge Phase 2."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
import ablation_settings as phase2_ablation


SCHEMA_VERSION = config.CONTRACT_SCHEMA_VERSION


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the newest complete regular or rolling recovery checkpoint."""

    checkpoint_dir = Path(checkpoint_dir)
    candidates = list(checkpoint_dir.glob("step_*.pt"))
    recovery = checkpoint_dir / "recovery_latest.pt"
    if recovery.is_file():
        candidates.append(recovery)
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    """Capture process RNGs used by training and DataLoader initialization."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if include_cuda and torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> bool:
    """Restore a captured RNG bundle; return False for legacy checkpoints."""

    if not state:
        return False
    required = {"python", "numpy", "torch_cpu"}
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"checkpoint RNG state is incomplete: missing={missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    return True


def _contract_without_validation(payload: dict[str, Any]) -> dict[str, Any]:
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("checkpoint has no architecture contract")
    return contract


def configure_attention_from_checkpoint(path: str | Path) -> dict[str, Any]:
    """Resolve inference architecture before construction; reject explicit drift."""
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    contract = _contract_without_validation(payload)
    phase2_ablation.configure_checkpoint(contract.get('phase2_ablation'))
    config.MOTION_DIM = phase2_ablation.motion_dim()
    fields = (
        ("motion_attention_mode", "MOTION_ATTENTION_MODE", "EDGE_MOTION_ATTENTION_MODE", "joint"),
        ("motion_io_dtype", "MOTION_IO_DTYPE", "EDGE_MOTION_IO_DTYPE", "bfloat16"),
        ("motion_norm_dtype", "MOTION_NORM_DTYPE", "EDGE_MOTION_NORM_DTYPE", "bfloat16"),
        ("reasoner_k_norm_trainable", "REASONER_K_NORM_TRAINABLE", "EDGE_REASONER_K_NORM_TRAINABLE", False),
        ("reasoner_k_norm_bypass", "REASONER_K_NORM_BYPASS", "EDGE_REASONER_K_NORM_BYPASS", False),
        ("reasoner_kv_projection", "REASONER_KV_PROJECTION", "EDGE_REASONER_KV_PROJECTION", "none"),
        ("reasoner_v_norm_trainable", "REASONER_V_NORM_TRAINABLE", "EDGE_REASONER_V_NORM_TRAINABLE", False),
        ("motion_self_attention_per_gap", "MOTION_SELF_ATTENTION_PER_GAP", "EDGE_MOTION_SELF_ATTENTION_PER_GAP", 0),
        ("motion_self_attention_before_joint", "MOTION_SELF_ATTENTION_BEFORE_JOINT", "EDGE_MOTION_SELF_ATTENTION_BEFORE_JOINT", False),
        ("bones_caption_subject_policy", "BONES_CAPTION_SUBJECT_POLICY", "EDGE_BONES_CAPTION_POLICY", "original"),
        ("text_packing", "TEXT_PACKING", "EDGE_TEXT_PACKING", "edge_native_generation_prefix"),
    )
    resolved = []
    for key, attribute, environment, default in fields:
        saved = contract.get(key, default)
        if environment in os.environ and getattr(config, attribute) != saved:
            raise RuntimeError(
                f"explicit {environment} conflicts with checkpoint: live={getattr(config, attribute)!r} saved={saved!r}"
            )
        resolved.append((attribute, saved))
    if contract.get("text_packing", "edge_native_generation_prefix") not in {
        "edge_native_generation_prefix", "nano_caption_only",
    }:
        raise RuntimeError(f"unsupported checkpoint text_packing={contract.get('text_packing')!r}")
    mode = contract.get("motion_attention_mode", "joint")
    cross_key_norm = (
        contract.get("cross_attention_key_normalization", "none")
        if mode == "self_cross_final_hidden" else "none"
    )
    if (
        "EDGE_CROSS_ATTENTION_K_NORM" in os.environ
        and config.CROSS_ATTENTION_K_NORM != cross_key_norm
    ):
        raise RuntimeError(
            "explicit EDGE_CROSS_ATTENTION_K_NORM conflicts with checkpoint: "
            f"live={config.CROSS_ATTENTION_K_NORM!r} saved={cross_key_norm!r}"
        )
    resolved.append(("CROSS_ATTENTION_K_NORM", cross_key_norm))
    for key, attribute, environment, default in (
        ("cross_attention_value_normalization", "CROSS_ATTENTION_V_NORM", "EDGE_CROSS_ATTENTION_V_NORM", "none"),
        ("cross_attention_reasoner_rope", "CROSS_ATTENTION_REASONER_ROPE", "EDGE_CROSS_ATTENTION_REASONER_ROPE", True),
    ):
        saved = contract.get(key, default) if mode == "self_cross_final_hidden" else default
        if environment in os.environ and getattr(config, attribute) != saved:
            raise RuntimeError(
                f"explicit {environment} conflicts with checkpoint: live={getattr(config, attribute)!r} saved={saved!r}"
            )
        resolved.append((attribute, saved))
    for attribute, value in resolved:
        setattr(config, attribute, value)
    config.MOTION_SELF_ATTENTION_AFTER_JOINT_LAYERS = (
        config.MOTION_LAYER_INDICES[:-1] if config.MOTION_SELF_ATTENTION_PER_GAP else ()
    )
    config.validate_attention_configuration()
    return dict(contract)


def validate_checkpoint_contract(payload: dict[str, Any]) -> None:
    saved = dict(_contract_without_validation(payload))
    # Existing schema-3 Grasp checkpoints predate this optional BONES policy.
    # They used original BONES wording; reject loading them under the new rule,
    # while keeping their original-policy reload behavior backward compatible.
    saved.setdefault("bones_caption_subject_policy", "original")
    # Older schema-3 runs contain only the seven joint-attention blocks.
    # Keep those loadable under the default architecture, but never silently
    # load a seven-block checkpoint into the opt-in 7-JA/6-SA experiment.
    saved.setdefault("motion_self_attention_per_gap", 0)
    saved.setdefault("motion_self_attention_before_joint", False)
    saved.setdefault("motion_self_attention_after_joint_layers", [])
    saved.setdefault("motion_io_dtype", "bfloat16")
    saved.setdefault("motion_norm_dtype", "bfloat16")
    saved.setdefault("reasoner_k_norm_trainable", False)
    saved.setdefault("reasoner_k_norm_bypass", False)
    saved.setdefault("reasoner_kv_projection", "none")
    saved.setdefault("reasoner_v_norm_trainable", False)
    # Legacy self/cross checkpoints used unnormalized V and rotated cross K.
    # Do not fill these defaults from the currently selected experiment.
    saved_mode = saved.get("motion_attention_mode", "joint")
    saved.setdefault("cross_attention_value_normalization", None if saved_mode == "joint" else "none")
    saved.setdefault("cross_attention_reasoner_rope", None if saved_mode == "joint" else True)
    for key, value in config.attention_mode_contract("joint").items():
        saved.setdefault(key, value)
    live = config.architecture_contract()
    if saved.get('phase2_ablation') != live.get('phase2_ablation'):
        raise RuntimeError('Checkpoint output codec/objective/statistics mismatch')
    mismatches = {
        key: (saved.get(key), value)
        for key, value in live.items()
        if saved.get(key) != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}: saved={actual!r} live={expected!r}"
            for key, (actual, expected) in mismatches.items()
        )
        raise RuntimeError(f"checkpoint architecture/data contract mismatch: {detail}")


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    args: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": int(step),
        "contract": config.architecture_contract(),
        "args": dict(args),
        "motion_state": {
            key: value.detach().cpu() for key, value in model.motion_state_dict().items()
        },
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "extra": dict(extra or {}),
    }
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported checkpoint schema {payload.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )
    validate_checkpoint_contract(payload)
    state = payload.get("motion_state")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint has no motion_state")
    model.load_motion_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer_state = payload.get("optimizer")
        if optimizer_state is None:
            raise RuntimeError("checkpoint has no optimizer state")
        optimizer.load_state_dict(optimizer_state)
    return payload


__all__ = [
    "configure_attention_from_checkpoint",
    "capture_rng_state",
    "latest_checkpoint",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "validate_checkpoint_contract",
]
