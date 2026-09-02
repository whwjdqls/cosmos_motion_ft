"""Strict, atomic motion-only checkpoints for Edge Phase 2."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config


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


def validate_checkpoint_contract(payload: dict[str, Any]) -> None:
    saved = _contract_without_validation(payload)
    live = config.architecture_contract()
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
    "capture_rng_state",
    "latest_checkpoint",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "validate_checkpoint_contract",
]
