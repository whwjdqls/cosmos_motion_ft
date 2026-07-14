"""Checkpoint adapters shared by joint training and sampling.

The historical joint-attention runs save a monolithic ``.pt`` payload.  Native
Cosmos Phase 1 saves a distributed checkpoint containing both ``net`` and
``net_ema``.  Phase 3 only needs the trained generator LoRA and action heads, so
loading the full 90+ GB native model would be wasteful.  This module reads only
those tensors and maps their names into ``JointMotionModel``'s ``cosmos.net``
namespace.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


_NATIVE_GEN_TOKENS = (
    "lora_A",
    "lora_B",
    "action2llm",
    "llm2action",
    "action_modality_embed",
)


def _native_model_dir(path: str | os.PathLike[str]) -> Path:
    root = Path(path)
    if (root / "model" / ".metadata").is_file():
        return root / "model"
    if (root / ".metadata").is_file():
        return root
    raise FileNotFoundError(
        f"{root} is not a native Cosmos model DCP directory "
        "(expected <checkpoint>/model/.metadata or <model>/.metadata)"
    )


def load_native_gen_dcp(
    path: str | os.PathLike[str],
    *,
    weights: str = "ema",
) -> dict[str, torch.Tensor]:
    """Load only Phase-1 generator adapters/heads from a native Cosmos DCP.

    Args:
        path: Native iteration root or its ``model`` child.
        weights: ``"ema"`` selects ``net_ema`` (the official inference path),
            while ``"regular"`` selects the optimizer-updated ``net`` copy.

    Returns:
        A CPU state dict named like ``JointMotionModel.named_all_parameters``.
    """
    if weights not in ("ema", "regular"):
        raise ValueError(f"weights must be 'ema' or 'regular', got {weights!r}")

    from torch.distributed.checkpoint import FileSystemReader, load_state_dict

    model_dir = _native_model_dir(path)
    reader = FileSystemReader(str(model_dir))
    metadata = reader.read_metadata()
    source_prefix = "net_ema." if weights == "ema" else "net."

    selected: dict[str, torch.Tensor] = {}
    for name, tensor_meta in metadata.state_dict_metadata.items():
        if not name.startswith(source_prefix):
            continue
        if not any(token in name for token in _NATIVE_GEN_TOKENS):
            continue
        selected[name] = torch.empty(
            tuple(tensor_meta.size),
            dtype=tensor_meta.properties.dtype,
            device="cpu",
        )

    if not selected:
        raise RuntimeError(
            f"native checkpoint {model_dir} has no {source_prefix} LoRA/action-head tensors"
        )

    # ``no_dist=True`` reconstructs complete tensors on each caller without a
    # process group.  Only ``selected`` keys are read from the DCP shards.
    load_state_dict(selected, storage_reader=reader, no_dist=True)
    mapped = {
        "cosmos.net." + name.removeprefix(source_prefix): tensor
        for name, tensor in selected.items()
    }
    return mapped


def load_joint_pt(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    """Return the model state from a historical joint ``.pt`` checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint {path} did not contain a state dict")
    return state


def load_gen_init_state(
    path: str | os.PathLike[str],
    *,
    native_weights: str = "ema",
) -> dict[str, torch.Tensor]:
    """Load a generator init from either native DCP or historical joint PT."""
    if Path(path).is_dir():
        return load_native_gen_dcp(path, weights=native_weights)
    return load_joint_pt(path)

