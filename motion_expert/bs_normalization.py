"""Normalization provenance and checkpoint-safe resolution for BONES UniEgo models."""
from __future__ import annotations

import hashlib
import os
import warnings
from typing import Any

import numpy as np


FEATURE_DIM = 283

KNOWN_NORMALIZATIONS = {
    (
        "f4f32d4f03cede93b35c46a3aeaef7282dabc9d57b428a4b672acfcf064a79d5",
        "559948ffc1d665a9e5c8e3a53f5b9ea024294fcb51b536c0f47bc7fb00ac9471",
    ): "bones_seed_proportional_20fps",
    (
        "bd1d6bdc9a3b026fe1e5b28899441655ee36672c69c3e6e6389e9baff4b400d3",
        "ee069e3aa9f3cd1a1e70135cc00bc751030f8045fae6bbfb7b4f5b32fa65f28c",
    ): "nymeria_grounded_uniego283",
}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_normalization(
    mean_path: str,
    std_path: str,
    *,
    tag: str | None = None,
    source: str = "explicit",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate one mean/std pair, returning JSON-safe provenance."""
    mean_path = os.path.realpath(os.path.abspath(mean_path))
    std_path = os.path.realpath(os.path.abspath(std_path))
    if not os.path.isfile(mean_path):
        raise FileNotFoundError(f"missing normalization mean: {mean_path}")
    if not os.path.isfile(std_path):
        raise FileNotFoundError(f"missing normalization std: {std_path}")

    mean_raw = np.load(mean_path, allow_pickle=False)
    std_raw = np.load(std_path, allow_pickle=False)
    if mean_raw.shape != (FEATURE_DIM,) or std_raw.shape != (FEATURE_DIM,):
        raise ValueError(
            f"normalization arrays must both be ({FEATURE_DIM},); "
            f"got mean={mean_raw.shape}, std={std_raw.shape}"
        )
    if not np.isfinite(mean_raw).all() or not np.isfinite(std_raw).all():
        raise ValueError("normalization arrays contain non-finite values")
    if np.any(std_raw <= 0):
        raise ValueError("normalization std must be strictly positive in every feature")

    mean_sha256 = file_sha256(mean_path)
    std_sha256 = file_sha256(std_path)
    resolved_tag = tag or KNOWN_NORMALIZATIONS.get(
        (mean_sha256, std_sha256),
        f"custom_{mean_sha256[:8]}_{std_sha256[:8]}",
    )
    metadata = {
        "tag": resolved_tag,
        "source": source,
        "mean_path": mean_path,
        "std_path": std_path,
        "mean_sha256": mean_sha256,
        "std_sha256": std_sha256,
        "mean_shape": list(mean_raw.shape),
        "std_shape": list(std_raw.shape),
        "mean_file_dtype": str(mean_raw.dtype),
        "std_file_dtype": str(std_raw.dtype),
    }
    return mean_raw.astype(np.float32), std_raw.astype(np.float32), metadata


def checkpoint_normalization_reference(checkpoint: dict) -> dict[str, Any] | None:
    """Read normalization metadata from a new or legacy BONES checkpoint."""
    metadata = checkpoint.get("normalization")
    if isinstance(metadata, dict) and metadata.get("mean_path") and metadata.get("std_path"):
        return dict(metadata)

    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        return None
    mean_path = checkpoint_args.get("mean")
    std_path = checkpoint_args.get("std")
    if not mean_path or not std_path:
        return None
    return {
        "tag": checkpoint_args.get("normalization_tag"),
        "source": "checkpoint_args",
        "mean_path": str(mean_path),
        "std_path": str(std_path),
    }


def _reference_hash(reference: dict[str, Any], key: str) -> str | None:
    recorded = reference.get(f"{key}_sha256")
    if recorded:
        return str(recorded)
    path = reference.get(f"{key}_path")
    if path and os.path.isfile(path):
        return file_sha256(path)
    return None


def resolve_checkpoint_normalization(
    checkpoint: dict,
    *,
    mean_override: str | None = None,
    std_override: str | None = None,
    fallback_mean: str | None = None,
    fallback_std: str | None = None,
    allow_override: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resolve stats for a checkpoint and reject silent normalization mismatches."""
    if (mean_override is None) != (std_override is None):
        raise ValueError("mean and std overrides must be supplied together")

    reference = checkpoint_normalization_reference(checkpoint)
    if mean_override is not None:
        mean_path, std_path = mean_override, std_override
        source = "explicit_override"
    elif reference is not None:
        mean_path = str(reference["mean_path"])
        std_path = str(reference["std_path"])
        source = "checkpoint_metadata"
        missing = [path for path in (mean_path, std_path) if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "checkpoint-recorded normalization files are unavailable: "
                + ", ".join(missing)
                + "; pass explicit mean/std copies"
            )
    else:
        if fallback_mean is None or fallback_std is None:
            raise ValueError(
                "checkpoint has no normalization metadata and no legacy fallback was provided"
            )
        mean_path, std_path = fallback_mean, fallback_std
        source = "legacy_default_fallback"
        warnings.warn(
            "checkpoint has no normalization metadata; using the legacy default stats",
            RuntimeWarning,
            stacklevel=2,
        )

    tag = reference.get("tag") if reference is not None and source != "explicit_override" else None
    mean, std, actual = load_normalization(mean_path, std_path, tag=tag, source=source)

    checkpoint_match: bool | None = None
    mismatch_reasons: list[str] = []
    if reference is not None:
        expected_mean_hash = _reference_hash(reference, "mean")
        expected_std_hash = _reference_hash(reference, "std")
        if expected_mean_hash is None or expected_std_hash is None:
            mismatch_reasons.append("checkpoint stats identity cannot be verified")
        else:
            if actual["mean_sha256"] != expected_mean_hash:
                mismatch_reasons.append("mean SHA-256 differs from checkpoint")
            if actual["std_sha256"] != expected_std_hash:
                mismatch_reasons.append("std SHA-256 differs from checkpoint")
        checkpoint_match = not mismatch_reasons
        if mismatch_reasons and not allow_override:
            raise ValueError(
                "normalization mismatch: "
                + "; ".join(mismatch_reasons)
                + "; use the checkpoint stats or explicitly allow a stats override"
            )

    if reference is not None and checkpoint_match:
        actual["tag"] = str(reference.get("tag") or actual["tag"])
    actual["checkpoint_match"] = checkpoint_match
    actual["override_allowed"] = bool(allow_override)
    if mismatch_reasons:
        actual["mismatch_reasons"] = mismatch_reasons
    return mean, std, actual
