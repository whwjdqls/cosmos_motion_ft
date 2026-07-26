"""Immutable metadata and validation for cached Wan-VAE training latents."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CACHE_CONTRACT_FILENAME = "latent_cache_contract.json"
CACHE_COMPLETE_FILENAME = "latent_cache_complete.json"
CACHE_CONTRACT_KIND = "nymeria_wan_latent_cache"
CACHE_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class LatentCacheContract:
    schema_version: int
    kind: str
    source_manifest: str
    split_file: str
    split: str
    source_window_count: int
    expected_file_count: int
    num_frames: int
    fps: float
    spatial_transform_resolution: str
    model_resolution_tier: str
    expected_image_hw: tuple[int, int]
    expected_latent_shape: tuple[int, int, int, int]
    expected_camera_shape: tuple[int, int]
    latent_dtype: str
    vae_path: str
    num_shards: int
    limit_per_shard: int | None

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported latent-cache schema {self.schema_version}; "
                f"expected {CACHE_CONTRACT_VERSION}"
            )
        if self.kind != CACHE_CONTRACT_KIND:
            raise ValueError(f"unexpected latent-cache kind: {self.kind!r}")
        if self.split not in {"all", "train", "test"}:
            raise ValueError(f"unsupported latent-cache split: {self.split!r}")
        if self.source_window_count <= 0:
            raise ValueError("source_window_count must be positive")
        if self.expected_file_count <= 0 or self.expected_file_count > self.source_window_count:
            raise ValueError(
                f"invalid expected_file_count={self.expected_file_count} for "
                f"source_window_count={self.source_window_count}"
            )
        if self.num_frames <= 0 or self.num_frames % 4 != 1:
            raise ValueError(f"num_frames must be positive and 4N+1, got {self.num_frames}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if len(self.expected_image_hw) != 2 or any(value <= 0 for value in self.expected_image_hw):
            raise ValueError(f"invalid expected_image_hw: {self.expected_image_hw}")
        if len(self.expected_latent_shape) != 4 or any(
            value <= 0 for value in self.expected_latent_shape
        ):
            raise ValueError(f"invalid expected_latent_shape: {self.expected_latent_shape}")
        if self.expected_latent_shape[1] != 1 + (self.num_frames - 1) // 4:
            raise ValueError(
                "latent temporal shape does not match the causal Wan-VAE contract: "
                f"{self.expected_latent_shape} for T={self.num_frames}"
            )
        if self.expected_camera_shape != (self.num_frames - 1, 9):
            raise ValueError(
                f"camera shape must be {(self.num_frames - 1, 9)}, "
                f"got {self.expected_camera_shape}"
            )
        if self.latent_dtype != "float16":
            raise ValueError(f"cached latents must be float16, got {self.latent_dtype!r}")
        if self.num_shards <= 0:
            raise ValueError(f"num_shards must be positive, got {self.num_shards}")
        if self.limit_per_shard is not None and self.limit_per_shard <= 0:
            raise ValueError(f"limit_per_shard must be positive, got {self.limit_per_shard}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LatentCacheContract":
        expected_fields = set(cls.__dataclass_fields__)
        missing = expected_fields - set(value)
        extra = set(value) - expected_fields
        if missing or extra:
            raise ValueError(
                f"invalid latent-cache contract fields: missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        return cls(
            schema_version=int(value["schema_version"]),
            kind=str(value["kind"]),
            source_manifest=str(value["source_manifest"]),
            split_file=str(value["split_file"]),
            split=str(value["split"]),
            source_window_count=int(value["source_window_count"]),
            expected_file_count=int(value["expected_file_count"]),
            num_frames=int(value["num_frames"]),
            fps=float(value["fps"]),
            spatial_transform_resolution=str(value["spatial_transform_resolution"]),
            model_resolution_tier=str(value["model_resolution_tier"]),
            expected_image_hw=tuple(int(item) for item in value["expected_image_hw"]),
            expected_latent_shape=tuple(int(item) for item in value["expected_latent_shape"]),
            expected_camera_shape=tuple(int(item) for item in value["expected_camera_shape"]),
            latent_dtype=str(value["latent_dtype"]),
            vae_path=str(value["vae_path"]),
            num_shards=int(value["num_shards"]),
            limit_per_shard=(
                None if value["limit_per_shard"] is None else int(value["limit_per_shard"])
            ),
        )


def load_latent_cache_contract(root: str | Path) -> LatentCacheContract:
    path = Path(root) / CACHE_CONTRACT_FILENAME
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read latent-cache contract {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"latent-cache contract must be a JSON object: {path}")
    return LatentCacheContract.from_dict(value)


def ensure_latent_cache_contract(
    root: str | Path,
    contract: LatentCacheContract,
) -> Path:
    """Create one shared contract atomically, or reject an incompatible resume."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    contract_path = root_path / CACHE_CONTRACT_FILENAME
    lock_path = root_path / f".{CACHE_CONTRACT_FILENAME}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if contract_path.exists():
            existing = load_latent_cache_contract(root_path)
            if existing != contract:
                raise RuntimeError(
                    f"latent-cache contract mismatch under {root_path}\n"
                    f"saved={existing.to_dict()}\ncurrent={contract.to_dict()}"
                )
            return contract_path

        temporary = contract_path.with_name(f".{contract_path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, contract_path)
    return contract_path


def validate_cached_sample(
    contract: LatentCacheContract,
    *,
    latents: np.ndarray,
    camera_action: np.ndarray,
    image_size: np.ndarray,
    context: str,
) -> None:
    """Validate one materialized cache record against immutable geometry."""

    if tuple(latents.shape) != contract.expected_latent_shape:
        raise ValueError(
            f"{context}: latent shape {tuple(latents.shape)} != "
            f"{contract.expected_latent_shape}"
        )
    if latents.dtype != np.float16:
        raise ValueError(f"{context}: latent dtype {latents.dtype} != float16")
    if tuple(camera_action.shape) != contract.expected_camera_shape:
        raise ValueError(
            f"{context}: camera shape {tuple(camera_action.shape)} != "
            f"{contract.expected_camera_shape}"
        )
    if image_size.shape != (4,):
        raise ValueError(f"{context}: image_size shape {image_size.shape} != (4,)")
    actual_image_hw = tuple(int(round(float(value))) for value in image_size[:2])
    if actual_image_hw != contract.expected_image_hw:
        raise ValueError(
            f"{context}: transformed image size {actual_image_hw} != "
            f"{contract.expected_image_hw}"
        )
    if not np.isfinite(latents).all():
        raise ValueError(f"{context}: non-finite latents")
    if not np.isfinite(camera_action).all():
        raise ValueError(f"{context}: non-finite camera action")
    if not np.isfinite(image_size).all():
        raise ValueError(f"{context}: non-finite image_size")
