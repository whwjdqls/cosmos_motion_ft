#!/usr/bin/env python
"""Validate a complete Nymeria Wan-latent cache before Phase-1 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from motion_expert_joint_attention.precompute_latents import (
    build_index,
    deduplicate_physical_windows,
    out_path,
)
from native_phase_training.latent_cache_contract import (
    CACHE_COMPLETE_FILENAME,
    CACHE_CONTRACT_FILENAME,
    load_latent_cache_contract,
    validate_cached_sample,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(contents)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=256,
        help="deterministically materialize and validate this many records",
    )
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error("--sample-count must be positive")

    contract = load_latent_cache_contract(args.root)
    if contract.limit_per_shard is not None:
        raise ValueError(
            "refusing to mark a limited smoke cache complete: "
            f"limit_per_shard={contract.limit_per_shard}"
        )

    source_rows = build_index(
        contract.source_manifest,
        contract.split_file,
        contract.split,
        contract.num_frames,
    )
    unique_rows = deduplicate_physical_windows(source_rows)
    if len(source_rows) != contract.source_window_count:
        raise ValueError(
            f"source row count changed: contract={contract.source_window_count} "
            f"current={len(source_rows)}"
        )
    if len(unique_rows) != contract.expected_file_count:
        raise ValueError(
            f"unique source count changed: contract={contract.expected_file_count} "
            f"current={len(unique_rows)}"
        )

    expected = {
        Path(out_path(str(args.root), item["uuid"], item["s"])).relative_to(args.root)
        for item in unique_rows
    }
    actual = {
        path.relative_to(args.root)
        for path in args.root.glob("*/*.npz")
        if path.is_file()
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "latent-cache file set is incomplete or contaminated: "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing={len(missing)} extra={len(extra)} "
            f"missing_examples={[str(path) for path in missing[:10]]} "
            f"extra_examples={[str(path) for path in extra[:10]]}"
        )

    ordered = sorted(actual)
    if len(ordered) <= args.sample_count:
        selected = ordered
    else:
        indexes = np.linspace(0, len(ordered) - 1, num=args.sample_count, dtype=np.int64)
        selected = [ordered[int(index)] for index in indexes]

    for relative_path in selected:
        path = args.root / relative_path
        with np.load(path) as record:
            latents = record["latents"]
            camera_action = record["camera_action"]
            image_size = record["image_size"]
            record_t = int(record["T"])
            record_fps = float(record["fps"])
        validate_cached_sample(
            contract,
            latents=latents,
            camera_action=camera_action,
            image_size=image_size,
            context=str(path),
        )
        if record_t != contract.num_frames:
            raise ValueError(f"{path}: T={record_t} != {contract.num_frames}")
        if abs(record_fps - contract.fps) > 1e-6:
            raise ValueError(f"{path}: fps={record_fps} != {contract.fps}")

    contract_path = args.root / CACHE_CONTRACT_FILENAME
    complete_path = args.root / CACHE_COMPLETE_FILENAME
    completion = {
        "schema_version": 1,
        "kind": "nymeria_wan_latent_cache_completion",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": _sha256(contract_path),
        "source_window_count": len(source_rows),
        "expected_file_count": len(expected),
        "actual_file_count": len(actual),
        "materialized_sample_count": len(selected),
        "model_resolution_tier": contract.model_resolution_tier,
        "spatial_transform_resolution": contract.spatial_transform_resolution,
        "expected_image_hw": list(contract.expected_image_hw),
        "expected_latent_shape": list(contract.expected_latent_shape),
    }
    _atomic_write(complete_path, json.dumps(completion, indent=2, sort_keys=True) + "\n")
    print(
        f"[latent-cache] COMPLETE files={len(actual)} sampled={len(selected)} "
        f"shape={contract.expected_latent_shape} marker={complete_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
