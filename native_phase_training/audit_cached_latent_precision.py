#!/usr/bin/env python
"""Measure numerical precision properties of cached Wan video latents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def _find_npz_files(root: Path, limit: int) -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.endswith(".npz"):
                paths.append(Path(dirpath) / filename)
                if len(paths) == limit:
                    return paths
    return paths


def audit(latent_root: Path, max_files: int) -> dict[str, object]:
    paths = _find_npz_files(latent_root, max_files)
    if not paths:
        raise RuntimeError(f"no .npz files found under {latent_root}")

    arrays: list[np.ndarray] = []
    shapes: set[tuple[int, ...]] = set()
    stored_dtypes: set[str] = set()
    for path in paths:
        with np.load(path) as data:
            latents = data["latents"]
        shapes.add(tuple(int(value) for value in latents.shape))
        stored_dtypes.add(str(latents.dtype))
        arrays.append(latents.reshape(-1))

    stored = np.concatenate(arrays)
    values = stored.astype(np.float32)
    finite = np.isfinite(values)

    if stored.dtype == np.float16:
        upper = np.nextafter(stored, np.float16(np.inf)).astype(np.float32)
        lower = np.nextafter(stored, np.float16(-np.inf)).astype(np.float32)
        half_ulp_bound = 0.5 * np.maximum(upper - values, values - lower)
    else:
        half_ulp_bound = np.zeros_like(values)

    tensor = torch.from_numpy(values)
    bf16_roundtrip = tensor.to(torch.bfloat16).float()
    bf16_error = (bf16_roundtrip - tensor).abs().numpy()

    abs_values = np.abs(values)
    mean_abs = float(abs_values.mean())
    return {
        "latent_root": str(latent_root.resolve()),
        "files": len(paths),
        "values": int(values.size),
        "stored_dtypes": sorted(stored_dtypes),
        "shapes": [list(shape) for shape in sorted(shapes)],
        "finite_count": int(finite.sum()),
        "all_finite": bool(finite.all()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_abs": mean_abs,
        "abs_p99": float(np.quantile(abs_values, 0.99)),
        "fp16_half_ulp_upper_bound_mean": float(half_ulp_bound.mean()),
        "fp16_half_ulp_upper_bound_p99": float(np.quantile(half_ulp_bound, 0.99)),
        "fp16_half_ulp_upper_bound_over_mean_abs": (
            float(half_ulp_bound.mean() / mean_abs) if mean_abs else 0.0
        ),
        "bf16_roundtrip_exact": bool(np.count_nonzero(bf16_error) == 0),
        "bf16_roundtrip_nonzero_count": int(np.count_nonzero(bf16_error)),
        "bf16_roundtrip_abs_error_mean": float(bf16_error.mean()),
        "bf16_roundtrip_abs_error_p99": float(np.quantile(bf16_error, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=32)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.max_files <= 0:
        parser.error("--max-files must be positive")

    result = audit(args.latent_root, args.max_files)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload)


if __name__ == "__main__":
    main()
