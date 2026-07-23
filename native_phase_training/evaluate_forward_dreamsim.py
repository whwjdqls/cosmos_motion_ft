#!/usr/bin/env python
"""Evaluate aligned forward-dynamics frames with the official DreamSim metric."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from dreamsim import dreamsim
import numpy as np
from PIL import Image
import torch

from native_phase_training.forward_video_metric_utils import (
    HORIZON_SUFFIX_SLICES,
    SUFFIX_RGB_INDICES,
    aggregate_scalars,
    iter_forward_video_pairs,
    load_forward_records,
    records_fingerprint,
    write_json,
)


@torch.inference_mode()
def _distances(
    model: torch.nn.Module,
    preprocess,
    gt: np.ndarray,
    generated: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(gt), batch_size):
        stop = min(len(gt), start + batch_size)
        gt_tensor = torch.cat([preprocess(Image.fromarray(frame)) for frame in gt[start:stop]])
        generated_tensor = torch.cat(
            [preprocess(Image.fromarray(frame)) for frame in generated[start:stop]]
        )
        distance = model(
            gt_tensor.to(device, non_blocking=True),
            generated_tensor.to(device, non_blocking=True),
        )
        values.append(distance.float().cpu().numpy().reshape(-1))
    result = np.concatenate(values).astype(np.float64)
    if result.shape != (len(gt),) or not np.isfinite(result).all():
        raise ValueError(f"invalid DreamSim result shape/values: {result.shape}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=71)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=Path("/weka/jungbin/model_cache/dreamsim"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    records = load_forward_records(args.eval_root, args.expected_count, args.max_samples)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, preprocess = dreamsim(
        pretrained=True,
        device=str(device),
        cache_dir=str(args.cache_dir),
        dreamsim_type="ensemble",
        normalize_embeds=True,
    )

    per_sequence: dict[str, dict[str, object]] = {}
    for index, pair in enumerate(iter_forward_video_pairs(args.inference_root, args.eval_root, records)):
        suffix_gt = pair.gt[SUFFIX_RGB_INDICES]
        suffix_generated = pair.generated[SUFFIX_RGB_INDICES]
        distances = _distances(model, preprocess, suffix_gt, suffix_generated, device, args.batch_size)
        per_sequence[pair.name] = {
            "mean": float(distances.mean()),
            "horizons": {
                name: float(distances[frame_slice].mean())
                for name, frame_slice in HORIZON_SUFFIX_SLICES.items()
            },
        }
        print(f"[dreamsim] {index + 1}/{len(records)}: {pair.name}", flush=True)

    aggregate = aggregate_scalars([float(row["mean"]) for row in per_sequence.values()])
    horizon_aggregate = {
        name: aggregate_scalars(
            [float(row["horizons"][name]) for row in per_sequence.values()]  # type: ignore[index]
        )
        for name in HORIZON_SUFFIX_SLICES
    }
    payload = {
        "metric": "DreamSim",
        "metric_version": version("dreamsim"),
        "model": "ensemble (DINO ViT-B/16 + CLIP ViT-B/16 + OpenCLIP ViT-B/16)",
        "distance": "cosine distance; lower is better",
        "n_sequences": len(per_sequence),
        "conditioned_frame_excluded": True,
        "evaluated_rgb_frame_indices": SUFFIX_RGB_INDICES.tolist(),
        "gt_preprocessing": (
            "native aspect-preserving bicubic antialias resize plus right/bottom reflection pad; "
            "then official DreamSim 224x224 bicubic preprocessing"
        ),
        "record_names_sha256": records_fingerprint(records),
        "aggregate": aggregate,
        "horizon_aggregate": horizon_aggregate,
        "per_sequence": per_sequence,
    }
    output = args.out or args.inference_root.parent / "analysis" / "dreamsim_metrics.json"
    write_json(output, payload)
    print(f"[dreamsim] complete: {output}", flush=True)


if __name__ == "__main__":
    main()

