#!/usr/bin/env python
"""Evaluate forward dynamics with CVPR 2024 VideoMAE-v2 content-debiased FVD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from native_phase_training.forward_video_metric_utils import (
    CDFVD_FRAME_INDICES,
    iter_forward_video_pairs,
    load_forward_records,
    records_fingerprint,
    write_json,
)


PINNED_CDFVD_REVISION = "a1e037ab7cb087debd2221d14ae4a001ec054201"
PINNED_VIDEOMAE_CHECKPOINT_SHA256 = "5a210a92f035dff30c53b46157b612e7a1a5d3c99700e1b2d71da5c399ca7e70"
DEFAULT_CDFVD_ROOT = Path("/home/jungbin_cho/.cache/cosmos_motion_ft/third_party/content-debiased-fvd")
DEFAULT_CHECKPOINT = Path("/weka/jungbin/model_cache/cdfvd/vit_g_hybrid_pt_1200e_ssv2_ft.pth")


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_stats(feature_stats_class):
    return feature_stats_class(capture_mean_cov=True, capture_all=False, max_items=None)


def _flush(
    evaluator,
    stats_by_segment: dict[str, Any],
    buffers: dict[str, list[np.ndarray]],
    batch_size: int,
) -> None:
    for segment, values in buffers.items():
        if not values:
            continue
        videos = np.stack(values)
        stats_by_segment[segment] = evaluator.feature_fn(
            stats_by_segment[segment],
            evaluator.model,
            videos,
            batchsize=batch_size,
            device=evaluator.device,
            model_dtype=evaluator.model_dtype,
        )
        values.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=71)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--cdfvd-root", type=Path, default=DEFAULT_CDFVD_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--real-cache-root", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.cdfvd_root.is_dir():
        parser.error(f"missing content-debiased-fvd checkout: {args.cdfvd_root}")
    revision = _git_revision(args.cdfvd_root)
    if revision != PINNED_CDFVD_REVISION:
        parser.error(
            f"content-debiased-fvd revision mismatch: expected {PINNED_CDFVD_REVISION}, got {revision}"
        )

    sys.path.insert(0, str(args.cdfvd_root))
    from cdfvd import fvd  # pylint: disable=import-outside-toplevel
    from cdfvd.utils.metric_utils import FeatureStats  # pylint: disable=import-outside-toplevel

    records = load_forward_records(args.eval_root, args.expected_count, args.max_samples)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    evaluator = fvd.cdfvd(
        model="videomae",
        n_real="full",
        n_fake="full",
        ckpt_path=str(args.checkpoint),
        device=args.device,
        half_precision=args.half_precision,
    )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"VideoMAE-v2 checkpoint was not created: {args.checkpoint}")
    checkpoint_sha256 = _sha256(args.checkpoint)
    if checkpoint_sha256 != PINNED_VIDEOMAE_CHECKPOINT_SHA256:
        raise ValueError(
            "VideoMAE-v2 checkpoint hash mismatch: "
            f"expected {PINNED_VIDEOMAE_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )

    cache_root = args.real_cache_root or args.eval_root / "metric_cache" / "cdfvd_videomae_v2_ssv2"
    cache_contract = {
        "metric": "CD-FVD VideoMAE-v2-SSv2",
        "cdfvd_revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_inference_dtype": "float16" if args.half_precision else "float32",
        "record_names_sha256": records_fingerprint(records),
        "n_sequences": len(records),
        "frame_indices": {name: values.tolist() for name, values in CDFVD_FRAME_INDICES.items()},
        "gt_preprocessing": "native aspect-preserving bicubic antialias resize plus right/bottom reflection pad",
    }
    cache_manifest = cache_root / "manifest.json"
    use_cache = args.max_samples == 0 and cache_manifest.is_file()
    if use_cache:
        use_cache = json.loads(cache_manifest.read_text()) == cache_contract
    cache_paths = {name: cache_root / f"{name}.pkl" for name in CDFVD_FRAME_INDICES}
    use_cache = use_cache and all(path.is_file() for path in cache_paths.values())

    if use_cache:
        real_stats = {name: FeatureStats.load(str(path)) for name, path in cache_paths.items()}
    else:
        real_stats = {name: _new_stats(FeatureStats) for name in CDFVD_FRAME_INDICES}
    fake_stats = {name: _new_stats(FeatureStats) for name in CDFVD_FRAME_INDICES}
    real_buffers = {name: [] for name in CDFVD_FRAME_INDICES}
    fake_buffers = {name: [] for name in CDFVD_FRAME_INDICES}

    for index, pair in enumerate(iter_forward_video_pairs(args.inference_root, args.eval_root, records)):
        for segment, frame_indices in CDFVD_FRAME_INDICES.items():
            if not use_cache:
                real_buffers[segment].append(pair.gt[frame_indices])
            fake_buffers[segment].append(pair.generated[frame_indices])
        if len(next(iter(fake_buffers.values()))) >= args.batch_size:
            if not use_cache:
                _flush(evaluator, real_stats, real_buffers, args.batch_size)
            _flush(evaluator, fake_stats, fake_buffers, args.batch_size)
        print(f"[cd-fvd] {index + 1}/{len(records)}: {pair.name}", flush=True)

    if not use_cache:
        _flush(evaluator, real_stats, real_buffers, args.batch_size)
    _flush(evaluator, fake_stats, fake_buffers, args.batch_size)
    for segment in CDFVD_FRAME_INDICES:
        if real_stats[segment].num_items != len(records) or fake_stats[segment].num_items != len(records):
            raise ValueError(
                f"{segment}: expected {len(records)} features, got "
                f"real={real_stats[segment].num_items} fake={fake_stats[segment].num_items}"
            )

    if not use_cache and args.max_samples == 0:
        cache_root.mkdir(parents=True, exist_ok=True)
        for segment, path in cache_paths.items():
            temporary = path.with_suffix(path.suffix + ".tmp")
            real_stats[segment].save(str(temporary))
            temporary.replace(path)
        write_json(cache_manifest, cache_contract)

    scores = {
        segment: float(evaluator.compute_fvd_from_stats(fake_stats[segment], real_stats[segment]))
        for segment in CDFVD_FRAME_INDICES
    }
    if not np.isfinite(list(scores.values())).all():
        raise ValueError(f"non-finite CD-FVD scores: {scores}")
    payload = {
        **cache_contract,
        "metric_note": (
            "CVPR 2024 content-debiased FVD with VideoMAE-v2-SSv2 features; "
            "not numerically comparable to legacy I3D FVD"
        ),
        "distance": "Frechet distance between VideoMAE-v2 feature distributions; lower is better",
        "sampling": (
            "all RGB frames in each reported suffix/horizon; "
            "full suffix uses frames 1-96 and the conditioned RGB frame 0 is always excluded"
        ),
        "conditioned_frame_excluded": True,
        "half_precision": bool(args.half_precision),
        "checkpoint": str(args.checkpoint),
        "n_feature_samples_per_score": len(records),
        "scores": scores,
        "real_stats_cache_used": use_cache,
    }
    output = args.out or args.inference_root.parent / "analysis" / "cdfvd_videomae_metrics.json"
    write_json(output, payload)
    print(f"[cd-fvd] complete: {output}", flush=True)


if __name__ == "__main__":
    main()
