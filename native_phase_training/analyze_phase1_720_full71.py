"""Compare one model's sharded 720-tier outputs with its canonical 256 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from native_phase_training.analyze_phase1_resolution_matrix import (
    QUALITY_KEYS,
    TEMPORAL_KEYS,
    _temporal_metrics,
)
from native_phase_training.evaluate_inverse_forward import (
    LPIPSAlex,
    _frame_metrics,
    _read_video_rgb,
    _resize_gt_like_native,
    _summarize_frame_metrics,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _aggregate_nested(rows: list[dict[str, Any]], section: str, keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: {
            "mean": float(np.mean([row[section][key] for row in rows])),
            "median": float(np.median([row[section][key] for row in rows])),
            "std": float(np.std([row[section][key] for row in rows])),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--high-root", type=Path, required=True)
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--low-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lpips-device", default="cuda:0")
    parser.add_argument("--lpips-batch-size", type=int, default=32)
    args = parser.parse_args()

    records = _read_jsonl(args.input_jsonl)
    low_metrics = json.loads(args.low_metrics.read_text())
    if len(records) != 71 or low_metrics.get("n") != 71:
        raise ValueError(f"expected 71 records and low metrics, got {len(records)} and {low_metrics.get('n')}")
    lpips_metric = LPIPSAlex(args.lpips_device, args.lpips_batch_size)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        name = record["name"]
        high_matches = list(args.high_root.glob(f"shard_*/{name}"))
        if len(high_matches) != 1:
            raise ValueError(f"expected one high-tier output for {name}, found {high_matches}")
        high_dir = high_matches[0]
        low_dir = args.low_root / name
        for sample_dir in (high_dir, low_dir):
            payload = json.loads((sample_dir / "sample_outputs.json").read_text())
            if payload.get("status") != "success":
                raise RuntimeError(f"unsuccessful output: {sample_dir}")

        base_name = name.removesuffix("_forward_dynamics")
        gt = _read_video_rgb(args.eval_root / "samples" / base_name / "gt_clip.mp4")
        high = _read_video_rgb(high_dir / "vision.mp4")
        low = _read_video_rgb(low_dir / "vision.mp4")
        if len(gt) != 97 or len(high) != 97 or len(low) != 97:
            raise ValueError(f"{name}: expected 97 frames, got GT={len(gt)} high={len(high)} low={len(low)}")

        gt_256 = _resize_gt_like_native(gt, 256, 256)
        high_256 = _resize_gt_like_native(high, 256, 256)
        high_quality = _summarize_frame_metrics(
            _frame_metrics(gt_256[1:], high_256[1:], lpips_metric)
        )
        rows.append(
            {
                "name": name,
                "high_video": str(high_dir / "vision.mp4"),
                "low_video": str(low_dir / "vision.mp4"),
                "high_common_256_quality": {key: high_quality[key] for key in QUALITY_KEYS},
                "low_temporal": _temporal_metrics(low),
                "high_temporal": _temporal_metrics(high),
            }
        )
        print(f"[full71-720-analysis] {args.model} {index + 1}/71 {name}", flush=True)

    paired_percent = {
        key: float(
            np.mean(
                [
                    (row["high_temporal"][key] - row["low_temporal"][key])
                    / row["low_temporal"][key]
                    * 100.0
                    for row in rows
                ]
            )
        )
        for key in TEMPORAL_KEYS
    }
    temporal_wins = {
        key: int(sum(row["high_temporal"][key] < row["low_temporal"][key] for row in rows))
        for key in TEMPORAL_KEYS
    }
    payload = {
        "schema_version": 1,
        "model": args.model,
        "n": len(rows),
        "conditioned_frame_excluded_from_quality": True,
        "quality_comparison_size": [256, 256],
        "low_256_quality": {
            key: low_metrics["aggregate"][key] for key in QUALITY_KEYS
        },
        "high_720_common_256_quality": _aggregate_nested(
            rows, "high_common_256_quality", QUALITY_KEYS
        ),
        "low_256_temporal": _aggregate_nested(rows, "low_temporal", TEMPORAL_KEYS),
        "high_720_temporal": _aggregate_nested(rows, "high_temporal", TEMPORAL_KEYS),
        "paired_temporal_percent_change": paired_percent,
        "high_temporal_win_count": temporal_wins,
        "per_sequence": rows,
        "paths": {
            "input_jsonl": str(args.input_jsonl),
            "eval_root": str(args.eval_root),
            "high_root": str(args.high_root),
            "low_root": str(args.low_root),
            "low_metrics": str(args.low_metrics),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[full71-720-analysis] complete: {args.output}")


if __name__ == "__main__":
    main()
