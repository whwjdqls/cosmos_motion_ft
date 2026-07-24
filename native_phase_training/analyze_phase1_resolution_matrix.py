"""Analyze matched Phase-1 256/720 forward-dynamics generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from native_phase_training.evaluate_inverse_forward import (
    LPIPSAlex,
    _frame_metrics,
    _read_video_rgb,
    _resize_gt_like_native,
    _summarize_frame_metrics,
)


MODELS = ("original", "A", "B", "D")
CELLS = ("r256_s3", "r720_s3", "r720_s10")
QUALITY_KEYS = ("psnr_db", "ssim", "lpips_alex")
TEMPORAL_KEYS = (
    "adjacent_rgb_mad",
    "second_temporal_difference",
    "second_to_first_ratio",
    "flow_compensated_rgb_mad_128",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _temporal_metrics(frames: np.ndarray) -> dict[str, float]:
    frames_256 = _resize_gt_like_native(frames, 256, 256).astype(np.float32) / 255.0
    first = np.abs(np.diff(frames_256, axis=0)).mean(axis=(1, 2, 3))
    second = np.abs(frames_256[2:] - 2.0 * frames_256[1:-1] + frames_256[:-2]).mean(
        axis=(1, 2, 3)
    )

    frames_128 = np.stack(
        [cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA) for frame in frames_256]
    )
    yy, xx = np.mgrid[:128, :128].astype(np.float32)
    residuals: list[float] = []
    for index in range(1, len(frames_128)):
        previous = cv2.cvtColor(
            np.rint(frames_128[index - 1] * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )
        current = cv2.cvtColor(
            np.rint(frames_128[index] * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )
        backward_flow = cv2.calcOpticalFlowFarneback(
            current, previous, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        warped_previous = cv2.remap(
            frames_128[index - 1],
            xx + backward_flow[..., 0],
            yy + backward_flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        residuals.append(float(np.abs(frames_128[index] - warped_previous).mean()))
    return {
        "adjacent_rgb_mad": float(first.mean()),
        "second_temporal_difference": float(second.mean()),
        "second_to_first_ratio": float(second.mean() / first.mean()),
        "flow_compensated_rgb_mad_128": float(np.mean(residuals)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "common_256_quality": {
            key: {
                "mean": float(np.mean([row["common_256_quality"][key] for row in rows])),
                "std": float(np.std([row["common_256_quality"][key] for row in rows])),
            }
            for key in QUALITY_KEYS
        },
        "temporal": {
            key: {
                "mean": float(np.mean([row["temporal"][key] for row in rows])),
                "std": float(np.std([row["temporal"][key] for row in rows])),
            }
            for key in TEMPORAL_KEYS
        },
    }


def _paired_changes(rows: dict[str, list[dict[str, Any]]], before: str, after: str) -> dict[str, Any]:
    before_rows = rows[before]
    after_rows = rows[after]
    if [row["source"] for row in before_rows] != [row["source"] for row in after_rows]:
        raise ValueError(f"sample order differs between {before} and {after}")

    quality_delta = {
        key: float(
            np.mean(
                [
                    after_rows[index]["common_256_quality"][key]
                    - before_rows[index]["common_256_quality"][key]
                    for index in range(len(before_rows))
                ]
            )
        )
        for key in QUALITY_KEYS
    }
    temporal_percent = {
        key: float(
            np.mean(
                [
                    (
                        after_rows[index]["temporal"][key]
                        - before_rows[index]["temporal"][key]
                    )
                    / before_rows[index]["temporal"][key]
                    * 100.0
                    for index in range(len(before_rows))
                ]
            )
        )
        for key in TEMPORAL_KEYS
    }
    return {
        "before": before,
        "after": after,
        "common_256_quality_delta": quality_delta,
        "temporal_percent_change": temporal_percent,
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase-1 Resolution Matrix",
        "",
        "Common-256 metrics resize both GT and generated videos to 256x256 and exclude frame 0.",
        "Temporal metrics are lower-better.",
        "",
        "| Model | Cell | PSNR | SSIM | LPIPS | Adj. MAD | Second diff. | Flow residual |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in payload["models"]:
        for cell in CELLS:
            row = payload["models"][model]["aggregate"][cell]
            quality = row["common_256_quality"]
            temporal = row["temporal"]
            lines.append(
                f"| {model} | {cell} | {quality['psnr_db']['mean']:.4f} | "
                f"{quality['ssim']['mean']:.6f} | {quality['lpips_alex']['mean']:.6f} | "
                f"{temporal['adjacent_rgb_mad']['mean']:.6f} | "
                f"{temporal['second_temporal_difference']['mean']:.6f} | "
                f"{temporal['flow_compensated_rgb_mad_128']['mean']:.6f} |"
            )
    lines.extend(["", "## Paired Changes", ""])
    for model in payload["models"]:
        changes = payload["models"][model]["paired_changes"]
        spatial = changes["r256_s3_to_r720_s3"]
        shift = changes["r720_s3_to_r720_s10"]
        lines.append(
            f"- **{model} 256->720 at shift 3:** "
            f"PSNR {spatial['common_256_quality_delta']['psnr_db']:+.4f} dB, "
            f"SSIM {spatial['common_256_quality_delta']['ssim']:+.6f}, "
            f"LPIPS {spatial['common_256_quality_delta']['lpips_alex']:+.6f}; "
            f"second difference {spatial['temporal_percent_change']['second_temporal_difference']:+.2f}%, "
            f"flow residual {spatial['temporal_percent_change']['flow_compensated_rgb_mad_128']:+.2f}%."
        )
        lines.append(
            f"- **{model} shift 3->10 at 720:** "
            f"PSNR {shift['common_256_quality_delta']['psnr_db']:+.4f} dB, "
            f"SSIM {shift['common_256_quality_delta']['ssim']:+.6f}, "
            f"LPIPS {shift['common_256_quality_delta']['lpips_alex']:+.6f}; "
            f"second difference {shift['temporal_percent_change']['second_temporal_difference']:+.2f}%, "
            f"flow residual {shift['temporal_percent_change']['flow_compensated_rgb_mad_128']:+.2f}%."
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--lpips-device", default="cuda:0")
    parser.add_argument("--lpips-batch-size", type=int, default=16)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.matrix_root / "inputs" / "manifest.json").read_text())
    source_records = _read_jsonl(Path(manifest["source"]))[: manifest["sample_count"]]
    lpips_metric = LPIPSAlex(args.lpips_device, args.lpips_batch_size)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "matrix_root": str(args.matrix_root),
        "eval_root": str(args.eval_root),
        "conditioned_frame_excluded": True,
        "common_comparison_size": [256, 256],
        "models": {},
    }
    for model in args.models:
        rows: dict[str, list[dict[str, Any]]] = {cell: [] for cell in CELLS}
        for cell in CELLS:
            for source in source_records:
                source_name = source["name"]
                sample_name = f"{source_name}__{cell}"
                sample_dir = args.matrix_root / "models" / model / sample_name
                output_payload = json.loads((sample_dir / "sample_outputs.json").read_text())
                if output_payload.get("status") != "success":
                    raise RuntimeError(f"unsuccessful output: {sample_dir}")
                predicted = _read_video_rgb(sample_dir / "vision.mp4")
                gt = _read_video_rgb(args.eval_root / "samples" / source_name / "gt_clip.mp4")
                if len(gt) != 97 or len(predicted) != 97:
                    raise ValueError(f"{sample_name}: expected 97 frames, got GT={len(gt)} pred={len(predicted)}")

                predicted_256 = _resize_gt_like_native(predicted, 256, 256)
                gt_256 = _resize_gt_like_native(gt, 256, 256)
                common_quality = _summarize_frame_metrics(
                    _frame_metrics(gt_256[1:], predicted_256[1:], lpips_metric)
                )
                rows[cell].append(
                    {
                        "source": source_name,
                        "video": str(sample_dir / "vision.mp4"),
                        "generated_size": [int(predicted.shape[2]), int(predicted.shape[1])],
                        "common_256_quality": {
                            key: common_quality[key] for key in QUALITY_KEYS
                        },
                        "temporal": _temporal_metrics(predicted),
                    }
                )
                print(f"[resolution-analysis] {model} {cell} {source_name}", flush=True)

        payload["models"][model] = {
            "aggregate": {cell: _aggregate(rows[cell]) for cell in CELLS},
            "paired_changes": {
                "r256_s3_to_r720_s3": _paired_changes(rows, "r256_s3", "r720_s3"),
                "r720_s3_to_r720_s10": _paired_changes(rows, "r720_s3", "r720_s10"),
            },
            "per_sample": rows,
        }

    analysis_dir = args.output_dir or args.matrix_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = analysis_dir / "resolution_matrix_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    _write_summary(analysis_dir / "SUMMARY.md", payload)
    (analysis_dir / "COMPLETE.json").write_text(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "summary": str(analysis_dir / "SUMMARY.md"),
                "models": list(args.models),
                "cells": list(CELLS),
                "samples_per_cell": manifest["sample_count"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[resolution-analysis] complete: {analysis_dir}")


if __name__ == "__main__":
    main()
