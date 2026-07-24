"""Merge Original/A/B/D Phase-1 full-71 720-tier reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("original", "A", "B", "D")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partials-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        model: json.loads((args.partials_root / f"{model}.json").read_text()) for model in MODELS
    }
    for model, report in reports.items():
        if report.get("model") != model or report.get("n") != 71:
            raise ValueError(f"invalid report for {model}: model={report.get('model')} n={report.get('n')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "full71_720_metrics.json"
    combined_path.write_text(
        json.dumps({"schema_version": 1, "models": reports}, indent=2) + "\n"
    )

    lines = [
        "# Phase-1 Full-71 720-Tier Evaluation",
        "",
        "Quality metrics compare both GT and generated videos at 256x256 and exclude frame 0.",
        "Temporal metrics use all 97 generated frames. Lower LPIPS and temporal values are better.",
        "",
        "| Model | Tier | PSNR | SSIM | LPIPS | Second diff. | Flow residual |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, report in reports.items():
        low_quality = report["low_256_quality"]
        low_temporal = report["low_256_temporal"]
        high_quality = report["high_720_common_256_quality"]
        high_temporal = report["high_720_temporal"]
        lines.append(
            f"| {model} | 256/s3 | {low_quality['psnr_db']['mean']:.4f} | "
            f"{low_quality['ssim']['mean']:.6f} | {low_quality['lpips_alex']['mean']:.6f} | "
            f"{low_temporal['second_temporal_difference']['mean']:.6f} | "
            f"{low_temporal['flow_compensated_rgb_mad_128']['mean']:.6f} |"
        )
        lines.append(
            f"| {model} | 720/s10 | {high_quality['psnr_db']['mean']:.4f} | "
            f"{high_quality['ssim']['mean']:.6f} | {high_quality['lpips_alex']['mean']:.6f} | "
            f"{high_temporal['second_temporal_difference']['mean']:.6f} | "
            f"{high_temporal['flow_compensated_rgb_mad_128']['mean']:.6f} |"
        )

    lines.extend(["", "## Paired Temporal Change", ""])
    for model, report in reports.items():
        change = report["paired_temporal_percent_change"]
        wins = report["high_temporal_win_count"]
        lines.append(
            f"- **{model}:** second difference {change['second_temporal_difference']:+.2f}% "
            f"({wins['second_temporal_difference']}/71 lower); flow residual "
            f"{change['flow_compensated_rgb_mad_128']:+.2f}% "
            f"({wins['flow_compensated_rgb_mad_128']}/71 lower)."
        )
    summary_path = args.output_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n")
    (args.output_dir / "COMPLETE.json").write_text(
        json.dumps(
            {
                "metrics": str(combined_path),
                "summary": str(summary_path),
                "models": list(MODELS),
                "sequences_per_model": 71,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[full71-720-analysis] merged reports into {args.output_dir}")


if __name__ == "__main__":
    main()
