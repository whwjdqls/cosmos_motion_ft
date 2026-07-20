#!/usr/bin/env python
"""Summarize full-71 inverse/forward metrics across Phase-1 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INVERSE_METRICS = (
    ("rot_deg", "Inverse rotation error (deg)", "lower"),
    ("trans_dir_cos", "Inverse translation direction cosine", "higher"),
    ("scale_ratio", "Inverse metric scale ratio", "target1"),
    ("trans_err_norm", "Inverse scale-normalized translation error (m)", "lower"),
    ("ate_m", "Inverse aligned ATE (m)", "lower"),
    ("len_ratio", "Inverse path-length ratio", "target1"),
)
FORWARD_METRICS = (
    ("psnr_db", "Forward PSNR (dB)", "higher"),
    ("ssim", "Forward SSIM", "higher"),
    ("lpips_alex", "Forward LPIPS (AlexNet)", "lower"),
)


def _iteration(path: Path) -> int:
    match = re.fullmatch(r"iter_(\d+)", path.name)
    if match is None:
        raise ValueError(f"invalid iteration directory: {path}")
    return int(match.group(1))


def _load_checkpoints(eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for iteration_dir in sorted(eval_root.glob("iter_*"), key=_iteration):
        analysis = iteration_dir / "analysis"
        if not (analysis / "COMPLETE.json").is_file():
            continue
        rows.append(
            {
                "step": _iteration(iteration_dir),
                "iteration": iteration_dir.name,
                "inverse": json.loads((analysis / "invdyn_metrics.json").read_text()),
                "forward": json.loads((analysis / "forward_metrics.json").read_text()),
            }
        )
    if not rows:
        raise RuntimeError(f"no completed checkpoint analyses under {eval_root}")
    return rows


def _plot_progression(rows: list[dict[str, Any]], output: Path) -> None:
    metrics = [("inverse", *metric) for metric in INVERSE_METRICS] + [
        ("forward", *metric) for metric in FORWARD_METRICS
    ]
    figure, axes = plt.subplots(3, 3, figsize=(15, 12))
    steps = np.asarray([row["step"] for row in rows])
    for axis, (group, key, title, target) in zip(axes.flat, metrics, strict=True):
        means = np.asarray([row[group]["aggregate"][key]["mean"] for row in rows])
        medians = np.asarray([row[group]["aggregate"][key]["median"] for row in rows])
        axis.plot(steps, means, marker="o", linewidth=1.8, label="mean")
        axis.plot(steps, medians, marker="s", linewidth=1.2, label="median")
        if target == "target1":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Native Phase-1 full-71 held-out evaluation", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _plot_boxplots(rows: list[dict[str, Any]], group: str, metrics: tuple[tuple[str, str, str], ...], output: Path) -> None:
    columns = 3
    row_count = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(row_count, columns, figsize=(15, 4.2 * row_count), squeeze=False)
    labels = [f"{row['step'] // 1000}k" for row in rows]
    for axis, (key, title, target) in zip(axes.flat, metrics, strict=False):
        values = [list(row[group]["per_sequence"][name][key] for name in row[group]["per_sequence"]) for row in rows]
        axis.boxplot(values, tick_labels=labels, showfliers=True)
        if target == "target1":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("checkpoint")
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.flat[len(metrics) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _plot_forward_horizons(rows: list[dict[str, Any]], output: Path) -> None:
    horizons = (
        ("early_frames_1_32", "frames 1-32"),
        ("middle_frames_33_64", "frames 33-64"),
        ("late_frames_65_96", "frames 65-96"),
    )
    steps = np.asarray([row["step"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, (key, title, _) in zip(axes, FORWARD_METRICS, strict=True):
        for horizon, label in horizons:
            values = [row["forward"]["horizon_aggregate"][horizon][key]["mean"] for row in rows]
            axis.plot(steps, values, marker="o", linewidth=1.5, label=label)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    output = args.out or args.eval_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_checkpoints(args.eval_root)

    summary = {
        "checkpoints": [
            {
                "step": row["step"],
                "iteration": row["iteration"],
                "inverse_n": row["inverse"]["n"],
                "forward_n": row["forward"]["n"],
                "inverse": row["inverse"]["aggregate"],
                "forward": row["forward"]["aggregate"],
                "forward_horizons": row["forward"]["horizon_aggregate"],
            }
            for row in rows
        ]
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    metric_columns = [f"inverse_{key}" for key, _, _ in INVERSE_METRICS] + [
        f"forward_{key}" for key, _, _ in FORWARD_METRICS
    ]
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "iteration", *metric_columns])
        writer.writeheader()
        for row in rows:
            values: dict[str, Any] = {"step": row["step"], "iteration": row["iteration"]}
            values.update(
                {f"inverse_{key}": row["inverse"]["aggregate"][key]["mean"] for key, _, _ in INVERSE_METRICS}
            )
            values.update(
                {f"forward_{key}": row["forward"]["aggregate"][key]["mean"] for key, _, _ in FORWARD_METRICS}
            )
            writer.writerow(values)

    markdown = [
        "# Full-71 Phase-1 Evaluation",
        "",
        "Forward metrics exclude the conditioned first frame and cover frames 1-96.",
        "",
        "| Step | Inv rot deg | Inv dir cos | Inv scale | Inv ATE m | FD PSNR | FD SSIM | FD LPIPS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        inv = row["inverse"]["aggregate"]
        fwd = row["forward"]["aggregate"]
        markdown.append(
            f"| {row['step']} | {inv['rot_deg']['mean']:.4f} | {inv['trans_dir_cos']['mean']:.4f} | "
            f"{inv['scale_ratio']['mean']:.4f} | {inv['ate_m']['mean']:.4f} | "
            f"{fwd['psnr_db']['mean']:.3f} | {fwd['ssim']['mean']:.4f} | {fwd['lpips_alex']['mean']:.4f} |"
        )
    (output / "summary.md").write_text("\n".join(markdown) + "\n")

    _plot_progression(rows, output / "metric_progression.png")
    _plot_boxplots(rows, "inverse", INVERSE_METRICS, output / "inverse_boxplots.png")
    _plot_boxplots(rows, "forward", FORWARD_METRICS, output / "forward_boxplots.png")
    _plot_forward_horizons(rows, output / "forward_horizon_progression.png")
    print(f"[full71-summary] summarized {len(rows)} checkpoints under {output}", flush=True)


if __name__ == "__main__":
    main()
