#!/usr/bin/env python3
"""Compare two completed Phase-3 V2M/M2V evaluations on identical windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MOTION_DIRECTIONS = {
    "mpjpe_m": "lower",
    "pa_mpjpe_m": "lower",
    "feat_mse": "lower",
    "accel_err": "lower",
    "root_err_m": "lower",
}
VIDEO_DIRECTIONS = {
    "psnr_db": "higher",
    "ssim": "higher",
    "lpips_alex": "lower",
}
HEAD_CAMERA_DIRECTIONS = {
    "translation_m": "lower",
    "rotation_deg": "lower",
}


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _metric_row(baseline: float, candidate: float, direction: str) -> dict:
    delta = candidate - baseline
    relative_change = 100.0 * delta / abs(baseline) if baseline else None
    signed_improvement = -delta if direction == "lower" else delta
    improvement = 100.0 * signed_improvement / abs(baseline) if baseline else None
    return {
        "direction": direction,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": delta,
        "relative_change_pct": relative_change,
        "candidate_improvement_pct": improvement,
    }


def _compare_aggregates(baseline: dict, candidate: dict, directions: dict) -> dict:
    return {
        key: _metric_row(
            float(baseline[key]["mean"]),
            float(candidate[key]["mean"]),
            direction,
        )
        for key, direction in directions.items()
    }


def _paired_wins(baseline_rows: dict, candidate_rows: dict, directions: dict) -> dict:
    if set(baseline_rows) != set(candidate_rows):
        missing = sorted(set(baseline_rows) - set(candidate_rows))
        extra = sorted(set(candidate_rows) - set(baseline_rows))
        raise ValueError(f"paired rows differ: missing={missing} extra={extra}")
    result = {}
    for key, direction in directions.items():
        wins = ties = 0
        for name in baseline_rows:
            base = float(baseline_rows[name][key])
            cand = float(candidate_rows[name][key])
            delta = cand - base
            if abs(delta) <= 1e-12:
                ties += 1
            elif (direction == "lower" and delta < 0) or (
                direction == "higher" and delta > 0
            ):
                wins += 1
        result[key] = {
            "candidate_wins": wins,
            "ties": ties,
            "baseline_wins": len(baseline_rows) - wins - ties,
            "n": len(baseline_rows),
        }
    return result


def _cohort_aggregate(summary: dict, section: str, cohort: str) -> dict:
    if section == "motion":
        info = summary["motion"]["video2motion"]
    elif section == "video":
        info = summary["video_metrics"]["motimg2video"]
    else:
        info = summary["head_camera"]
    key = {
        "all71": "aggregate",
        "floor_valid66": "floor_valid_aggregate",
        "motion_clean71": "motion_clean71_aggregate",
    }[cohort]
    return info[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline_root = args.baseline_root.resolve()
    candidate_root = args.candidate_root.resolve()
    baseline_summary = _load(baseline_root / "summary.json")
    candidate_summary = _load(candidate_root / "summary.json")
    if baseline_summary.get("sampling") != candidate_summary.get("sampling"):
        raise ValueError("sampling contracts differ between evaluations")

    comparison = {
        "baseline_checkpoint": baseline_summary["ckpt"],
        "candidate_checkpoint": candidate_summary["ckpt"],
        "candidate_label": args.candidate_label,
        "sampling": baseline_summary["sampling"],
        "cohorts": {},
    }
    for cohort in ("all71", "floor_valid66", "motion_clean71"):
        comparison["cohorts"][cohort] = {
            "motion": _compare_aggregates(
                _cohort_aggregate(baseline_summary, "motion", cohort),
                _cohort_aggregate(candidate_summary, "motion", cohort),
                MOTION_DIRECTIONS,
            ),
            "video": _compare_aggregates(
                _cohort_aggregate(baseline_summary, "video", cohort),
                _cohort_aggregate(candidate_summary, "video", cohort),
                VIDEO_DIRECTIONS,
            ),
            "head_camera": _compare_aggregates(
                _cohort_aggregate(baseline_summary, "head_camera", cohort),
                _cohort_aggregate(candidate_summary, "head_camera", cohort),
                HEAD_CAMERA_DIRECTIONS,
            ),
        }

    clean_paths = {
        "motion": "motion_recon/video2motion/motion_recon_metrics_motion_clean71.json",
        "video": "video/motimg2video_metrics_motion_clean71.json",
        "head_camera": (
            "motion_recon/video2motion/"
            "head_camera_alignment_metrics_motion_clean71.json"
        ),
    }
    directions = {
        "motion": MOTION_DIRECTIONS,
        "video": VIDEO_DIRECTIONS,
        "head_camera": HEAD_CAMERA_DIRECTIONS,
    }
    comparison["motion_clean71_paired_wins"] = {}
    for section, relative_path in clean_paths.items():
        baseline_rows = _load(baseline_root / relative_path)["per_sequence"]
        candidate_rows = _load(candidate_root / relative_path)["per_sequence"]
        comparison["motion_clean71_paired_wins"][section] = _paired_wins(
            baseline_rows, candidate_rows, directions[section]
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(comparison, f, indent=2)
    print(json.dumps(comparison["cohorts"]["motion_clean71"], indent=2))
    print(f"comparison -> {args.out}")


if __name__ == "__main__":
    main()
