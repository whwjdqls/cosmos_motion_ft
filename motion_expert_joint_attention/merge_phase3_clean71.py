#!/usr/bin/env python3
"""Merge Phase-3 floor-valid66 metrics with five deterministic clean replacements."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np


MOTION_KEYS = (
    "mpjpe_m",
    "pa_mpjpe_m",
    "feat_mse",
    "accel_err",
    "accel_pred",
    "accel_gt",
    "root_err_m",
)
VIDEO_KEYS = ("psnr_db", "ssim", "lpips_alex")
HEAD_CAMERA_KEYS = (
    "translation_m",
    "rotation_deg",
    "gt_calibration_translation_m",
    "gt_calibration_rotation_deg",
)


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _dump(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _combine_rows(valid: dict, replacements: dict, label: str) -> dict:
    overlap = set(valid) & set(replacements)
    if overlap:
        raise ValueError(f"{label}: duplicate sequence keys: {sorted(overlap)}")
    rows = {**valid, **replacements}
    if len(valid) != 66 or len(replacements) != 5 or len(rows) != 71:
        raise ValueError(
            f"{label}: expected 66 + 5 = 71 rows, got "
            f"{len(valid)} + {len(replacements)} = {len(rows)}"
        )
    return rows


def _motion_payload(rows: dict) -> dict:
    aggregate = {}
    for key in MOTION_KEYS:
        values = np.asarray([row[key] for row in rows.values()], dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            values = np.asarray([np.nan])
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }
    return {"n": len(rows), "generative": False, "aggregate": aggregate, "per_sequence": rows}


def _stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("video metrics contain non-finite values")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _video_payload(rows: dict, template: dict) -> dict:
    horizons = tuple(next(iter(rows.values()))["horizons"])
    horizon_aggregate = {
        horizon: {
            key: {
                "mean": float(np.mean([row["horizons"][horizon][key] for row in rows.values()])),
                "median": float(
                    np.median([row["horizons"][horizon][key] for row in rows.values()])
                ),
            }
            for key in VIDEO_KEYS
        }
        for horizon in horizons
    }
    return {
        "n": len(rows),
        "task": template["task"],
        "conditioned_frame_excluded": template["conditioned_frame_excluded"],
        "evaluated_frame_range": template["evaluated_frame_range"],
        "gt_preprocessing": template["gt_preprocessing"],
        "aggregate": {
            key: _stats([row[key] for row in rows.values()]) for key in VIDEO_KEYS
        },
        "horizon_aggregate": horizon_aggregate,
        "per_sequence": rows,
    }


def _head_camera_payload(rows: dict, template: dict) -> dict:
    aggregate = {}
    for key in HEAD_CAMERA_KEYS:
        values = np.asarray([row[key] for row in rows.values()], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"head-camera metric {key} contains non-finite values")
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
        }
    return {
        "n": len(rows),
        "task": template["task"],
        "representation": template["representation"],
        "absolute_pose_used": template["absolute_pose_used"],
        "aggregate": aggregate,
        "per_sequence": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    args = parser.parse_args()

    full_root = args.full_root.resolve()
    replacement_root = args.replacement_root.resolve()
    template_root = args.template_root.resolve()

    full_summary_path = full_root / "summary.json"
    replacement_summary_path = replacement_root / "summary.json"
    full_summary = _load(full_summary_path)
    replacement_summary = _load(replacement_summary_path)

    expected_sampling = {
        "steps": 30,
        "cfg": 1.0,
        "seed": 0,
        "motion_schedule": "native",
        "motion_shift": 3.0,
        "motion_native_solver": "unipc",
        "gen_schedule": "native",
        "gen_shift": 3.0,
        "gen_native_solver": "unipc",
    }
    for label, summary, expected_n in (
        ("full", full_summary, 71),
        ("replacement", replacement_summary, 5),
    ):
        if summary.get("n") != expected_n:
            raise ValueError(f"{label}: expected n={expected_n}, got {summary.get('n')}")
        if summary.get("tasks") != ["video2motion", "motimg2video"]:
            raise ValueError(f"{label}: unexpected tasks {summary.get('tasks')}")
        if summary.get("sampling") != expected_sampling:
            raise ValueError(f"{label}: unexpected sampling contract {summary.get('sampling')}")

    full_motion_path = (
        full_root / "motion_recon/video2motion/motion_recon_metrics_floor_valid.json"
    )
    replacement_motion_path = (
        replacement_root / "motion_recon/video2motion/motion_recon_metrics.json"
    )
    full_video_path = full_root / "video/motimg2video_metrics_floor_valid.json"
    replacement_video_path = replacement_root / "video/motimg2video_metrics.json"

    full_motion = _load(full_motion_path)
    replacement_motion = _load(replacement_motion_path)
    motion_rows = _combine_rows(
        full_motion["per_sequence"], replacement_motion["per_sequence"], "motion"
    )
    motion_payload = _motion_payload(motion_rows)
    clean_motion_path = (
        full_root / "motion_recon/video2motion/motion_recon_metrics_motion_clean71.json"
    )
    _dump(clean_motion_path, motion_payload)

    full_video = _load(full_video_path)
    replacement_video = _load(replacement_video_path)
    video_rows = _combine_rows(
        full_video["per_sequence"], replacement_video["per_sequence"], "video"
    )
    video_payload = _video_payload(video_rows, full_video)
    clean_video_path = full_root / "video/motimg2video_metrics_motion_clean71.json"
    _dump(clean_video_path, video_payload)

    clean_head_camera_path = None
    full_head_camera_path = (
        full_root
        / "motion_recon/video2motion/head_camera_alignment_metrics_floor_valid.json"
    )
    replacement_head_camera_path = (
        replacement_root
        / "motion_recon/video2motion/head_camera_alignment_metrics.json"
    )
    if full_head_camera_path.is_file() or replacement_head_camera_path.is_file():
        if not full_head_camera_path.is_file() or not replacement_head_camera_path.is_file():
            raise FileNotFoundError(
                "head-camera clean71 merge requires both floor-valid66 and replacement5 metrics"
            )
        full_head_camera = _load(full_head_camera_path)
        replacement_head_camera = _load(replacement_head_camera_path)
        head_camera_rows = _combine_rows(
            full_head_camera["per_sequence"],
            replacement_head_camera["per_sequence"],
            "head-camera",
        )
        head_camera_payload = _head_camera_payload(head_camera_rows, full_head_camera)
        clean_head_camera_path = (
            full_root
            / "motion_recon/video2motion/head_camera_alignment_metrics_motion_clean71.json"
        )
        _dump(clean_head_camera_path, head_camera_payload)

    clean_windows_path = full_root / "motion_clean71_windows.json"
    replacement_windows_path = full_root / "motion_clean_replacement5_windows.json"
    shutil.copyfile(template_root / "motion_clean71_windows.json", clean_windows_path)
    shutil.copyfile(
        template_root / "motion_clean_replacement5_windows.json", replacement_windows_path
    )

    provenance = copy.deepcopy(_load(template_root / "motion_clean71_provenance.json"))
    provenance.update(
        {
            "floor_valid66_motion_metrics": str(full_motion_path),
            "floor_valid66_video_metrics": str(full_video_path),
            "replacement5_windows_json": str(replacement_windows_path),
            "replacement5_eval_root": str(replacement_root),
            "replacement5_motion_metrics": str(replacement_motion_path),
            "replacement5_video_metrics": str(replacement_video_path),
            "motion_clean71_windows_json": str(clean_windows_path),
            "motion_clean71_motion_metrics": str(clean_motion_path),
            "motion_clean71_video_metrics": str(clean_video_path),
        }
    )
    if clean_head_camera_path is not None:
        provenance.update(
            {
                "floor_valid66_head_camera_metrics": str(full_head_camera_path),
                "replacement5_head_camera_metrics": str(replacement_head_camera_path),
                "motion_clean71_head_camera_metrics": str(clean_head_camera_path),
            }
        )
    _dump(full_root / "motion_clean71_provenance.json", provenance)

    full_summary["motion"]["video2motion"].update(
        {
            "floor_valid_aggregate": full_motion["aggregate"],
            "motion_clean71_metrics_json": str(clean_motion_path),
            "motion_clean71_n": 71,
            "motion_clean71_aggregate": motion_payload["aggregate"],
        }
    )
    full_summary["video_metrics"]["motimg2video"].update(
        {
            "motion_clean71_metrics_json": str(clean_video_path),
            "motion_clean71_n": 71,
            "motion_clean71_aggregate": video_payload["aggregate"],
        }
    )
    if clean_head_camera_path is not None:
        full_summary["head_camera"].update(
            {
                "floor_valid_aggregate": full_head_camera["aggregate"],
                "motion_clean71_metrics_json": str(clean_head_camera_path),
                "motion_clean71_n": 71,
                "motion_clean71_aggregate": head_camera_payload["aggregate"],
            }
        )
    _dump(full_summary_path, full_summary)

    print(f"merged motion-clean71 metrics into {full_root}")
    result = {
        "motion": motion_payload["aggregate"],
        "video": video_payload["aggregate"],
    }
    if clean_head_camera_path is not None:
        result["head_camera"] = head_camera_payload["aggregate"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
