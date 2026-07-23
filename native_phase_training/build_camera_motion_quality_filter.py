#!/usr/bin/env python3
"""Build a versioned Nymeria T97 camera/motion quality exclusion manifest.

The native Phase-1 dataset uses cached video/camera windows but is later paired with a human-motion
expert.  This audit therefore removes windows that cannot be explained by one rigid Head-to-camera
transform over 97 frames.  It distinguishes source corruption from valid actor/session extrinsics:

* one-frame camera, Head, or cross-modal jumps are hard failures;
* impossible direct camera-to-Head separation is a hard failure;
* smooth rotational non-rigidity is measured after fitting a fixed rotation per window;
* smooth translational non-rigidity is measured after fitting a fixed Head-frame lever per window.

Disagreement with one train-global calibration and raw origin-relative trajectory drift are report-
only diagnostics.  They are not filters because a valid fixed extrinsic may differ by actor/session,
and a rigid camera's nonzero lever arm legitimately changes its origin trajectory during rotation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_expert_joint_attention.audit_nymeria_camera_motion import (  # noqa: E402
    ARIA_Z_UP_TO_KIMODO_Y_UP,
    _decode_uniego_head,
)


MOTION_ROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")
DEFAULT_DETAILS = Path(
    "/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/"
    "details_all.jsonl"
)
DEFAULT_OUTPUT = MOTION_ROOT / "metadata/camera_motion_quality_filter_v1_T97.json"
FILTER_KIND = "nymeria_camera_motion_quality_filter"
FILTER_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot summarize an empty metric")
    if not np.isfinite(values).all():
        raise ValueError("metric summary contains non-finite values")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _project_so3_batched(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vh
    reflected = np.linalg.det(rotation) < 0.0
    if np.any(reflected):
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        rotation = u @ vh
    return rotation


def _rotation_angle_deg(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    return np.rad2deg(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def _latent_path(uuid: str, start: int, root: Path) -> Path:
    subject = uuid.split("/", 1)[0] if "/" in uuid else "_misc"
    return root / subject / f"{uuid.replace('/', '__')}_{start}.npz"


def _load_pose_streams(uuid: str) -> dict[str, np.ndarray]:
    subject, sequence = uuid.split("/", 1)
    with np.load(MOTION_ROOT / "uniego_rep" / subject / f"{sequence}.npz") as data:
        features = data["features"].astype(np.float64)
    with np.load(MOTION_ROOT / "camera_rgb" / subject / f"{sequence}.npz") as data:
        camera_position = data["cam_world_pos_upright"].astype(np.float64)
        camera_rotation = data["cam_world_rot_upright"].astype(np.float64)
    head_rotation, head_position, _ = _decode_uniego_head(features)
    if len(head_position) != len(camera_position):
        raise ValueError(
            f"pose length mismatch for {uuid}: Head={len(head_position)} camera={len(camera_position)}"
        )
    return {
        "head_rotation": head_rotation,
        "head_position": head_position,
        "camera_rotation": ARIA_Z_UP_TO_KIMODO_Y_UP @ camera_rotation,
        "camera_position": (ARIA_Z_UP_TO_KIMODO_Y_UP @ camera_position.T).T,
    }


def _build_dataset_rows(
    *,
    manifest_path: Path,
    split_path: Path,
    latent_root: Path,
    num_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    split_payload = json.loads(split_path.read_text())
    split_for = {
        uuid: split
        for split in ("train", "test")
        for uuid in split_payload[split]
    }
    # Weka metadata latency makes 130k independent is_file calls very slow. Scan each subject
    # directory once, then use in-memory membership exactly as the earlier window-impact audit.
    cached_latents = {
        (subject_dir.name, path.name)
        for subject_dir in latent_root.glob("S*")
        if subject_dir.is_dir()
        for path in subject_dir.glob("*.npz")
    }
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for record in _read_jsonl(manifest_path):
        uuid = record.get("uuid")
        split = split_for.get(uuid)
        if split is None or not record.get("camera_path") or not record.get("vision_path"):
            continue
        frame_count = int(record.get("nb_frames", 0))
        for caption_window in record.get("t2w_windows", []):
            if not caption_window.get("usable", False) or not caption_window.get("caption"):
                continue
            start = int(caption_window["start_frame"])
            end = min(int(caption_window["end_frame"]), frame_count)
            while start + num_frames <= end:
                counts[f"{split}_candidates"] += 1
                latent = _latent_path(uuid, start, latent_root)
                latent_key = (latent.parent.name, latent.name)
                if latent_key in cached_latents:
                    rows.append(
                        {
                            "split": split,
                            "uuid": uuid,
                            "start": start,
                            "end": start + num_frames,
                        }
                    )
                    counts[f"{split}_cached_rows"] += 1
                else:
                    counts[f"{split}_missing_latents"] += 1
                start += num_frames
    return rows, dict(counts)


def _compute_window_metrics(
    rows: list[dict[str, Any]], *, num_frames: int
) -> dict[tuple[str, int, int], dict[str, float]]:
    grouped: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        grouped[row["uuid"]].add(int(row["start"]))

    offsets = np.arange(num_frames, dtype=np.int64)
    transition_offsets = np.arange(num_frames - 1, dtype=np.int64)
    metrics: dict[tuple[str, int, int], dict[str, float]] = {}
    for sequence_index, uuid in enumerate(sorted(grouped)):
        data = _load_pose_streams(uuid)
        head_r = data["head_rotation"]
        head_p = data["head_position"]
        camera_r = data["camera_rotation"]
        camera_p = data["camera_position"]
        starts = np.asarray(sorted(grouped[uuid]), dtype=np.int64)
        frame_indices = starts[:, None] + offsets[None, :]
        transition_indices = starts[:, None] + transition_offsets[None, :]

        relation = np.swapaxes(head_r, -1, -2) @ camera_r
        relation_windows = relation[frame_indices]
        fitted_rotation = _project_so3_batched(relation_windows.mean(axis=1))
        rotation_residual = _rotation_angle_deg(
            np.swapaxes(fitted_rotation, -1, -2)[:, None] @ relation_windows
        )

        offset_world = camera_p - head_p
        head_frame_offset = (
            np.swapaxes(head_r, -1, -2) @ offset_world[..., None]
        )[..., 0]
        offset_windows = offset_world[frame_indices]
        head_frame_offset_windows = head_frame_offset[frame_indices]
        fitted_lever = head_frame_offset_windows.mean(axis=1)
        lever_residual = np.linalg.norm(
            head_frame_offset_windows - fitted_lever[:, None], axis=-1
        )
        raw_trajectory_error = np.linalg.norm(
            offset_windows - offset_windows[:, :1], axis=-1
        )

        camera_delta_p = np.diff(camera_p, axis=0)
        head_delta_p = np.diff(head_p, axis=0)
        camera_delta_r = np.swapaxes(camera_r[:-1], -1, -2) @ camera_r[1:]
        head_delta_r = np.swapaxes(head_r[:-1], -1, -2) @ head_r[1:]
        camera_step_translation = np.linalg.norm(camera_delta_p, axis=-1)
        head_step_translation = np.linalg.norm(head_delta_p, axis=-1)
        camera_step_rotation = _rotation_angle_deg(camera_delta_r)
        head_step_rotation = _rotation_angle_deg(head_delta_r)
        direct_step_error = np.linalg.norm(camera_delta_p - head_delta_p, axis=-1)
        separation = np.linalg.norm(offset_world, axis=-1)

        for local_index, start in enumerate(starts.tolist()):
            key = (uuid, start, start + num_frames)
            transition_selector = transition_indices[local_index]
            frame_selector = frame_indices[local_index]
            metrics[key] = {
                "max_camera_translation_step_m": float(
                    camera_step_translation[transition_selector].max()
                ),
                "max_camera_rotation_step_deg": float(
                    camera_step_rotation[transition_selector].max()
                ),
                "max_head_translation_step_m": float(
                    head_step_translation[transition_selector].max()
                ),
                "max_head_rotation_step_deg": float(
                    head_step_rotation[transition_selector].max()
                ),
                "max_direct_step_translation_error_m": float(
                    direct_step_error[transition_selector].max()
                ),
                "max_head_camera_separation_m": float(separation[frame_selector].max()),
                "window_fixed_rotation_residual_mean_deg": float(
                    rotation_residual[local_index].mean()
                ),
                "window_fixed_rotation_residual_max_deg": float(
                    rotation_residual[local_index].max()
                ),
                "window_fixed_lever_residual_rmse_m": float(
                    np.sqrt(np.mean(lever_residual[local_index] ** 2))
                ),
                "window_fixed_lever_residual_max_m": float(
                    lever_residual[local_index].max()
                ),
                "raw_relative_trajectory_rmse_m": float(
                    np.sqrt(np.mean(raw_trajectory_error[local_index] ** 2))
                ),
                "fitted_lever_norm_m": float(np.linalg.norm(fitted_lever[local_index])),
            }
        if (sequence_index + 1) % 50 == 0 or sequence_index + 1 == len(grouped):
            print(
                f"[quality-filter] metrics {sequence_index + 1}/{len(grouped)} sequences, "
                f"{len(metrics)} unique windows",
                flush=True,
            )
    return metrics


def _threshold_contract(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "camera_translation_jump": {
            "metric": "max_camera_translation_step_m",
            "comparison": ">=",
            "value": args.max_camera_translation_step_m,
            "rationale": "0.25 m in one 20-FPS step is >=5 m/s and indicates a camera-pose discontinuity",
        },
        "camera_rotation_jump": {
            "metric": "max_camera_rotation_step_deg",
            "comparison": ">=",
            "value": args.max_camera_rotation_step_deg,
            "rationale": "30 degrees in one 20-FPS step is >=600 deg/s and is an intentionally conservative action-target gate",
        },
        "head_translation_jump": {
            "metric": "max_head_translation_step_m",
            "comparison": ">=",
            "value": args.max_head_translation_step_m,
            "rationale": "0.25 m in one frame is incompatible with a clean fitted Head trajectory",
        },
        "head_rotation_jump": {
            "metric": "max_head_rotation_step_deg",
            "comparison": ">=",
            "value": args.max_head_rotation_step_deg,
            "rationale": "30 degrees in one fitted Head frame removes extreme pose-fit discontinuities",
        },
        "cross_modal_translation_jump": {
            "metric": "max_direct_step_translation_error_m",
            "comparison": ">=",
            "value": args.max_direct_step_translation_error_m,
            "rationale": "camera and Head world displacements cannot differ by 0.25 m in one rigidly mounted frame",
        },
        "head_camera_separation": {
            "metric": "max_head_camera_separation_m",
            "comparison": ">",
            "value": args.max_head_camera_separation_m,
            "rationale": "the normal camera-origin lever is about 0.14 m; 0.5 m is physically implausible",
        },
        "smooth_rotation_nonrigid": {
            "metric": "window_fixed_rotation_residual_mean_deg",
            "comparison": ">",
            "value": args.max_window_rotation_residual_mean_deg,
            "rationale": "after fitting the best fixed per-window extrinsic, >25 deg mean residual cannot represent a rigid head-mounted camera",
        },
        "smooth_translation_nonrigid": {
            "metric": "window_fixed_lever_residual_rmse_m",
            "comparison": ">",
            "value": args.max_window_lever_residual_rmse_m,
            "rationale": "after fitting the best fixed per-window Head-frame lever, >5 cm RMS residual is excessive relative to the normal 14 cm lever",
        },
    }


def _reasons_for_metrics(
    metrics: dict[str, float], thresholds: dict[str, dict[str, Any]]
) -> list[str]:
    if not all(np.isfinite(value) for value in metrics.values()):
        return ["nonfinite_metrics"]
    reasons = []
    for reason, rule in thresholds.items():
        value = metrics[rule["metric"]]
        threshold = float(rule["value"])
        if rule["comparison"] == ">=" and value >= threshold:
            reasons.append(reason)
        elif rule["comparison"] == ">" and value > threshold:
            reasons.append(reason)
    return reasons


def _top_windows(
    metrics: dict[tuple[str, int, int], dict[str, float]], metric: str, count: int = 20
) -> list[dict[str, Any]]:
    ranked = sorted(metrics.items(), key=lambda item: item[1][metric], reverse=True)
    return [
        {
            "uuid": key[0],
            "start": key[1],
            "end": key[2],
            "value": values[metric],
        }
        for key, values in ranked[:count]
    ]


def _threshold_sensitivity(
    metrics: dict[tuple[str, int, int], dict[str, float]],
    split_for_key: dict[tuple[str, int, int], str],
    row_multiplicity: Counter,
) -> dict[str, Any]:
    """Report nearby cutoffs without changing the pinned filter contract."""
    sweeps = {
        "max_camera_rotation_step_deg": [30.0, 45.0, 60.0],
        "max_head_rotation_step_deg": [30.0, 45.0, 60.0],
        "window_fixed_rotation_residual_mean_deg": [20.0, 25.0, 30.0],
        "window_fixed_lever_residual_rmse_m": [0.02, 0.05, 0.10],
    }
    report: dict[str, Any] = {}
    for metric, cutoffs in sweeps.items():
        report[metric] = {}
        for cutoff in cutoffs:
            by_split = {}
            for split in ("train", "test"):
                selected = [
                    key
                    for key, values in metrics.items()
                    if split_for_key[key] == split and values[metric] >= cutoff
                ]
                by_split[split] = {
                    "unique_physical_windows": len(selected),
                    "dataset_rows": int(sum(row_multiplicity[key] for key in selected)),
                }
            report[metric][str(cutoff)] = by_split
    return report


def _subject_summary(
    metrics: dict[tuple[str, int, int], dict[str, float]],
    split_for_key: dict[tuple[str, int, int], str],
    row_multiplicity: Counter,
    reasons_for: dict[tuple[str, int, int], list[str]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"train": {}, "test": {}}
    subjects = sorted({key[0].split("/", 1)[0] for key in metrics})
    for split in ("train", "test"):
        for subject in subjects:
            keys = [
                key
                for key in metrics
                if split_for_key[key] == split and key[0].split("/", 1)[0] == subject
            ]
            if not keys:
                continue
            excluded = [key for key in keys if reasons_for[key]]
            input_rows = int(sum(row_multiplicity[key] for key in keys))
            excluded_rows = int(sum(row_multiplicity[key] for key in excluded))
            reason_rows = Counter()
            for key in excluded:
                for reason in reasons_for[key]:
                    reason_rows[reason] += row_multiplicity[key]
            summary[split][subject] = {
                "input_dataset_rows": input_rows,
                "input_unique_physical_windows": len(keys),
                "excluded_dataset_rows": excluded_rows,
                "excluded_unique_physical_windows": len(excluded),
                "excluded_dataset_row_fraction": excluded_rows / input_rows,
                "reason_dataset_rows": dict(sorted(reason_rows.items())),
            }
    return summary


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows, construction_counts = _build_dataset_rows(
        manifest_path=args.manifest,
        split_path=args.split_file,
        latent_root=args.latent_root,
        num_frames=args.num_frames,
    )
    metrics = _compute_window_metrics(rows, num_frames=args.num_frames)
    thresholds = _threshold_contract(args)
    reasons_for = {
        key: _reasons_for_metrics(values, thresholds) for key, values in metrics.items()
    }

    row_multiplicity = Counter(
        (row["uuid"], int(row["start"]), int(row["end"])) for row in rows
    )
    split_for_key = {
        (row["uuid"], int(row["start"]), int(row["end"])): row["split"] for row in rows
    }
    reason_rows = {split: Counter() for split in ("train", "test")}
    reason_unique = {split: Counter() for split in ("train", "test")}
    union_rows = Counter()
    union_unique = Counter()
    exclusions = []
    for key in sorted(metrics):
        reasons = reasons_for[key]
        if not reasons:
            continue
        split = split_for_key[key]
        multiplicity = row_multiplicity[key]
        union_rows[split] += multiplicity
        union_unique[split] += 1
        for reason in reasons:
            reason_rows[split][reason] += multiplicity
            reason_unique[split][reason] += 1
        exclusions.append(
            {
                "split": split,
                "uuid": key[0],
                "start": key[1],
                "end": key[2],
                "dataset_row_multiplicity": multiplicity,
                "reasons": reasons,
                "metrics": metrics[key],
            }
        )

    metric_names = sorted(next(iter(metrics.values())))
    distributions = {}
    for split in ("train", "test"):
        unique_keys = [key for key in metrics if split_for_key[key] == split]
        distributions[split] = {
            "unique_physical_windows": {
                metric: _stats(np.asarray([metrics[key][metric] for key in unique_keys]))
                for metric in metric_names
            },
            "dataset_rows": {
                metric: _stats(
                    np.asarray(
                        [
                            metrics[key][metric]
                            for key in unique_keys
                            for _ in range(row_multiplicity[key])
                        ]
                    )
                )
                for metric in metric_names
            },
        }

    summary_by_split = {}
    for split in ("train", "test"):
        total_rows = int(construction_counts.get(f"{split}_cached_rows", 0))
        total_unique = sum(1 for key in metrics if split_for_key[key] == split)
        summary_by_split[split] = {
            "input_dataset_rows": total_rows,
            "input_unique_physical_windows": total_unique,
            "excluded_dataset_rows": int(union_rows[split]),
            "excluded_unique_physical_windows": int(union_unique[split]),
            "kept_dataset_rows": total_rows - int(union_rows[split]),
            "kept_unique_physical_windows": total_unique - int(union_unique[split]),
            "excluded_dataset_row_fraction": (
                float(union_rows[split] / total_rows) if total_rows else 0.0
            ),
            "reason_dataset_rows": dict(reason_rows[split]),
            "reason_unique_physical_windows": dict(reason_unique[split]),
        }

    top_metric_names = (
        "window_fixed_rotation_residual_mean_deg",
        "window_fixed_lever_residual_rmse_m",
        "raw_relative_trajectory_rmse_m",
        "max_camera_translation_step_m",
        "max_camera_rotation_step_deg",
        "max_head_translation_step_m",
        "max_head_rotation_step_deg",
    )
    return {
        "kind": FILTER_KIND,
        "version": FILTER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_frames": args.num_frames,
        "fps": 20.0,
        "population_contract": (
            "Exact native_phase_training.build_cached_index construction over usable+captioned "
            "non-overlapping T97 subwindows whose cached latent exists; no motion floor-drop list "
            "is applied because native Phase 1 does not consume grounded motion"
        ),
        "filter_contract": (
            "Exclude exact physical (uuid,start,end) windows when any hard discontinuity or "
            "per-window rigid Head-camera consistency threshold fails. Duplicate caption rows "
            "sharing one physical window are all excluded."
        ),
        "explicit_non_filters": {
            "train_global_rotation_disagreement": (
                "not filtered because actor/session fixed extrinsics may legitimately differ"
            ),
            "raw_relative_trajectory_rmse_m": (
                "report-only because a rigid nonzero camera lever changes origin trajectory during rotation"
            ),
            "motion_floor_drop_list": (
                "not applied to camera/video-only Phase 1; Phase 2/3 retain their separate floor filter"
            ),
        },
        "inputs": {
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "split_file": str(args.split_file),
            "split_file_sha256": _sha256(args.split_file),
            "latent_root": str(args.latent_root),
            "source_audit_details": str(args.details),
            "source_audit_details_sha256": _sha256(args.details),
        },
        "thresholds": thresholds,
        "construction_counts": construction_counts,
        "summary_by_split": summary_by_split,
        "summary_by_subject": _subject_summary(
            metrics, split_for_key, row_multiplicity, reasons_for
        ),
        "threshold_sensitivity": _threshold_sensitivity(
            metrics, split_for_key, row_multiplicity
        ),
        "metric_distributions": distributions,
        "top_windows": {
            metric: _top_windows(metrics, metric) for metric in top_metric_names
        },
        "excluded_windows": exclusions,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=MOTION_ROOT / "video/manifest_video.jsonl"
    )
    parser.add_argument(
        "--split-file", type=Path, default=MOTION_ROOT / "train_test_split.json"
    )
    parser.add_argument(
        "--latent-root", type=Path, default=MOTION_ROOT / "joint_latents_T97"
    )
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--max-camera-translation-step-m", type=float, default=0.25)
    parser.add_argument("--max-camera-rotation-step-deg", type=float, default=30.0)
    parser.add_argument("--max-head-translation-step-m", type=float, default=0.25)
    parser.add_argument("--max-head-rotation-step-deg", type=float, default=30.0)
    parser.add_argument("--max-direct-step-translation-error-m", type=float, default=0.25)
    parser.add_argument("--max-head-camera-separation-m", type=float, default=0.5)
    parser.add_argument(
        "--max-window-rotation-residual-mean-deg", type=float, default=25.0
    )
    parser.add_argument(
        "--max-window-lever-residual-rmse-m", type=float, default=0.05
    )
    args = parser.parse_args()
    if args.num_frames <= 1:
        raise ValueError("--num-frames must be greater than one")

    payload = _build_report(args)
    _write_json_atomic(args.output, payload)
    output_hash = _sha256(args.output)
    print(json.dumps({
        "output": str(args.output),
        "sha256": output_hash,
        "summary_by_split": payload["summary_by_split"],
        "thresholds": payload["thresholds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
