#!/usr/bin/env python3
"""Map source-level Nymeria camera/motion defects into actual training windows.

Input is the JSONL emitted by ``audit_nymeria_camera_motion.py``. This script applies the same
caption/usable/full-T window construction as the Phase-1 and Phase-3 datasets and reports which
T=97 windows intersect a catastrophic camera step, motion-Head step, direct cross-modal step
disagreement, or a >0.5 m camera-to-Head separation. Phase 1 does not apply motion-floor filtering;
Phase 2/3 do, so both populations are reported separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


MOTION_ROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")
DEFAULT_DETAILS = Path(
    "/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/"
    "details_all.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/"
    "training_window_impact_T97.json"
)
DEFAULT_LATENTS = MOTION_ROOT / "joint_latents_T97"
DEFAULT_FLOOR_CALIBRATION = MOTION_ROOT / "metadata/floor_calibration.json"


def _overlaps_range(start: int, end: int, ranges: list[dict[str, int]]) -> bool:
    return any(int(item["start"]) < end and int(item["end"]) > start for item in ranges)


def _overlaps_indices(start: int, end: int, indices: set[int]) -> bool:
    return any(start <= index < end for index in indices)


def _latent_key(uuid: str, start: int) -> tuple[str, str]:
    subject = uuid.split("/", 1)[0]
    return subject, f"{uuid.replace('/', '__')}_{start}.npz"


def _flags(result: dict[str, Any], start: int, end: int) -> dict[str, bool]:
    # Frames are [start,end); relative-action transition i is i -> i+1 and is present when
    # start <= i < end-1.
    shared = result.get("shared_world", {})
    camera = result.get("camera_preprocess", {})
    return {
        "camera_step": _overlaps_indices(
            start,
            end - 1,
            set(map(int, camera.get("implausible_camera_step_indices", []))),
        ),
        "camera_translation_step": _overlaps_indices(
            start,
            end - 1,
            set(map(int, camera.get("camera_translation_step_over_0p25m_indices", []))),
        ),
        "camera_rotation_step": _overlaps_indices(
            start,
            end - 1,
            set(map(int, camera.get("camera_rotation_step_over_30deg_indices", []))),
        ),
        "motion_head_step": _overlaps_indices(
            start,
            end - 1,
            set(map(int, camera.get("implausible_motion_head_step_indices", []))),
        ),
        "motion_head_translation_step": _overlaps_indices(
            start,
            end - 1,
            set(
                map(
                    int,
                    camera.get("motion_head_translation_step_over_0p25m_indices", []),
                )
            ),
        ),
        "motion_head_rotation_step": _overlaps_indices(
            start,
            end - 1,
            set(map(int, camera.get("motion_head_rotation_step_over_30deg_indices", []))),
        ),
        "direct_cross_modal_step": _overlaps_indices(
            start,
            end - 1,
            set(map(int, shared.get("direct_step_error_over_0p25m_indices", []))),
        ),
        "head_camera_separation": _overlaps_range(
            start,
            end,
            shared.get("head_camera_distance_over_0p5m_ranges", []),
        ),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    sequences = set()
    examples: list[dict[str, Any]] = []
    for row in rows:
        counts["windows"] += 1
        if row["cached_latent"]:
            counts["cached_latent"] += 1
        active = [name for name, value in row["defects"].items() if value]
        for name in active:
            counts[name] += 1
        if active:
            counts["any_defect"] += 1
            sequences.add(row["uuid"])
            if len(examples) < 200:
                examples.append(row)
        else:
            counts["clean"] += 1
    return {
        **dict(counts),
        "affected_sequences": len(sequences),
        "affected_window_fraction": (
            counts["any_defect"] / counts["windows"] if counts["windows"] else 0.0
        ),
        "affected_cached_window_fraction": (
            sum(row["cached_latent"] and any(row["defects"].values()) for row in rows)
            / counts["cached_latent"]
            if counts["cached_latent"]
            else 0.0
        ),
        "affected_examples_first_200": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument(
        "--manifest", type=Path, default=MOTION_ROOT / "video/manifest_video.jsonl"
    )
    parser.add_argument(
        "--split-file", type=Path, default=MOTION_ROOT / "train_test_split.json"
    )
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENTS)
    parser.add_argument(
        "--floor-calibration",
        type=Path,
        default=DEFAULT_FLOOR_CALIBRATION,
        help="Phase-2/3 floor calibration and whole-caption-window exclusion list",
    )
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audit = {row["uuid"]: row for row in map(json.loads, args.details.read_text().splitlines())}
    # One directory scan is substantially cheaper on Weka than issuing an is_file/stat call for
    # every manifest window.
    cached_latents = {
        (subject_dir.name, path.name)
        for subject_dir in args.latent_root.glob("S*")
        if subject_dir.is_dir()
        for path in subject_dir.glob("*.npz")
    }
    split_data = json.loads(args.split_file.read_text())
    split_for = {
        uuid: split_name
        for split_name in ("train", "test")
        for uuid in split_data[split_name]
    }
    floor_data = json.loads(args.floor_calibration.read_text())
    floor_drops = {
        uuid: {
            (int(entry[0]), int(entry[1])): (
                str(entry[2]) if len(entry) > 2 else "unknown"
            )
            for entry in entries
        }
        for uuid, entries in floor_data.get("dropped_windows", {}).items()
    }
    phase1_aligned: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    phase23_aligned: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    phase1_native: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    phase23_native: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    floor_drop_counts = Counter()

    for record in map(json.loads, args.manifest.read_text().splitlines()):
        uuid = record.get("uuid")
        split = split_for.get(uuid)
        result = audit.get(uuid)
        if split is None or result is None or "shared_world" not in result:
            continue
        frame_count = int(record.get("nb_frames", 0))
        for window in record.get("t2w_windows", []):
            if not window.get("usable", False) or not window.get("caption"):
                continue
            window_start = int(window["start_frame"])
            window_end = min(int(window["end_frame"]), frame_count)
            if window_end <= window_start:
                continue
            floor_drop_reason = floor_drops.get(uuid, {}).get(
                (int(window["start_frame"]), int(window["end_frame"]))
            )
            if floor_drop_reason is not None:
                floor_drop_counts[(split, floor_drop_reason)] += 1

            native_row = {
                "uuid": uuid,
                "start": window_start,
                "end": window_end,
                "cached_latent": False,
                "defects": _flags(result, window_start, window_end),
            }
            phase1_native[split].append(native_row)
            if floor_drop_reason is None:
                phase23_native[split].append(native_row)
            start = window_start
            while start + args.num_frames <= window_end:
                end = start + args.num_frames
                aligned_row = {
                    "uuid": uuid,
                    "start": start,
                    "end": end,
                    "cached_latent": _latent_key(uuid, start) in cached_latents,
                    "defects": _flags(result, start, end),
                }
                phase1_aligned[split].append(aligned_row)
                if floor_drop_reason is None:
                    phase23_aligned[split].append(aligned_row)
                start += args.num_frames

    report = {
        "contract": {
            "num_frames": args.num_frames,
            "camera_step_bad": "translation >=0.25 m or rotation >=30 deg in one 20-FPS step",
            "motion_head_step_bad": "translation >=0.25 m or rotation >=30 deg in one 20-FPS step",
            "direct_cross_modal_step_bad": "|delta camera world - delta Head world| >0.25 m",
            "head_camera_separation_bad": "direct shared-world origin distance >0.5 m",
            "aligned_window_construction": (
                "same usable+captioned full-T non-overlapping construction as active datasets; "
                "Phase-2/3 populations additionally remove whole caption spans named in the "
                "floor-calibration drop list"
            ),
        },
        "inputs": {
            "details": str(args.details),
            "manifest": str(args.manifest),
            "split_file": str(args.split_file),
            "latent_root": str(args.latent_root),
            "floor_calibration": str(args.floor_calibration),
        },
        "floor_filter": {
            "dropped_caption_windows_by_split_reason": {
                split: {
                    reason: floor_drop_counts[(split, reason)]
                    for reason in sorted(
                        {key_reason for key_split, key_reason in floor_drop_counts if key_split == split}
                    )
                }
                for split in ("train", "test")
            }
        },
        "phase1_aligned_T97_unfiltered": {
            split: _summarize(rows) for split, rows in phase1_aligned.items()
        },
        "phase23_aligned_T97_floor_filtered": {
            split: _summarize(rows) for split, rows in phase23_aligned.items()
        },
        "phase1_native_caption_spans_unfiltered": {
            split: _summarize(rows) for split, rows in phase1_native.items()
        },
        "phase23_native_caption_spans_floor_filtered": {
            split: _summarize(rows) for split, rows in phase23_native.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aligned_affected_path = args.output.with_name("affected_aligned_T97.jsonl")
    native_affected_path = args.output.with_name("affected_native_caption_spans.jsonl")
    phase23_aligned_path = args.output.with_name(
        "affected_phase23_aligned_T97_floor_filtered.jsonl"
    )
    phase23_native_path = args.output.with_name(
        "affected_phase23_native_caption_spans_floor_filtered.jsonl"
    )
    with aligned_affected_path.open("w") as handle:
        for split, rows in phase1_aligned.items():
            for row in rows:
                if any(row["defects"].values()):
                    handle.write(json.dumps({"split": split, **row}) + "\n")
    with native_affected_path.open("w") as handle:
        for split, rows in phase1_native.items():
            for row in rows:
                if any(row["defects"].values()):
                    handle.write(json.dumps({"split": split, **row}) + "\n")
    with phase23_aligned_path.open("w") as handle:
        for split, rows in phase23_aligned.items():
            for row in rows:
                if any(row["defects"].values()):
                    handle.write(json.dumps({"split": split, **row}) + "\n")
    with phase23_native_path.open("w") as handle:
        for split, rows in phase23_native.items():
            for row in rows:
                if any(row["defects"].values()):
                    handle.write(json.dumps({"split": split, **row}) + "\n")
    report["outputs"] = {
        "affected_phase1_aligned_T97_unfiltered": str(aligned_affected_path),
        "affected_phase1_native_caption_spans_unfiltered": str(native_affected_path),
        "affected_phase23_aligned_T97_floor_filtered": str(phase23_aligned_path),
        "affected_phase23_native_caption_spans_floor_filtered": str(phase23_native_path),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
