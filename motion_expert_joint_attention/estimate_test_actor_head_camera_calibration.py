#!/usr/bin/env python3
"""Fit a diagnostic SOMA-Head -> camera transform for each held-out test actor.

This is intentionally an oracle diagnostic: each actor transform is fitted from synchronized
ground-truth motion and ground-truth camera data in the requested test windows. It must not be
used as a training input, a deployable test-time condition, or a leakage-free model metric.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import config as C
from estimate_head_camera_calibration import (
    DEFAULT_CAMERA,
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_UNIEGO,
    calibration_sample_from_arrays,
    fit_head_camera_calibration,
    load_calibration_window,
    optimize_head_camera_transform_from_relative_actions,
)
from head_camera_alignment import (
    DEFAULT_CALIBRATION,
    actor_id_from_uuid,
    load_head_camera_calibration,
)


def _calibration_entry(result: dict, windows: list[dict]) -> dict:
    return {
        "rotation_head_to_upright_camera": result["rotation"].float().tolist(),
        "camera_origin_in_head_m": result["lever"].float().tolist(),
        "counts": {
            "windows": len(windows),
            "sequences": len({row["uuid"] for row in windows}),
            **result["counts"],
        },
        "in_sample_gt_fit": result["fit"],
        "windows": windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--windows-json")
    source.add_argument(
        "--all-test-caption-windows",
        action="store_true",
        help="fit all unique usable 97-frame annotation windows in split_file[test]",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT)
    parser.add_argument("--uniego-root", default=DEFAULT_UNIEGO)
    parser.add_argument("--camera-root", default=DEFAULT_CAMERA)
    parser.add_argument("--window-frames", type=int, default=97)
    parser.add_argument("--orientation-stride", type=int, default=4)
    parser.add_argument("--train-global-calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--rotation-fit-max-samples", type=int, default=50_000)
    parser.add_argument(
        "--floor-calibration-json",
        default=C.FLOOR_CALIBRATION_JSON,
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.orientation_stride <= 0:
        raise ValueError("--orientation-stride must be positive")
    with open(args.split_file) as f:
        split_payload = json.load(f)
    test_uuids = set(split_payload["test"])
    if args.windows_json:
        with open(args.windows_json) as f:
            requested = json.load(f)
        if not isinstance(requested, list) or not requested:
            raise ValueError(f"{args.windows_json}: expected a non-empty JSON list")
    else:
        dropped = {}
        floor_path = Path(args.floor_calibration_json)
        if floor_path.is_file():
            with floor_path.open() as f:
                floor_payload = json.load(f)
            dropped = {
                uuid: {(int(row[0]), int(row[1])) for row in rows}
                for uuid, rows in floor_payload.get("dropped_windows", {}).items()
            }
        else:
            raise FileNotFoundError(
                f"floor calibration is required for oracle fitting: {floor_path}"
            )
        manifest_uuids = []
        requested = []
        seen = set()
        with open(args.manifest) as f:
            for line in f:
                record = json.loads(line)
                uuid = record.get("uuid")
                if uuid not in test_uuids:
                    continue
                manifest_uuids.append(uuid)
                n_sequence_frames = int(record.get("nb_frames", 0))
                for window in record.get("t2w_windows", []):
                    if not window.get("usable", False) or not window.get("caption"):
                        continue
                    start = int(window["start_frame"])
                    end = int(window["end_frame"])
                    key = (uuid, start)
                    if key in seen or (start, end) in dropped.get(uuid, set()):
                        continue
                    if start + args.window_frames > min(end, n_sequence_frames):
                        continue
                    seen.add(key)
                    requested.append({
                        "uuid": uuid,
                        "start": start,
                        "num_frames": args.window_frames,
                    })
        missing_manifest = sorted(test_uuids - set(manifest_uuids))
        if missing_manifest:
            raise ValueError(
                f"manifest is missing {len(missing_manifest)} test UUIDs: {missing_manifest[:5]}"
            )
        requested.sort(key=lambda row: (row["uuid"], row["start"]))
    keys = [(row["uuid"], int(row["start"])) for row in requested]
    if len(keys) != len(set(keys)):
        source_name = args.windows_json or args.manifest
        raise ValueError(f"{source_name}: duplicate (uuid,start) windows")
    non_test = sorted({uuid for uuid, _ in keys if uuid not in test_uuids})
    if non_test:
        raise ValueError(
            f"oracle test calibration received {len(non_test)} non-test UUIDs: {non_test[:5]}"
        )

    samples_by_actor = defaultdict(list)
    windows_by_actor = defaultdict(list)
    pooled_samples = []
    normalized_rows = []
    requested_by_uuid = defaultdict(list)
    for row in requested:
        requested_by_uuid[row["uuid"]].append(row)
    loaded_windows = 0
    for sequence_index, (uuid, sequence_rows) in enumerate(sorted(requested_by_uuid.items()), 1):
        if args.windows_json:
            for row in sequence_rows:
                start = int(row["start"])
                n_frames = int(row.get("num_frames", args.window_frames))
                sample = load_calibration_window(
                    Path(args.uniego_root) / f"{uuid}.npz",
                    Path(args.camera_root) / f"{uuid}.npz",
                    start,
                    n_frames,
                    args.orientation_stride,
                )
                normalized = {"uuid": uuid, "start": start, "num_frames": n_frames}
                actor = actor_id_from_uuid(uuid)
                samples_by_actor[actor].append(sample)
                windows_by_actor[actor].append(normalized)
                pooled_samples.append(sample)
                normalized_rows.append(normalized)
                loaded_windows += 1
        else:
            with np.load(Path(args.uniego_root) / f"{uuid}.npz") as data:
                features = data["features"].astype(np.float64)
            with np.load(Path(args.camera_root) / f"{uuid}.npz") as data:
                camera_rotation = data["cam_world_rot_upright"].astype(np.float64)
                camera_action = data["cam_action_upright_k1"].astype(np.float64)
            actor = actor_id_from_uuid(uuid)
            for row in sequence_rows:
                start = int(row["start"])
                n_frames = int(row.get("num_frames", args.window_frames))
                sample = calibration_sample_from_arrays(
                    features,
                    camera_rotation,
                    camera_action,
                    start,
                    n_frames,
                    args.orientation_stride,
                )
                normalized = {"uuid": uuid, "start": start, "num_frames": n_frames}
                samples_by_actor[actor].append(sample)
                windows_by_actor[actor].append(normalized)
                pooled_samples.append(sample)
                normalized_rows.append(normalized)
                loaded_windows += 1
        if sequence_index % 10 == 0 or sequence_index == len(requested_by_uuid):
            print(
                "[oracle-test-actor-calibration] loaded "
                f"{sequence_index}/{len(requested_by_uuid)} sequences, "
                f"{loaded_windows}/{len(requested)} windows",
                flush=True,
            )

    train_global_rotation, train_global_lever, _ = load_head_camera_calibration(
        args.train_global_calibration
    )
    actor_payload = {}
    for actor in sorted(samples_by_actor):
        actor_rotation, actor_lever, optimizer = (
            optimize_head_camera_transform_from_relative_actions(
            samples_by_actor[actor],
            train_global_rotation,
            train_global_lever,
            max_samples=args.rotation_fit_max_samples,
            )
        )
        result = fit_head_camera_calibration(
            samples_by_actor[actor],
            rotation_override=actor_rotation,
            lever_override=actor_lever,
        )
        actor_payload[actor] = _calibration_entry(result, windows_by_actor[actor])
        actor_payload[actor]["relative_transform_optimization"] = optimizer
        fit = result["fit"]
        print(
            f"[oracle-test-actor-calibration] {actor}: "
            f"n={len(windows_by_actor[actor])} "
            f"GT={fit['relative_translation_error_m']['mean']:.6f}m/"
            f"{fit['relative_rotation_error_deg']['mean']:.3f}deg",
            flush=True,
        )

    pooled_rotation, pooled_lever, pooled_optimizer = (
        optimize_head_camera_transform_from_relative_actions(
        pooled_samples,
        train_global_rotation,
        train_global_lever,
        max_samples=args.rotation_fit_max_samples,
        )
    )
    pooled = fit_head_camera_calibration(
        pooled_samples,
        rotation_override=pooled_rotation,
        lever_override=pooled_lever,
    )
    canonical_windows = json.dumps(
        normalized_rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = {
        "schema_version": 1,
        "kind": "oracle_test_actor_head_camera_calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "scope": (
            "one rigid calibration per actor over all unique usable floor-valid held-out "
            "annotation windows"
            if args.all_test_caption_windows
            else "one rigid calibration per actor over the exact requested held-out windows"
        ),
        "leakage_contract": {
            "uses_test_gt_motion": True,
            "uses_test_gt_camera": True,
            "fit_and_evaluation_windows_are_identical": bool(args.windows_json),
            "diagnostic_only": True,
            "valid_as_leakage_free_model_metric": False,
            "allowed_uses": [
                "estimate the actor-calibrated GT motion-to-camera representation floor",
                "diagnose how much global calibration contributes to V2M camera error",
            ],
            "forbidden_uses": [
                "training or model selection",
                "conditioning the model at test time",
                "reporting as a deployable or leakage-free benchmark metric",
            ],
        },
        "coordinate_contract": {
            "motion": "SOMA-30 UniEgo Head joint index 6, Kimodo Y-up",
            "camera": "upright RGB/OpenCV frame used by cam_action_upright_k1",
            "action": "inv(T_t) @ T_t+1; translation plus rotation columns 0/1",
            "absolute_translation_used": False,
            "world_basis_for_rotation_only": "kimodo(x,y,z)=(aria_x,aria_z,-aria_y)",
        },
        "sources": {
            "windows_json": (
                str(Path(args.windows_json).resolve()) if args.windows_json else None
            ),
            "manifest": str(Path(args.manifest).resolve()),
            "all_test_caption_windows": bool(args.all_test_caption_windows),
            "floor_calibration_json": str(Path(args.floor_calibration_json).resolve()),
            "windows_sha256": hashlib.sha256(canonical_windows).hexdigest(),
            "split_file": str(Path(args.split_file).resolve()),
            "uniego_root": str(Path(args.uniego_root).resolve()),
            "camera_root": str(Path(args.camera_root).resolve()),
            "orientation_stride": args.orientation_stride,
            "train_global_calibration": str(
                Path(args.train_global_calibration).resolve()
            ),
            "rotation_fit_max_samples": args.rotation_fit_max_samples,
        },
        "counts": {
            "actors": len(actor_payload),
            "windows": len(normalized_rows),
            "sequences": len({row["uuid"] for row in normalized_rows}),
        },
        "pooled_test_fit_reference": {
            **_calibration_entry(pooled, normalized_rows),
            "relative_transform_optimization": pooled_optimizer,
        },
        "actors": actor_payload,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"[oracle-test-actor-calibration] wrote {len(actor_payload)} actors to {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
