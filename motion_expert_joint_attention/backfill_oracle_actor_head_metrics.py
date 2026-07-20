#!/usr/bin/env python3
"""Add test-GT-derived per-actor head-camera diagnostics to saved Phase-3 V2M samples."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import config as C
from eval_all import _aggregate_head_camera_metrics, build_full_index, seq_name
from head_camera_alignment import (
    DEFAULT_CALIBRATION,
    actor_id_from_uuid,
    head_camera_errors,
    load_head_camera_calibration,
    load_oracle_actor_head_camera_calibrations,
    motion_to_camera_action,
)
from nymeria_joint_dataset import _load_rgb_cam, rel_action_from_window


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(temporary, path)


def _artifact_window_keys(payload: dict) -> set[tuple[str, int]]:
    return {
        (row["uuid"], int(row["start"]))
        for entry in payload["actors"].values()
        for row in entry["windows"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--windows-json", required=True)
    parser.add_argument("--oracle-test-actor-calibration", required=True)
    parser.add_argument("--head-camera-calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--manifest", default=C.NYMERIA_MANIFEST)
    parser.add_argument("--split-file", default=C.NYMERIA_SPLIT_FILE)
    parser.add_argument(
        "--uniego-root",
        default="/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep",
    )
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--check-global-tolerance-m", type=float, default=2e-5)
    parser.add_argument("--check-global-tolerance-deg", type=float, default=0.02)
    args = parser.parse_args()

    eval_root = Path(args.eval_root).resolve()
    metrics_path = (
        eval_root / "motion_recon/video2motion/head_camera_alignment_metrics.json"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open() as f:
        old_payload = json.load(f)

    index = build_full_index(
        args.manifest,
        args.split_file,
        "test",
        args.num_frames,
        latent_root="",
        uniego_root=args.uniego_root,
        require_latents=False,
        windows_json=args.windows_json,
    )
    if len(index) != len(old_payload["per_sequence"]):
        raise ValueError(
            f"saved metric/index mismatch: {len(old_payload['per_sequence'])} rows vs "
            f"{len(index)} requested windows"
        )

    global_rotation, global_lever, _ = load_head_camera_calibration(
        args.head_camera_calibration
    )
    actor_calibrations, oracle_payload = load_oracle_actor_head_camera_calibrations(
        args.oracle_test_actor_calibration
    )
    mean = torch.from_numpy(np.load(C.MOTION_STATS_MEAN).astype(np.float32))
    std = torch.from_numpy(np.load(C.MOTION_STATS_STD).astype(np.float32))

    rows = {}
    max_global_translation_delta = 0.0
    max_global_rotation_delta = 0.0
    for i, item in enumerate(index):
        name = seq_name(i, item["uuid"], item["s"])
        if name not in old_payload["per_sequence"]:
            raise KeyError(f"saved metrics do not contain expected row {name}")
        pred_path = eval_root / "motion_recon/video2motion/pred" / f"{name}.npy"
        gt_path = eval_root / "motion_recon/video2motion/gt" / f"{name}.npy"
        pred_z = torch.from_numpy(np.ascontiguousarray(np.load(pred_path))).float().unsqueeze(0)
        gt_z = torch.from_numpy(np.ascontiguousarray(np.load(gt_path))).float().unsqueeze(0)
        if pred_z.shape != gt_z.shape or pred_z.shape[-1] != len(mean):
            raise ValueError(
                f"{name}: bad saved motion shapes pred={tuple(pred_z.shape)} "
                f"gt={tuple(gt_z.shape)} stats={tuple(mean.shape)}"
            )
        n_frames = pred_z.shape[1]
        camera_position, camera_rotation = _load_rgb_cam(item["rgb"])
        camera_target = torch.from_numpy(rel_action_from_window(
            camera_position[item["s"]:item["s"] + n_frames],
            camera_rotation[item["s"]:item["s"] + n_frames],
        )).float().unsqueeze(0)
        transition_mask = torch.ones(camera_target.shape[:2], dtype=torch.bool)
        pred_motion = pred_z * std + mean
        gt_motion = gt_z * std + mean

        with torch.no_grad():
            global_pred = motion_to_camera_action(
                pred_motion, global_rotation, global_lever
            )
            global_gt = motion_to_camera_action(gt_motion, global_rotation, global_lever)
            global_pred_trans, global_pred_rot = head_camera_errors(
                global_pred, camera_target, transition_mask
            )
            global_gt_trans, global_gt_rot = head_camera_errors(
                global_gt, camera_target, transition_mask
            )

        old_row = old_payload["per_sequence"][name]
        checks = {
            "translation_m": float(global_pred_trans),
            "rotation_deg": float(global_pred_rot),
            "gt_calibration_translation_m": float(global_gt_trans),
            "gt_calibration_rotation_deg": float(global_gt_rot),
        }
        max_global_translation_delta = max(
            max_global_translation_delta,
            abs(checks["translation_m"] - old_row["translation_m"]),
            abs(
                checks["gt_calibration_translation_m"]
                - old_row["gt_calibration_translation_m"]
            ),
        )
        max_global_rotation_delta = max(
            max_global_rotation_delta,
            abs(checks["rotation_deg"] - old_row["rotation_deg"]),
            abs(
                checks["gt_calibration_rotation_deg"]
                - old_row["gt_calibration_rotation_deg"]
            ),
        )

        actor_id = actor_id_from_uuid(item["uuid"])
        if actor_id not in actor_calibrations:
            raise KeyError(f"oracle calibration has no entry for {actor_id}")
        actor_rotation, actor_lever = actor_calibrations[actor_id]
        with torch.no_grad():
            actor_pred = motion_to_camera_action(
                pred_motion, actor_rotation, actor_lever
            )
            actor_gt = motion_to_camera_action(gt_motion, actor_rotation, actor_lever)
            actor_pred_trans, actor_pred_rot = head_camera_errors(
                actor_pred, camera_target, transition_mask
            )
            actor_gt_trans, actor_gt_rot = head_camera_errors(
                actor_gt, camera_target, transition_mask
            )
        rows[name] = {
            **old_row,
            "actor_id": actor_id,
            "oracle_actor_translation_m": float(actor_pred_trans),
            "oracle_actor_rotation_deg": float(actor_pred_rot),
            "gt_oracle_actor_translation_m": float(actor_gt_trans),
            "gt_oracle_actor_rotation_deg": float(actor_gt_rot),
        }

    if max_global_translation_delta > args.check_global_tolerance_m:
        raise ValueError(
            "saved/recomputed global translation metric mismatch: "
            f"{max_global_translation_delta:.8f}m"
        )
    if max_global_rotation_delta > args.check_global_tolerance_deg:
        raise ValueError(
            "saved/recomputed global rotation metric mismatch: "
            f"{max_global_rotation_delta:.6f}deg"
        )

    fit_keys = _artifact_window_keys(oracle_payload)
    key_by_name = {
        seq_name(i, item["uuid"], item["s"]): (item["uuid"], int(item["s"]))
        for i, item in enumerate(index)
    }

    def oracle_metadata_for(names) -> dict:
        eval_keys = {key_by_name[name] for name in names}
        return {
            "path": str(Path(args.oracle_test_actor_calibration).resolve()),
            "kind": oracle_payload["kind"],
            "diagnostic_only": True,
            "uses_test_gt_motion": True,
            "uses_test_gt_camera": True,
            "fit_and_evaluation_windows_are_identical": bool(
                oracle_payload["leakage_contract"].get(
                    "fit_and_evaluation_windows_are_identical", False
                )
            ),
            "fit_windows": int(oracle_payload.get("counts", {}).get("windows", 0)),
            "fit_window_overlap": len(fit_keys & eval_keys),
            "evaluation_windows": len(eval_keys),
        }

    oracle_metadata = oracle_metadata_for(rows)
    payload = _aggregate_head_camera_metrics(rows)
    payload["train_global_calibration"] = str(
        Path(args.head_camera_calibration).resolve()
    )
    payload["oracle_test_actor_calibration"] = oracle_metadata
    _write_json(metrics_path, payload)

    floor_path = metrics_path.with_name("head_camera_alignment_metrics_floor_valid.json")
    floor_payload = None
    if floor_path.is_file():
        with floor_path.open() as f:
            old_floor_payload = json.load(f)
        floor_names = set(old_floor_payload["per_sequence"])
        missing = floor_names - set(rows)
        if missing:
            raise KeyError(f"floor-valid metrics contain unknown rows: {sorted(missing)}")
        floor_payload = _aggregate_head_camera_metrics({
            name: rows[name] for name in rows if name in floor_names
        })
        floor_payload["train_global_calibration"] = payload["train_global_calibration"]
        floor_payload["oracle_test_actor_calibration"] = oracle_metadata_for(floor_names)
        _write_json(floor_path, floor_payload)

    summary_path = eval_root / "summary.json"
    if summary_path.is_file():
        with summary_path.open() as f:
            summary = json.load(f)
        head = summary.setdefault("head_camera", {})
        head.update({
            "metrics_json": str(metrics_path),
            "aggregate": payload["aggregate"],
            "calibration": payload["train_global_calibration"],
            "oracle_test_actor_calibration": oracle_metadata,
            "oracle_test_actor_diagnostic_only": True,
        })
        if floor_payload is not None:
            head.update({
                "floor_valid_metrics_json": str(floor_path),
                "floor_valid_n": floor_payload["n"],
                "floor_valid_aggregate": floor_payload["aggregate"],
            })
        _write_json(summary_path, summary)

    print(
        "[backfill-oracle-head] global reproduction max delta: "
        f"{max_global_translation_delta:.8f}m/{max_global_rotation_delta:.6f}deg",
        flush=True,
    )
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"[backfill-oracle-head] updated {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
