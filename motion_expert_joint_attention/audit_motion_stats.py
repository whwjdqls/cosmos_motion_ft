#!/usr/bin/env python
"""Recompute candidate motion stats after floor calibration and quality filtering.

This is an audit utility. It never replaces the active stats in ``motion_expert/stats``;
existing checkpoints require those original normalization semantics. A future clean-stats
experiment must use a versioned output directory and retrain the motion expert.

Run in the Cosmos environment on a compute node::

    bash run.sh audit_motion_stats.py --split train --out-dir /weka/.../motion_stats_audit
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

import numpy as np

import config
from uniego_layout import FEAT_DIM, canonicalize_frame0, ground_features


DEFAULT_UNIEGO_ROOT = "/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep"


def _load_kept_windows(args):
    split_data = json.load(open(args.split_file))
    if args.split not in split_data or not isinstance(split_data[args.split], list):
        raise ValueError(f"{args.split!r} is not a sequence-list split in {args.split_file}")
    keep_uuids = set(split_data[args.split])

    calibration = json.load(open(args.calibration))
    deltas = {uuid: float(value) for uuid, value in calibration["deltas"].items()}
    drop_entries = calibration.get("dropped_windows", {})
    drop_map = {
        uuid: {(int(entry[0]), int(entry[1])): str(entry[2]) for entry in entries}
        for uuid, entries in drop_entries.items()
    }

    by_sequence = collections.defaultdict(list)
    raw_windows = 0
    dropped = collections.Counter()
    with open(args.manifest) as manifest_file:
        for line in manifest_file:
            record = json.loads(line)
            uuid = record.get("uuid")
            if uuid not in keep_uuids:
                continue
            if uuid not in deltas:
                raise ValueError(f"calibration has no per-sequence delta for {uuid}")
            num_frames = int(record.get("nb_frames", 0))
            for window in record.get("t2w_windows", []):
                if not window.get("usable", False) or not window.get("caption"):
                    continue
                raw_windows += 1
                start = int(window["start_frame"])
                raw_end = int(window["end_frame"])
                reason = drop_map.get(uuid, {}).get((start, raw_end))
                if reason is not None:
                    dropped[reason] += 1
                    continue
                end = min(raw_end, num_frames)
                if end <= start:
                    continue
                ground_offset = window.get("ground_offset_y")
                if ground_offset is None:
                    raise ValueError(f"kept window has no ground_offset_y: {uuid}@{start}")
                calibrated_offset = float(ground_offset) + deltas[uuid]
                by_sequence[uuid].append((start, end, calibrated_offset))

    return by_sequence, raw_windows, dropped


def _compute_stats(args, by_sequence, old_mean, old_std):
    count = 0
    stats_windows = 0
    sum_1 = np.zeros(FEAT_DIM, dtype=np.float64)
    sum_2 = np.zeros(FEAT_DIM, dtype=np.float64)
    old_zmax = []
    started = time.time()

    for index, (uuid, windows) in enumerate(sorted(by_sequence.items()), 1):
        path = os.path.join(args.uniego_root, uuid + ".npz")
        with np.load(path) as archive:
            all_features = archive["features"]
            for start, end, offset in windows:
                features = all_features[start:end].astype(np.float32)
                features = canonicalize_frame0(ground_features(features, offset))
                features_64 = features.astype(np.float64)
                zmax = float(np.abs((features_64 - old_mean) / old_std).max())
                old_zmax.append(zmax)
                if args.max_old_z > 0 and zmax > args.max_old_z:
                    continue
                sum_1 += features_64.sum(axis=0)
                sum_2 += np.square(features_64).sum(axis=0)
                count += len(features_64)
                stats_windows += 1
        if index % args.progress_every == 0 or index == len(by_sequence):
            print(
                f"[motion-stats-audit] {index}/{len(by_sequence)} sequences "
                f"frames={count} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if count == 0:
        raise ValueError("runtime feature guard rejected every candidate stats window")
    mean = sum_1 / count
    std = np.sqrt(np.maximum(sum_2 / count - np.square(mean), 0.0))
    std[std < args.const_eps] = 1.0
    return (
        mean.astype(np.float32),
        std.astype(np.float32),
        np.asarray(old_zmax),
        count,
        stats_windows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=config.NYMERIA_MANIFEST)
    parser.add_argument("--split-file", default=config.NYMERIA_SPLIT_FILE)
    parser.add_argument("--split", default="train")
    parser.add_argument("--calibration", default=config.FLOOR_CALIBRATION_JSON)
    parser.add_argument("--uniego-root", default=DEFAULT_UNIEGO_ROOT)
    parser.add_argument("--old-mean", default=config.MOTION_STATS_MEAN)
    parser.add_argument("--old-std", default=config.MOTION_STATS_STD)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--const-eps", type=float, default=1e-6)
    parser.add_argument(
        "--max-old-z",
        type=float,
        default=20.0,
        help="exclude windows rejected by the active runtime guard; <=0 disables",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    active_stats_dir = os.path.realpath(os.path.dirname(config.MOTION_STATS_MEAN))
    if os.path.realpath(args.out_dir) == active_stats_dir:
        raise ValueError("refusing to overwrite active checkpoint-compatible motion stats")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")

    old_mean = np.load(args.old_mean).astype(np.float64)
    old_std = np.load(args.old_std).astype(np.float64)
    if old_mean.shape != (FEAT_DIM,) or old_std.shape != (FEAT_DIM,):
        raise ValueError(f"expected {FEAT_DIM}-D old stats")
    if not np.isfinite(old_mean).all() or not np.isfinite(old_std).all() or (old_std <= 0).any():
        raise ValueError("old stats must be finite with strictly positive std")

    by_sequence, raw_windows, dropped = _load_kept_windows(args)
    if not by_sequence:
        raise ValueError(f"no kept windows found for split {args.split!r}")
    mean, std, old_zmax, frame_count, stats_windows = _compute_stats(
        args, by_sequence, old_mean, old_std
    )

    os.makedirs(args.out_dir, exist_ok=True)
    mean_path = os.path.join(args.out_dir, "clean_calibrated_uniego283_mean.npy")
    std_path = os.path.join(args.out_dir, "clean_calibrated_uniego283_std.npy")
    np.save(mean_path, mean)
    np.save(std_path, std)

    joint_y = np.asarray([joint * 9 + 7 for joint in range(30)])
    summary = {
        "manifest": args.manifest,
        "split_file": args.split_file,
        "split": args.split,
        "calibration": args.calibration,
        "raw_usable_captioned_windows": raw_windows,
        "dropped_windows": int(sum(dropped.values())),
        "drop_reasons": dict(sorted(dropped.items())),
        "kept_windows_after_floor_filter": int(len(old_zmax)),
        "runtime_guard_max_old_z": args.max_old_z,
        "runtime_guard_rejected": int(len(old_zmax) - stats_windows),
        "stats_windows": int(stats_windows),
        "frames": int(frame_count),
        "candidate_mean": mean_path,
        "candidate_std": std_path,
        "old_mean": args.old_mean,
        "old_std": args.old_std,
        "mean_abs_change_all": float(np.abs(mean - old_mean).mean()),
        "std_ratio_all_median": float(np.median(std / old_std)),
        "joint_y": {
            "old_mean_median": float(np.median(old_mean[joint_y])),
            "clean_mean_median": float(np.median(mean[joint_y])),
            "mean_shift_median": float(np.median(mean[joint_y] - old_mean[joint_y])),
            "old_std_median": float(np.median(old_std[joint_y])),
            "clean_std_median": float(np.median(std[joint_y])),
            "std_ratio_median": float(np.median(std[joint_y] / old_std[joint_y])),
        },
        "old_normalization_zmax_per_kept_caption_window": {
            "median": float(np.median(old_zmax)),
            "p90": float(np.percentile(old_zmax, 90)),
            "p95": float(np.percentile(old_zmax, 95)),
            "p99": float(np.percentile(old_zmax, 99)),
            "max": float(old_zmax.max()),
            "gt20_count": int((old_zmax > 20).sum()),
            "gt20_fraction": float((old_zmax > 20).mean()),
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    temporary_path = summary_path + ".tmp"
    with open(temporary_path, "w") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, summary_path)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
