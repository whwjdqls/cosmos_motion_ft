#!/usr/bin/env python
"""Audit BONES training windows under one or more versioned UniEgo normalizers.

Rows are sampled uniformly from the actual Phase-2 BONES pairs JSONL. Duplicate
physical windows with different captions retain their row multiplicity because
they are distinct training samples. Each window follows the runtime transform:
slice -> frame-0 canonicalization -> z-score; BONES is already floor-grounded.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from uniego_layout import CANON_DELTA_SLICE, FEAT_DIM, IDENTITY_DELTA9


JOINT_NAMES = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin",
    "RightFoot", "RightToeBase",
)
CHANNEL_NAMES = ("r6d0", "r6d1", "r6d2", "r6d3", "r6d4", "r6d5", "x", "y", "z")


def _feature_name(index: int) -> str:
    if index < 270:
        joint, channel = divmod(index, 9)
        return f"{JOINT_NAMES[joint]}.{CHANNEL_NAMES[channel]}"
    if index < 279:
        return f"canon_delta.{CHANNEL_NAMES[index - 270]}"
    return f"foot_contact_{index - 279}"


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stats(specs: list[list[str]]) -> dict[str, tuple[np.ndarray, np.ndarray, dict]]:
    loaded = {}
    for label, mean_path, std_path in specs:
        if label in loaded:
            raise ValueError(f"duplicate stats label: {label}")
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        if mean.shape != (FEAT_DIM,) or std.shape != (FEAT_DIM,):
            raise ValueError(f"{label}: expected {FEAT_DIM}-D stats")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"{label}: stats must be finite with positive std")
        loaded[label] = (
            mean,
            std,
            {
                "mean": str(Path(mean_path).resolve()),
                "std": str(Path(std_path).resolve()),
                "mean_sha256": _file_sha256(mean_path),
                "std_sha256": _file_sha256(std_path),
            },
        )
    return loaded


def _sample_rows(path: str, max_rows: int, seed: int):
    """Uniform reservoir sample; max_rows<=0 retains the full row population."""
    rng = random.Random(seed)
    selected: list[tuple[str, int, int]] = []
    total = 0
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            value = (str(row["uniego_path"]), int(row["start"]), int(row["end"]))
            total += 1
            if max_rows <= 0 or len(selected) < max_rows:
                selected.append(value)
            else:
                replacement = rng.randrange(total)
                if replacement < max_rows:
                    selected[replacement] = value
    return selected, total


def _weighted_percentiles(values: list[float], weights: list[int]) -> dict[str, float]:
    expanded = np.repeat(
        np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.int64)
    )
    return {
        name: float(np.percentile(expanded, percentile))
        for name, percentile in (
            ("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99),
            ("p99_9", 99.9), ("max", 100),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument(
        "--stats", nargs=3, action="append", required=True,
        metavar=("LABEL", "MEAN_NPY", "STD_NPY"),
    )
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--thresholds", default="10,15,20,30,50")
    parser.add_argument("--progress-every", type=int, default=5_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    thresholds = sorted({float(value) for value in args.thresholds.split(",")})
    if not thresholds or thresholds[0] <= 0:
        parser.error("--thresholds must contain positive values")
    stats = _load_stats(args.stats)
    started = time.time()
    rows, population_rows = _sample_rows(args.pairs, args.max_rows, args.seed)
    by_path: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for path, start, end in rows:
        if end <= start or end - start > 200:
            raise ValueError(f"invalid Phase-2 BONES window {path}[{start}:{end}]")
        by_path[path][(start, end)] += 1

    state = {}
    for label in stats:
        state[label] = {
            "zmax": [],
            "weights": [],
            "threshold_rows": np.zeros(len(thresholds), dtype=np.int64),
            "feature_rows_over_20": np.zeros(FEAT_DIM, dtype=np.int64),
            "max_feature_over_20": np.zeros(FEAT_DIM, dtype=np.int64),
            "worst": [],
        }

    missing_rows = 0
    invalid_rows = 0
    valid_rows = 0
    unique_windows = 0
    lengths = []
    length_weights = []
    for path_index, (path, windows) in enumerate(sorted(by_path.items()), 1):
        try:
            with np.load(path) as archive:
                all_features = np.asarray(archive["features"])
                for (start, end), multiplicity in windows.items():
                    unique_windows += 1
                    features = np.asarray(all_features[start:end], dtype=np.float32).copy()
                    if features.shape != (end - start, FEAT_DIM) or not np.isfinite(features).all():
                        invalid_rows += multiplicity
                        continue
                    features[0, CANON_DELTA_SLICE] = IDENTITY_DELTA9
                    valid_rows += multiplicity
                    lengths.append(len(features))
                    length_weights.append(multiplicity)
                    for label, (mean, std, _provenance) in stats.items():
                        feature_peak = np.max(np.abs((features - mean) / std), axis=0)
                        max_feature = int(np.argmax(feature_peak))
                        zmax = float(feature_peak[max_feature])
                        current = state[label]
                        current["zmax"].append(zmax)
                        current["weights"].append(multiplicity)
                        current["threshold_rows"] += multiplicity * (
                            zmax > np.asarray(thresholds)
                        )
                        over_20 = feature_peak > 20.0
                        current["feature_rows_over_20"] += multiplicity * over_20
                        if zmax > 20.0:
                            current["max_feature_over_20"][max_feature] += multiplicity
                        candidate = (zmax, path, start, end, max_feature)
                        if len(current["worst"]) < 20:
                            heapq.heappush(current["worst"], candidate)
                        elif zmax > current["worst"][0][0]:
                            heapq.heapreplace(current["worst"], candidate)
        except (OSError, KeyError, ValueError, EOFError):
            missing_rows += sum(windows.values())
        if path_index % args.progress_every == 0 or path_index == len(by_path):
            print(
                f"[bones-normalization] paths={path_index}/{len(by_path)} "
                f"valid_rows={valid_rows}/{len(rows)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if valid_rows == 0:
        raise RuntimeError("no valid BONES rows were audited")
    report = {
        "protocol": {
            "pairs": str(Path(args.pairs).resolve()),
            "population_rows": population_rows,
            "sample_rows_requested": args.max_rows,
            "sample_rows_selected": len(rows),
            "sample_seed": args.seed,
            "sampling": "uniform reservoir over pair rows",
            "row_semantics": (
                "caption rows retain multiplicity, matching Phase-2 BONES sample weighting"
            ),
            "transform": "raw BONES slice -> canonicalize frame 0 -> z-score",
            "grounding": "none; BONES UniEgo is already grounded",
            "T_max": 200,
            "thresholds": thresholds,
        },
        "coverage": {
            "sampled_paths": len(by_path),
            "sampled_unique_physical_windows": unique_windows,
            "valid_rows": valid_rows,
            "missing_rows": missing_rows,
            "invalid_rows": invalid_rows,
            "length_frames": _weighted_percentiles(lengths, length_weights),
        },
        "stats": {},
        "elapsed_sec": time.time() - started,
    }
    for label, (_mean, _std, provenance) in stats.items():
        current = state[label]
        threshold_counts = current["threshold_rows"]
        feature_counts = current["feature_rows_over_20"]
        max_counts = current["max_feature_over_20"]
        top_trigger = np.argsort(feature_counts)[::-1][:20]
        top_max = np.argsort(max_counts)[::-1][:20]
        report["stats"][label] = {
            "provenance": provenance,
            "zmax_percentiles": _weighted_percentiles(
                current["zmax"], current["weights"]
            ),
            "thresholds": {
                str(value): {
                    "rows": int(count),
                    "fraction": float(count / valid_rows),
                }
                for value, count in zip(thresholds, threshold_counts)
            },
            "top_features_with_any_abs_z_gt20": [
                {
                    "index": int(index),
                    "feature": _feature_name(int(index)),
                    "rows": int(feature_counts[index]),
                    "fraction": float(feature_counts[index] / valid_rows),
                }
                for index in top_trigger if feature_counts[index] > 0
            ],
            "top_argmax_features_for_rejected_rows": [
                {
                    "index": int(index),
                    "feature": _feature_name(int(index)),
                    "rows": int(max_counts[index]),
                    "fraction": float(max_counts[index] / valid_rows),
                }
                for index in top_max if max_counts[index] > 0
            ],
            "worst_windows": [
                {
                    "zmax": float(zmax),
                    "path": path,
                    "start": int(start),
                    "end": int(end),
                    "feature_index": int(feature),
                    "feature": _feature_name(int(feature)),
                }
                for zmax, path, start, end, feature in sorted(
                    current["worst"], reverse=True
                )
            ],
        }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, out)
    print(f"[bones-normalization] wrote {out}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
