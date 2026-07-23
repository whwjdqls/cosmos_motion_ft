#!/usr/bin/env python
"""Evaluate the fixed-prefix native Phase-1 checkpoint suite.

Video metrics are computed only on the generated RGB suffix.  Each suffix is
split into three equal-duration relative horizons, so a 49-frame condition is
not compared using horizon bins defined for a one-frame condition.  Policy
camera metrics are reported both for all 96 transitions and for the future
suffix re-anchored at the last clean RGB frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from native_phase_training.evaluate_inverse_forward import (
    LPIPSAlex,
    METRIC_KEYS_FORWARD,
    METRIC_KEYS_INVERSE,
    _aggregate,
    _frame_metrics,
    _load_successful_output,
    _read_jsonl,
    _read_video_rgb,
    _resize_gt_like_native,
    _write_json,
)
from nymeria_world.eval_inverse_dynamics import eval_seq, gt_abs


VISUAL_MODES = ("forward_dynamics", "policy", "image2video")
INPUT_FILES = {
    "inverse_dynamics": "invdyn_input.jsonl",
    "forward_dynamics": "fd_input.jsonl",
    "policy": "policy_input.jsonl",
    "image2video": "i2v_input.jsonl",
}
HORIZON_NAMES = ("early", "middle", "late")


def _parse_prefixes(raw: str) -> tuple[int, ...]:
    try:
        prefixes = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"invalid prefix list {raw!r}") from error
    if not prefixes or len(prefixes) != len(set(prefixes)):
        raise ValueError(f"prefix list must be non-empty and unique, got {prefixes}")
    return prefixes


def _source_name(record: dict[str, Any]) -> str:
    source = record.get("source_name")
    if isinstance(source, str) and source:
        return source
    name = record.get("name")
    mode = record.get("model_mode")
    suffix = f"_{mode}"
    if (
        any(key in record for key in ("rgb_prefix_length", "latent_prefix_length"))
        or not isinstance(name, str)
        or not isinstance(mode, str)
        or not name.endswith(suffix)
    ):
        raise ValueError(f"prefix evaluation record is missing source_name: {name!r}")
    # Legacy fixed-prefix fixtures predate explicit source/prefix metadata.
    return name[: -len(suffix)]


def _rgb_prefix_length(record: dict[str, Any]) -> int:
    value = record.get("rgb_prefix_length")
    if value is None and "source_name" not in record:
        return 1
    if not isinstance(value, int):
        raise ValueError(f"invalid RGB prefix length in record {record.get('name')!r}: {value!r}")
    return value


def _action_from_output(payload: dict[str, Any], sample_dir: Path) -> np.ndarray:
    action = np.asarray(payload["outputs"][0]["content"].get("action"), dtype=np.float64)
    if action.shape != (96, 9) or not np.isfinite(action).all():
        raise ValueError(f"{sample_dir}: expected finite action [96,9], got {action.shape}")
    return action


def _relative_horizon_summary(values: dict[str, np.ndarray], rgb_prefix: int) -> dict[str, Any]:
    suffix_length = len(next(iter(values.values())))
    if suffix_length != 97 - rgb_prefix:
        raise ValueError(
            f"prefix {rgb_prefix}: expected suffix length {97 - rgb_prefix}, got {suffix_length}"
        )
    boundaries = np.linspace(0, suffix_length, 4, dtype=np.int64)
    horizons: dict[str, Any] = {}
    for name, start, stop in zip(HORIZON_NAMES, boundaries[:-1], boundaries[1:], strict=True):
        if stop <= start:
            raise ValueError(f"empty {name} horizon for prefix {rgb_prefix}")
        horizons[name] = {
            **{key: float(metric[start:stop].mean()) for key, metric in values.items()},
            "suffix_offset_range": [int(start), int(stop - 1)],
            "rgb_frame_range": [int(rgb_prefix + start), int(rgb_prefix + stop - 1)],
            "frames": int(stop - start),
        }
    return {
        **{key: float(metric.mean()) for key, metric in values.items()},
        "rgb_prefix_length": rgb_prefix,
        "evaluated_rgb_frame_range": [rgb_prefix, 96],
        "evaluated_frames": suffix_length,
        "horizons": horizons,
    }


def _aggregate_video_rows(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scalar_rows = {
        source: {key: float(row[key]) for key in METRIC_KEYS_FORWARD}
        for source, row in rows.items()
    }
    horizon_aggregate = {
        horizon: {
            key: {
                "mean": float(np.mean([row["horizons"][horizon][key] for row in rows.values()])),
                "median": float(np.median([row["horizons"][horizon][key] for row in rows.values()])),
                "std": float(np.std([row["horizons"][horizon][key] for row in rows.values()])),
            }
            for key in METRIC_KEYS_FORWARD
        }
        for horizon in HORIZON_NAMES
    }
    return {
        "n": len(rows),
        "aggregate": _aggregate(scalar_rows, METRIC_KEYS_FORWARD),
        "horizon_aggregate": horizon_aggregate,
        "per_sequence": rows,
    }


def _validate_grid(
    records: list[dict[str, Any]],
    mode: str,
    expected_sources: set[str],
    expected_prefixes: tuple[int, ...],
) -> None:
    observed: dict[str, set[int]] = {}
    pairs: set[tuple[str, int]] = set()
    for record in records:
        source = _source_name(record)
        prefix = _rgb_prefix_length(record)
        pair = (source, prefix)
        if pair in pairs:
            raise ValueError(f"{mode}: duplicate source/prefix pair {pair}")
        pairs.add(pair)
        observed.setdefault(source, set()).add(prefix)
    if set(observed) != expected_sources:
        raise ValueError(
            f"{mode}: source set differs from inverse records: "
            f"missing={sorted(expected_sources - set(observed))}, extra={sorted(set(observed) - expected_sources)}"
        )
    expected_set = set(expected_prefixes)
    invalid = {source: sorted(prefixes) for source, prefixes in observed.items() if prefixes != expected_set}
    if invalid:
        raise ValueError(f"{mode}: incomplete prefix grid: {invalid}")
    expected_count = len(expected_sources) * len(expected_prefixes)
    if len(records) != expected_count:
        raise ValueError(f"{mode}: expected {expected_count} records, got {len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--prefix-lengths", default="1,9,17,33,49")
    parser.add_argument("--expected-sources", type=int, default=0)
    parser.add_argument("--lpips-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips-batch-size", type=int, default=16)
    args = parser.parse_args()
    expected_prefixes = _parse_prefixes(args.prefix_lengths)
    output_root = args.out or args.inference_root / "metrics"

    inverse_records = _read_jsonl(args.eval_root / INPUT_FILES["inverse_dynamics"], "inverse_dynamics")
    if args.expected_sources and len(inverse_records) != args.expected_sources:
        raise ValueError(f"expected {args.expected_sources} inverse sources, got {len(inverse_records)}")
    expected_sources = {_source_name(record) for record in inverse_records}
    if len(expected_sources) != len(inverse_records):
        raise ValueError("inverse records contain duplicate source_name values")

    inverse_rows: dict[str, dict[str, float]] = {}
    for index, record in enumerate(inverse_records):
        source = _source_name(record)
        sample_dir, payload = _load_successful_output(args.inference_root, record)
        action = _action_from_output(payload, sample_dir)
        gt_camera = args.eval_root / "samples" / source / "gt_camera_cosmos.npz"
        inverse_rows[source] = eval_seq(action, gt_abs(gt_camera))
        print(f"[prefix-eval] inverse {index + 1}/{len(inverse_records)}: {source}", flush=True)

    inverse_payload = {
        "n": len(inverse_rows),
        "aggregate": _aggregate(inverse_rows, METRIC_KEYS_INVERSE),
        "per_sequence": inverse_rows,
    }
    _write_json(output_root / "inverse_camera_metrics.json", inverse_payload)

    lpips_metric = LPIPSAlex(args.lpips_device, args.lpips_batch_size)
    video_payload: dict[str, dict[str, Any]] = {}
    policy_camera_payload: dict[str, dict[str, Any]] = {}
    for mode in VISUAL_MODES:
        records = _read_jsonl(args.eval_root / INPUT_FILES[mode], mode)
        _validate_grid(records, mode, expected_sources, expected_prefixes)
        rows_by_prefix: dict[int, dict[str, dict[str, Any]]] = {
            prefix: {} for prefix in expected_prefixes
        }
        policy_full_by_prefix: dict[int, dict[str, dict[str, float]]] = {
            prefix: {} for prefix in expected_prefixes
        }
        policy_suffix_by_prefix: dict[int, dict[str, dict[str, float]]] = {
            prefix: {} for prefix in expected_prefixes
        }

        for index, record in enumerate(records):
            source = _source_name(record)
            prefix = _rgb_prefix_length(record)
            sample_dir, payload = _load_successful_output(args.inference_root, record)
            gt_video_path = args.eval_root / "samples" / source / "gt_clip.mp4"
            gt_frames = _read_video_rgb(gt_video_path)
            generated_frames = _read_video_rgb(sample_dir / "vision.mp4")
            if len(gt_frames) != 97 or len(generated_frames) != 97:
                raise ValueError(
                    f"{record['name']}: expected 97 frames, got GT={len(gt_frames)}, "
                    f"generated={len(generated_frames)}"
                )
            gt_frames = _resize_gt_like_native(
                gt_frames,
                generated_frames.shape[1],
                generated_frames.shape[2],
            )
            metrics = _frame_metrics(gt_frames[prefix:], generated_frames[prefix:], lpips_metric)
            rows_by_prefix[prefix][source] = _relative_horizon_summary(metrics, prefix)

            if mode == "policy":
                action = _action_from_output(payload, sample_dir)
                gt_camera_path = args.eval_root / "samples" / source / "gt_camera_cosmos.npz"
                gt_poses = gt_abs(gt_camera_path)
                policy_full_by_prefix[prefix][source] = eval_seq(action, gt_poses)
                action_start = prefix - 1
                policy_suffix_by_prefix[prefix][source] = eval_seq(
                    action[action_start:],
                    gt_poses[action_start:],
                )

            print(
                f"[prefix-eval] {mode} {index + 1}/{len(records)}: {source} prefix={prefix}",
                flush=True,
            )

        video_payload[mode] = {
            str(prefix): {
                "rgb_prefix_length": prefix,
                "generated_suffix_range": [prefix, 96],
                **_aggregate_video_rows(rows_by_prefix[prefix]),
            }
            for prefix in expected_prefixes
        }
        if mode == "policy":
            policy_camera_payload = {
                str(prefix): {
                    "rgb_prefix_length": prefix,
                    "full_trajectory": {
                        "n": len(policy_full_by_prefix[prefix]),
                        "aggregate": _aggregate(policy_full_by_prefix[prefix], METRIC_KEYS_INVERSE),
                        "per_sequence": policy_full_by_prefix[prefix],
                    },
                    "generated_suffix_reanchored": {
                        "action_index_range": [prefix - 1, 95],
                        "n": len(policy_suffix_by_prefix[prefix]),
                        "aggregate": _aggregate(policy_suffix_by_prefix[prefix], METRIC_KEYS_INVERSE),
                        "per_sequence": policy_suffix_by_prefix[prefix],
                    },
                }
                for prefix in expected_prefixes
            }

    _write_json(output_root / "video_prefix_metrics.json", video_payload)
    _write_json(output_root / "policy_camera_prefix_metrics.json", policy_camera_payload)
    _write_json(
        output_root / "METRICS_COMPLETE.json",
        {
            "sources": len(expected_sources),
            "prefix_lengths": list(expected_prefixes),
            "video_modes": list(VISUAL_MODES),
            "video_metrics": list(METRIC_KEYS_FORWARD),
            "camera_metrics": list(METRIC_KEYS_INVERSE),
            "suffix_only_video_metrics": True,
            "relative_horizons": list(HORIZON_NAMES),
        },
    )
    print(f"[prefix-eval] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
