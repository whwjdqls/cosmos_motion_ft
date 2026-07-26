#!/usr/bin/env python
"""Reject Phase-1 evaluation JSONL files that conflict with a checkpoint contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ACTION_MODES = frozenset({"forward_dynamics", "inverse_dynamics", "policy"})
SUPPORTED_MODES = ACTION_MODES | {"image2video"}
TIER_INPUT_CONTRACTS = {
    "256": {"action_image_size": 256, "vision_resolution": "256"},
    "720": {"action_image_size": 480, "vision_resolution": "480"},
}


def validate_record(
    record: dict,
    *,
    context: str,
    expected_shift: float,
    expected_resolution: str,
    expected_num_frames: int,
    expected_fps: int = 20,
) -> None:
    if expected_resolution not in TIER_INPUT_CONTRACTS:
        raise ValueError(f"unsupported model resolution contract: {expected_resolution!r}")
    tier = TIER_INPUT_CONTRACTS[expected_resolution]
    mode = record.get("model_mode")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"{context}: unsupported model_mode={mode!r}")

    shift = float(record.get("shift", float("nan")))
    if abs(shift - expected_shift) > 1e-9:
        raise ValueError(
            f"{context}: shift={shift} conflicts with checkpoint shift={expected_shift}"
        )
    if "num_frames" not in record:
        raise ValueError(
            f"{context}: num_frames is required; omitting it can inherit a "
            "different released-tier temporal default"
        )
    if int(record["num_frames"]) != expected_num_frames:
        raise ValueError(
            f"{context}: num_frames={record['num_frames']} conflicts with "
            f"checkpoint T={expected_num_frames}"
        )
    if int(record.get("fps", -1)) != expected_fps:
        raise ValueError(
            f"{context}: fps={record.get('fps')!r} conflicts with expected {expected_fps}"
        )

    if mode in ACTION_MODES:
        if int(record.get("image_size", -1)) != tier["action_image_size"]:
            raise ValueError(
                f"{context}: action image_size={record.get('image_size')!r} conflicts "
                f"with tier {expected_resolution} value {tier['action_image_size']}"
            )
        if int(record.get("action_chunk_size", -1)) != expected_num_frames - 1:
            raise ValueError(
                f"{context}: action_chunk_size={record.get('action_chunk_size')!r} "
                f"!= {expected_num_frames - 1}"
            )
        if expected_resolution == "256":
            if record.get("resolution") != "256" or record.get("aspect_ratio") != "1,1":
                raise ValueError(
                    f"{context}: 256-tier action input must retain "
                    "resolution='256', aspect_ratio='1,1'"
                )
        elif "resolution" in record or "aspect_ratio" in record:
            raise ValueError(
                f"{context}: high-tier action inputs must use image_size, not "
                "generic resolution/aspect_ratio fields"
            )
        return

    if str(record.get("resolution")) != tier["vision_resolution"]:
        raise ValueError(
            f"{context}: I2V resolution={record.get('resolution')!r} conflicts with "
            f"tier {expected_resolution} output bucket {tier['vision_resolution']!r}"
        )
    if record.get("aspect_ratio") != "1,1":
        raise ValueError(
            f"{context}: I2V aspect_ratio={record.get('aspect_ratio')!r} != '1,1'"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-shift", type=float, required=True)
    parser.add_argument("--expected-resolution", required=True)
    parser.add_argument("--expected-num-frames", type=int, required=True)
    parser.add_argument("--expected-fps", type=int, default=20)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    total = 0
    for filename in args.files:
        path = args.input_dir / filename
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"empty evaluation input: {path}")
        for index, record in enumerate(records):
            context = f"{path}:{index + 1}"
            validate_record(
                record,
                context=context,
                expected_shift=args.expected_shift,
                expected_resolution=args.expected_resolution,
                expected_num_frames=args.expected_num_frames,
                expected_fps=args.expected_fps,
            )
        total += len(records)
    print(
        f"[eval-contract] validated {total} records across {len(args.files)} files: "
        f"resolution={args.expected_resolution} shift={args.expected_shift} "
        f"T={args.expected_num_frames}",
        flush=True,
    )


if __name__ == "__main__":
    main()
