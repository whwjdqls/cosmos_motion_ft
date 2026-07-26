#!/usr/bin/env python
"""Convert canonical Phase-1 inputs to a released-Nano resolution/shift tier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RELEASED_TIER_CONTRACTS = {
    "256": {
        "shift": 3.0,
        "action_image_size": 256,
        "vision_resolution": "256",
        "output_hw": 256,
    },
    "720": {
        "shift": 10.0,
        "action_image_size": 480,
        "vision_resolution": "480",
        "output_hw": 640,
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"empty source JSONL: {path}")
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    )


def _ensure_samples_link(source_dir: Path, output_dir: Path) -> None:
    source_samples = (source_dir / "samples").resolve()
    if not source_samples.is_dir():
        raise FileNotFoundError(f"source samples directory is missing: {source_samples}")
    output_samples = output_dir / "samples"
    if output_samples.exists() or output_samples.is_symlink():
        if output_samples.resolve() != source_samples:
            raise ValueError(
                f"existing samples path points elsewhere: {output_samples} -> "
                f"{output_samples.resolve()}"
            )
        return
    os.symlink(source_samples, output_samples, target_is_directory=True)


def convert_record(
    source: dict[str, Any],
    *,
    resolution_tier: str,
    shift: float,
) -> dict[str, Any]:
    """Apply the exact model-tier and per-sample output-bucket contract."""

    tier = RELEASED_TIER_CONTRACTS[resolution_tier]
    record = dict(source)
    record["shift"] = float(shift)
    record["num_frames"] = 97
    mode = record.get("model_mode")
    if mode == "image2video":
        # Generic vision inference uses VIDEO_RES_SIZE_INFO. The 480/1:1
        # output bucket is 640x640, while the loaded model remains tier 720.
        record["resolution"] = str(tier["vision_resolution"])
        record["aspect_ratio"] = "1,1"
        record.pop("image_size", None)
    else:
        # Native action inference ignores vision resolution/aspect ratio and
        # routes image_size through find_closest_target_size.
        if resolution_tier == "256":
            record["resolution"] = "256"
            record["aspect_ratio"] = "1,1"
        else:
            record.pop("resolution", None)
            record.pop("aspect_ratio", None)
        record["image_size"] = int(tier["action_image_size"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution-tier", choices=sorted(RELEASED_TIER_CONTRACTS), required=True)
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="explicit override; default is the released Nano tier mapping",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["fd_input.jsonl", "invdyn_input.jsonl", "policy_input.jsonl", "i2v_input.jsonl"],
    )
    args = parser.parse_args()

    tier = RELEASED_TIER_CONTRACTS[args.resolution_tier]
    shift = tier["shift"] if args.shift is None else args.shift
    if shift <= 0:
        raise ValueError("--shift must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_samples_link(args.source_dir, args.output_dir)

    manifest_files: dict[str, Any] = {}
    total = 0
    for filename in args.files:
        source_path = args.source_dir / filename
        records = _read_jsonl(source_path)
        converted = [
            convert_record(
                source,
                resolution_tier=args.resolution_tier,
                shift=float(shift),
            )
            for source in records
        ]

        output_path = args.output_dir / filename
        _write_jsonl(output_path, converted)
        manifest_files[filename] = {
            "source": str(source_path.resolve()),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "output": str(output_path.resolve()),
            "count": len(converted),
        }
        total += len(converted)

    manifest = {
        "schema_version": 1,
        "kind": "native_phase1_eval_resolution_tier",
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "resolution_tier": int(args.resolution_tier),
        "released_nano_shift": float(shift),
        "action_image_size": int(tier["action_image_size"]),
        "vision_resolution": str(tier["vision_resolution"]),
        "vision_aspect_ratio": "1,1",
        "expected_output_hw": [int(tier["output_hw"]), int(tier["output_hw"])],
        "num_frames": 97,
        "fps": 20,
        "files": manifest_files,
        "total_records": total,
        "samples_symlink": str((args.output_dir / "samples").resolve()),
    }
    (args.output_dir / "tier_contract.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[eval-tier] wrote {total} records for tier={args.resolution_tier} "
        f"shift={shift} output_hw={tier['output_hw']} under {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
