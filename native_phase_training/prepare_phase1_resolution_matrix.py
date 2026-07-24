"""Prepare matched forward-dynamics inputs for the Phase-1 resolution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CELLS = {
    "r256_s3": {"shift": 3.0, "explicit_256": True},
    "r720_s3": {"shift": 3.0, "explicit_256": False},
    "r720_s10": {"shift": 10.0, "explicit_256": False},
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"input JSONL is empty: {path}")
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    source_records = _read_jsonl(args.source)
    if len(source_records) < args.count:
        raise ValueError(f"requested {args.count} records but {args.source} contains {len(source_records)}")
    source_records = source_records[: args.count]

    for record in source_records:
        if record.get("model_mode") != "forward_dynamics":
            raise ValueError(f"expected forward_dynamics record, got {record.get('model_mode')!r}")
        if record.get("action_chunk_size") != 96:
            raise ValueError(f"expected action_chunk_size=96 in {record.get('name')!r}")
        if not Path(record["vision_path"]).is_file() or not Path(record["action_path"]).is_file():
            raise FileNotFoundError(f"missing conditioning input for {record.get('name')!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for cell, settings in CELLS.items():
        cell_records: list[dict[str, Any]] = []
        for source in source_records:
            record = dict(source)
            record["name"] = f"{source['name']}__{cell}"
            record["shift"] = settings["shift"]
            if settings["explicit_256"]:
                record.update(
                    {
                        "num_frames": 97,
                        "resolution": "256",
                        "aspect_ratio": "1,1",
                        "image_size": 256,
                    }
                )
            else:
                # Preserve the historical high-tier contract. The source JSONL
                # omits these fields and inherits the model's 720-tier defaults.
                for key in ("num_frames", "resolution", "aspect_ratio"):
                    record.pop(key, None)
                record["image_size"] = 480
            cell_records.append(record)
        output = args.output_dir / f"fd_{cell}.jsonl"
        _write_jsonl(output, cell_records)
        outputs[cell] = str(output)

    manifest = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "sample_count": args.count,
        "model_mode": "forward_dynamics",
        "cells": {
            "r256_s3": {"resolution_tier": 256, "shift": 3.0, "expected_output_size": [256, 256]},
            "r720_s3": {"resolution_tier": 720, "shift": 3.0, "expected_output_size": [640, 640]},
            "r720_s10": {"resolution_tier": 720, "shift": 10.0, "expected_output_size": [640, 640]},
        },
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[resolution-matrix] wrote {args.count} matched samples x {len(CELLS)} cells to {args.output_dir}")


if __name__ == "__main__":
    main()
