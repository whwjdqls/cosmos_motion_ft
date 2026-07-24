#!/usr/bin/env python
"""Validate and split an explicit evaluation-window JSON list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, default=97)
    args = parser.parse_args()

    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    rows = json.loads(args.input.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{args.input}: expected a nonempty JSON list")

    keys = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "uuid" not in row or "start" not in row:
            raise ValueError(f"{args.input}: malformed row {index}: {row!r}")
        frames = int(row.get("num_frames", args.expected_frames))
        if frames != args.expected_frames:
            raise ValueError(
                f"{args.input}: row {index} has num_frames={frames}, "
                f"expected {args.expected_frames}"
            )
        keys.append((str(row["uuid"]), int(row["start"])))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{args.input}: duplicate (uuid, start) windows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_id in range(args.num_shards):
        shard_rows = rows[shard_id :: args.num_shards]
        path = args.out_dir / f"shard_{shard_id}.json"
        path.write_text(json.dumps(shard_rows, indent=2) + "\n")
        shards.append({"shard_id": shard_id, "path": str(path), "n": len(shard_rows)})

    summary = {
        "source": str(args.input),
        "n": len(rows),
        "num_shards": args.num_shards,
        "expected_frames": args.expected_frames,
        "shards": shards,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[shard-windows] wrote {len(rows)} windows across {args.num_shards} "
        f"shards -> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
