"""Prepare sharded 720-tier forward-dynamics inputs for all held-out sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=71)
    args = parser.parse_args()

    if args.shards <= 0:
        raise ValueError("--shards must be positive")
    records = _read_jsonl(args.source)
    if len(records) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} records, found {len(records)} in {args.source}")
    if len({record["name"] for record in records}) != len(records):
        raise ValueError(f"duplicate sample names in {args.source}")

    transformed: list[dict[str, Any]] = []
    for source in records:
        if source.get("model_mode") != "forward_dynamics":
            raise ValueError(f"expected forward_dynamics record, got {source.get('model_mode')!r}")
        if source.get("action_chunk_size") != 96:
            raise ValueError(f"expected action_chunk_size=96 in {source.get('name')!r}")
        if not Path(source["vision_path"]).is_file() or not Path(source["action_path"]).is_file():
            raise FileNotFoundError(f"missing condition input for {source.get('name')!r}")
        record = dict(source)
        for key in (
            "num_frames",
            "resolution",
            "aspect_ratio",
            "source_name",
            "rgb_prefix_length",
            "latent_prefix_length",
        ):
            record.pop(key, None)
        record.update(
            {
                "image_size": 480,
                "shift": 10.0,
                "num_steps": 30,
                "guidance": 1.0,
                "seed": 0,
            }
        )
        transformed.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_size = math.ceil(len(transformed) / args.shards)
    shard_manifest: list[dict[str, Any]] = []
    for index in range(args.shards):
        shard_records = transformed[index * shard_size : (index + 1) * shard_size]
        if not shard_records:
            raise ValueError(f"empty shard {index}; reduce --shards")
        path = args.output_dir / f"fd_shard_{index:02d}.jsonl"
        _write_jsonl(path, shard_records)
        shard_manifest.append(
            {
                "index": index,
                "path": str(path),
                "count": len(shard_records),
                "first_name": shard_records[0]["name"],
                "last_name": shard_records[-1]["name"],
            }
        )

    manifest = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "sample_count": len(transformed),
        "resolution_tier": 720,
        "expected_output_size": [640, 640],
        "shift": 10.0,
        "num_steps": 30,
        "guidance": 1.0,
        "seed": 0,
        "shards": shard_manifest,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[full71-720] wrote {len(transformed)} records across "
        f"{len(shard_manifest)} shards to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
