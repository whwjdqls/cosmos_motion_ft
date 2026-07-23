#!/usr/bin/env python
"""Strip local evaluation metadata from official Cosmos inference JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INPUT_FILES = {
    "forward_dynamics": "fd_input.jsonl",
    "inverse_dynamics": "invdyn_input.jsonl",
    "policy": "policy_input.jsonl",
    "image2video": "i2v_input.jsonl",
}
LOCAL_METADATA_FIELDS = frozenset({"rgb_prefix_length", "latent_prefix_length", "source_name"})
VISUAL_MODES = frozenset({"forward_dynamics", "policy", "image2video"})


def sanitize_record(record: dict[str, Any], expected_mode: str) -> dict[str, Any]:
    """Return an official-schema record while validating local metric metadata."""
    if record.get("model_mode") != expected_mode:
        raise ValueError(
            f"expected model_mode={expected_mode!r}, got {record.get('model_mode')!r}"
        )
    sample_name = record.get("name")
    legacy_suffix = f"_{expected_mode}"
    source_name = record.get("source_name")
    if not isinstance(source_name, str) or not source_name:
        local_fields = LOCAL_METADATA_FIELDS & record.keys()
        if local_fields:
            raise ValueError(
                f"{expected_mode}: incomplete local metadata; missing source_name "
                f"with fields={sorted(local_fields)}"
            )
        if not isinstance(sample_name, str) or not sample_name.endswith(legacy_suffix):
            raise ValueError(
                f"{expected_mode}: legacy sample name {sample_name!r} must end with "
                f"{legacy_suffix!r}"
            )
        # Fixed-prefix fixtures created before local metric metadata already
        # match the official inference schema and condition on RGB frame 0.
        return dict(record)
    if not isinstance(sample_name, str) or not sample_name.startswith(f"{source_name}_"):
        raise ValueError(
            f"{expected_mode}: sample name {sample_name!r} is inconsistent with "
            f"source_name {source_name!r}"
        )
    if expected_mode in VISUAL_MODES:
        for key in ("rgb_prefix_length", "latent_prefix_length"):
            if not isinstance(record.get(key), int):
                raise ValueError(f"{expected_mode}: missing integer {key}")
        indexes = record.get("condition_frame_indexes_vision")
        if indexes != list(range(int(record["latent_prefix_length"]))):
            raise ValueError(
                f"{expected_mode}: condition indexes do not match latent prefix "
                f"{record['latent_prefix_length']}"
            )
        expected_rgb_prefix = 1 + 4 * (int(record["latent_prefix_length"]) - 1)
        if record["rgb_prefix_length"] != expected_rgb_prefix:
            raise ValueError(
                f"{expected_mode}: RGB prefix {record['rgb_prefix_length']} does not "
                f"match latent prefix {record['latent_prefix_length']}"
            )
    return {key: value for key, value in record.items() if key not in LOCAL_METADATA_FIELDS}


def sanitize_input_directory(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write all four schema-clean JSONLs and return per-mode record counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for mode, filename in INPUT_FILES.items():
        source = input_dir / filename
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty evaluation input: {source}")
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        sanitized = [sanitize_record(record, mode) for record in records]
        names = [record.get("name") for record in sanitized]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{source}: every record must have a non-empty name")
        if len(names) != len(set(names)):
            raise ValueError(f"{source}: duplicate sample names")
        destination = output_dir / filename
        destination.write_text("".join(json.dumps(record) + "\n" for record in sanitized))
        counts[mode] = len(sanitized)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    counts = sanitize_input_directory(args.input_dir, args.output_dir)
    print(f"[nativeviz] sanitized official inference inputs: {counts}", flush=True)


if __name__ == "__main__":
    main()
