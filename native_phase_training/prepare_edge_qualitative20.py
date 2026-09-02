#!/usr/bin/env python
"""Materialize the frozen Edge qualitative-20 inference cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from native_phase_training.nymeria_i2v_prompt import (
    DEFAULT_NEGATIVE_TEMPLATE_PATH,
    DEFAULT_TEMPLATE_PATH,
    normalize_native_structured_prompt,
)
from native_phase_training.sanitize_prefix_inference_inputs import sanitize_input_directory
from runtime_paths import resolve_legacy_path


INPUT_SPECS = {
    "fd_input.jsonl": ("forward_dynamics", 10.0, 30, 1.0),
    "invdyn_input.jsonl": ("inverse_dynamics", 10.0, 30, 1.0),
    "policy_input.jsonl": ("policy", 10.0, 30, 1.0),
    "i2v_input.jsonl": ("image2video", 12.0, 20, 6.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: expected non-empty JSON-object records")
    return records


def _base_name(record: dict[str, Any], mode: str, path: Path) -> str:
    name = record.get("name")
    suffix = f"_{mode}"
    if not isinstance(name, str) or not name.endswith(suffix):
        raise ValueError(f"{path}: invalid {mode} sample name {name!r}")
    return name[: -len(suffix)]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))


def _resolve_record_paths(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    for key, value in tuple(result.items()):
        if key.endswith("_path") and isinstance(value, str):
            resolved = Path(resolve_legacy_path(value))
            if not resolved.is_file():
                raise FileNotFoundError(f"resolved {key} does not exist: {resolved}")
            result[key] = str(resolved.resolve())
    return result


def _ensure_samples_link(source_root: Path, output_dir: Path) -> None:
    source = (source_root / "samples").resolve()
    destination = output_dir / "samples"
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise ValueError(f"{destination} points to {destination.resolve()}, expected {source}")
        return
    os.symlink(source, destination, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort_path = args.cohort.resolve()
    cohort = json.loads(cohort_path.read_text())
    if not isinstance(cohort, dict) or cohort.get("schema_version") != 1:
        raise ValueError(f"{cohort_path}: unsupported cohort schema")
    sample_names = cohort.get("samples")
    if (
        not isinstance(sample_names, list)
        or len(sample_names) != 20
        or len(set(sample_names)) != 20
        or not all(isinstance(name, str) and name for name in sample_names)
    ):
        raise ValueError(f"{cohort_path}: expected exactly 20 unique sample names")

    source_root = Path(str(cohort["source_root"])).resolve()
    source_hashes = cohort.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError(f"{cohort_path}: source_sha256 must be an object")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = output_dir / "canonical_inputs"
    sanitized_dir = output_dir / "inference_inputs"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    _ensure_samples_link(source_root, canonical_dir)

    file_manifest: dict[str, Any] = {}
    for filename, (mode, shift, steps, guidance) in INPUT_SPECS.items():
        source_path = source_root / filename
        expected_hash = source_hashes.get(filename)
        actual_hash = _sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{source_path}: SHA-256 changed; expected {expected_hash}, got {actual_hash}"
            )
        source_records = _read_jsonl(source_path)
        observed_first20 = [_base_name(record, mode, source_path) for record in source_records[:20]]
        if observed_first20 != sample_names:
            raise ValueError(f"{source_path}: first-20 order differs from frozen cohort")

        records: list[dict[str, Any]] = []
        for source_record in source_records[:20]:
            record = _resolve_record_paths(source_record)
            record.update(
                {
                    "num_frames": 97,
                    "fps": 20,
                    "resolution": "256",
                    "aspect_ratio": "1,1",
                    "seed": 0,
                    "shift": shift,
                    "num_steps": steps,
                    "guidance": guidance,
                }
            )
            if mode != "image2video":
                record.update(
                    {
                        "domain_name": "camera_pose",
                        "view_point": "ego_view",
                        "image_size": 256,
                        "action_chunk_size": 96,
                    }
                )
            records.append(record)

        destination = canonical_dir / filename
        _write_jsonl(destination, records)
        file_manifest[filename] = {
            "mode": mode,
            "count": len(records),
            "source": str(source_path),
            "source_sha256": actual_hash,
            "canonical_sha256": _sha256(destination),
            "sampler": {"shift": shift, "num_steps": steps, "guidance": guidance},
        }

    counts = sanitize_input_directory(
        canonical_dir,
        sanitized_dir,
        model_family="edge",
        replace_standalone_c=True,
    )
    if counts != {mode: 20 for mode, _shift, _steps, _guidance in INPUT_SPECS.values()}:
        raise ValueError(f"unexpected sanitized counts: {counts}")

    # The native inference path parses structured positive prompts, refreshes
    # media metadata, then serializes with json.dumps defaults immediately
    # before tokenization. Store that exact representation in the common JSONL
    # so Diffusers receives the identical text string.
    i2v_path = sanitized_dir / "i2v_input.jsonl"
    i2v_records = _read_jsonl(i2v_path)
    for record in i2v_records:
        record["prompt"] = normalize_native_structured_prompt(
            record["prompt"],
            num_frames=int(record["num_frames"]),
            fps=float(record["fps"]),
            height=int(record["resolution"]),
            width=int(record["resolution"]),
            aspect_ratio=str(record["aspect_ratio"]),
        )
        record["negative_metadata_mode"] = "none"
        record["negative_prompt_keep_metadata"] = False
    _write_jsonl(i2v_path, i2v_records)

    for filename in INPUT_SPECS:
        file_manifest[filename]["inference_sha256"] = _sha256(sanitized_dir / filename)

    output_manifest = {
        "schema_version": 1,
        "kind": "cosmos3_edge_frozen_qualitative_cohort",
        "name": cohort["name"],
        "cohort": str(cohort_path),
        "cohort_sha256": _sha256(cohort_path),
        "source_root": str(source_root),
        "samples": sample_names,
        "count": len(sample_names),
        "media": {"resolution": [256, 256], "num_frames": 97, "fps": 20, "seed": 0},
        "action": {
            "domain_name": "camera_pose",
            "view_point": "ego_view",
            "image_size": 256,
            "action_chunk_size": 96,
            "raw_action_dim": 9,
            "runtime_policy_mode": "wam",
        },
        "i2v": {
            "backends": ["official_native_framework", "pinned_diffusers_0.40"],
            "positive_template": str(DEFAULT_TEMPLATE_PATH.resolve()),
            "positive_template_sha256": _sha256(DEFAULT_TEMPLATE_PATH),
            "negative_template": str(DEFAULT_NEGATIVE_TEMPLATE_PATH.resolve()),
            "negative_template_sha256": _sha256(DEFAULT_NEGATIVE_TEMPLATE_PATH),
            "native_prompt_upsampling": False,
            "shared_input_jsonl": True,
            "native_effective_positive_serialization": "json.dumps_defaults",
        },
        "files": file_manifest,
    }
    manifest_path = output_dir / "cohort_contract.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote frozen qualitative cohort to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
