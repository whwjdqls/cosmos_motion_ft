#!/usr/bin/env python
"""Inventory reusable Phase-1 outputs and prepare only missing resolution pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from native_phase_training.sanitize_prefix_inference_inputs import sanitize_record
except ModuleNotFoundError:
    from sanitize_prefix_inference_inputs import sanitize_record


CORE_256_FIELDS = (
    "num_frames",
    "resolution",
    "aspect_ratio",
    "fps",
    "shift",
    "seed",
    "domain_name",
    "view_point",
    "action_chunk_size",
    "image_size",
    "num_steps",
    "guidance",
    "model_mode",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta(record: dict[str, Any]) -> tuple[tuple[str, int], dict[str, Any]]:
    meta_path = Path(record["vision_path"]).parent / "meta.json"
    value = json.loads(meta_path.read_text())
    key = (str(value["uuid"]), int(value["start_frame"]))
    return key, value


def _index(records: list[dict[str, Any]], label: str) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for record in records:
        if record.get("model_mode") != "forward_dynamics":
            raise ValueError(f"{label}: expected forward_dynamics, got {record.get('model_mode')}")
        key, _value = _meta(record)
        if key in result:
            raise ValueError(f"{label}: duplicate window {key}")
        result[key] = record
    return result


def _successful_video(root: Path, name: str) -> Path:
    sample_dir = root / name
    payload = json.loads((sample_dir / "sample_outputs.json").read_text())
    if payload.get("status") != "success":
        raise RuntimeError(f"unsuccessful inference output: {sample_dir}")
    video = sample_dir / "vision.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    return video


def _high_record(source: dict[str, Any]) -> dict[str, Any]:
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
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-input-root", type=Path, required=True)
    parser.add_argument("--canonical-input-root", type=Path, required=True)
    parser.add_argument("--canonical-256-output-root", type=Path, required=True)
    parser.add_argument("--canonical-720-output-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=71)
    args = parser.parse_args()

    exact_path = args.exact_input_root / "fd_input.jsonl"
    canonical_path = args.canonical_input_root / "fd_input.jsonl"
    exact_rows = _read_jsonl(exact_path)
    canonical_rows = _read_jsonl(canonical_path)
    if len(exact_rows) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} exact rows, found {len(exact_rows)}"
        )
    exact = _index(exact_rows, "exact")
    canonical = _index(canonical_rows, "canonical")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing_256 = []
    missing_720 = []
    inventory = []
    for key, exact_record in exact.items():
        canonical_record = canonical.get(key)
        reusable = canonical_record is not None
        mismatch: list[str] = []
        exact_vision = Path(exact_record["vision_path"])
        exact_action = Path(exact_record["action_path"])
        condition = {
            "first_frame_sha256": _sha256(exact_vision),
            "camera_action_sha256": _sha256(exact_action),
            "prompt_sha256": hashlib.sha256(exact_record["prompt"].encode()).hexdigest(),
        }
        existing_256 = None
        existing_720 = None
        if canonical_record is not None:
            canonical_vision = Path(canonical_record["vision_path"])
            canonical_action = Path(canonical_record["action_path"])
            if condition["first_frame_sha256"] != _sha256(canonical_vision):
                mismatch.append("first_frame")
            if condition["camera_action_sha256"] != _sha256(canonical_action):
                mismatch.append("camera_action")
            if exact_record["prompt"] != canonical_record["prompt"]:
                mismatch.append("prompt")
            for field in CORE_256_FIELDS:
                if exact_record.get(field) != canonical_record.get(field):
                    mismatch.append(field)
            reusable = not mismatch
            if reusable:
                existing_256 = _successful_video(
                    args.canonical_256_output_root, canonical_record["name"]
                )
                existing_720 = _successful_video(
                    args.canonical_720_output_root, canonical_record["name"]
                )

        if not reusable:
            official_record = sanitize_record(exact_record, "forward_dynamics")
            missing_256.append(official_record)
            missing_720.append(_high_record(official_record))

        inventory.append(
            {
                "uuid": key[0],
                "start": key[1],
                "exact_name": exact_record["name"],
                "reused": reusable,
                "mismatch": mismatch,
                "condition": condition,
                "phase1_256": str(existing_256) if existing_256 else None,
                "phase1_720": str(existing_720) if existing_720 else None,
            }
        )

    missing_256_path = args.out_dir / "fd_missing_256_s3.jsonl"
    missing_720_path = args.out_dir / "fd_missing_720_s10.jsonl"
    _write_jsonl(missing_256_path, missing_256)
    _write_jsonl(missing_720_path, missing_720)
    manifest = {
        "schema_version": 1,
        "exact_input": str(exact_path),
        "exact_input_sha256": _sha256(exact_path),
        "canonical_input": str(canonical_path),
        "canonical_input_sha256": _sha256(canonical_path),
        "expected_count": args.expected_count,
        "reused": sum(row["reused"] for row in inventory),
        "missing": len(missing_256),
        "missing_256_jsonl": str(missing_256_path),
        "missing_720_jsonl": str(missing_720_path),
        "inventory": inventory,
    }
    (args.out_dir / "reuse_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[phase1-motion-clean-reuse] exact={len(inventory)} "
        f"reused={manifest['reused']} missing={manifest['missing']} -> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
