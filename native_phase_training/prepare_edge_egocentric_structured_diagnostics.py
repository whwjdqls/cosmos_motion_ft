#!/usr/bin/env python
"""Prepare frozen-JSON I2V and action-conditioned Edge diagnostics.

The generated records keep the media, seed, frame count, FPS, and checkpoint
contract fixed.  The I2V pair varies only the released sampler recipe.  The
forward-dynamics pair varies only whether the action subject is described as a
generic person or explicitly as the camera wearer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


STANDALONE_C = re.compile(r"(?<!\w)C(?!\w)")
EXPECTED_COMMON = {
    "num_frames": 97,
    "resolution": "256",
    "aspect_ratio": "1,1",
    "fps": 20,
    "seed": 0,
}


def _read_one_jsonl(path: Path, expected_mode: str) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"{path}: expected exactly one JSON-object record")
    record = records[0]
    mismatches = {
        key: (record.get(key), expected)
        for key, expected in {**EXPECTED_COMMON, "model_mode": expected_mode}.items()
        if record.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{path}: source contract mismatch: {mismatches}")
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_subject(caption: str, *, sentence_initial: str, otherwise: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        prefix = match.string[: match.start()].rstrip()
        return sentence_initial if not prefix or prefix[-1] in ".!?" else otherwise

    return STANDALONE_C.sub(replacement, caption)


def _load_frozen_prompt(path: Path) -> dict[str, Any]:
    prompt = json.loads(path.read_text())
    if not isinstance(prompt, dict):
        raise ValueError(f"{path}: structured prompt must be a JSON object")
    cinematography = prompt.get("cinematography")
    actions = prompt.get("actions")
    if not isinstance(cinematography, dict) or not str(cinematography.get("camera_motion", "")).strip():
        raise ValueError(f"{path}: structured prompt needs cinematography.camera_motion")
    if not isinstance(actions, list) or not actions or not all(isinstance(action, dict) for action in actions):
        raise ValueError(f"{path}: structured prompt needs at least one action object")

    # Match NVIDIA's JSON training/inference metadata representation.  Native
    # inference overwrites these fields from the request with the same values.
    prompt["duration"] = f"{int(EXPECTED_COMMON['num_frames'] / EXPECTED_COMMON['fps'])}s"
    prompt["fps"] = float(EXPECTED_COMMON["fps"])
    prompt["resolution"] = {"H": 256, "W": 256}
    prompt["aspect_ratio"] = EXPECTED_COMMON["aspect_ratio"]
    return prompt


def _base_name(record: dict[str, Any], suffix: str) -> str:
    name = str(record.get("name", ""))
    if not name.endswith(suffix):
        raise ValueError(f"sample name {name!r} must end with {suffix!r}")
    return name[: -len(suffix)]


def _build_i2v_records(
    source: dict[str, Any],
    structured_prompt: dict[str, Any],
    negative_prompt: dict[str, Any],
) -> list[dict[str, Any]]:
    base_name = _base_name(source, "_image2video")
    common = {
        key: source[key]
        for key in ("num_frames", "resolution", "aspect_ratio", "fps", "seed", "model_mode", "vision_path")
    }
    common.update(
        {
            "prompt": json.dumps(structured_prompt, separators=(",", ":")),
            "negative_prompt": json.dumps(negative_prompt, separators=(",", ":")),
            "negative_metadata_mode": "none",
            "negative_prompt_keep_metadata": False,
        }
    )
    return [
        {
            **common,
            "name": f"{base_name}_i2v_egojson_native_s10_n35_g6_nocache",
            "shift": 10.0,
            "num_steps": 35,
            "guidance": 6.0,
        },
        {
            **common,
            "name": f"{base_name}_i2v_egojson_refreshed_s12_n20_g6_nocache",
            "shift": 12.0,
            "num_steps": 20,
            "guidance": 6.0,
        },
    ]


def _build_forward_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    base_name = _base_name(source, "_forward_dynamics")
    common = {
        key: source[key]
        for key in (
            "num_frames",
            "resolution",
            "aspect_ratio",
            "fps",
            "seed",
            "domain_name",
            "view_point",
            "action_chunk_size",
            "image_size",
            "model_mode",
            "vision_path",
            "action_path",
        )
    }
    common.update({"shift": 10.0, "num_steps": 30, "guidance": 1.0})
    raw_caption = str(source["prompt"])
    return [
        {
            **common,
            "name": f"{base_name}_fd_person_s10_n30_g1_nocache",
            "prompt": _replace_subject(
                raw_caption,
                sentence_initial="The person",
                otherwise="the person",
            ),
        },
        {
            **common,
            "name": f"{base_name}_fd_camera_wearer_s10_n30_g1_nocache",
            "prompt": _replace_subject(
                raw_caption,
                sentence_initial="The camera wearer",
                otherwise="the camera wearer",
            ),
        },
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-i2v", type=Path, required=True)
    parser.add_argument("--source-forward", type=Path, required=True)
    parser.add_argument("--structured-prompt", type=Path, required=True)
    parser.add_argument("--negative-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_i2v = _read_one_jsonl(args.source_i2v, "image2video")
    source_forward = _read_one_jsonl(args.source_forward, "forward_dynamics")
    if source_i2v["vision_path"] != source_forward["vision_path"]:
        raise ValueError("I2V and forward controls must use the same conditioning frame")

    structured_prompt = _load_frozen_prompt(args.structured_prompt)
    negative_prompt = json.loads(args.negative_prompt.read_text())
    if not isinstance(negative_prompt, dict):
        raise ValueError(f"{args.negative_prompt}: negative prompt must be a JSON object")

    i2v_records = _build_i2v_records(source_i2v, structured_prompt, negative_prompt)
    forward_records = _build_forward_records(source_forward)
    _write_jsonl(args.output_dir / "i2v_structured.jsonl", i2v_records)
    _write_jsonl(args.output_dir / "forward_controls.jsonl", forward_records)

    vision_path = Path(source_i2v["vision_path"])
    gt_video = vision_path.with_name("gt_clip.mp4")
    if not gt_video.is_file():
        raise FileNotFoundError(f"missing GT video next to conditioning frame: {gt_video}")

    variants = []
    for record in [*i2v_records, *forward_records]:
        variants.append(
            {
                "name": record["name"],
                "model_mode": record["model_mode"],
                "shift": record["shift"],
                "num_steps": record["num_steps"],
                "guidance": record["guidance"],
                "prompt": record["prompt"],
            }
        )
    manifest = {
        "kind": "cosmos3_edge_egocentric_structured_diagnostics",
        "checkpoint": "raw downloaded Cosmos3-Edge",
        "global_diffusion_cache": False,
        "fixed_contract": {**EXPECTED_COMMON, "conditioning_image": source_i2v["vision_path"]},
        "gt_video": str(gt_video.resolve()),
        "camera_action": str(Path(source_forward["action_path"]).resolve()),
        "source_i2v": str(args.source_i2v.resolve()),
        "source_i2v_sha256": _sha256(args.source_i2v),
        "source_forward": str(args.source_forward.resolve()),
        "source_forward_sha256": _sha256(args.source_forward),
        "structured_prompt": str(args.structured_prompt.resolve()),
        "structured_prompt_sha256": _sha256(args.structured_prompt),
        "negative_prompt": str(args.negative_prompt.resolve()),
        "negative_prompt_sha256": _sha256(args.negative_prompt),
        "variants": variants,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(i2v_records)} I2V and {len(forward_records)} forward-dynamics records")


if __name__ == "__main__":
    main()
