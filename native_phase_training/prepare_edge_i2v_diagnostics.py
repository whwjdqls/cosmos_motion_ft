#!/usr/bin/env python
"""Build controlled Cosmos3-Edge I2V diagnostic inputs from one eval sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


STANDALONE_C = re.compile(r"(?<!\w)C(?!\w)")
VARIANT_KEYS = (
    "current",
    "sampler",
    "refreshed",
    "guidance1",
    "camera_wearer",
    "egocentric_viewpoint",
    "egocentric_upsampled",
)


def _replace_standalone_c(
    caption: str,
    *,
    sentence_initial: str,
    otherwise: str,
) -> str:
    def replacement(match: re.Match[str]) -> str:
        prefix = match.string[: match.start()].rstrip()
        return sentence_initial if not prefix or prefix[-1] in ".!?" else otherwise

    return STANDALONE_C.sub(replacement, caption)


def _replace_standalone_c_with_person(caption: str) -> str:
    return _replace_standalone_c(
        caption,
        sentence_initial="The person",
        otherwise="the person",
    )


def _replace_standalone_c_with_camera_wearer(caption: str) -> str:
    return _replace_standalone_c(
        caption,
        sentence_initial="The person wearing the camera",
        otherwise="the person wearing the camera",
    )


def _make_egocentric_viewpoint_prompt(caption: str) -> str:
    action = _replace_standalone_c(
        caption,
        sentence_initial="The camera wearer",
        otherwise="the camera wearer",
    )
    return (
        "A first-person egocentric video from a head-mounted camera. "
        f"{action} "
        "The camera viewpoint moves with the camera wearer's head and body."
    )


def _read_one_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one JSONL record, found {len(rows)}")
    if not isinstance(rows[0], dict):
        raise ValueError(f"{path}: expected a JSON object")
    return rows[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(source: dict[str, Any], path: Path) -> None:
    expected = {
        "model_mode": "image2video",
        "num_frames": 97,
        "resolution": "256",
        "aspect_ratio": "1,1",
        "fps": 20,
        "seed": 0,
    }
    mismatches = {
        key: (source.get(key), value)
        for key, value in expected.items()
        if source.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: source contract mismatch: {mismatches}")
    vision_path = Path(str(source.get("vision_path", "")))
    if not vision_path.is_file():
        raise FileNotFoundError(f"{path}: conditioning image does not exist: {vision_path}")
    if not str(source.get("prompt", "")).strip():
        raise ValueError(f"{path}: prompt must be nonempty")


def _common_source_fields(source: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "num_frames",
        "resolution",
        "aspect_ratio",
        "fps",
        "seed",
        "model_mode",
        "vision_path",
    )
    common = {key: source[key] for key in fields}
    common["prompt"] = _replace_standalone_c_with_person(str(source["prompt"]))
    return common


def build_records(source: dict[str, Any], refreshed_negative: dict[str, Any]) -> list[dict[str, Any]]:
    common = _common_source_fields(source)
    camera_wearer_common = {
        **common,
        "prompt": _replace_standalone_c_with_camera_wearer(str(source["prompt"])),
    }
    egocentric_viewpoint_common = {
        **common,
        "prompt": _make_egocentric_viewpoint_prompt(str(source["prompt"])),
    }
    base_name = str(source["name"]).removesuffix("_image2video")

    # The CLI-wide --no-diffusion-cache switch is part of the matching Slurm
    # launcher. These rows deliberately vary only the fields named below.
    current_no_cache = {
        **common,
        "name": f"{base_name}_i2v_current_s10_n35_g6_nocache",
        "shift": 10.0,
        "num_steps": 35,
        "guidance": 6.0,
    }
    refreshed_sampler = {
        **common,
        "name": f"{base_name}_i2v_sampler_s12_n20_g6_nocache",
        "shift": 12.0,
        "num_steps": 20,
        "guidance": 6.0,
    }
    refreshed_contract = {
        **common,
        "name": f"{base_name}_i2v_refreshed_s12_n20_g6_nocache",
        "shift": 12.0,
        "num_steps": 20,
        "guidance": 6.0,
        "negative_prompt": json.dumps(refreshed_negative, separators=(",", ":")),
        "negative_metadata_mode": "none",
        "negative_prompt_keep_metadata": False,
        "prompt_upsampling": True,
        "prompt_upsampler_max_tokens": 4096,
        "prompt_upsampler_temperature": 0.7,
        "prompt_upsampler_top_p": 0.8,
        "prompt_upsampler_top_k": 20,
        "prompt_upsampler_repetition_penalty": 1.0,
        "prompt_upsampler_presence_penalty": 1.5,
        "prompt_upsampler_seed": 3407,
    }
    guidance_one = {
        **common,
        "name": f"{base_name}_i2v_guidance1_s10_n35_nocache",
        "shift": 10.0,
        "num_steps": 35,
        "guidance": 1.0,
    }
    camera_wearer = {
        **camera_wearer_common,
        "name": f"{base_name}_i2v_camera_wearer_s10_n35_g6_nocache",
        "shift": 10.0,
        "num_steps": 35,
        "guidance": 6.0,
    }
    egocentric_viewpoint = {
        **egocentric_viewpoint_common,
        "name": f"{base_name}_i2v_egocentric_viewpoint_s10_n35_g6_nocache",
        "shift": 10.0,
        "num_steps": 35,
        "guidance": 6.0,
    }
    egocentric_upsampled = {
        **egocentric_viewpoint_common,
        "name": f"{base_name}_i2v_egocentric_upsampled_s12_n20_g6_nocache",
        "shift": 12.0,
        "num_steps": 20,
        "guidance": 6.0,
        "negative_prompt": json.dumps(refreshed_negative, separators=(",", ":")),
        "negative_metadata_mode": "none",
        "negative_prompt_keep_metadata": False,
        "prompt_upsampling": True,
        "prompt_upsampler_max_tokens": 4096,
        "prompt_upsampler_temperature": 0.7,
        "prompt_upsampler_top_p": 0.8,
        "prompt_upsampler_top_k": 20,
        "prompt_upsampler_repetition_penalty": 1.0,
        "prompt_upsampler_presence_penalty": 1.5,
        "prompt_upsampler_seed": 3407,
    }
    return [
        current_no_cache,
        refreshed_sampler,
        refreshed_contract,
        guidance_one,
        camera_wearer,
        egocentric_viewpoint,
        egocentric_upsampled,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--negative-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_KEYS,
        default=list(VARIANT_KEYS),
        help="Subset of controlled variants to emit, in the requested order.",
    )
    args = parser.parse_args()

    source = _read_one_jsonl(args.source_input)
    _validate_source(source, args.source_input)
    refreshed_negative = json.loads(args.negative_prompt.read_text())
    if not isinstance(refreshed_negative, dict):
        raise ValueError(f"{args.negative_prompt}: expected a JSON object")

    all_records = dict(zip(VARIANT_KEYS, build_records(source, refreshed_negative), strict=True))
    if len(set(args.variants)) != len(args.variants):
        raise ValueError(f"duplicate diagnostic variants requested: {args.variants}")
    records = [all_records[key] for key in args.variants]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records))

    manifest = {
        "source_input": str(args.source_input.resolve()),
        "source_input_sha256": _sha256(args.source_input),
        "negative_prompt": str(args.negative_prompt.resolve()),
        "negative_prompt_sha256": _sha256(args.negative_prompt),
        "global_diffusion_cache": False,
        "fixed_contract": {
            "checkpoint": "raw downloaded Cosmos3-Edge",
            "resolution": "256",
            "aspect_ratio": "1,1",
            "num_frames": 97,
            "fps": 20,
            "seed": 0,
            "conditioning_image": source["vision_path"],
            "prompt": records[0]["prompt"],
        },
        "variants": [
            {
                "name": row["name"],
                "shift": row["shift"],
                "num_steps": row["num_steps"],
                "guidance": row["guidance"],
                "native_prompt_upsampling": bool(row.get("prompt_upsampling", False)),
                "negative_prompt_contract": "refreshed_snapshot"
                if "negative_prompt" in row
                else "framework_default",
            }
            for row in records
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(records)} controlled I2V records to {args.output}")


if __name__ == "__main__":
    main()
