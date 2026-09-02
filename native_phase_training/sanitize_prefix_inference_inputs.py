#!/usr/bin/env python
"""Strip local evaluation metadata from official Cosmos inference JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from native_phase_training.nymeria_i2v_prompt import (
    build_nymeria_i2v_negative_prompt,
    build_nymeria_i2v_prompt,
    compact_prompt,
    parse_json_object,
)


INPUT_FILES = {
    "forward_dynamics": "fd_input.jsonl",
    "inverse_dynamics": "invdyn_input.jsonl",
    "policy": "policy_input.jsonl",
    "image2video": "i2v_input.jsonl",
}
LOCAL_METADATA_FIELDS = frozenset({"rgb_prefix_length", "latent_prefix_length", "source_name"})
VISUAL_MODES = frozenset({"forward_dynamics", "policy", "image2video"})
# Nymeria uses uppercase ``C`` as the camera-wearer label. Lowercase ``c`` and
# embedded identifiers are deliberately left untouched.
_STANDALONE_C_PATTERN = re.compile(r"(?<!\w)C(?!\w)")


def _replace_standalone_c_with_camera_wearer(caption: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        prefix = match.string[: match.start()].rstrip()
        return "The camera wearer" if not prefix or prefix[-1] in ".!?" else "the camera wearer"

    return _STANDALONE_C_PATTERN.sub(replacement, caption)


def _replace_standalone_c_with_person(caption: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        prefix = match.string[: match.start()].rstrip()
        return "The person" if not prefix or prefix[-1] in ".!?" else "the person"

    return _STANDALONE_C_PATTERN.sub(replacement, caption)


def _format_edge_nymeria_i2v_prompt(record: dict[str, Any]) -> str:
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Edge Nymeria image2video requires a non-empty prompt")
    # Preserve intentionally supplied structured prompts, such as controlled
    # diagnostics. Ordinary Nymeria captions are deterministically templated.
    if parse_json_object(prompt) is not None:
        return prompt
    aspect_ratio = str(record.get("aspect_ratio", ""))
    if aspect_ratio != "1,1":
        raise ValueError(
            "the Nymeria I2V template currently supports the square inference contract only; "
            f"got aspect_ratio={aspect_ratio!r}"
        )
    try:
        size = int(record["resolution"])
        num_frames = int(record["num_frames"])
        fps = float(record["fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Edge Nymeria I2V media contract: {error}") from error
    return compact_prompt(
        build_nymeria_i2v_prompt(
            prompt,
            num_frames=num_frames,
            fps=fps,
            height=size,
            width=size,
        )
    )


def _format_edge_nymeria_i2v_negative_prompt(record: dict[str, Any]) -> str:
    try:
        size = int(record["resolution"])
        num_frames = int(record["num_frames"])
        fps = float(record["fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Edge Nymeria I2V media contract: {error}") from error
    return compact_prompt(
        build_nymeria_i2v_negative_prompt(
            num_frames=num_frames,
            fps=fps,
            height=size,
            width=size,
        )
    )


def runtime_model_mode(canonical_mode: str, model_family: str) -> str:
    """Translate repository task names to the selected framework's public enum."""
    if model_family == "edge" and canonical_mode == "policy":
        return "wam"
    return canonical_mode


def runtime_mode_matches(*, actual_mode: Any, canonical_mode: str) -> bool:
    """Accept the renewed Edge ``wam`` spelling for canonical policy outputs."""
    return actual_mode == canonical_mode or (
        canonical_mode == "policy" and actual_mode == "wam"
    )


def _with_runtime_mode(
    record: dict[str, Any], *, canonical_mode: str, model_family: str
) -> dict[str, Any]:
    output = dict(record)
    output["model_mode"] = runtime_model_mode(canonical_mode, model_family)
    return output


def sanitize_record(
    record: dict[str, Any],
    expected_mode: str,
    *,
    model_family: str = "nano",
    replace_standalone_c: bool = False,
    standalone_c_subject: str = "person",
) -> dict[str, Any]:
    """Return an official-schema record while validating local metric metadata."""
    if model_family not in {"nano", "edge"}:
        raise ValueError(f"unsupported model family: {model_family!r}")
    if standalone_c_subject not in {"person", "camera_wearer"}:
        raise ValueError(
            "standalone_c_subject must be 'person' or 'camera_wearer', "
            f"got {standalone_c_subject!r}"
        )
    record = dict(record)
    if replace_standalone_c and isinstance(record.get("prompt"), str):
        replacement = (
            _replace_standalone_c_with_camera_wearer
            if standalone_c_subject == "camera_wearer"
            else _replace_standalone_c_with_person
        )
        record["prompt"] = replacement(record["prompt"])
    if model_family == "edge" and expected_mode == "image2video":
        record["prompt"] = _format_edge_nymeria_i2v_prompt(record)
        if "negative_prompt" not in record:
            record["negative_prompt"] = _format_edge_nymeria_i2v_negative_prompt(record)
            record["negative_metadata_mode"] = "none"
            record["negative_prompt_keep_metadata"] = False
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
        return _with_runtime_mode(
            record,
            canonical_mode=expected_mode,
            model_family=model_family,
        )
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
    sanitized = {key: value for key, value in record.items() if key not in LOCAL_METADATA_FIELDS}
    return _with_runtime_mode(
        sanitized,
        canonical_mode=expected_mode,
        model_family=model_family,
    )


def sanitize_input_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    model_family: str = "nano",
    replace_standalone_c: bool = False,
    standalone_c_subject: str = "person",
) -> dict[str, int]:
    """Write all four schema-clean JSONLs and return per-mode record counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for mode, filename in INPUT_FILES.items():
        source = input_dir / filename
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty evaluation input: {source}")
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        sanitized = [
            sanitize_record(
                record,
                mode,
                model_family=model_family,
                replace_standalone_c=replace_standalone_c,
                standalone_c_subject=standalone_c_subject,
            )
            for record in records
        ]
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
    parser.add_argument(
        "--model-family",
        choices=("nano", "edge"),
        default=os.environ.get("NATIVEP1_MODEL_FAMILY", "nano").strip().lower(),
    )
    parser.add_argument(
        "--replace-standalone-c",
        action="store_true",
        help=(
            "replace only whole-token uppercase C with the selected sentence-aware subject"
        ),
    )
    parser.add_argument(
        "--standalone-c-subject",
        choices=("person", "camera_wearer"),
        default="person",
    )
    args = parser.parse_args()
    counts = sanitize_input_directory(
        args.input_dir,
        args.output_dir,
        model_family=args.model_family,
        replace_standalone_c=args.replace_standalone_c,
        standalone_c_subject=args.standalone_c_subject,
    )
    print(f"[nativeviz] sanitized official inference inputs: {counts}", flush=True)


if __name__ == "__main__":
    main()
