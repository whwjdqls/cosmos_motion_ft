"""Deterministic structured I2V prompts for Nymeria inference.

Only ``actions[0].description`` depends on the source caption. Everything else
is either a conservative Nymeria-wide visual prior or request metadata. This is
an inference transform; it is intentionally not used by the training dataset.
"""

from __future__ import annotations

import copy
import hashlib
import json
from math import gcd
from pathlib import Path
from typing import Any


DESCRIPTION_PLACEHOLDER = "{{DESCRIPTION}}"
DEFAULT_TEMPLATE_PATH = (
    Path(__file__).with_name("prompts") / "nymeria_i2v_prompt_template_v2_1.json"
)
DEFAULT_NEGATIVE_TEMPLATE_PATH = (
    Path(__file__).with_name("prompts") / "nymeria_i2v_negative_prompt_template_v2.json"
)


def _duration_seconds(num_frames: int, fps: float) -> int:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    # Match the released Cosmos training/inference metadata convention: T121 at
    # 24 FPS is labelled 5s and Nymeria T97 at 20 FPS is labelled 4s.
    return max(1, int(num_frames / fps))


def _aspect_ratio(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {(width, height)}")
    divisor = gcd(width, height)
    return f"{width // divisor},{height // divisor}"


def load_template(path: Path | None = None) -> dict[str, Any]:
    template_path = (path or DEFAULT_TEMPLATE_PATH).resolve()
    value = json.loads(template_path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{template_path}: template must be a JSON object")
    actions = value.get("actions")
    if (
        not isinstance(actions, list)
        or len(actions) != 1
        or not isinstance(actions[0], dict)
        or not isinstance(actions[0].get("description"), str)
        or DESCRIPTION_PLACEHOLDER not in actions[0]["description"]
    ):
        raise ValueError(
            f"{template_path}: actions must contain exactly one object whose description contains "
            f"{DESCRIPTION_PLACEHOLDER!r} exactly once"
        )
    encoded = json.dumps(value, ensure_ascii=False)
    if encoded.count(DESCRIPTION_PLACEHOLDER) != 1:
        raise ValueError(f"{template_path}: description placeholder must occur exactly once")
    return value


def build_nymeria_i2v_prompt(
    description: str,
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    template_path: Path | None = None,
) -> dict[str, Any]:
    """Render the frozen template with one caption-dependent field."""
    description = description.strip()
    if not description:
        raise ValueError("Nymeria I2V description must be non-empty")

    prompt = copy.deepcopy(load_template(template_path))
    seconds = _duration_seconds(num_frames, fps)
    prompt["actions"][0]["description"] = prompt["actions"][0]["description"].replace(
        DESCRIPTION_PLACEHOLDER,
        description,
    )
    prompt["actions"][0]["time"] = f"0:00-0:{seconds:02d}"
    prompt["duration"] = f"{seconds}s"
    prompt["fps"] = float(fps)
    prompt["resolution"] = {"H": int(height), "W": int(width)}
    prompt["aspect_ratio"] = _aspect_ratio(width, height)
    return prompt


def build_nymeria_i2v_negative_prompt(
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    template_path: Path | None = None,
) -> dict[str, Any]:
    """Render the fixed Nymeria I2V failure-mode prompt with request metadata."""
    path = (template_path or DEFAULT_NEGATIVE_TEMPLATE_PATH).resolve()
    prompt = json.loads(path.read_text())
    if not isinstance(prompt, dict):
        raise ValueError(f"{path}: negative prompt template must be a JSON object")
    if DESCRIPTION_PLACEHOLDER in json.dumps(prompt, ensure_ascii=False):
        raise ValueError(f"{path}: negative prompt must not contain a description placeholder")
    seconds = _duration_seconds(num_frames, fps)
    actions = prompt.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if isinstance(action, dict) and "time" in action:
                action["time"] = f"0:00-0:{seconds:02d}"
    segments = prompt.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, dict) and "time_range" in segment:
                segment["time_range"] = f"0:00-0:{seconds:02d}"
    prompt["duration"] = f"{seconds}s"
    prompt["fps"] = float(fps)
    prompt["resolution"] = {"H": int(height), "W": int(width)}
    prompt["aspect_ratio"] = _aspect_ratio(width, height)
    return prompt


def compact_prompt(prompt: dict[str, Any]) -> str:
    return json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))


def parse_json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_native_structured_prompt(
    value: str,
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    aspect_ratio: str | None = None,
) -> str:
    """Match the official native inference string passed to the tokenizer.

    Native Cosmos parses a JSON-object positive prompt, overwrites its request
    metadata, and serializes it with the default ``json.dumps`` formatting.
    Keeping that exact serialization in the shared input JSONL makes native and
    Diffusers receive the same positive-prompt string, not merely equivalent
    JSON objects.
    """
    prompt = parse_json_object(value)
    if prompt is None:
        raise ValueError("native structured-prompt normalization requires a JSON object")
    prompt = copy.deepcopy(prompt)
    if num_frames > 1:
        prompt.update(
            {
                "duration": f"{_duration_seconds(num_frames, fps)}s",
                "fps": float(fps),
            }
        )
    else:
        prompt.pop("duration", None)
        prompt.pop("fps", None)
    prompt["resolution"] = {"H": int(height), "W": int(width)}
    if aspect_ratio is not None:
        prompt["aspect_ratio"] = str(aspect_ratio)
    return json.dumps(prompt)


def write_prompt_artifacts(
    output_dir: Path,
    *,
    positive_prompt: str,
    negative_prompt: str | None = None,
) -> dict[str, Any]:
    """Save the exact effective prompt(s) beside a generated sample."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for kind, value in (("positive", positive_prompt), ("negative", negative_prompt)):
        if value is None:
            continue
        parsed = parse_json_object(value)
        suffix = "json" if parsed is not None else "txt"
        path = output_dir / f"{kind}_prompt.{suffix}"
        content = (
            json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
            if parsed is not None
            else value.rstrip() + "\n"
        )
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"refusing to overwrite a different prompt artifact: {path}")
        path.write_text(content)
        manifest[kind] = {
            "path": path.name,
            "format": "structured_json" if parsed is not None else "plain_text",
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    manifest_path = output_dir / "prompt_manifest.json"
    manifest_content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != manifest_content:
        raise FileExistsError(f"refusing to overwrite a different prompt manifest: {manifest_path}")
    manifest_path.write_text(manifest_content)
    return manifest
