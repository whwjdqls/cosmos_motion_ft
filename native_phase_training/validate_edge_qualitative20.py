#!/usr/bin/env python
"""Validate the frozen Edge qualitative-20 native/Diffusers result contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from native_phase_training.nymeria_i2v_prompt import normalize_native_structured_prompt
from native_phase_training.sanitize_prefix_inference_inputs import runtime_mode_matches


INPUT_FILES = {
    "forward_dynamics": "fd_input.jsonl",
    "inverse_dynamics": "invdyn_input.jsonl",
    "policy": "policy_input.jsonl",
    "image2video": "i2v_input.jsonl",
}
VIDEO_MODES = frozenset({"forward_dynamics", "policy", "image2video"})
EXPECTED_SAMPLERS = {
    "forward_dynamics": (10.0, 30, 1.0),
    "inverse_dynamics": (10.0, 30, 1.0),
    "policy": (10.0, 30, 1.0),
    "image2video": (12.0, 20, 6.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_success(path: Path, mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "success":
        raise ValueError(f"{path}: inference was not successful")
    sample_args = payload.get("args")
    if not isinstance(sample_args, dict) or not runtime_mode_matches(
        actual_mode=sample_args.get("model_mode"), canonical_mode=mode
    ):
        raise ValueError(f"{path}: mode mismatch for {mode}")
    return payload, sample_args


def _validate_settings(sample_args: dict[str, Any], mode: str, path: Path) -> None:
    shift, steps, guidance = EXPECTED_SAMPLERS[mode]
    expected = {
        "shift": shift,
        "num_steps": steps,
        "guidance": guidance,
        "num_frames": 97,
        "fps": 20,
        "resolution": "256",
        "seed": 0,
        "native_prompt_upsampling": False,
    }
    mismatches = {
        key: (sample_args.get(key), value)
        for key, value in expected.items()
        if sample_args.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: settings mismatch: {mismatches}")


def _probe_video(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    metadata_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    observed = (metadata_frames, round(fps, 4), width, height)
    expected = (97, 20.0, 256, 256)
    if observed != expected:
        raise ValueError(f"{path}: media mismatch, expected {expected}, got {observed}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "num_frames": metadata_frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--diffusers-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prepared_root = args.prepared_root.resolve()
    input_root = prepared_root / "inference_inputs"
    contract = json.loads((prepared_root / "cohort_contract.json").read_text())
    cohort_names = contract["samples"]
    if len(cohort_names) != 20:
        raise ValueError("cohort contract is not the frozen 20-sample set")

    by_mode: dict[str, list[dict[str, Any]]] = {}
    native_videos: list[dict[str, Any]] = []
    action_outputs = 0
    for mode, filename in INPUT_FILES.items():
        records = _read_jsonl(input_root / filename)
        if len(records) != 20:
            raise ValueError(f"{filename}: expected 20 records, got {len(records)}")
        by_mode[mode] = records
        for record in records:
            sample_dir = args.native_root.resolve() / record["name"]
            payload, sample_args = _load_success(sample_dir / "sample_outputs.json", mode)
            _validate_settings(sample_args, mode, sample_dir / "sample_args.json")
            if mode in VIDEO_MODES:
                native_videos.append(_probe_video(sample_dir / "vision.mp4"))
            if mode in {"inverse_dynamics", "policy"}:
                action = np.asarray(payload["outputs"][0]["content"].get("action"), dtype=np.float64)
                if action.shape != (96, 9) or not np.isfinite(action).all():
                    raise ValueError(f"{sample_dir}: invalid action tensor {action.shape}")
                action_outputs += 1

    i2v_pairs: list[dict[str, Any]] = []
    diffusers_videos: list[dict[str, Any]] = []
    for base_name, record in zip(cohort_names, by_mode["image2video"], strict=True):
        expected_name = f"{base_name}_image2video"
        if record["name"] != expected_name:
            raise ValueError(f"I2V order mismatch: expected {expected_name}, got {record['name']}")
        expected_prompt = normalize_native_structured_prompt(
            record["prompt"],
            num_frames=97,
            fps=20,
            height=256,
            width=256,
            aspect_ratio="1,1",
        )
        if expected_prompt != record["prompt"]:
            raise ValueError(f"{expected_name}: shared positive prompt is not native-normalized")

        native_dir = args.native_root.resolve() / expected_name
        diffusers_dir = args.diffusers_root.resolve() / expected_name
        native_args = json.loads((native_dir / "sample_args.json").read_text())
        diffusers_args = json.loads((diffusers_dir / "sample_args.json").read_text())
        for key in ("prompt", "negative_prompt", "vision_path"):
            expected_value = record[key]
            values = (native_args.get(key), diffusers_args.get(key))
            if values != (expected_value, expected_value):
                raise ValueError(f"{expected_name}: backend {key} mismatch")
        _validate_settings(native_args, "image2video", native_dir / "sample_args.json")
        _validate_settings(diffusers_args, "image2video", diffusers_dir / "sample_args.json")
        if diffusers_args.get("scheduler_class") != "FlowUniPCMultistepScheduler":
            raise ValueError(f"{diffusers_dir}: Diffusers did not use the native UniPC scheduler")
        image_path = Path(record["vision_path"])
        image_hash = _sha256(image_path)
        diff_complete = json.loads((diffusers_dir / "COMPLETE.json").read_text())
        if diff_complete.get("input_image_sha256") != image_hash:
            raise ValueError(f"{diffusers_dir}: conditioning-image hash mismatch")
        diff_video = _probe_video(diffusers_dir / "vision.mp4")
        diffusers_videos.append(diff_video)
        i2v_pairs.append(
            {
                "sample": base_name,
                "native": str(native_dir / "vision.mp4"),
                "diffusers": str(diffusers_dir / "vision.mp4"),
                "conditioning_image": str(image_path),
                "conditioning_image_sha256": image_hash,
                "positive_prompt_exact_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                "negative_prompt_exact_sha256": hashlib.sha256(
                    record["negative_prompt"].encode()
                ).hexdigest(),
                "settings": {"shift": 12.0, "num_steps": 20, "guidance": 6.0},
            }
        )

    result = {
        "status": "complete",
        "kind": "cosmos3_edge_qualitative20_validation",
        "cohort": contract["name"],
        "count": 20,
        "native_result_count": 80,
        "native_video_count": len(native_videos),
        "native_action_count": action_outputs,
        "diffusers_i2v_video_count": len(diffusers_videos),
        "i2v_backends_share_exact_inputs": True,
        "backend_specific_implementation_remains": True,
        "i2v_pairs": i2v_pairs,
    }
    if (len(native_videos), action_outputs, len(diffusers_videos)) != (60, 40, 20):
        raise ValueError("unexpected validated artifact totals")
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"validated 80 native results and 20 matched Diffusers I2V results: {args.out}")


if __name__ == "__main__":
    main()
