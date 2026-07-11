#!/usr/bin/env python
"""Build official-inference inputs for the held-out native Phase 1 test set.

This entrypoint intentionally pins the training contract: 97 frames, 96 camera
actions, 20 FPS, the 256 resolution tier, and sampler shift 3.0. It does not
reuse the historical ``nymeria_world/prep_test_eval.py`` defaults (480/shift 10).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from nymeria_camera_dataset import decode_window_pyav
from nymeria_camera_rgb_dataset import _rgb_path, rel_action_from_window


MANIFEST = Path("/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl")
SPLIT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json")

NUM_FRAMES = 97
ACTION_CHUNK_SIZE = NUM_FRAMES - 1
FPS = 20
RESOLUTION = "256"
IMAGE_SIZE = 256
SHIFT = 3.0


def pick_test_windows(n: int = 0, seed: int = 0) -> list[dict[str, Any]]:
    """Choose at most one T97 window per held-out sequence.

    Walking/turning captions are preferred for camera-control visibility. If a
    sequence has no such caption, its first usable full-length window is used so
    the quantitative set is not silently reduced by a qualitative text filter.
    ``n=0`` means all available held-out sequences (71 in the current split).
    """
    split_data = json.loads(SPLIT.read_text())
    test_uuids = list(split_data["test"])
    records: dict[str, dict[str, Any]] = {}
    with MANIFEST.open() as f:
        for line in f:
            record = json.loads(line)
            uuid = record.get("uuid")
            if uuid in split_data["test"]:
                records[uuid] = record

    selected: list[dict[str, Any]] = []
    skipped: list[str] = []
    for uuid in test_uuids:
        record = records.get(uuid)
        if record is None or not record.get("camera_path") or not record.get("vision_path"):
            skipped.append(uuid)
            continue

        rgb_path = _rgb_path(record["camera_path"])
        if not os.path.isfile(rgb_path) or not os.path.isfile(record["vision_path"]):
            skipped.append(uuid)
            continue

        nb_frames = int(record.get("nb_frames", 0))
        candidates: list[dict[str, Any]] = []
        for window in record.get("t2w_windows", []):
            caption = (window.get("caption") or "").strip()
            start = int(window.get("start_frame", 0))
            end = min(int(window.get("end_frame", nb_frames)), nb_frames)
            if not window.get("usable", False) or not caption or start + NUM_FRAMES > end:
                continue
            candidates.append(
                {
                    "uuid": uuid,
                    "start": start,
                    "vision_path": record["vision_path"],
                    "rgb_path": rgb_path,
                    "caption": caption,
                }
            )

        if not candidates:
            skipped.append(uuid)
            continue
        preferred = [
            candidate
            for candidate in candidates
            if any(word in candidate["caption"].lower() for word in ("walk", "turn"))
        ]
        selected.append((preferred or candidates)[0])

    random.Random(seed).shuffle(selected)
    if n > 0:
        selected = selected[:n]
    print(
        f"[native-eval] selected={len(selected)} held_out={len(test_uuids)} "
        f"unavailable={len(skipped)} seed={seed}",
        flush=True,
    )
    if skipped:
        print(f"[native-eval] unavailable UUIDs: {', '.join(skipped)}", flush=True)
    return selected


def build_inference_records(
    *, name: str, first_frame: Path, gt_clip: Path, action_path: Path, caption: str, seed: int
) -> dict[str, dict[str, Any]]:
    """Build and validate the four native Phase 1 inference records."""
    sampling_common: dict[str, Any] = {
        "num_frames": NUM_FRAMES,
        "resolution": RESOLUTION,
        "aspect_ratio": "1,1",
        "fps": FPS,
        "shift": SHIFT,
        "seed": seed,
    }
    action_common: dict[str, Any] = {
        **sampling_common,
        "domain_name": "camera_pose",
        "view_point": "ego_view",
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "image_size": IMAGE_SIZE,
        "num_steps": 30,
        "guidance": 1.0,
    }
    records = {
        "inverse_dynamics": {
            **action_common,
            "model_mode": "inverse_dynamics",
            "name": f"{name}_inverse_dynamics",
            "vision_path": str(gt_clip),
            "prompt": "",
        },
        "forward_dynamics": {
            **action_common,
            "model_mode": "forward_dynamics",
            "name": f"{name}_forward_dynamics",
            "vision_path": str(first_frame),
            "action_path": str(action_path),
            "prompt": caption,
        },
        "policy": {
            **action_common,
            "model_mode": "policy",
            "name": f"{name}_policy",
            "vision_path": str(first_frame),
            "prompt": caption,
        },
        "image2video": {
            **sampling_common,
            "model_mode": "image2video",
            "name": f"{name}_image2video",
            "vision_path": str(first_frame),
            "prompt": caption,
            "aspect_ratio": "1,1",
            "num_steps": 35,
            "guidance": 6.0,
        },
    }
    for mode, record in records.items():
        validate_inference_record(record, mode)
    return records


def validate_inference_record(record: dict[str, Any], expected_mode: str) -> None:
    """Fail before GPU inference if a record drifts from the Phase 1 contract."""
    expected: dict[str, Any] = {
        "model_mode": expected_mode,
        "num_frames": NUM_FRAMES,
        "resolution": RESOLUTION,
        "aspect_ratio": "1,1",
        "fps": FPS,
        "shift": SHIFT,
    }
    if expected_mode == "image2video":
        expected.update({"num_steps": 35, "guidance": 6.0})
    else:
        expected.update(
            {
                "action_chunk_size": ACTION_CHUNK_SIZE,
                "image_size": IMAGE_SIZE,
                "num_steps": 30,
                "guidance": 1.0,
            }
        )
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{expected_mode}: expected {key}={value!r}, got {record.get(key)!r}")
    if expected_mode == "inverse_dynamics" and record.get("prompt") != "":
        raise ValueError("inverse_dynamics prompt must be exactly empty")
    if expected_mode == "forward_dynamics" and not record.get("action_path"):
        raise ValueError("forward_dynamics requires action_path")
    if expected_mode != "forward_dynamics" and "action_path" in record:
        raise ValueError(f"{expected_mode} must not provide action_path")
    if expected_mode == "image2video" and any(
        key in record for key in ("domain_name", "view_point", "action_chunk_size", "image_size")
    ):
        raise ValueError("image2video must not contain action-only fields")


def _write_json(path: Path, value: Any, *, indent: int | None = None) -> None:
    path.write_text(json.dumps(value, indent=indent) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs"))
    parser.add_argument("--n", type=int, default=0, help="Number of held-out sequences; 0 uses all available")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.n < 0:
        parser.error("--n must be non-negative")

    picks = pick_test_windows(n=args.n, seed=args.seed)
    if not picks:
        raise RuntimeError("no held-out T97 windows are available")

    samples_root = args.out / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    by_mode: dict[str, list[dict[str, Any]]] = {
        "inverse_dynamics": [],
        "forward_dynamics": [],
        "policy": [],
        "image2video": [],
    }

    for index, pick in enumerate(picks):
        name = f"t{index:02d}_{pick['uuid'].replace('/', '_')}"
        sample_dir = samples_root / name
        sample_dir.mkdir(parents=True, exist_ok=True)

        frames = decode_window_pyav(pick["vision_path"], pick["start"], NUM_FRAMES, FPS)
        if frames.shape[0] != NUM_FRAMES:
            raise ValueError(f"{name}: decoded {frames.shape[0]} frames, expected {NUM_FRAMES}")
        first_frame = sample_dir / "first_frame.png"
        gt_clip = sample_dir / "gt_clip.mp4"
        iio.imwrite(first_frame, frames[0])
        iio.imwrite(gt_clip, frames, fps=FPS, codec="libx264")

        with np.load(pick["rgb_path"]) as camera:
            start = pick["start"]
            stop = start + NUM_FRAMES
            position = camera["cam_world_pos_upright"][start:stop].astype(np.float32)
            rotation = camera["cam_world_rot_upright"][start:stop].astype(np.float32)
        action = rel_action_from_window(position, rotation).astype(np.float32)
        if action.shape != (ACTION_CHUNK_SIZE, 9) or not np.isfinite(action).all():
            raise ValueError(f"{name}: invalid camera action shape/values {action.shape}")

        np.savez(sample_dir / "gt_camera_cosmos.npz", cam_world_pos=position, cam_world_rot=rotation)
        action_path = sample_dir / "camera_action.json"
        _write_json(action_path, action.tolist())
        _write_json(
            sample_dir / "meta.json",
            {
                "uuid": pick["uuid"],
                "start_frame": pick["start"],
                "caption": pick["caption"],
                "num_frames": NUM_FRAMES,
                "action_chunk_size": ACTION_CHUNK_SIZE,
                "fps": FPS,
                "resolution": RESOLUTION,
                "shift": SHIFT,
            },
            indent=2,
        )

        records = build_inference_records(
            name=name,
            first_frame=first_frame.resolve(),
            gt_clip=gt_clip.resolve(),
            action_path=action_path.resolve(),
            caption=pick["caption"],
            seed=args.seed,
        )
        for mode, record in records.items():
            by_mode[mode].append(record)
        print(f"[native-eval] [{index + 1}/{len(picks)}] {name}: {pick['caption'][:72]}", flush=True)

    _write_jsonl(args.out / "invdyn_input.jsonl", by_mode["inverse_dynamics"])
    _write_jsonl(args.out / "fd_input.jsonl", by_mode["forward_dynamics"])
    _write_jsonl(args.out / "policy_input.jsonl", by_mode["policy"])
    _write_jsonl(args.out / "i2v_input.jsonl", by_mode["image2video"])
    print(f"[native-eval] wrote {len(picks)} records for each of four modes under {args.out}", flush=True)


if __name__ == "__main__":
    main()
