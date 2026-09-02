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
from native_phase_training.latent_nymeria_dataset import (
    load_quality_filter_exclusions,
    replace_standalone_c,
    rgb_prefix_to_latent_frames,
    validate_prefix_sampling,
)


MANIFEST = Path("/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl")
SPLIT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json")

NUM_FRAMES = 97
ACTION_CHUNK_SIZE = NUM_FRAMES - 1
FPS = 20
RESOLUTION = "256"
IMAGE_SIZE = 256
SHIFT = 3.0


def pick_test_windows(
    n: int = 0,
    seed: int = 0,
    quality_filter_path: str = "",
) -> list[dict[str, Any]]:
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

    exclusions = load_quality_filter_exclusions(quality_filter_path, NUM_FRAMES)
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
            exclusion = exclusions.get((uuid, start, start + NUM_FRAMES))
            if exclusion is not None:
                if exclusion["split"] != "test":
                    raise ValueError(
                        f"quality filter split mismatch for {(uuid, start, start + NUM_FRAMES)}"
                    )
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


def pick_explicit_test_windows(
    windows_json: Path,
    quality_filter_path: str = "",
) -> list[dict[str, Any]]:
    """Resolve an ordered explicit ``[{uuid,start}]`` list against the test manifest."""
    rows = json.loads(windows_json.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{windows_json}: expected a nonempty JSON list")

    split_data = json.loads(SPLIT.read_text())
    test_uuids = set(split_data["test"])
    requested: list[tuple[str, int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "uuid" not in row or "start" not in row:
            raise ValueError(f"{windows_json}: malformed row {index}: {row!r}")
        uuid = str(row["uuid"])
        start = int(row["start"])
        frames = int(row.get("num_frames", NUM_FRAMES))
        if frames != NUM_FRAMES:
            raise ValueError(
                f"{windows_json}: row {index} has num_frames={frames}, expected {NUM_FRAMES}"
            )
        if uuid not in test_uuids:
            raise ValueError(f"{windows_json}: {uuid!r} is not in the held-out split")
        requested.append((uuid, start))
    if len(set(requested)) != len(requested):
        raise ValueError(f"{windows_json}: duplicate (uuid, start) windows")

    wanted_uuids = {uuid for uuid, _start in requested}
    records: dict[str, dict[str, Any]] = {}
    with MANIFEST.open() as f:
        for line in f:
            record = json.loads(line)
            uuid = record.get("uuid")
            if uuid in wanted_uuids:
                records[uuid] = record

    exclusions = load_quality_filter_exclusions(quality_filter_path, NUM_FRAMES)
    selected: list[dict[str, Any]] = []
    for uuid, start in requested:
        record = records.get(uuid)
        if record is None or not record.get("camera_path") or not record.get("vision_path"):
            raise ValueError(f"{windows_json}: missing manifest camera/video record for {uuid}")
        rgb_path = _rgb_path(record["camera_path"])
        if not os.path.isfile(rgb_path) or not os.path.isfile(record["vision_path"]):
            raise FileNotFoundError(
                f"{windows_json}: missing RGB trajectory or video for {(uuid, start)}"
            )
        nb_frames = int(record.get("nb_frames", 0))
        if start < 0 or start + NUM_FRAMES > nb_frames:
            raise ValueError(
                f"{windows_json}: {(uuid, start)} exceeds sequence length {nb_frames}"
            )
        exclusion = exclusions.get((uuid, start, start + NUM_FRAMES))
        if exclusion is not None:
            if exclusion["split"] != "test":
                raise ValueError(
                    f"quality filter split mismatch for {(uuid, start, start + NUM_FRAMES)}"
                )
            raise ValueError(
                f"{windows_json}: explicitly requested window is excluded by the quality filter: "
                f"{(uuid, start, start + NUM_FRAMES)}"
            )

        captions = []
        for window in record.get("t2w_windows", []):
            caption = (window.get("caption") or "").strip()
            window_start = int(window.get("start_frame", 0))
            window_end = min(int(window.get("end_frame", nb_frames)), nb_frames)
            if (
                window_start == start
                and window.get("usable", False)
                and caption
                and start + NUM_FRAMES <= window_end
            ):
                captions.append(caption)
        if not captions:
            raise ValueError(
                f"{windows_json}: no usable captioned T97 manifest window for {(uuid, start)}"
            )
        selected.append(
            {
                "uuid": uuid,
                "start": start,
                "vision_path": record["vision_path"],
                "rgb_path": rgb_path,
                "caption": captions[0],
            }
        )

    print(
        f"[native-eval] resolved {len(selected)}/{len(requested)} explicit held-out windows "
        f"from {windows_json}",
        flush=True,
    )
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


def build_prefix_inference_records(
    *,
    name: str,
    gt_clip: Path,
    prefix_paths: dict[int, Path],
    action_path: Path,
    caption: str,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Build one inverse record and fixed-prefix records for all visual tasks."""
    if not prefix_paths:
        raise ValueError("prefix_paths must not be empty")
    records: dict[str, list[dict[str, Any]]] = {
        "inverse_dynamics": [],
        "forward_dynamics": [],
        "policy": [],
        "image2video": [],
    }
    first_prefix = min(prefix_paths)
    base = build_inference_records(
        name=name,
        first_frame=prefix_paths[first_prefix],
        gt_clip=gt_clip,
        action_path=action_path,
        caption=caption,
        seed=seed,
    )
    inverse = base["inverse_dynamics"]
    inverse["source_name"] = name
    records["inverse_dynamics"].append(inverse)

    for prefix_length, prefix_path in sorted(prefix_paths.items()):
        latent_prefix_length = rgb_prefix_to_latent_frames(prefix_length, NUM_FRAMES)
        condition_indexes = list(range(latent_prefix_length))
        for mode in ("forward_dynamics", "policy", "image2video"):
            record = dict(base[mode])
            record.update(
                {
                    "name": f"{name}_p{prefix_length:03d}_{mode}",
                    "vision_path": str(prefix_path),
                    "condition_frame_indexes_vision": condition_indexes,
                    "rgb_prefix_length": prefix_length,
                    "latent_prefix_length": latent_prefix_length,
                    "source_name": name,
                }
            )
            validate_inference_record(record, mode)
            records[mode].append(record)
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
    if "rgb_prefix_length" in record:
        prefix_length = int(record["rgb_prefix_length"])
        latent_prefix_length = rgb_prefix_to_latent_frames(prefix_length, NUM_FRAMES)
        expected_indexes = list(range(latent_prefix_length))
        if record.get("latent_prefix_length") != latent_prefix_length:
            raise ValueError(f"{expected_mode}: latent prefix mismatch for RGB prefix {prefix_length}")
        if record.get("condition_frame_indexes_vision") != expected_indexes:
            raise ValueError(
                f"{expected_mode}: expected condition indexes {expected_indexes}, "
                f"got {record.get('condition_frame_indexes_vision')}"
            )


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
    parser.add_argument(
        "--prefix-lengths",
        default="1",
        help="Comma-separated exact causal RGB prefix lengths (each must be 1+4N)",
    )
    parser.add_argument("--quality-filter", default="")
    parser.add_argument("--replace-standalone-c", action="store_true")
    parser.add_argument(
        "--standalone-c-subject",
        choices=("person", "camera_wearer"),
        default="person",
        help="semantic subject used when --replace-standalone-c is enabled",
    )
    parser.add_argument(
        "--windows-json",
        type=Path,
        default=None,
        help="ordered explicit [{uuid,start,num_frames?}] held-out windows; preserves list order",
    )
    args = parser.parse_args()
    if args.n < 0:
        parser.error("--n must be non-negative")
    if args.windows_json is not None and args.n != 0:
        parser.error("--n cannot be combined with --windows-json")

    try:
        requested_prefixes = [int(value.strip()) for value in args.prefix_lengths.split(",") if value.strip()]
    except ValueError as error:
        parser.error(f"invalid --prefix-lengths: {error}")
    prefix_lengths, _ = validate_prefix_sampling(requested_prefixes, None, NUM_FRAMES)

    if args.windows_json is not None:
        picks = pick_explicit_test_windows(
            args.windows_json,
            quality_filter_path=args.quality_filter,
        )
    else:
        picks = pick_test_windows(
            n=args.n,
            seed=args.seed,
            quality_filter_path=args.quality_filter,
        )
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
        if args.windows_json is not None:
            name += f"_s{pick['start']}"
        sample_dir = samples_root / name
        sample_dir.mkdir(parents=True, exist_ok=True)

        frames = decode_window_pyav(pick["vision_path"], pick["start"], NUM_FRAMES, FPS)
        if frames.shape[0] != NUM_FRAMES:
            raise ValueError(f"{name}: decoded {frames.shape[0]} frames, expected {NUM_FRAMES}")
        first_frame = sample_dir / "first_frame.png"
        gt_clip = sample_dir / "gt_clip.mp4"
        iio.imwrite(first_frame, frames[0])
        iio.imwrite(gt_clip, frames, fps=FPS, codec="libx264")
        prefix_paths: dict[int, Path] = {}
        for prefix_length in prefix_lengths:
            if prefix_length == 1:
                prefix_paths[prefix_length] = first_frame.resolve()
            else:
                prefix_path = sample_dir / f"prefix_{prefix_length:03d}.mp4"
                iio.imwrite(prefix_path, frames[:prefix_length], fps=FPS, codec="libx264")
                prefix_paths[prefix_length] = prefix_path.resolve()

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

        eval_caption = (
            replace_standalone_c(pick["caption"], args.standalone_c_subject)
            if args.replace_standalone_c
            else pick["caption"]
        )
        records = build_prefix_inference_records(
            name=name,
            gt_clip=gt_clip.resolve(),
            prefix_paths=prefix_paths,
            action_path=action_path.resolve(),
            caption=eval_caption,
            seed=args.seed,
        )
        for mode, mode_records in records.items():
            by_mode[mode].extend(mode_records)
        print(f"[native-eval] [{index + 1}/{len(picks)}] {name}: {pick['caption'][:72]}", flush=True)

    _write_jsonl(args.out / "invdyn_input.jsonl", by_mode["inverse_dynamics"])
    _write_jsonl(args.out / "fd_input.jsonl", by_mode["forward_dynamics"])
    _write_jsonl(args.out / "policy_input.jsonl", by_mode["policy"])
    _write_jsonl(args.out / "i2v_input.jsonl", by_mode["image2video"])
    print(
        f"[native-eval] wrote inverse={len(by_mode['inverse_dynamics'])} and "
        f"visual={len(by_mode['forward_dynamics'])} records/mode for prefixes={list(prefix_lengths)} "
        f"under {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
