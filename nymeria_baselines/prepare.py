"""Prepare Nymeria-SOMA-30 camera/motion arrays for upstream baseline loaders.

Runs on an owned compute node. Source NPZ files are never modified. The common
index keeps Nymeria's original recording-level 71-case test split intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


T = 97
FPS = 20
JOINTS = 30
FEATURES = 283
Y_FROM_Z = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
IDENTITY_DELTA = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
JOINT_HEIGHT = np.arange(JOINTS) * 9 + 7


def _matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    return rotation[..., :, :2].swapaxes(-1, -2).reshape(*rotation.shape[:-2], 6)


def camera_trajectory18(pose_yup: np.ndarray) -> np.ndarray:
    """Official v4-style local camera pose + residual floor-yaw pose, Y up."""
    rotation = pose_yup[:, :3, :3].astype(np.float64)
    position = pose_yup[:, :3, 3].astype(np.float64)
    forward = rotation[:, :, 2]
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    co, si = np.cos(yaw), np.sin(yaw)
    canonical = np.zeros_like(rotation)
    canonical[:, 0, 0] = co
    canonical[:, 0, 2] = si
    canonical[:, 1, 1] = 1
    canonical[:, 2, 0] = -si
    canonical[:, 2, 2] = co
    inv = canonical.swapaxes(-1, -2)
    local_rotation = inv @ rotation
    local_position = np.einsum("tij,tj->ti", inv, position - position * [1, 0, 1])
    delta_rotation = np.empty_like(rotation)
    delta_position = np.empty_like(position)
    delta_rotation[0] = canonical[0]
    delta_position[0] = position[0] * [1, 0, 1]
    delta_rotation[1:] = inv[:-1] @ canonical[1:]
    delta_position[1:] = np.einsum(
        "tij,tj->ti", inv[:-1], (position[1:] - position[:-1]) * [1, 0, 1]
    )
    return np.concatenate(
        [_matrix_to_6d(local_rotation), local_position,
         _matrix_to_6d(delta_rotation), delta_position], axis=-1
    ).astype(np.float32)


def prepare_window(motion: np.ndarray, pose_yup: np.ndarray,
                   ground_offset: float | None) -> tuple[np.ndarray, np.ndarray]:
    if motion.shape != (T, FEATURES) or pose_yup.shape != (T, 4, 4):
        raise ValueError(f"unaligned shapes: {motion.shape}, {pose_yup.shape}")
    motion = motion.astype(np.float32, copy=True)
    pose = pose_yup.astype(np.float32, copy=True)
    if ground_offset is not None:
        motion[:, JOINT_HEIGHT] -= float(ground_offset)
        pose[:, 1, 3] -= float(ground_offset)
    motion[0, 270:279] = IDENTITY_DELTA
    camera = camera_trajectory18(pose)
    camera[0, 9:18] = IDENTITY_DELTA
    if not np.isfinite(motion).all() or not np.isfinite(camera).all():
        raise ValueError("non-finite motion/camera window")
    return motion, camera


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in os.uname().nodename:
        raise RuntimeError("Prepare the Nymeria baseline cache on an owned compute node")
    if args.out.exists():
        raise FileExistsError(f"cache destination already exists: {args.out}")
    manifest = args.root / "video/manifest_video.jsonl"
    split_path = args.root / "train_test_split.json"
    split = json.loads(split_path.read_text())
    train_names = sorted(split["train"])
    test_names = set(split["test"])
    if set(train_names) & test_names or len(test_names) != 71:
        raise ValueError("Nymeria train/test recording split changed")
    rng = np.random.default_rng(42)
    val_count = max(1, round(len(train_names) * args.validation_fraction))
    val_names = set(rng.choice(train_names, size=val_count, replace=False).tolist())
    training_names = set(train_names) - val_names
    build = args.out.with_name(args.out.name + f".building.{os.getpid()}")
    build.mkdir(parents=True, exist_ok=False)
    (build / "records").mkdir()
    rows: list[dict] = []
    rejected: list[dict] = []
    sums = np.zeros(FEATURES + 18, dtype=np.float64)
    squares = np.zeros_like(sums)
    n_frames = 0
    counts = {"train": 0, "val": 0, "test": 0}
    with manifest.open() as source:
        for line in source:
            rec = json.loads(line)
            uid = rec["uuid"]
            if uid in training_names:
                group = "train"
            elif uid in val_names:
                group = "val"
            elif uid in test_names:
                group = "test"
            else:
                continue
            motion_path = args.root / "uniego_rep_camhead_v1" / (uid + ".npz")
            camera_path = args.root / "camera_rgb" / (uid + ".npz")
            if not motion_path.is_file() or not camera_path.is_file():
                rejected.append({"uuid": uid, "reason": "missing motion or upright camera"})
                continue
            with np.load(motion_path, allow_pickle=False) as archive:
                motion = archive["features"].astype(np.float32)
                neutral = archive["neutral_joints"].astype(np.float32)
                motion_fps = int(archive["fps"])
            with np.load(camera_path, allow_pickle=False) as archive:
                cam_pos = archive["cam_world_pos_upright"].astype(np.float64)
                cam_rot = archive["cam_world_rot_upright"].astype(np.float64)
                camera_fps = int(archive["fps"])
            if motion_fps != FPS or camera_fps != FPS or motion.shape[1:] != (FEATURES,) or neutral.shape != (JOINTS, 3):
                rejected.append({"uuid": uid, "reason": "representation/FPS mismatch"})
                continue
            length = min(len(motion), len(cam_pos), len(cam_rot), int(rec["nb_frames"]))
            pose_zup = np.tile(np.eye(4, dtype=np.float32), (length, 1, 1))
            pose_zup[:, :3, :3] = cam_rot[:length].astype(np.float32)
            pose_zup[:, :3, 3] = cam_pos[:length].astype(np.float32)
            pose_yup = np.tile(np.eye(4, dtype=np.float32), (length, 1, 1))
            pose_yup[:, :3, :3] = (Y_FROM_Z @ cam_rot[:length]).astype(np.float32)
            pose_yup[:, :3, 3] = (Y_FROM_Z @ cam_pos[:length].T).T.astype(np.float32)
            key = hashlib.sha256(uid.encode()).hexdigest()[:24]
            np.save(build / "records" / f"{key}.motion.npy", motion[:length], allow_pickle=False)
            np.save(build / "records" / f"{key}.pose_yup.npy", pose_yup, allow_pickle=False)
            np.save(build / "records" / f"{key}.pose_zup.npy", pose_zup, allow_pickle=False)
            np.save(build / "records" / f"{key}.neutral.npy", neutral, allow_pickle=False)
            for window in rec.get("t2w_windows", []):
                if not window.get("usable") or not window.get("caption"):
                    continue
                start = int(window["start_frame"])
                end = min(int(window["end_frame"]), length)
                while start + T <= end:
                    try:
                        ground = window.get("ground_offset_y")
                        x_motion, x_camera = prepare_window(
                            motion[start:start + T], pose_yup[start:start + T], ground
                        )
                        combined = np.concatenate((x_motion, x_camera), axis=-1).astype(np.float64)
                        if group == "train":
                            sums += combined.sum(axis=0)
                            squares += (combined * combined).sum(axis=0)
                            n_frames += T
                        rows.append({"uuid": uid, "array": key, "start": start,
                                     "ground_offset_y": ground, "split": group,
                                     "caption": window["caption"]})
                        counts[group] += 1
                    except (ValueError, TypeError) as error:
                        rejected.append({"uuid": uid, "start": start,
                                         "reason": str(error)})
                    start += T
    if not n_frames or not all(counts.values()):
        raise RuntimeError(f"empty baseline split: {counts}")
    mean = sums / n_frames
    std = np.sqrt(np.maximum(squares / n_frames - mean * mean, 0.0))
    std[std < 1e-8] = 1.0
    np.savez(build / "train_stats.npz", motion_mean=mean[:FEATURES].astype(np.float32),
             motion_std=std[:FEATURES].astype(np.float32),
             camera_mean=mean[FEATURES:].astype(np.float32),
             camera_std=std[FEATURES:].astype(np.float32), train_frames=n_frames)
    payload = {
        "schema": 1, "source_manifest_sha256": _sha256(manifest),
        "source_split_sha256": _sha256(split_path), "motion_representation": "SOMA30 camhead-v1 283D",
        "camera_source": "camera_rgb upright", "camera_yup_from_zup": Y_FROM_Z.tolist(),
        "fps": FPS, "window": T, "train_recordings": len(training_names),
        "val_recordings": len(val_names), "test_recordings": len(test_names),
        "windows": counts, "rows": rows, "rejected": rejected,
    }
    (build / "index.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.rename(build, args.out)
    print(json.dumps({"stage": "complete", "cache": str(args.out),
                      "windows": counts, "rejected": len(rejected), "train_frames": n_frames}), flush=True)


if __name__ == "__main__":
    main()
