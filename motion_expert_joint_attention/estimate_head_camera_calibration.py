#!/usr/bin/env python3
"""Estimate the train-split rigid SOMA-Head -> upright-camera calibration.

This intentionally uses only train sequences. Absolute rotations estimate the fixed frame
rotation; synchronized relative actions estimate the camera-origin lever arm. Absolute
translations are never used because some proportional-UniEgo sequences do not preserve a
trustworthy common world origin with the Aria trajectory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from decode_uniego_torch import cont6d_to_matrix, decode_transforms
from head_camera_alignment import HEAD_JOINT_IDX
from uniego_layout import canonicalize_frame0


DEFAULT_MANIFEST = "/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl"
DEFAULT_SPLIT = "/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json"
DEFAULT_UNIEGO = "/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep"
DEFAULT_CAMERA = "/weka/jungbin/nymeriaplus_kimodo_proportional/camera_rgb"

ARIA_Z_UP_TO_KIMODO_Y_UP = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=torch.float64,
)


def _project_so3(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(matrix)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def _rotation_angle_deg(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine) * (180.0 / torch.pi)


def _summary(values: torch.Tensor) -> dict:
    values = values.detach().double().cpu()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _discover_windows(manifest: str, split_file: str) -> list[tuple[str, int]]:
    train_uuids = set(json.load(open(split_file))["train"])
    windows = []
    with open(manifest) as f:
        for line in f:
            record = json.loads(line)
            uuid = record.get("uuid")
            if uuid not in train_uuids:
                continue
            starts = [
                int(window["start_frame"])
                for window in record.get("t2w_windows", [])
                if window.get("usable", False) and window.get("caption")
            ]
            if starts:
                windows.append((uuid, starts[0]))
    return sorted(windows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT)
    parser.add_argument("--uniego-root", default=DEFAULT_UNIEGO)
    parser.add_argument("--camera-root", default=DEFAULT_CAMERA)
    parser.add_argument("--window-frames", type=int, default=97)
    parser.add_argument("--orientation-stride", type=int, default=4)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    windows = _discover_windows(args.manifest, args.split_file)
    if args.max_sequences > 0:
        windows = windows[: args.max_sequences]

    frame_rotations = []
    head_relative_rotations = []
    head_relative_translations = []
    camera_relative_rotations = []
    camera_relative_translations = []
    used = []
    skipped = []

    for index, (uuid, start) in enumerate(windows):
        uniego_path = Path(args.uniego_root) / f"{uuid}.npz"
        camera_path = Path(args.camera_root) / f"{uuid}.npz"
        try:
            with np.load(uniego_path) as data:
                features = data["features"][start:start + args.window_frames].astype(np.float64)
            with np.load(camera_path) as data:
                camera_rotation_world = data["cam_world_rot_upright"][
                    start:start + args.window_frames
                ].astype(np.float64)
                camera_action = data["cam_action_upright_k1"][
                    start:start + args.window_frames - 1
                ].astype(np.float64)
            n_frames = min(len(features), len(camera_rotation_world), len(camera_action) + 1)
            if n_frames < 2:
                raise ValueError(f"short aligned window ({n_frames} frames)")
            features = features[:n_frames]
            camera_rotation_world = camera_rotation_world[:n_frames]
            camera_action = camera_action[:n_frames - 1]
            if not (
                np.isfinite(features).all()
                and np.isfinite(camera_rotation_world).all()
                and np.isfinite(camera_action).all()
            ):
                raise ValueError("non-finite aligned data")

            raw = torch.from_numpy(features).double().unsqueeze(0)
            raw_head = decode_transforms(raw)[0, :, HEAD_JOINT_IDX]
            world_camera_rotation = (
                ARIA_Z_UP_TO_KIMODO_Y_UP
                @ torch.from_numpy(camera_rotation_world).double()
            )
            x_rotation = raw_head[:, :3, :3].transpose(-1, -2) @ world_camera_rotation
            frame_rotations.append(x_rotation[:: args.orientation_stride])

            canonical = torch.from_numpy(canonicalize_frame0(features)).double().unsqueeze(0)
            head = decode_transforms(canonical)[0, :, HEAD_JOINT_IDX]
            head_rotation = head[:, :3, :3]
            head_position = head[:, :3, 3]
            relative_head_rotation = (
                head_rotation[:-1].transpose(-1, -2) @ head_rotation[1:]
            )
            relative_head_translation = (
                head_rotation[:-1].transpose(-1, -2)
                @ (head_position[1:] - head_position[:-1]).unsqueeze(-1)
            ).squeeze(-1)
            camera_action_t = torch.from_numpy(camera_action).double()
            head_relative_rotations.append(relative_head_rotation)
            head_relative_translations.append(relative_head_translation)
            camera_relative_rotations.append(cont6d_to_matrix(camera_action_t[:, 3:9]))
            camera_relative_translations.append(camera_action_t[:, :3])
            used.append(uuid)
        except Exception as error:  # noqa: BLE001 - calibration must report every exclusion.
            skipped.append({"uuid": uuid, "start": start, "error": f"{type(error).__name__}: {error}"})
        if (index + 1) % 100 == 0:
            print(f"[head-camera-calibration] scanned {index + 1}/{len(windows)}", flush=True)

    if not used:
        raise RuntimeError("no usable train sequences for head-camera calibration")

    x_rotations = torch.cat(frame_rotations, dim=0)
    rotation = _project_so3(x_rotations.mean(dim=0))
    rotation_keep = torch.ones(len(x_rotations), dtype=torch.bool)
    for _ in range(5):
        rotation_error = _rotation_angle_deg(rotation.T @ x_rotations)
        rotation_keep = rotation_error <= torch.quantile(rotation_error, 0.90)
        rotation = _project_so3(x_rotations[rotation_keep].mean(dim=0))
    rotation_error = _rotation_angle_deg(rotation.T @ x_rotations)

    head_r = torch.cat(head_relative_rotations, dim=0)
    head_t = torch.cat(head_relative_translations, dim=0)
    camera_r = torch.cat(camera_relative_rotations, dim=0)
    camera_t = torch.cat(camera_relative_translations, dim=0)
    eye = torch.eye(3, dtype=torch.float64)
    equation_a = head_r - eye
    equation_b = (rotation @ camera_t.unsqueeze(-1)).squeeze(-1) - head_t

    # Exclude implausible per-frame jumps before robust least squares. The thresholds are far
    # above normal 20-FPS motion and only remove broken synchronization/fits.
    plausible = (
        torch.isfinite(equation_a).all(dim=(-1, -2))
        & torch.isfinite(equation_b).all(dim=-1)
        & (head_t.norm(dim=-1) < 0.25)
        & (camera_t.norm(dim=-1) < 0.25)
        & (_rotation_angle_deg(head_r) < 30.0)
    )
    lever_keep = plausible.clone()
    lever = torch.zeros(3, dtype=torch.float64)
    for _ in range(5):
        lever = torch.linalg.lstsq(
            equation_a[lever_keep].reshape(-1, 3),
            equation_b[lever_keep].reshape(-1, 1),
        ).solution[:, 0]
        residual = (equation_a @ lever - equation_b).norm(dim=-1)
        cutoff = torch.quantile(residual[plausible], 0.90)
        lever_keep = plausible & (residual <= cutoff)

    predicted_rotation = rotation.T @ head_r @ rotation
    predicted_translation_no_lever = (
        rotation.T @ head_t.unsqueeze(-1)
    ).squeeze(-1)
    predicted_translation = (
        rotation.T
        @ (head_t + ((head_r - eye) @ lever.view(3, 1)).squeeze(-1)).unsqueeze(-1)
    ).squeeze(-1)
    rotation_action_error = _rotation_angle_deg(
        predicted_rotation.transpose(-1, -2) @ camera_r
    )
    translation_error = (predicted_translation - camera_t).norm(dim=-1)
    translation_error_no_lever = (predicted_translation_no_lever - camera_t).norm(dim=-1)

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "rotation_head_to_upright_camera": rotation.float().tolist(),
        "camera_origin_in_head_m": lever.float().tolist(),
        "coordinate_contract": {
            "motion": "SOMA-30 UniEgo Head joint index 6, Kimodo Y-up",
            "camera": "upright RGB/OpenCV frame used by cam_action_upright_k1",
            "action": "inv(T_t) @ T_t+1; translation plus rotation columns 0/1",
            "absolute_translation_used": False,
            "world_basis_for_rotation_only": "kimodo(x,y,z)=(aria_x,aria_z,-aria_y)",
        },
        "sources": {
            "manifest": args.manifest,
            "split_file": args.split_file,
            "uniego_root": args.uniego_root,
            "camera_root": args.camera_root,
        },
        "counts": {
            "candidate_sequences": len(windows),
            "used_sequences": len(used),
            "skipped_sequences": len(skipped),
            "orientation_samples": len(x_rotations),
            "orientation_samples_kept": int(rotation_keep.sum()),
            "relative_action_samples": len(head_r),
            "lever_samples_kept": int(lever_keep.sum()),
        },
        "train_fit": {
            "head_to_camera_rotation_deviation_deg": _summary(rotation_error),
            "relative_rotation_error_deg": _summary(rotation_action_error[plausible]),
            "relative_translation_error_m": _summary(translation_error[plausible]),
            "relative_translation_error_without_lever_m": _summary(
                translation_error_no_lever[plausible]
            ),
        },
        "skipped": skipped,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(json.dumps({k: payload[k] for k in ("counts", "train_fit")}, indent=2))
    print(f"[head-camera-calibration] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
