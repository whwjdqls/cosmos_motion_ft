#!/usr/bin/env python3
"""Estimate the train-split rigid SOMA-Head -> upright-camera calibration.

This intentionally uses only train sequences. Absolute rotations estimate the fixed frame
rotation; synchronized relative actions estimate the camera-origin lever arm. Absolute
translations are not used by this historical production fit. A later source audit established
that clean intervals normally do share a metric world frame and a stable translation lever, but
also found sparse upstream discontinuities and >0.5 m registration failures; keeping this fitter
relative preserves its checkpoint contract and avoids those unfiltered intervals.
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


def load_calibration_window(
    uniego_path: str | Path,
    camera_path: str | Path,
    start: int,
    window_frames: int,
    orientation_stride: int,
) -> dict[str, torch.Tensor]:
    """Load one synchronized GT motion/camera window for rigid calibration fitting."""
    with np.load(uniego_path) as data:
        features = data["features"].astype(np.float64)
    with np.load(camera_path) as data:
        camera_rotation_world = data["cam_world_rot_upright"].astype(np.float64)
        camera_action = data["cam_action_upright_k1"].astype(np.float64)
    return calibration_sample_from_arrays(
        features,
        camera_rotation_world,
        camera_action,
        start,
        window_frames,
        orientation_stride,
    )


def calibration_sample_from_arrays(
    features: np.ndarray,
    camera_rotation_world: np.ndarray,
    camera_action: np.ndarray,
    start: int,
    window_frames: int,
    orientation_stride: int,
) -> dict[str, torch.Tensor]:
    """Build one calibration sample from already-loaded sequence arrays."""
    if start < 0:
        raise ValueError(f"negative calibration-window start {start}")
    feature_window = features[start:start + window_frames]
    camera_rotation_window = camera_rotation_world[start:start + window_frames]
    camera_action_window = camera_action[start:start + window_frames - 1]
    n_frames = min(
        len(feature_window), len(camera_rotation_window), len(camera_action_window) + 1
    )
    if n_frames < 2:
        raise ValueError(f"short aligned window ({n_frames} frames)")
    feature_window = feature_window[:n_frames]
    camera_rotation_window = camera_rotation_window[:n_frames]
    camera_action_window = camera_action_window[:n_frames - 1]
    # UniEgo canon_delta[0] is absolute only at sequence frame 0; all later rows are relative.
    # Decode the prefix before slicing absolute head orientations. Decoding a nonzero-start slice
    # directly incorrectly treats delta[start] as an absolute world transform.
    feature_prefix = features[:start + n_frames]
    if not (
        np.isfinite(feature_prefix).all()
        and np.isfinite(camera_rotation_window).all()
        and np.isfinite(camera_action_window).all()
    ):
        raise ValueError("non-finite aligned data")

    raw = torch.from_numpy(feature_prefix).double().unsqueeze(0)
    raw_head = decode_transforms(raw)[0, start:start + n_frames, HEAD_JOINT_IDX]
    world_camera_rotation = (
        ARIA_Z_UP_TO_KIMODO_Y_UP
        @ torch.from_numpy(camera_rotation_window).double()
    )
    frame_rotations = (
        raw_head[:, :3, :3].transpose(-1, -2) @ world_camera_rotation
    )[::orientation_stride]

    canonical = torch.from_numpy(canonicalize_frame0(feature_window)).double().unsqueeze(0)
    head = decode_transforms(canonical)[0, :, HEAD_JOINT_IDX]
    head_rotation = head[:, :3, :3]
    head_position = head[:, :3, 3]
    head_relative_rotation = head_rotation[:-1].transpose(-1, -2) @ head_rotation[1:]
    head_relative_translation = (
        head_rotation[:-1].transpose(-1, -2)
        @ (head_position[1:] - head_position[:-1]).unsqueeze(-1)
    ).squeeze(-1)
    camera_action_t = torch.from_numpy(camera_action_window).double()
    return {
        "frame_rotations": frame_rotations,
        "head_relative_rotations": head_relative_rotation,
        "head_relative_translations": head_relative_translation,
        "camera_relative_rotations": cont6d_to_matrix(camera_action_t[:, 3:9]),
        "camera_relative_translations": camera_action_t[:, :3],
    }


def fit_head_camera_calibration(
    samples: list[dict[str, torch.Tensor]],
    *,
    rotation_override: torch.Tensor | None = None,
    lever_override: torch.Tensor | None = None,
) -> dict:
    """Robustly fit one rigid head-to-camera transform from synchronized GT samples."""
    if not samples:
        raise ValueError("cannot fit head-camera calibration without samples")
    x_rotations = torch.cat([sample["frame_rotations"] for sample in samples], dim=0)
    if rotation_override is None:
        rotation = _project_so3(x_rotations.mean(dim=0))
        rotation_keep = torch.ones(len(x_rotations), dtype=torch.bool)
        for _ in range(5):
            rotation_error = _rotation_angle_deg(rotation.T @ x_rotations)
            rotation_keep = rotation_error <= torch.quantile(rotation_error, 0.90)
            rotation = _project_so3(x_rotations[rotation_keep].mean(dim=0))
    else:
        rotation = _project_so3(rotation_override.detach().double().cpu())
        rotation_error = _rotation_angle_deg(rotation.T @ x_rotations)
        rotation_keep = rotation_error <= torch.quantile(rotation_error, 0.90)
    rotation_error = _rotation_angle_deg(rotation.T @ x_rotations)

    head_r = torch.cat([sample["head_relative_rotations"] for sample in samples], dim=0)
    head_t = torch.cat([sample["head_relative_translations"] for sample in samples], dim=0)
    camera_r = torch.cat([sample["camera_relative_rotations"] for sample in samples], dim=0)
    camera_t = torch.cat([sample["camera_relative_translations"] for sample in samples], dim=0)
    eye = torch.eye(3, dtype=torch.float64)
    equation_a = head_r - eye
    equation_b = (rotation @ camera_t.unsqueeze(-1)).squeeze(-1) - head_t

    # These thresholds are far above normal 20-FPS motion and reject only broken
    # synchronization/fits before robust least squares.
    plausible = (
        torch.isfinite(equation_a).all(dim=(-1, -2))
        & torch.isfinite(equation_b).all(dim=-1)
        & (head_t.norm(dim=-1) < 0.25)
        & (camera_t.norm(dim=-1) < 0.25)
        & (_rotation_angle_deg(head_r) < 30.0)
    )
    if int(plausible.sum()) < 3:
        raise ValueError(
            f"insufficient plausible relative actions ({int(plausible.sum())}/{len(plausible)})"
        )
    if lever_override is None:
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
    else:
        lever = lever_override.detach().double().cpu()
        if lever.shape != (3,) or not torch.isfinite(lever).all():
            raise ValueError(f"invalid lever override: shape={tuple(lever.shape)}")
        lever_keep = plausible.clone()

    predicted_rotation = rotation.T @ head_r @ rotation
    predicted_translation_no_lever = (rotation.T @ head_t.unsqueeze(-1)).squeeze(-1)
    predicted_translation = (
        rotation.T
        @ (head_t + ((head_r - eye) @ lever.view(3, 1)).squeeze(-1)).unsqueeze(-1)
    ).squeeze(-1)
    rotation_action_error = _rotation_angle_deg(
        predicted_rotation.transpose(-1, -2) @ camera_r
    )
    translation_error = (predicted_translation - camera_t).norm(dim=-1)
    translation_error_no_lever = (predicted_translation_no_lever - camera_t).norm(dim=-1)
    return {
        "rotation": rotation,
        "lever": lever,
        "counts": {
            "orientation_samples": len(x_rotations),
            "orientation_samples_kept": int(rotation_keep.sum()),
            "relative_action_samples": len(head_r),
            "plausible_relative_action_samples": int(plausible.sum()),
            "lever_samples_kept": int(lever_keep.sum()),
        },
        "fit": {
            "head_to_camera_rotation_deviation_deg": _summary(rotation_error),
            "relative_rotation_error_deg": _summary(rotation_action_error[plausible]),
            "relative_translation_error_m": _summary(translation_error[plausible]),
            "relative_translation_error_without_lever_m": _summary(
                translation_error_no_lever[plausible]
            ),
        },
    }


def optimize_head_camera_transform_from_relative_actions(
    samples: list[dict[str, torch.Tensor]],
    initial_rotation: torch.Tensor,
    initial_lever: torch.Tensor,
    *,
    max_samples: int = 50_000,
    max_rotation_correction_deg: float = 15.0,
    translation_scale_m: float = 0.02,
    rotation_scale_deg: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit an oracle rigid transform using the same normalized geometry loss as Phase 3."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    head_rotation = torch.cat(
        [sample["head_relative_rotations"] for sample in samples], dim=0
    ).double()
    head_translation = torch.cat(
        [sample["head_relative_translations"] for sample in samples], dim=0
    ).double()
    camera_rotation = torch.cat(
        [sample["camera_relative_rotations"] for sample in samples], dim=0
    ).double()
    camera_translation = torch.cat(
        [sample["camera_relative_translations"] for sample in samples], dim=0
    ).double()
    valid = (
        torch.isfinite(head_rotation).all(dim=(-1, -2))
        & torch.isfinite(head_translation).all(dim=-1)
        & torch.isfinite(camera_rotation).all(dim=(-1, -2))
        & torch.isfinite(camera_translation).all(dim=-1)
        & (head_translation.norm(dim=-1) < 0.25)
        & (camera_translation.norm(dim=-1) < 0.25)
        & (_rotation_angle_deg(head_rotation) < 30.0)
        & (_rotation_angle_deg(camera_rotation) < 30.0)
    )
    indices = torch.where(valid)[0]
    if len(indices) < 3:
        raise ValueError(
            f"insufficient relative actions for oracle optimization ({len(indices)})"
        )
    if len(indices) > max_samples:
        selection = torch.linspace(0, len(indices) - 1, max_samples).round().long()
        indices = indices[selection]
    head_r = head_rotation[indices].numpy()
    head_t = head_translation[indices].numpy()
    camera_r = camera_rotation[indices].numpy()
    camera_t = camera_translation[indices].numpy()
    initial_r = _project_so3(initial_rotation.detach().double().cpu()).numpy()
    initial_l = initial_lever.detach().double().cpu().numpy()
    eye = np.eye(3, dtype=np.float64)
    rotation_chord_scale = 2.0 * np.sqrt(2.0) * np.sin(
        0.5 * np.deg2rad(float(rotation_scale_deg))
    )

    def smooth_l1(values: np.ndarray) -> np.ndarray:
        absolute = np.abs(values)
        return np.where(absolute < 1.0, 0.5 * values ** 2, absolute - 0.5)

    def metrics(parameters: np.ndarray) -> tuple[float, dict]:
        correction = Rotation.from_rotvec(parameters[:3]).as_matrix()
        rotation = initial_r @ correction
        lever = parameters[3:]
        predicted_rotation = np.einsum(
            "ij,njk,kl->nil", rotation.T, head_r, rotation, optimize=True
        )
        lever_velocity = np.einsum(
            "nij,j->ni", head_r - eye, lever, optimize=True
        )
        predicted_translation = np.einsum(
            "ij,nj->ni", rotation.T, head_t + lever_velocity, optimize=True
        )
        translation_delta = predicted_translation - camera_t
        rotation_chord = np.linalg.norm(
            predicted_rotation - camera_r, axis=(-2, -1)
        )
        translation_loss = float(np.mean(smooth_l1(
            translation_delta / float(translation_scale_m)
        )))
        rotation_loss = float(np.mean(smooth_l1(
            rotation_chord / rotation_chord_scale
        )))
        relative = np.einsum(
            "nij,njk->nik",
            np.swapaxes(predicted_rotation, -1, -2),
            camera_r,
            optimize=True,
        )
        cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
        summary = {
            "loss": translation_loss + rotation_loss,
            "translation_loss": translation_loss,
            "rotation_loss": rotation_loss,
            "translation_error_m": float(np.mean(np.linalg.norm(translation_delta, axis=-1))),
            "rotation_error_deg": float(np.mean(np.rad2deg(np.arccos(cosine)))),
        }
        return summary["loss"], summary

    initial_parameters = np.concatenate([np.zeros(3), initial_l])
    initial_objective, initial_summary = metrics(initial_parameters)
    rotation_bound = np.deg2rad(float(max_rotation_correction_deg))
    bounds = [(-rotation_bound, rotation_bound)] * 3 + [(-0.5, 0.5)] * 3
    optimized = minimize(
        lambda parameters: metrics(parameters)[0],
        initial_parameters,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8, "maxls": 30},
    )
    final_objective, final_summary = metrics(optimized.x)
    if not optimized.success:
        raise RuntimeError(f"oracle joint transform fit failed: {optimized.message}")
    if final_objective > initial_objective + 1e-10:
        raise RuntimeError("oracle joint transform fit worsened its initialization")
    correction = Rotation.from_rotvec(optimized.x[:3]).as_matrix()
    rotation = torch.from_numpy(initial_r @ correction).double()
    lever = torch.from_numpy(optimized.x[3:].copy()).double()
    return rotation, lever, {
        "method": (
            "bounded joint optimization of the Phase-3 normalized SmoothL1 translation and "
            "rotation-chord geometry losses"
        ),
        "initialization": "train-global rigid calibration",
        "samples": len(indices),
        "max_samples": int(max_samples),
        "max_rotation_correction_deg_per_axis": float(max_rotation_correction_deg),
        "translation_scale_m": float(translation_scale_m),
        "rotation_scale_deg": float(rotation_scale_deg),
        "correction_rotvec_deg": np.rad2deg(optimized.x[:3]).tolist(),
        "correction_angle_deg": float(np.linalg.norm(np.rad2deg(optimized.x[:3]))),
        "initial": initial_summary,
        "final": final_summary,
        "optimizer_iterations": int(optimized.nit),
        "optimizer_evaluations": int(optimized.nfev),
    }


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
            sample = load_calibration_window(
                uniego_path,
                camera_path,
                start,
                args.window_frames,
                args.orientation_stride,
            )
            frame_rotations.append(sample["frame_rotations"])
            head_relative_rotations.append(sample["head_relative_rotations"])
            head_relative_translations.append(sample["head_relative_translations"])
            camera_relative_rotations.append(sample["camera_relative_rotations"])
            camera_relative_translations.append(sample["camera_relative_translations"])
            used.append(uuid)
        except Exception as error:  # noqa: BLE001 - calibration must report every exclusion.
            skipped.append({"uuid": uuid, "start": start, "error": f"{type(error).__name__}: {error}"})
        if (index + 1) % 100 == 0:
            print(f"[head-camera-calibration] scanned {index + 1}/{len(windows)}", flush=True)

    if not used:
        raise RuntimeError("no usable train sequences for head-camera calibration")

    fit_result = fit_head_camera_calibration([
        {
            "frame_rotations": frame_rotation,
            "head_relative_rotations": head_rotation,
            "head_relative_translations": head_translation,
            "camera_relative_rotations": camera_rotation,
            "camera_relative_translations": camera_translation,
        }
        for frame_rotation, head_rotation, head_translation, camera_rotation, camera_translation
        in zip(
            frame_rotations,
            head_relative_rotations,
            head_relative_translations,
            camera_relative_rotations,
            camera_relative_translations,
        )
    ])
    rotation = fit_result["rotation"]
    lever = fit_result["lever"]

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
            **fit_result["counts"],
        },
        "train_fit": fit_result["fit"],
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
