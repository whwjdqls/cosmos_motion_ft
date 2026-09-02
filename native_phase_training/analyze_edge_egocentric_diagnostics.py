#!/usr/bin/env python
"""Measure temporal and background-motion diagnostics for Edge ego runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _read_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return np.stack(frames), fps


def _global_flow(frames: np.ndarray) -> np.ndarray:
    gray = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (128, 128))
        for frame in frames
    ]
    medians = []
    for previous, current in zip(gray[:-1], gray[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        medians.append(float(np.median(np.linalg.norm(flow, axis=2))))
    return np.asarray(medians, dtype=np.float64)


def _resize_like(frames: np.ndarray, target: np.ndarray) -> np.ndarray:
    if frames.shape[1:3] == target.shape[1:3]:
        return frames
    target_h, target_w = target.shape[1:3]
    return np.stack([cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC) for frame in frames])


def _metrics(frames: np.ndarray, fps: float, gt: np.ndarray, gt_flow_mean: float) -> dict[str, Any]:
    if len(frames) != 97 or abs(fps - 20.0) > 1e-3:
        raise ValueError(f"expected 97 frames at 20 FPS, got {len(frames)} at {fps}")
    frames_float = frames.astype(np.float32) / 255.0
    adjacent = np.abs(np.diff(frames_float, axis=0)).mean(axis=(1, 2, 3))
    flow = _global_flow(frames)
    gt_resized = _resize_like(gt, frames)
    mse = np.square(frames_float[1:] - gt_resized[1:].astype(np.float32) / 255.0).mean(axis=(1, 2, 3))
    psnr = -10.0 * np.log10(np.maximum(mse, 1e-12))
    ssim = [
        structural_similarity(gt_resized[index], frames[index], channel_axis=2, data_range=255)
        for index in range(1, len(frames))
    ]
    return {
        "frames": len(frames),
        "fps": fps,
        "rgb_adjacent_delta_mean": float(adjacent.mean()),
        "rgb_adjacent_delta_max": float(adjacent.max()),
        "rgb_adjacent_delta_max_frame": int(adjacent.argmax() + 1),
        "global_flow_px_at_128_mean": float(flow.mean()),
        "global_flow_px_at_128_p95": float(np.percentile(flow, 95)),
        "global_flow_px_at_128_max": float(flow.max()),
        "global_flow_ratio_to_gt": float(flow.mean() / gt_flow_mean),
        "global_flow_early_1_32": float(flow[:32].mean()),
        "global_flow_middle_33_64": float(flow[32:64].mean()),
        "global_flow_late_65_96": float(flow[64:].mean()),
        "frame_aligned_psnr_db": float(psnr.mean()),
        "frame_aligned_ssim": float(np.mean(ssim)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.inference_root / "inference_inputs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    gt_path = Path(manifest["gt_video"])
    gt, gt_fps = _read_video(gt_path)
    gt_flow = _global_flow(gt)
    result: dict[str, Any] = {
        "contract": {
            "conditioned_frame_excluded_from_psnr_ssim": True,
            "flow_definition": "median Farneback magnitude per adjacent frame after resize to 128x128",
            "frame_aligned_metrics_warning": "I2V is stochastic; PSNR/SSIM do not measure prompt adherence.",
        },
        "gt": _metrics(gt, gt_fps, gt, float(gt_flow.mean())),
        "variants": {},
    }
    for variant in manifest["variants"]:
        video_path = args.inference_root / variant["name"] / "vision.mp4"
        frames, fps = _read_video(video_path)
        result["variants"][variant["name"]] = {
            "model_mode": variant["model_mode"],
            "video": str(video_path.resolve()),
            **_metrics(frames, fps, gt, float(gt_flow.mean())),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
