#!/usr/bin/env python
"""Evaluate native Phase-1 inverse and forward dynamics outputs.

The evaluator expects outputs from the official Cosmos inference entrypoint and
the held-out inputs produced by ``prep_test_eval.py``. The conditioned first
video frame is excluded from forward-dynamics metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torchvision.transforms import functional as transforms_F
from torchvision.transforms.functional import InterpolationMode

from native_phase_training.visualize_checkpoint import (
    _make_video_comparison,
    _plot_camera_comparison,
)
from native_phase_training.sanitize_prefix_inference_inputs import runtime_mode_matches
from nymeria_world.eval_inverse_dynamics import eval_seq, gt_abs


HORIZONS = {
    "early_frames_1_32": slice(0, 32),
    "middle_frames_33_64": slice(32, 64),
    "late_frames_65_96": slice(64, 96),
}
METRIC_KEYS_INVERSE = (
    "rot_deg",
    "trans_dir_cos",
    "scale_ratio",
    "trans_err_norm",
    "ate_m",
    "len_ratio",
)
METRIC_KEYS_FORWARD = ("psnr_db", "ssim", "lpips_alex")


def _read_jsonl(path: Path, expected_mode: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"no records in {path}")
    names: set[str] = set()
    for record in records:
        if record.get("model_mode") != expected_mode:
            raise ValueError(f"{path}: expected model_mode={expected_mode!r}")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: every record needs a non-empty name")
        if name in names:
            raise ValueError(f"{path}: duplicate sample name {name!r}")
        names.add(name)
    return records


def _base_name(name: str, mode: str) -> str:
    suffix = f"_{mode}"
    if not name.endswith(suffix):
        raise ValueError(f"sample name {name!r} must end with {suffix!r}")
    return name[: -len(suffix)]


def _load_successful_output(inference_root: Path, record: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    sample_dir = inference_root / record["name"]
    output_path = sample_dir / "sample_outputs.json"
    if not output_path.is_file():
        raise FileNotFoundError(f"missing inference output: {output_path}")
    payload = json.loads(output_path.read_text())
    if payload.get("status") != "success":
        raise RuntimeError(f"failed inference output {output_path}: {payload.get('message')}")
    if not runtime_mode_matches(
        actual_mode=payload.get("args", {}).get("model_mode"),
        canonical_mode=record["model_mode"],
    ):
        raise ValueError(f"mode mismatch in {output_path}")
    return sample_dir, payload


def _read_video_rgb(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return np.stack(frames)


def _resize_gt_like_native(gt: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Apply Cosmos's aspect-preserving bicubic resize and right/bottom pad."""
    if gt.ndim != 4 or gt.shape[-1] != 3:
        raise ValueError(f"expected GT video [T,H,W,3], got {gt.shape}")
    original_height, original_width = gt.shape[1:3]
    scaling_ratio = min(target_width / original_width, target_height / original_height, 1.0)
    resized_height = int(scaling_ratio * original_height + 0.5)
    resized_width = int(scaling_ratio * original_width + 0.5)
    tensor = torch.from_numpy(gt).permute(0, 3, 1, 2)
    if (resized_height, resized_width) != (original_height, original_width):
        tensor = transforms_F.resize(
            tensor,
            size=[resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    padding_right = target_width - resized_width
    padding_bottom = target_height - resized_height
    if padding_right or padding_bottom:
        padding_mode = "replicate" if padding_right >= resized_width or padding_bottom >= resized_height else "reflect"
        tensor = torch.nn.functional.pad(tensor, [0, padding_right, 0, padding_bottom], mode=padding_mode)
    return tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _aggregate(rows: dict[str, dict[str, float]], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows.values()], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class LPIPSAlex:
    def __init__(self, device: str, batch_size: int) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, gt: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        for start in range(0, len(gt), self.batch_size):
            stop = start + self.batch_size
            gt_tensor = torch.from_numpy(gt[start:stop]).permute(0, 3, 1, 2).float().div_(255.0)
            pred_tensor = torch.from_numpy(predicted[start:stop]).permute(0, 3, 1, 2).float().div_(255.0)
            score = self.model(
                pred_tensor.to(self.device, non_blocking=True),
                gt_tensor.to(self.device, non_blocking=True),
                normalize=True,
            )
            values.append(score.flatten().float().cpu().numpy())
        return np.concatenate(values)


def _frame_metrics(gt: np.ndarray, predicted: np.ndarray, lpips_metric: LPIPSAlex) -> dict[str, np.ndarray]:
    if gt.shape != predicted.shape:
        raise ValueError(f"GT/generated video shape mismatch: {gt.shape} vs {predicted.shape}")
    if gt.ndim != 4 or gt.shape[-1] != 3:
        raise ValueError(f"expected video [T,H,W,3], got {gt.shape}")

    gt_float = gt.astype(np.float32) / 255.0
    predicted_float = predicted.astype(np.float32) / 255.0
    mse = np.square(gt_float - predicted_float).mean(axis=(1, 2, 3))
    psnr = np.where(mse > 0.0, 10.0 * np.log10(1.0 / np.maximum(mse, 1.0e-12)), 100.0)
    ssim = np.asarray(
        [
            structural_similarity(g, p, channel_axis=2, data_range=1.0)
            for g, p in zip(gt_float, predicted_float, strict=True)
        ],
        dtype=np.float64,
    )
    lpips_values = lpips_metric(gt, predicted).astype(np.float64)
    return {"psnr_db": psnr.astype(np.float64), "ssim": ssim, "lpips_alex": lpips_values}


def _summarize_frame_metrics(values: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {key: float(metric.mean()) for key, metric in values.items()}
    result["horizons"] = {
        horizon: {key: float(metric[frame_slice].mean()) for key, metric in values.items()}
        for horizon, frame_slice in HORIZONS.items()
    }
    result["evaluated_frames"] = int(len(next(iter(values.values()))))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--visualize-limit", type=int, default=0, help="0 visualizes every sequence")
    parser.add_argument("--lpips-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.max_samples < 0 or args.visualize_limit < 0 or args.expected_count < 0:
        parser.error("count arguments must be non-negative")

    output_root = args.out or args.inference_root.parent / "analysis"
    inverse_viz = output_root / "viz" / "inverse_dynamics"
    forward_viz = output_root / "viz" / "forward_dynamics"
    inverse_viz.mkdir(parents=True, exist_ok=True)
    forward_viz.mkdir(parents=True, exist_ok=True)

    inverse_records = _read_jsonl(args.eval_root / "invdyn_input.jsonl", "inverse_dynamics")
    forward_records = _read_jsonl(args.eval_root / "fd_input.jsonl", "forward_dynamics")
    if args.expected_count and (len(inverse_records) != args.expected_count or len(forward_records) != args.expected_count):
        raise ValueError(
            f"expected {args.expected_count} records per mode, got "
            f"inverse={len(inverse_records)}, forward={len(forward_records)}"
        )
    if args.max_samples:
        inverse_records = inverse_records[: args.max_samples]
        forward_records = forward_records[: args.max_samples]

    inverse_rows: dict[str, dict[str, float]] = {}
    for index, record in enumerate(inverse_records):
        sample_dir, payload = _load_successful_output(args.inference_root, record)
        base_name = _base_name(record["name"], "inverse_dynamics")
        action = np.asarray(payload["outputs"][0]["content"].get("action"), dtype=np.float64)
        if action.shape != (96, 9) or not np.isfinite(action).all():
            raise ValueError(f"{sample_dir}: expected finite action [96,9], got {action.shape}")
        gt_camera = args.eval_root / "samples" / base_name / "gt_camera_cosmos.npz"
        inverse_rows[base_name] = eval_seq(action, gt_abs(gt_camera))
        if args.visualize_limit == 0 or index < args.visualize_limit:
            _plot_camera_comparison(
                action=action,
                gt_camera=gt_camera,
                output=inverse_viz / f"{base_name}.png",
                title=f"{base_name} inverse_dynamics",
                n_cameras=7,
            )
        print(f"[full71-eval] inverse {index + 1}/{len(inverse_records)}: {base_name}", flush=True)

    inverse_payload = {
        "n": len(inverse_rows),
        "aggregate": _aggregate(inverse_rows, METRIC_KEYS_INVERSE),
        "per_sequence": inverse_rows,
    }
    _write_json(output_root / "invdyn_metrics.json", inverse_payload)

    lpips_metric = LPIPSAlex(args.lpips_device, args.lpips_batch_size)
    forward_rows: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(forward_records):
        sample_dir, _ = _load_successful_output(args.inference_root, record)
        base_name = _base_name(record["name"], "forward_dynamics")
        gt_video = args.eval_root / "samples" / base_name / "gt_clip.mp4"
        generated_video = sample_dir / "vision.mp4"
        gt_frames = _read_video_rgb(gt_video)
        generated_frames = _read_video_rgb(generated_video)
        if len(gt_frames) != 97 or len(generated_frames) != 97:
            raise ValueError(
                f"{base_name}: expected 97 frames, got GT={len(gt_frames)} generated={len(generated_frames)}"
            )
        gt_frames = _resize_gt_like_native(gt_frames, generated_frames.shape[1], generated_frames.shape[2])
        values = _frame_metrics(gt_frames[1:], generated_frames[1:], lpips_metric)
        forward_rows[base_name] = _summarize_frame_metrics(values)
        if args.visualize_limit == 0 or index < args.visualize_limit:
            _make_video_comparison(
                gt_video=gt_video,
                generated_video=generated_video,
                output=forward_viz / f"{base_name}.mp4",
                label="forward_dynamics",
                prefix_length=1,
            )
        print(f"[full71-eval] forward {index + 1}/{len(forward_records)}: {base_name}", flush=True)

    forward_scalar_rows = {
        name: {key: float(row[key]) for key in METRIC_KEYS_FORWARD} for name, row in forward_rows.items()
    }
    horizon_aggregate = {
        horizon: {
            key: {
                "mean": float(np.mean([row["horizons"][horizon][key] for row in forward_rows.values()])),
                "median": float(np.median([row["horizons"][horizon][key] for row in forward_rows.values()])),
            }
            for key in METRIC_KEYS_FORWARD
        }
        for horizon in HORIZONS
    }
    forward_payload = {
        "n": len(forward_rows),
        "conditioned_frame_excluded": True,
        "evaluated_frame_range": [1, 96],
        "gt_preprocessing": "native aspect-preserving bicubic antialias resize plus right/bottom reflection pad",
        "aggregate": _aggregate(forward_scalar_rows, METRIC_KEYS_FORWARD),
        "horizon_aggregate": horizon_aggregate,
        "per_sequence": forward_rows,
    }
    _write_json(output_root / "forward_metrics.json", forward_payload)
    _write_json(
        output_root / "COMPLETE.json",
        {
            "inverse_sequences": len(inverse_rows),
            "forward_sequences": len(forward_rows),
            "inference_root": str(args.inference_root),
            "eval_root": str(args.eval_root),
            "lpips_backbone": "alex",
            "conditioned_frame_excluded": True,
        },
    )
    print(f"[full71-eval] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
