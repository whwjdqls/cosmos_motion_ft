#!/usr/bin/env python
"""Measure one-GPU Phase-2 training memory without writing checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import config
from data import build_phase2_dataset, collate_joint
from flow import add_noise_x0_masked, sample_training_sigma
from losses import t2m_reconstruction_loss
from model import EdgePhase2MotionExpert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="2,4,8,12,16,24,32")
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--ti2m-frames", type=int, default=97)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def build_rows(mode: str, count: int, args: argparse.Namespace) -> list[dict]:
    dataset = build_phase2_dataset(
        split="train",
        train=False,
        num_frames=args.T,
        ti2m_frames=args.ti2m_frames,
        task_weights={mode: 1.0},
        bones_frac=0.0,
        cfg_dropout=0.0,
        reasoner_image_size=config.REASONER_IMAGE_SIZE,
        max_samples=max(count * 4, count),
        seed=args.seed,
    )
    rows = [dataset[index] for index in range(count)]
    if len(rows) != count:
        raise RuntimeError(f"could not materialize {count} {mode} rows")
    return rows


def run_update(
    *,
    model: EdgePhase2MotionExpert,
    optimizer: torch.optim.Optimizer,
    rows: list[dict],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict:
    batch = collate_joint(rows)
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    x0 = batch["motion"].to(device=device, dtype=torch.float32)
    pad = batch["motion_pad_mask"].to(device=device)
    neutral = batch["neutral_joints"].to(device=device)
    sigma = sample_training_sigma(x0.shape[0], device)
    x_sigma, sigma, target, noised = add_noise_x0_masked(x0, pad, sigma)
    conditions = model.prepare_conditions(
        batch["caption"],
        modes=batch["mode"],
        reasoner_images=batch["reasoner_image"],
        image_size=config.REASONER_IMAGE_SIZE,
    )
    prediction = model(
        reasoner_inputs=conditions,
        x_sigma=x_sigma,
        sigma=sigma,
        neutral_joints=neutral,
        motion_pad_mask=pad,
    )
    loss, _ = t2m_reconstruction_loss(
        prediction,
        target,
        noised,
        mean,
        std,
        w_feat=1.0,
        w_joint=10.0,
        w_smooth=50.0,
        fps=config.FPS,
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "batch_size": len(rows),
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "valid_motion_tokens": int((~pad).sum()),
        "seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }
    del batch, x0, pad, neutral, sigma, x_sigma, target, noised, conditions, prediction, loss
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    device = torch.device("cuda", 0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    config.validate_artifacts()

    max_batch = max(batch_sizes)
    rows_by_mode = {
        mode: build_rows(mode, max_batch, args)
        for mode in ("text2motion", "textimg2motion")
    }
    model = EdgePhase2MotionExpert(device=device, dtype=torch.bfloat16)
    model.train()
    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=2e-4,
        betas=(0.9, 0.95),
        fused=True,
    )
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).to(device=device)
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).to(device=device)
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
        "T": args.T,
        "ti2m_frames": args.ti2m_frames,
        "results": [],
    }

    stop = False
    for batch_size in batch_sizes:
        for mode in ("text2motion", "textimg2motion"):
            try:
                result = run_update(
                    model=model,
                    optimizer=optimizer,
                    rows=rows_by_mode[mode][:batch_size],
                    mean=mean,
                    std=std,
                    device=device,
                )
                result.update(mode=mode, status="pass")
                report["results"].append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
            except torch.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                result = {
                    "batch_size": batch_size,
                    "mode": mode,
                    "status": "oom",
                    "error": str(error),
                }
                report["results"].append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
                stop = True
                break
        if stop:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[batch-benchmark] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
