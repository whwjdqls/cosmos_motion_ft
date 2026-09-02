"""Deterministic held-out T2M/TI2M visualization for Edge Phase-2 training.

The first invocation selects a fixed, caption-diverse Nymeria test cohort and
materializes its exact inputs under ``<run>/visualizations/fixed_samples``.
Later checkpoints and resumed Slurm jobs load those frozen arrays instead of
re-querying the dataset.  This makes both the sample identity and initial
diffusion noise invariant across the whole training run.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from data import build_phase2_dataset
from decode_uniego_torch import decode_joints
from flow import sample_x0_unipc
from render_motion import render_conditioned_motion_mp4, render_motion_mp4


FIXED_SUITE_SCHEMA = 1
TASKS = ("text2motion", "textimg2motion")


def _task_seed(seed: int, task: str) -> int:
    if task not in TASKS:
        raise ValueError(f"unsupported visualization task: {task}")
    return int(seed) + (0 if task == "text2motion" else 1_000_003)


def _sample_seed(seed: int, task: str, slot: int) -> int:
    return _task_seed(seed, task) + 10_007 * (int(slot) + 1)


def _sample_id(task: str, dataset_index: int, caption: str) -> str:
    digest = hashlib.sha256(
        f"{task}\0{int(dataset_index)}\0{caption}".encode("utf-8")
    ).hexdigest()[:12]
    prefix = "t2m" if task == "text2motion" else "ti2m"
    return f"{prefix}_{int(dataset_index):06d}_{digest}"


def _suite_contract(
    *,
    T: int,
    ti2m_frames: int,
    reasoner_image_size: int,
    samples_per_task: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": FIXED_SUITE_SCHEMA,
        "motion_representation": config.MOTION_REPRESENTATION,
        "T": int(T),
        "ti2m_frames": int(min(ti2m_frames, T)),
        "reasoner_image_size": int(reasoner_image_size),
        "samples_per_task": int(samples_per_task),
        "seed": int(seed),
        "tasks": list(TASKS),
        "split": "test",
        "source": "nymeria",
    }


def _write_fixed_sample(
    fixed_dir: Path,
    *,
    task: str,
    slot: int,
    dataset_index: int,
    row: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    caption = str(row.get("caption") or "").strip()
    if not caption:
        raise ValueError("held-out visualization row has an empty caption")
    motion = torch.as_tensor(row["motion"]).cpu().numpy().astype(np.float32)
    pad = torch.as_tensor(row["motion_pad_mask"]).cpu().numpy().astype(bool)
    neutral = torch.as_tensor(row["neutral_joints"]).cpu().numpy().astype(np.float32)
    if motion.ndim != 2 or motion.shape[1] != config.MOTION_DIM:
        raise ValueError(f"invalid fixed motion shape: {motion.shape}")
    if pad.shape != motion.shape[:1] or bool(pad.all()):
        raise ValueError(f"invalid fixed motion pad mask: {pad.shape}")
    if neutral.shape != (config.NUM_JOINTS, 3):
        raise ValueError(f"invalid fixed neutral-joint shape: {neutral.shape}")

    image = row.get("reasoner_image")
    if task == "textimg2motion":
        if image is None:
            raise ValueError("fixed TI2M sample is missing its conditioning image")
        image_np = torch.as_tensor(image).cpu().numpy().astype(np.uint8)
    else:
        if image is not None:
            raise ValueError("fixed T2M sample unexpectedly carries an image")
        image_np = np.empty((0,), dtype=np.uint8)

    sample_id = _sample_id(task, dataset_index, caption)
    relative_npz = Path("fixed_samples") / f"{task}_{slot:02d}_{sample_id}.npz"
    np.savez_compressed(
        fixed_dir.parent / relative_npz,
        motion=motion,
        motion_pad_mask=pad,
        neutral_joints=neutral,
        reasoner_image=image_np,
    )
    return {
        "task": task,
        "slot": int(slot),
        "sample_id": sample_id,
        "dataset_index": int(dataset_index),
        "source": str(row.get("source", "nymeria")),
        "prompt": caption,
        "prompt_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
        "sampling_seed": _sample_seed(seed, task, slot),
        "valid_frames": int((~pad).sum()),
        "arrays_npz": str(relative_npz),
    }


def _select_task_samples(
    *,
    task: str,
    count: int,
    T: int,
    ti2m_frames: int,
    reasoner_image_size: int,
    seed: int,
    fixed_dir: Path,
) -> list[dict[str, Any]]:
    dataset = build_phase2_dataset(
        split="test",
        train=False,
        num_frames=T,
        ti2m_frames=ti2m_frames,
        task_weights={task: 1.0},
        bones_frac=0.0,
        cfg_dropout=0.0,
        reasoner_image_size=reasoner_image_size,
        seed=_task_seed(seed, task),
    )
    rng = random.Random(_task_seed(seed, task))
    candidate_count = min(len(dataset), max(512, count * 64))
    candidates = rng.sample(range(len(dataset)), k=candidate_count)
    selected: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for dataset_index in candidates:
        try:
            row = dataset[dataset_index]
        except Exception as error:  # the dataset already exhausts its guarded retry path
            print(
                f"[viz] skip candidate task={task} index={dataset_index}: "
                f"{type(error).__name__}: {str(error)[:160]}",
                flush=True,
            )
            continue
        caption = str(row.get("caption") or "").strip()
        if not caption or caption in seen_prompts or row.get("source") != "nymeria":
            continue
        if task == "textimg2motion" and row.get("reasoner_image") is None:
            continue
        selected.append(
            _write_fixed_sample(
                fixed_dir,
                task=task,
                slot=len(selected),
                dataset_index=dataset_index,
                row=row,
                seed=seed,
            )
        )
        seen_prompts.add(caption)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"could only select {len(selected)}/{count} fixed {task} samples"
        )
    return selected


def build_or_load_fixed_suite(
    output_root: str | Path,
    *,
    T: int,
    ti2m_frames: int,
    reasoner_image_size: int,
    samples_per_task: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create once, then strictly reuse the exact fixed held-out arrays."""

    if samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    viz_root = Path(output_root) / "visualizations"
    fixed_dir = viz_root / "fixed_samples"
    manifest_path = fixed_dir / "manifest.json"
    expected = _suite_contract(
        T=T,
        ti2m_frames=ti2m_frames,
        reasoner_image_size=reasoner_image_size,
        samples_per_task=samples_per_task,
        seed=seed,
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        actual = manifest.get("contract")
        if actual != expected:
            raise RuntimeError(
                "fixed visualization-suite contract drift: "
                f"saved={actual!r} live={expected!r}"
            )
        records = manifest.get("samples", [])
    else:
        fixed_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for task in TASKS:
            records.extend(
                _select_task_samples(
                    task=task,
                    count=samples_per_task,
                    T=T,
                    ti2m_frames=ti2m_frames,
                    reasoner_image_size=reasoner_image_size,
                    seed=seed,
                    fixed_dir=fixed_dir,
                )
            )
        manifest = {"contract": expected, "samples": records}
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)

    expected_count = samples_per_task * len(TASKS)
    if len(records) != expected_count:
        raise RuntimeError(
            f"fixed visualization suite has {len(records)} rows, expected {expected_count}"
        )
    for task in TASKS:
        task_rows = [record for record in records if record.get("task") == task]
        if len(task_rows) != samples_per_task:
            raise RuntimeError(
                f"fixed visualization suite has {len(task_rows)} {task} rows, "
                f"expected {samples_per_task}"
            )
    items: list[dict[str, Any]] = []
    for record in records:
        arrays_path = viz_root / record["arrays_npz"]
        if not arrays_path.is_file():
            raise FileNotFoundError(f"fixed visualization arrays missing: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            item = dict(record)
            item.update(
                motion=torch.from_numpy(arrays["motion"].copy()).float(),
                motion_pad_mask=torch.from_numpy(
                    arrays["motion_pad_mask"].copy()
                ).bool(),
                neutral_joints=torch.from_numpy(
                    arrays["neutral_joints"].copy()
                ).float(),
                reasoner_image=(
                    None
                    if arrays["reasoner_image"].size == 0
                    else torch.from_numpy(arrays["reasoner_image"].copy())
                ),
            )
        if hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest() != item[
            "prompt_sha256"
        ]:
            raise RuntimeError(f"fixed prompt hash mismatch for {item['sample_id']}")
        items.append(item)
    return manifest, items


def _fixed_initial_noise(
    items: list[dict[str, Any]],
    *,
    frames: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for item in items:
        generator = torch.Generator(device=device).manual_seed(int(item["sampling_seed"]))
        rows.append(
            torch.randn(
                frames,
                config.MOTION_DIM,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        )
    return torch.stack(rows, dim=0)


@torch.no_grad()
def render_fixed_suite(
    *,
    model,
    items: list[dict[str, Any]],
    output_root: str | Path,
    step: int,
    T: int,
    reasoner_image_size: int,
    sampling_steps: int,
    guidance: float,
    frame_stride: int,
    device: torch.device,
) -> tuple[Path, list[dict[str, Any]]]:
    """Sample the fixed suite with fixed noise, save arrays, and render MP4s."""

    if sampling_steps <= 0 or frame_stride <= 0:
        raise ValueError("sampling_steps and frame_stride must be positive")
    step_dir = Path(output_root) / "visualizations" / f"step_{int(step):09d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).to(
        device=device, dtype=torch.float32
    )
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).to(
        device=device, dtype=torch.float32
    )
    records: list[dict[str, Any]] = []
    was_training = bool(model.training)
    model.eval()
    try:
        for task in TASKS:
            task_items = sorted(
                (item for item in items if item["task"] == task),
                key=lambda item: int(item["slot"]),
            )
            if not task_items:
                continue
            frames = (
                int(T)
                if task == "text2motion"
                else int((~task_items[0]["motion_pad_mask"]).sum())
            )
            if task == "textimg2motion" and any(
                int((~item["motion_pad_mask"]).sum()) != frames for item in task_items
            ):
                raise RuntimeError("fixed TI2M samples do not share one aligned frame count")
            neutral = torch.stack(
                [item["neutral_joints"] for item in task_items], dim=0
            ).to(device=device, dtype=torch.float32)
            images = [item["reasoner_image"] for item in task_items]
            captions = [item["prompt"] for item in task_items]
            modes = [task] * len(task_items)
            conditional = model.prepare_conditions(
                captions,
                modes=modes,
                reasoner_images=images,
                image_size=reasoner_image_size,
            )
            null = model.prepare_conditions(
                [""] * len(task_items),
                modes=modes,
                reasoner_images=images,
                image_size=reasoner_image_size,
            )
            pad = torch.zeros(
                (len(task_items), frames), dtype=torch.bool, device=device
            )

            def predict(reasoner_inputs):
                return lambda noisy, sigma: model(
                    reasoner_inputs=reasoner_inputs,
                    x_sigma=noisy,
                    sigma=sigma,
                    neutral_joints=neutral,
                    motion_pad_mask=pad,
                )

            generated = sample_x0_unipc(
                predict(conditional),
                predict_null=predict(null),
                T=frames,
                motion_dim=config.MOTION_DIM,
                steps=sampling_steps,
                guidance=guidance,
                batch=len(task_items),
                device=device,
                initial_noise=_fixed_initial_noise(
                    task_items, frames=frames, device=device
                ),
            ).float()
            if not torch.isfinite(generated).all():
                raise FloatingPointError(
                    f"non-finite {task} visualization sample at step {step}"
                )
            generated_features = generated * std + mean
            generated_joints = decode_joints(generated_features).cpu().numpy()

            for batch_index, item in enumerate(task_items):
                valid = ~item["motion_pad_mask"]
                gt_normalized = item["motion"][valid].to(device=device).unsqueeze(0)
                gt_joints = decode_joints(gt_normalized * std + mean).cpu().numpy()[0]
                generated_normalized = generated[batch_index].cpu().numpy()
                generated_features_np = generated_features[batch_index].cpu().numpy()
                generated_joints_np = generated_joints[batch_index]
                compare_frames = min(len(gt_joints), len(generated_joints_np))
                stem = f"{task}_{int(item['slot']):02d}_{item['sample_id']}"
                arrays_path = step_dir / f"{stem}.npz"
                np.savez_compressed(
                    arrays_path,
                    generated_normalized=generated_normalized,
                    generated_features=generated_features_np,
                    generated_joints=generated_joints_np,
                    gt_normalized=gt_normalized.cpu().numpy()[0],
                    gt_joints=gt_joints,
                )
                video_path = step_dir / f"{stem}.mp4"
                condition_path = None
                render_fps = max(1, round(float(config.FPS) / frame_stride))
                title = f"Prompt: {item['prompt']}"
                if task == "textimg2motion":
                    condition_path = step_dir / f"{stem}_condition.png"
                    render_conditioned_motion_mp4(
                        condition_image=item["reasoner_image"],
                        gen_joints=generated_joints_np[:compare_frames],
                        gt_joints=gt_joints[:compare_frames],
                        out_path=str(video_path),
                        condition_out_path=str(condition_path),
                        caption=title,
                        fps=render_fps,
                        frame_stride=frame_stride,
                    )
                else:
                    render_motion_mp4(
                        generated_joints_np[:compare_frames],
                        str(video_path),
                        caption=title,
                        fps=render_fps,
                        gt_joints=gt_joints[:compare_frames],
                        frame_stride=frame_stride,
                    )
                records.append(
                    {
                        "step": int(step),
                        "task": task,
                        "slot": int(item["slot"]),
                        "sample_id": item["sample_id"],
                        "dataset_index": int(item["dataset_index"]),
                        "source": item["source"],
                        "prompt": item["prompt"],
                        "sampling_seed": int(item["sampling_seed"]),
                        "sampling_steps": int(sampling_steps),
                        "guidance": float(guidance),
                        "generated_frames": int(len(generated_joints_np)),
                        "gt_frames": int(len(gt_joints)),
                        "rendered_comparison_frames": int(compare_frames),
                        "render_frame_stride": int(frame_stride),
                        "render_fps": int(render_fps),
                        "arrays_npz": str(arrays_path),
                        "video": str(video_path),
                        "condition_image": (
                            None if condition_path is None else str(condition_path)
                        ),
                    }
                )
    finally:
        model.train(was_training)
        torch.cuda.empty_cache()

    manifest = {
        "step": int(step),
        "fixed_sample_manifest": str(
            Path(output_root) / "visualizations" / "fixed_samples" / "manifest.json"
        ),
        "sampling_steps": int(sampling_steps),
        "guidance": float(guidance),
        "samples": records,
    }
    manifest_path = step_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, records


def log_suite_to_wandb(wandb_run, records: list[dict[str, Any]], *, step: int) -> None:
    """Upload every MP4, TI2M condition image, and full prompt to W&B."""

    if wandb_run is None:
        return
    import wandb

    payload: dict[str, Any] = {"train/step": int(step)}
    table_rows = []
    for record in records:
        task = record["task"]
        slot = int(record["slot"])
        key = f"visualizations/{task}/sample_{slot:02d}"
        caption = (
            f"step={step} | {record['sample_id']} | seed={record['sampling_seed']}\n"
            f"Prompt: {record['prompt']}"
        )
        payload[key] = wandb.Video(
            record["video"],
            fps=int(record["render_fps"]),
            format="mp4",
            caption=caption,
        )
        if record.get("condition_image"):
            payload[f"{key}_condition_image"] = wandb.Image(
                record["condition_image"], caption=caption
            )
        table_rows.append(
            [
                task,
                slot,
                record["sample_id"],
                record["source"],
                record["prompt"],
                int(record["sampling_seed"]),
                int(record["gt_frames"]),
                int(record["generated_frames"]),
                record["video"],
            ]
        )
    payload["visualizations/sample_manifest"] = wandb.Table(
        columns=[
            "task",
            "slot",
            "sample_id",
            "source",
            "prompt",
            "sampling_seed",
            "gt_frames",
            "generated_frames",
            "local_video",
        ],
        data=table_rows,
    )
    wandb_run.log(payload)
    wandb_run.summary["last_visualization_step"] = int(step)


__all__ = [
    "FIXED_SUITE_SCHEMA",
    "TASKS",
    "build_or_load_fixed_suite",
    "log_suite_to_wandb",
    "render_fixed_suite",
]
