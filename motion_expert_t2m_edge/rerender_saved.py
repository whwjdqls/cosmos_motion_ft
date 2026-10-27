"""Re-render saved Edge visualization arrays without resampling or moving joints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "motion_expert_joint_attention"))
from render_motion import (
    head_rotation_from_features, head_rotation_summary,
    render_conditioned_motion_mp4, render_motion_mp4,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mean", type=Path)
    parser.add_argument("--std", type=Path)
    parser.add_argument("--samples-per-task", type=int, default=1)
    args = parser.parse_args()
    if "login" in socket.gethostname() or socket.gethostname() in {"cs-gpu-01", "node31", "node33"}:
        raise RuntimeError("Render on a compute node, not a login/excluded node")
    if args.samples_per_task < 1:
        raise ValueError("--samples-per-task must be positive")
    manifest = json.loads(args.manifest.read_text())
    selected, counts = [], {}
    for row in sorted(manifest["samples"], key=lambda r: (r["task"], r["slot"])):
        task = row["task"]
        if counts.get(task, 0) < args.samples_per_task:
            selected.append(row)
            counts[task] = counts.get(task, 0) + 1
    mean = np.load(args.mean).astype(np.float32) if args.mean is not None else None
    std = np.load(args.std).astype(np.float32) if args.std is not None else None
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected:
        arrays_path = Path(row["arrays_npz"])
        if not arrays_path.is_absolute():
            arrays_path = args.manifest.parent / arrays_path
        with np.load(arrays_path, allow_pickle=False) as arrays:
            generated = arrays["generated_joints"].copy()
            gt = arrays["gt_joints"].copy()
            generated_features = arrays["generated_features"][:1].copy()
            if "gt_features" in arrays:
                gt_features = arrays["gt_features"][:1].copy()
            else:
                if mean is None or std is None:
                    raise ValueError("Old NPZ requires its generator --mean and --std")
                gt_features = arrays["gt_normalized"][:1].astype(np.float32) * std + mean
        count = min(len(gt), len(generated))
        if count < 1:
            raise ValueError("Empty saved motion comparison")
        gen_rotation = head_rotation_from_features(generated_features)
        gt_rotation = head_rotation_from_features(gt_features)
        stem = f"{row['task']}_{int(row['slot']):02d}"
        video = args.out / (stem + ".mp4")
        first = args.out / (stem + "_frame0.png")
        options = dict(
            gt_joints=gt[:count], out_path=str(video),
            caption=f"Prompt: {row['prompt']}", fps=int(row["render_fps"]),
            frame_stride=int(row["render_frame_stride"]), max_frames=None,
            gen_head_rotation0=gen_rotation, gt_head_rotation0=gt_rotation,
            first_frame_path=str(first),
        )
        if row["task"] == "textimg2motion":
            image_path = Path(row["condition_image"])
            if not image_path.is_absolute():
                image_path = args.manifest.parent / image_path
            with Image.open(image_path) as image:
                condition = np.asarray(image.convert("RGB")).copy()
            render_conditioned_motion_mp4(
                condition_image=condition, gen_joints=generated[:count], **options,
            )
        else:
            render_motion_mp4(generated[:count], **options)
        record = {
            "task":row["task"], "sample_id":row["sample_id"], "source_arrays":str(arrays_path),
            "prompt":row["prompt"], "step":manifest["step"], "frames":count,
            "video":str(video), "first_frame":str(first),
            "generated_head0":head_rotation_summary(gen_rotation),
            "gt_head0":head_rotation_summary(gt_rotation),
            "generated_head0_position_world":generated[0,6].tolist(),
            "gt_head0_position_world":gt[0,6].tolist(),
            "generated_frame0_min_joint_y":float(generated[0,:,1].min()),
            "gt_frame0_min_joint_y":float(gt[0,:,1].min()),
            "motion_coordinates_unchanged":True,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.out / "manifest.json").write_text(json.dumps({
        "source_manifest":str(args.manifest), "samples":records,
        "mean":str(args.mean) if args.mean else None,
        "std":str(args.std) if args.std else None,
        "note":"Rendering only; source arrays and all model predictions are unchanged.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
