"""Sample an Edge Phase-2 T2M/TI2M checkpoint and optionally render it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
from checkpoint import load_checkpoint
from data import build_phase2_dataset
from decode_uniego_torch import decode_joints
from flow import sample_x0_unipc
from model import EdgePhase2MotionExpert
from render_motion import render_conditioned_motion_mp4, render_motion_mp4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--mode", choices=("text2motion", "textimg2motion"), default="text2motion")
    parser.add_argument("--source", choices=("nymeria", "bones"), default="nymeria")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--T", type=int, default=config.DEFAULT_T)
    parser.add_argument("--ti2m-frames", type=int, default=config.TI2M_FRAMES)
    parser.add_argument("--reasoner-image-size", type=int, default=config.REASONER_IMAGE_SIZE)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    if args.mode == "textimg2motion" and args.source == "bones":
        raise ValueError("BONES is motion-only and cannot supply TI2M images")
    task_weights = {args.mode: 1.0}
    bones_frac = 1.0 if args.source == "bones" else 0.0
    dataset = build_phase2_dataset(
        split="test",
        train=False,
        num_frames=args.T,
        ti2m_frames=args.ti2m_frames,
        task_weights=task_weights,
        bones_frac=bones_frac,
        cfg_dropout=0.0,
        reasoner_image_size=args.reasoner_image_size,
        max_samples=args.sample_index + 1,
        seed=args.seed,
    )
    row = dataset[args.sample_index]
    prompt = row["caption"] if args.prompt is None else args.prompt
    neutral = row["neutral_joints"].unsqueeze(0).to(device)
    valid_gt = ~row["motion_pad_mask"]
    sample_frames = int(valid_gt.sum()) if args.mode == "textimg2motion" else args.T
    pad = torch.zeros((1, sample_frames), dtype=torch.bool, device=device)
    model = EdgePhase2MotionExpert(device=device, dtype=torch.bfloat16)
    payload = load_checkpoint(args.checkpoint, model=model)
    trained_args = payload.get("args", {})
    trained_weights = trained_args.get("task_weights", config.TASK_WEIGHTS)
    if float(trained_weights.get(args.mode, 0.0)) <= 0.0:
        raise RuntimeError(
            f"checkpoint was not trained for {args.mode}: task_weights={trained_weights}"
        )
    if args.source == "bones" and float(trained_args.get("bones_frac", 0.0)) <= 0.0:
        raise RuntimeError("checkpoint was not trained with BONES")
    model.eval()
    images = [row.get("reasoner_image")]
    conditional = model.prepare_conditions(
        [prompt], modes=[args.mode], reasoner_images=images, image_size=args.reasoner_image_size
    )
    # TI2M classifier-free guidance drops text only and deliberately retains
    # the identical frame-0 image in the null branch.
    null = model.prepare_conditions(
        [""], modes=[args.mode], reasoner_images=images, image_size=args.reasoner_image_size
    )

    def predict(reasoner_inputs):
        return lambda noisy, sigma: model(
            reasoner_inputs=reasoner_inputs,
            x_sigma=noisy,
            sigma=sigma,
            neutral_joints=neutral,
            motion_pad_mask=pad,
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    generated = sample_x0_unipc(
        predict(conditional),
        predict_null=predict(null),
        T=sample_frames,
        motion_dim=config.MOTION_DIM,
        steps=args.steps,
        guidance=args.guidance,
        device=device,
        generator=generator,
    )
    if not torch.isfinite(generated).all():
        raise FloatingPointError("sample contains non-finite values")
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).to(device)
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).to(device)
    generated_features = generated * std + mean
    generated_joints = decode_joints(generated_features).cpu().numpy()[0]
    gt_normalized = row["motion"][valid_gt].unsqueeze(0).to(device)
    gt_joints = decode_joints(gt_normalized * std + mean).cpu().numpy()[0]

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "sample.npz",
        generated_normalized=generated.cpu().numpy()[0],
        generated_features=generated_features.cpu().numpy()[0],
        generated_joints=generated_joints,
        gt_normalized=gt_normalized.cpu().numpy()[0],
        gt_joints=gt_joints,
    )
    if not args.no_render:
        n = min(len(gt_joints), len(generated_joints))
        if args.mode == "textimg2motion":
            render_conditioned_motion_mp4(
                condition_image=row["reasoner_image"],
                gen_joints=generated_joints[:n],
                gt_joints=gt_joints[:n],
                out_path=str(args.out / "image_gt_vs_generated.mp4"),
                condition_out_path=str(args.out / "condition.png"),
                caption=f"Prompt: {prompt}",
                fps=int(config.FPS),
            )
        else:
            render_motion_mp4(
                generated_joints[:n],
                str(args.out / "gt_vs_generated.mp4"),
                caption=f"Prompt: {prompt}",
                fps=int(config.FPS),
                gt_joints=gt_joints[:n],
            )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "prompt": prompt,
        "mode": args.mode,
        "source": args.source,
        "T_capacity": args.T,
        "output_frames": sample_frames,
        "reasoner_image_conditioned": args.mode == "textimg2motion",
        "sampling_steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "finite": True,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
