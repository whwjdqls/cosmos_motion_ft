#!/usr/bin/env python
"""Run Cosmos3-Edge I2V through the Diffusers pipeline used by its model card."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline, __version__ as diffusers_version
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

from nymeria_i2v_prompt import (
    DEFAULT_NEGATIVE_TEMPLATE_PATH,
    DEFAULT_TEMPLATE_PATH,
    build_nymeria_i2v_negative_prompt,
    build_nymeria_i2v_prompt,
    compact_prompt,
    write_prompt_artifacts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusers-source", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-json", type=Path)
    prompt_group.add_argument("--prompt-text-file", type=Path)
    parser.add_argument("--use-nymeria-template", action="store_true")
    parser.add_argument("--nymeria-template-json", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--use-nymeria-negative-template", action="store_true")
    parser.add_argument("--negative-prompt-json", type=Path)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-safety-checker", action="store_true")
    args = parser.parse_args()
    nymeria_template_path = args.nymeria_template_json.resolve()

    model = args.model.resolve()
    assets = model / "assets"
    image_path = (args.image or assets / "example_i2v_input.jpg").resolve()
    prompt_path = (args.prompt_json or args.prompt_text_file or assets / "example_i2v_prompt.json").resolve()
    negative_path = (
        args.negative_prompt_json
        or (DEFAULT_NEGATIVE_TEMPLATE_PATH if args.use_nymeria_negative_template else assets / "negative_prompt.json")
    ).resolve()
    for required in (
        model / "modular_model_index.json",
        image_path,
        prompt_path,
        negative_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.use_nymeria_template and not nymeria_template_path.is_file():
        raise FileNotFoundError(nymeria_template_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "edge_i2v_diffusers.mp4"
    complete_path = output_dir / "COMPLETE.json"
    if complete_path.is_file():
        print(f"already complete: {output_dir}")
        return
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite incomplete output: {output_path}")

    if args.prompt_text_file is not None:
        source_description = prompt_path.read_text().strip()
        if not source_description:
            raise ValueError(f"plain-text prompt is empty: {prompt_path}")
        if args.use_nymeria_template:
            prompt = compact_prompt(
                build_nymeria_i2v_prompt(
                    source_description,
                    num_frames=args.num_frames,
                    fps=args.fps,
                    height=args.height,
                    width=args.width,
                    template_path=nymeria_template_path,
                )
            )
            prompt_format = "nymeria_template_json"
        else:
            prompt = source_description
            prompt_format = "plain_text"
    else:
        if args.use_nymeria_template:
            raise ValueError("--use-nymeria-template requires --prompt-text-file")
        prompt = json.dumps(json.loads(prompt_path.read_text()))
        prompt_format = "structured_json"
    if args.use_nymeria_negative_template:
        negative_prompt = compact_prompt(
            build_nymeria_i2v_negative_prompt(
                num_frames=args.num_frames,
                fps=args.fps,
                height=args.height,
                width=args.width,
                template_path=negative_path,
            )
        )
        negative_prompt_format = "nymeria_template_json"
    else:
        negative_prompt = json.dumps(json.loads(negative_path.read_text()))
        negative_prompt_format = "structured_json"
    safety_checker = not args.disable_safety_checker

    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(model),
        torch_dtype=torch.bfloat16,
        enable_safety_checker=safety_checker,
    )
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=args.flow_shift,
        use_karras_sigmas=False,
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=load_image(str(image_path)),
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        add_resolution_template=False,
        add_duration_template=False,
    )
    export_to_video(result.video, str(output_path), fps=args.fps, macro_block_size=1)
    prompt_artifacts = write_prompt_artifacts(
        output_dir,
        positive_prompt=prompt,
        negative_prompt=negative_prompt,
    )

    diffusers_source = args.diffusers_source.resolve()
    metadata = {
        "status": "complete",
        "kind": "cosmos3_edge_model_card_diffusers_i2v",
        "model": str(model),
        "output": str(output_path),
        "diffusers_version": diffusers_version,
        "diffusers_source": str(diffusers_source),
        "diffusers_commit": subprocess.check_output(
            ["git", "-C", str(diffusers_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "safety_checker": safety_checker,
        "prompt_sha256": _sha256(prompt_path),
        "prompt_format": prompt_format,
        "positive_template": str(nymeria_template_path)
        if args.use_nymeria_template
        else None,
        "positive_template_sha256": _sha256(nymeria_template_path)
        if args.use_nymeria_template
        else None,
        "native_prompt_upsampling": False,
        "prompt_artifacts": prompt_artifacts,
        "negative_prompt_sha256": _sha256(negative_path),
        "negative_prompt_format": negative_prompt_format,
        "negative_template": str(negative_path)
        if args.use_nymeria_negative_template
        else None,
        "input_image_sha256": _sha256(image_path),
        "settings": {
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "flow_shift": args.flow_shift,
            "use_karras_sigmas": False,
            "seed": args.seed,
            "add_resolution_template": False,
            "add_duration_template": False,
        },
    }
    complete_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
