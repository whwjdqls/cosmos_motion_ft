#!/usr/bin/env python
"""Run a prepared Cosmos3-Edge I2V JSONL through pinned Diffusers once."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from diffusers import Cosmos3OmniPipeline, __version__ as diffusers_version
from diffusers.utils import export_to_video, load_image

from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
)

from nymeria_i2v_prompt import parse_json_object, write_prompt_artifacts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"{path}: no records")
    names: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or record.get("model_mode") != "image2video":
            raise ValueError(f"{path}:{index}: expected image2video JSON object")
        name = record.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"{path}:{index}: invalid or duplicate name {name!r}")
        names.add(name)
        for prompt_key in ("prompt", "negative_prompt"):
            prompt = record.get(prompt_key)
            if not isinstance(prompt, str) or parse_json_object(prompt) is None:
                raise ValueError(f"{path}:{index}: {prompt_key} must be structured JSON text")
        image = Path(str(record.get("vision_path", "")))
        if not image.is_file():
            raise FileNotFoundError(f"{path}:{index}: missing conditioning image {image}")
    return records


def _same_setting(records: list[dict[str, Any]], key: str) -> Any:
    values = [record.get(key) for record in records]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"I2V batch requires one {key}, got {values}")
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diffusers-source", type=Path, required=True)
    parser.add_argument("--native-framework", type=Path, required=True)
    parser.add_argument("--disable-safety-checker", action="store_true")
    args = parser.parse_args()

    model = args.model.resolve()
    input_path = args.input.resolve()
    output_root = args.output_root.resolve()
    diffusers_source = args.diffusers_source.resolve()
    native_framework = args.native_framework.resolve()
    records = _load_records(input_path)
    if len(records) != 20:
        raise ValueError(f"frozen qualitative batch requires 20 records, got {len(records)}")

    settings = {
        "num_frames": int(_same_setting(records, "num_frames")),
        "height": int(_same_setting(records, "resolution")),
        "width": int(_same_setting(records, "resolution")),
        "fps": float(_same_setting(records, "fps")),
        "num_inference_steps": int(_same_setting(records, "num_steps")),
        "guidance_scale": float(_same_setting(records, "guidance")),
        "flow_shift": float(_same_setting(records, "shift")),
        "seed": int(_same_setting(records, "seed")),
        "use_karras_sigmas": False,
        "add_resolution_template": False,
        "add_duration_template": False,
        "native_prompt_upsampling": False,
        "use_system_prompt": False,
        "enable_sound": False,
        "enable_safety_check": False,
    }
    expected = {
        "num_frames": 97,
        "height": 256,
        "width": 256,
        "fps": 20.0,
        "num_inference_steps": 20,
        "guidance_scale": 6.0,
        "flow_shift": 12.0,
        "seed": 0,
    }
    mismatches = {key: (settings[key], value) for key, value in expected.items() if settings[key] != value}
    if mismatches:
        raise ValueError(f"I2V qualitative contract mismatch: {mismatches}")

    output_root.mkdir(parents=True, exist_ok=True)
    batch_complete = output_root / "I2V_DIFFUSERS_COMPLETE.json"
    if batch_complete.is_file():
        print(f"I2V batch already complete: {batch_complete}")
        return

    safety_checker = not args.disable_safety_checker
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(model),
        torch_dtype=torch.bfloat16,
        enable_safety_checker=safety_checker,
    )
    pipe.to("cuda")
    # Use the exact scheduler implementation used by official native Cosmos.
    # The pipelines still differ, but their 20 timesteps and UniPC updates do
    # not become an avoidable confound in the backend comparison.
    pipe.scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=settings["flow_shift"],
        use_dynamic_shifting=False,
    )

    completed: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        sample_dir = output_root / record["name"]
        video_path = sample_dir / "vision.mp4"
        sample_complete = sample_dir / "COMPLETE.json"
        if sample_complete.is_file():
            if not video_path.is_file():
                raise FileNotFoundError(f"completion exists without video: {sample_complete}")
            completed.append(json.loads(sample_complete.read_text()))
            print(f"[edge-i2v-batch] {index}/20 already complete: {record['name']}", flush=True)
            continue
        if video_path.exists():
            raise FileExistsError(f"refusing to overwrite incomplete final video: {video_path}")
        sample_dir.mkdir(parents=True, exist_ok=True)
        partial_video = sample_dir / "vision.partial.mp4"
        if partial_video.exists():
            partial_video.unlink()

        result = pipe(
            prompt=record["prompt"],
            negative_prompt=record["negative_prompt"],
            image=load_image(record["vision_path"]),
            num_frames=settings["num_frames"],
            height=settings["height"],
            width=settings["width"],
            fps=settings["fps"],
            num_inference_steps=settings["num_inference_steps"],
            guidance_scale=settings["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(settings["seed"]),
            add_resolution_template=False,
            add_duration_template=False,
            use_system_prompt=False,
            enable_sound=False,
            enable_safety_check=False,
        )
        if len(result.video) != settings["num_frames"]:
            raise ValueError(f"{record['name']}: expected 97 frames, got {len(result.video)}")
        export_to_video(result.video, str(partial_video), fps=settings["fps"], macro_block_size=1)
        partial_video.replace(video_path)

        prompt_artifacts = write_prompt_artifacts(
            sample_dir,
            positive_prompt=record["prompt"],
            negative_prompt=record["negative_prompt"],
        )
        sample_args = {
            **record,
            "backend": "diffusers",
            "diffusers_version": diffusers_version,
            "native_prompt_upsampling": False,
            "use_karras_sigmas": False,
            "scheduler_class": "FlowUniPCMultistepScheduler",
            "add_resolution_template": False,
            "add_duration_template": False,
            "use_system_prompt": False,
            "enable_sound": False,
            "enable_safety_check": False,
        }
        _write_json(sample_dir / "sample_args.json", sample_args)
        _write_json(
            sample_dir / "sample_outputs.json",
            {
                "status": "success",
                "args": sample_args,
                "outputs": [{"content": {"vision": str(video_path)}}],
            },
        )
        sample_record = {
            "status": "complete",
            "name": record["name"],
            "video": str(video_path),
            "video_sha256": _sha256(video_path),
            "input_image_sha256": _sha256(Path(record["vision_path"])),
            "prompt_artifacts": prompt_artifacts,
            "settings": settings,
        }
        _write_json(sample_complete, sample_record)
        completed.append(sample_record)
        print(f"[edge-i2v-batch] {index}/20 complete: {record['name']}", flush=True)

    batch_record = {
        "status": "complete",
        "kind": "cosmos3_edge_diffusers_i2v_qualitative20",
        "model": str(model),
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "diffusers_version": diffusers_version,
        "diffusers_source": str(diffusers_source),
        "diffusers_commit": subprocess.check_output(
            ["git", "-C", str(diffusers_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "native_framework": str(native_framework),
        "native_framework_commit": subprocess.check_output(
            ["git", "-C", str(native_framework), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "safety_checker": safety_checker,
        "settings": settings,
        "count": len(completed),
        "samples": [record["name"] for record in completed],
    }
    _write_json(batch_complete, batch_record)
    print(f"[edge-i2v-batch] complete: {batch_complete}", flush=True)


if __name__ == "__main__":
    main()
