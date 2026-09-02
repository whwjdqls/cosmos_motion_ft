#!/usr/bin/env python
"""Materialize effective prompts beside official Cosmos inference videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from native_phase_training.nymeria_i2v_prompt import write_prompt_artifacts


def save_inference_prompts(inference_root: Path) -> int:
    count = 0
    for sample_args_path in sorted(inference_root.rglob("sample_args.json")):
        sample_dir = sample_args_path.parent
        if not (sample_dir / "vision.mp4").is_file() and not (sample_dir / "sample_outputs.json").is_file():
            continue
        sample_args = json.loads(sample_args_path.read_text())
        positive = sample_args.get("prompt")
        if not isinstance(positive, str):
            raise ValueError(f"{sample_args_path}: prompt must be a string")
        negative = sample_args.get("negative_prompt")
        if negative is not None and not isinstance(negative, str):
            raise ValueError(f"{sample_args_path}: negative_prompt must be a string or null")
        write_prompt_artifacts(
            sample_dir,
            positive_prompt=positive,
            negative_prompt=negative,
        )
        count += 1
    if count == 0:
        raise ValueError(f"no completed sample directories found under {inference_root}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, required=True)
    args = parser.parse_args()
    count = save_inference_prompts(args.inference_root.resolve())
    print(f"saved effective prompt artifacts beside {count} inference outputs", flush=True)


if __name__ == "__main__":
    main()
