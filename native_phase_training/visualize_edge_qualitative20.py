#!/usr/bin/env python
"""Render fixed GT/FD/WAM/native-I2V/Diffusers-I2V qualitative grids."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from native_phase_training.visualize_checkpoint import _annotated_video_filter


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_grid(inputs: list[tuple[str, Path, int | None]], output: Path) -> None:
    if len(inputs) != 5:
        raise ValueError(f"expected five qualitative videos, got {len(inputs)}")
    filters: list[str] = []
    for index, (label, _path, prefix_length) in enumerate(inputs):
        filters.append(
            _annotated_video_filter(
                input_index=index,
                output_label=f"v{index}",
                width=256,
                height=256,
                header_height=28,
                font_size=12,
                label=label,
                prefix_length=prefix_length,
            )
        )
    layout = ("0_0", "w0_0", "w0+w1_0", "0_h0", "w0_h0")
    stack_inputs = "".join(f"[v{index}]" for index in range(5))
    filters.append(f"{stack_inputs}xstack=inputs=5:layout={'|'.join(layout)}:fill=black[out]")
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for _label, path, _prefix_length in inputs:
        command.extend(("-i", str(path)))
    command.extend(
        (
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-r",
            "20",
            "-frames:v",
            "97",
            "-pix_fmt",
            "yuv420p",
            str(output),
        )
    )
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--diffusers-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prepared = args.prepared_root.resolve()
    canonical = prepared / "canonical_inputs"
    contract = json.loads((prepared / "cohort_contract.json").read_text())
    fd_records = _read_jsonl(canonical / "fd_input.jsonl")
    policy_records = _read_jsonl(canonical / "policy_input.jsonl")
    i2v_records = _read_jsonl(canonical / "i2v_input.jsonl")
    if not (len(fd_records) == len(policy_records) == len(i2v_records) == 20):
        raise ValueError("qualitative visualization requires the frozen 20-sample cohort")

    output_root = args.out.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for index, base_name in enumerate(contract["samples"]):
        inputs = [
            ("GT REFERENCE", canonical / "samples" / base_name / "gt_clip.mp4", None),
            ("NATIVE FD | 10/30/1", args.native_root.resolve() / fd_records[index]["name"] / "vision.mp4", 1),
            (
                "NATIVE WAM | 10/30/1",
                args.native_root.resolve() / policy_records[index]["name"] / "vision.mp4",
                1,
            ),
            (
                "NATIVE I2V | 12/20/6",
                args.native_root.resolve() / i2v_records[index]["name"] / "vision.mp4",
                1,
            ),
            (
                "DIFFUSERS I2V | 12/20/6",
                args.diffusers_root.resolve() / i2v_records[index]["name"] / "vision.mp4",
                1,
            ),
        ]
        missing = [str(path) for _label, path, _prefix in inputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{base_name}: missing grid inputs: {missing}")
        output = output_root / f"{base_name}_five_way.mp4"
        _make_grid(inputs, output)
        manifest.append(
            {
                "index": index,
                "sample": base_name,
                "output": str(output),
                "tiles": [
                    {"label": label, "path": str(path), "condition_prefix_frames": prefix}
                    for label, path, prefix in inputs
                ],
            }
        )
        print(f"[edge-qual20-viz] {index + 1}/20 {base_name}", flush=True)

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote 20 fixed five-way grids to {output_root}")


if __name__ == "__main__":
    main()
