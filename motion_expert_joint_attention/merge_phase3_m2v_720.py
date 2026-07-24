#!/usr/bin/env python
"""Merge independently generated Phase-3 high-tier M2V shards."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import eval_all as EA


def _probe_video(path: Path) -> dict:
    payload = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    if len(payload.get("streams", [])) != 1:
        raise RuntimeError(f"{path}: expected one video stream")
    return payload["streams"][0]


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-parts", type=int, default=5)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    parts = sorted(path for path in args.parts_root.glob("part_*") if path.is_dir())
    if len(parts) != args.expected_parts:
        raise RuntimeError(
            f"expected {args.expected_parts} part directories under {args.parts_root}, "
            f"found {len(parts)}"
        )

    rows: dict[str, dict] = {}
    sampling = None
    checkpoint = None
    artifacts = {}
    for part in parts:
        if not (part / "EVALUATION_COMPLETE").is_file():
            raise RuntimeError(f"incomplete shard: {part}")
        summary = json.loads((part / "summary.json").read_text())
        part_sampling = summary["sampling"]
        if sampling is None:
            sampling = part_sampling
            checkpoint = summary["ckpt"]
        elif part_sampling != sampling or summary["ckpt"] != checkpoint:
            raise RuntimeError(f"sampling/checkpoint mismatch in {part}")

        metrics_path = part / "video" / "motimg2video_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("n") != 1 or len(metrics.get("per_sequence", {})) != 1:
            raise RuntimeError(f"{metrics_path}: expected exactly one sequence")
        name, row = next(iter(metrics["per_sequence"].items()))
        if name in rows:
            raise RuntimeError(f"duplicate sequence name {name}")
        rows[name] = row

        latent = part / "video" / "motimg2video" / f"{name}.npz"
        gt = part / "viz" / f"motimg2video_{name}_gt.mp4"
        gen = part / "viz" / f"motimg2video_{name}_gen.mp4"
        comparison = part / "viz" / f"motimg2video_{name}.mp4"
        for path in (latent, gt, gen, comparison):
            if not path.is_file():
                raise FileNotFoundError(path)
        gen_probe = _probe_video(gen)
        if (
            int(gen_probe["width"]) != 640
            or int(gen_probe["height"]) != 640
            or int(gen_probe["nb_frames"]) != 97
            or gen_probe["r_frame_rate"] != "20/1"
        ):
            raise RuntimeError(f"{gen}: unexpected video contract {gen_probe}")
        artifacts[name] = {
            "part": str(part),
            "latents": str(latent),
            "gt_video": str(gt),
            "generated_video": str(gen),
            "comparison_video": str(comparison),
        }

    aggregate = EA._aggregate_video_metrics(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = args.out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = analysis_dir / "motimg2video_metrics.json"
    metrics_out.write_text(json.dumps(aggregate, indent=2) + "\n")

    for name, paths in artifacts.items():
        _link(Path(paths["latents"]), args.out_dir / "latents" / f"{name}.npz")
        _link(Path(paths["gt_video"]), args.out_dir / "videos" / f"{name}_gt.mp4")
        _link(Path(paths["generated_video"]), args.out_dir / "videos" / f"{name}_gen.mp4")
        _link(
            Path(paths["comparison_video"]),
            args.out_dir / "comparisons" / f"{name}.mp4",
        )

    result = {
        "schema_version": 1,
        "label": args.label,
        "checkpoint": checkpoint,
        "n": len(rows),
        "sampling": sampling,
        "video_contract": {
            "rgb_frames": 97,
            "fps": 20,
            "output_size": [640, 640],
            "latent_shape": [48, 25, 40, 40],
            "conditioned_rgb_frames": [0],
            "generated_rgb_frames": [1, 96],
        },
        "aggregate": aggregate["aggregate"],
        "metrics_json": str(metrics_out),
        "artifacts": artifacts,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    complete_path = args.out_dir / "COMPLETE.json"
    complete_path.write_text(
        json.dumps(
            {
                "summary": str(summary_path),
                "metrics": str(metrics_out),
                "parts_root": str(args.parts_root),
                "parts": len(parts),
                "samples": len(rows),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[merge-phase3-m2v-720] {args.label}: merged {len(rows)} samples -> "
        f"{args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
