#!/usr/bin/env python
"""Build matched GT/256/720 Phase-3 M2V comparisons and metric deltas."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def _identity(name: str) -> str:
    match = re.match(r"^t\d+_(.+)$", name)
    if not match:
        raise ValueError(f"unexpected sequence name {name!r}")
    return match.group(1)


def _ffmpeg_compare(gt: Path, low: Path, base: Path, head: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    labels = ("GT", "BASE 256/S3", "BASE 720/S10", "HEADCAM 720/S10")
    for index, label in enumerate(labels):
        chain = (
            f"[{index}:v]scale=256:256,"
            f"drawtext=text='{label}':x=8:y=8:fontsize=18:fontcolor=white:"
            "box=1:boxcolor=black@0.7"
        )
        if index == 0:
            chain += ",drawbox=x=0:y=0:w=iw:h=ih:color=green:t=6"
        else:
            chain += (
                ",drawbox=x=0:y=0:w=iw:h=ih:color=green:t=6:enable='eq(n,0)'"
                ",drawbox=x=0:y=0:w=iw:h=ih:color=red:t=6:enable='gte(n,1)'"
            )
        filters.append(f"{chain}[v{index}]")
    filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|256_0|512_0|768_0[out]")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(gt),
            "-i",
            str(low),
            "-i",
            str(base),
            "-i",
            str(head),
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "20",
            str(out),
        ],
        check=True,
    )


def _low_videos(root: Path) -> dict[str, Path]:
    result = {}
    for path in (root / "viz").glob("motimg2video_*_gen.mp4"):
        name = path.name.removeprefix("motimg2video_").removesuffix("_gen.mp4")
        result[_identity(name)] = path
    return result


def _aggregate(summary: dict) -> dict:
    video_metrics = summary.get("video_metrics", {}).get("motimg2video", {})
    aggregate = video_metrics.get("aggregate")
    if aggregate is None:
        raise KeyError("summary does not contain video_metrics.motimg2video.aggregate")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-720", type=Path, required=True)
    parser.add_argument("--headcam-720", type=Path, required=True)
    parser.add_argument("--baseline-256", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads((args.baseline_720 / "summary.json").read_text())
    head = json.loads((args.headcam_720 / "summary.json").read_text())
    low = json.loads((args.baseline_256 / "summary.json").read_text())
    if base["n"] != head["n"]:
        raise RuntimeError(f"sample-count mismatch: baseline={base['n']} headcam={head['n']}")

    base_by_id = {_identity(name): (name, row) for name, row in base["artifacts"].items()}
    head_by_id = {_identity(name): (name, row) for name, row in head["artifacts"].items()}
    low_by_id = _low_videos(args.baseline_256)
    identities = sorted(base_by_id)
    if set(identities) != set(head_by_id) or set(identities) - set(low_by_id):
        raise RuntimeError("baseline/headcam/256 sequence identities do not match")

    videos = {}
    for identity in identities:
        base_name, base_artifacts = base_by_id[identity]
        _head_name, head_artifacts = head_by_id[identity]
        out = args.out_dir / "viz" / f"{identity}.mp4"
        _ffmpeg_compare(
            Path(base_artifacts["gt_video"]),
            low_by_id[identity],
            Path(base_artifacts["generated_video"]),
            Path(head_artifacts["generated_video"]),
            out,
        )
        videos[identity] = str(out)

    metric_keys = ("psnr_db", "ssim", "lpips_alex")
    comparison = {}
    low_aggregate = _aggregate(low)
    for key in metric_keys:
        low_value = float(low_aggregate[key]["mean"])
        base_value = float(base["aggregate"][key]["mean"])
        head_value = float(head["aggregate"][key]["mean"])
        comparison[key] = {
            "baseline_256": low_value,
            "baseline_720": base_value,
            "headcam_720": head_value,
            "baseline_720_minus_baseline_256": base_value - low_value,
            "headcam_720_minus_baseline_256": head_value - low_value,
            "headcam_minus_baseline": head_value - base_value,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "n": len(identities),
        "baseline_720": str(args.baseline_720),
        "headcam_720": str(args.headcam_720),
        "baseline_256": str(args.baseline_256),
        "frame_provenance": {
            "green": "GT tile, or clean conditioned frame 0 in a generated tile",
            "red": "generated suffix frames 1-96",
        },
        "metric_comparison": comparison,
        "videos": videos,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out_dir / "COMPLETE.json").write_text(
        json.dumps({"summary": str(args.out_dir / "summary.json"), "videos": len(videos)}, indent=2)
        + "\n"
    )
    print(f"[compare-phase3-m2v-720] wrote {len(videos)} comparisons -> {args.out_dir}")


if __name__ == "__main__":
    main()
