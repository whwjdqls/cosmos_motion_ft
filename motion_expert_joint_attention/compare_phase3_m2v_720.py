#!/usr/bin/env python
"""Build matched Phase-3 M2V resolution and optional headcam comparisons."""
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


def _ffmpeg_compare(
    tiles: list[tuple[str, Path]],
    out: Path,
    *,
    tile_size: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    for index, (label, _path) in enumerate(tiles):
        chain = (
            f"[{index}:v]scale={tile_size}:{tile_size},"
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
    inputs = "".join(f"[v{index}]" for index in range(len(tiles)))
    layout = "|".join(f"{index * tile_size}_0" for index in range(len(tiles)))
    filters.append(f"{inputs}xstack=inputs={len(tiles)}:layout={layout}[out]")
    input_args = [arg for _label, path in tiles for arg in ("-i", str(path))]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *input_args,
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


def _low_videos(roots: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in roots:
        for path in (root / "viz").glob("motimg2video_*_gen.mp4"):
            name = path.name.removeprefix("motimg2video_").removesuffix("_gen.mp4")
            identity = _identity(name)
            if identity in result:
                raise RuntimeError(
                    f"duplicate 256-tier sequence identity {identity}: "
                    f"{result[identity]} and {path}"
                )
            result[identity] = path
    return result


def _default_metrics_path(root: Path, summary: dict) -> Path:
    video_metrics = summary.get("video_metrics", {}).get("motimg2video", {})
    path = video_metrics.get("metrics_json")
    if not path:
        raise KeyError("summary does not contain video_metrics.motimg2video.metrics_json")
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-720", type=Path, required=True)
    parser.add_argument(
        "--headcam-720",
        type=Path,
        default=None,
        help="optional headcam high-tier result; omit for a dedicated GT/256/720 comparison",
    )
    parser.add_argument(
        "--baseline-256",
        type=Path,
        nargs="+",
        required=True,
        help="one or more 256-tier eval roots containing matched generated videos",
    )
    parser.add_argument(
        "--baseline-256-metrics",
        type=Path,
        default=None,
        help="optional consolidated metrics JSON, required when multiple eval roots form one set",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads((args.baseline_720 / "summary.json").read_text())
    low_summaries = [
        json.loads((root / "summary.json").read_text()) for root in args.baseline_256
    ]
    for root, low_summary in zip(args.baseline_256, low_summaries):
        if low_summary.get("ckpt") != base.get("checkpoint"):
            raise RuntimeError(
                f"checkpoint mismatch: high={base.get('checkpoint')} "
                f"low[{root}]={low_summary.get('ckpt')}"
            )
    base_by_id = {_identity(name): (name, row) for name, row in base["artifacts"].items()}
    if base.get("n") != len(base_by_id):
        raise RuntimeError(
            f"high-tier summary count mismatch: n={base.get('n')} "
            f"artifacts={len(base_by_id)}"
        )
    low_by_id = _low_videos(args.baseline_256)
    identities = sorted(base_by_id)
    if set(identities) - set(low_by_id):
        raise RuntimeError("baseline 720/256 sequence identities do not match")

    low_metrics_path = args.baseline_256_metrics
    if low_metrics_path is None:
        if len(args.baseline_256) != 1:
            raise RuntimeError(
                "--baseline-256-metrics is required when combining multiple 256 eval roots"
            )
        low_metrics_path = _default_metrics_path(
            args.baseline_256[0], low_summaries[0]
        )
    low_metrics = json.loads(low_metrics_path.read_text())
    low_metric_ids = {
        _identity(name) for name in low_metrics.get("per_sequence", {})
    }
    if low_metrics.get("n") != len(identities) or low_metric_ids != set(identities):
        raise RuntimeError(
            f"256 metrics do not describe the exact high-tier sample set: "
            f"n={low_metrics.get('n')} metric_ids={len(low_metric_ids)} "
            f"high_ids={len(identities)}"
        )

    head = None
    head_by_id = None
    if args.headcam_720 is not None:
        head = json.loads((args.headcam_720 / "summary.json").read_text())
        if base["n"] != head["n"]:
            raise RuntimeError(
                f"sample-count mismatch: baseline={base['n']} headcam={head['n']}"
            )
        head_by_id = {
            _identity(name): (name, row) for name, row in head["artifacts"].items()
        }
        if set(identities) != set(head_by_id):
            raise RuntimeError("baseline/headcam sequence identities do not match")

    videos = {}
    for identity in identities:
        _base_name, base_artifacts = base_by_id[identity]
        out = args.out_dir / "viz" / f"{identity}.mp4"
        tiles = [
            ("GT", Path(base_artifacts["gt_video"])),
            ("BASE 256/S3", low_by_id[identity]),
            ("BASE 720/S10", Path(base_artifacts["generated_video"])),
        ]
        tile_size = 320
        if head_by_id is not None:
            _head_name, head_artifacts = head_by_id[identity]
            tiles.append(("HEADCAM 720/S10", Path(head_artifacts["generated_video"])))
            tile_size = 256
        _ffmpeg_compare(
            tiles,
            out,
            tile_size=tile_size,
        )
        videos[identity] = str(out)

    metric_keys = ("psnr_db", "ssim", "lpips_alex")
    comparison = {}
    low_aggregate = low_metrics["aggregate"]
    for key in metric_keys:
        low_value = float(low_aggregate[key]["mean"])
        base_value = float(base["aggregate"][key]["mean"])
        comparison[key] = {
            "baseline_256": low_value,
            "baseline_720": base_value,
            "baseline_720_minus_baseline_256": base_value - low_value,
        }
        if head is not None:
            head_value = float(head["aggregate"][key]["mean"])
            comparison[key].update(
                {
                    "headcam_720": head_value,
                    "headcam_720_minus_baseline_256": head_value - low_value,
                    "headcam_minus_baseline": head_value - base_value,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "n": len(identities),
        "baseline_720": str(args.baseline_720),
        "baseline_256": [str(root) for root in args.baseline_256],
        "baseline_256_metrics": str(low_metrics_path),
        "headcam_720": str(args.headcam_720) if args.headcam_720 is not None else None,
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
