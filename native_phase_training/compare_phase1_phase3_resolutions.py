#!/usr/bin/env python
"""Build exact-window GT | Phase-1 | Phase-3 resolution comparisons."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


EXPECTED_FRAMES = 97
EXPECTED_FPS = Fraction(20, 1)
TILE_SIZE = 256


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _canonical_identity(uuid: str, start: int) -> str:
    return f"{uuid.replace('/', '_')}_{start}"


def _task_identity(name: str) -> str:
    match = re.match(r"^t\d+_(.+)$", name)
    if not match:
        raise ValueError(f"unexpected task name {name!r}")
    return match.group(1)


def _exact_records(path: Path, expected_count: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        meta_path = Path(record["vision_path"]).parent / "meta.json"
        meta = json.loads(meta_path.read_text())
        identity = _canonical_identity(str(meta["uuid"]), int(meta["start_frame"]))
        if identity in result:
            raise RuntimeError(f"duplicate exact window {identity}")
        gt_video = meta_path.parent / "gt_clip.mp4"
        if not gt_video.is_file():
            raise FileNotFoundError(gt_video)
        result[identity] = {
            "record": record,
            "meta": meta,
            "gt_video": gt_video,
        }
    if len(result) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} exact windows, found {len(result)} in {path}"
        )
    return result


def _phase1_videos(
    manifest_path: Path,
    missing_256_root: Path,
    missing_720_root: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, int]]:
    manifest = json.loads(manifest_path.read_text())
    phase1_256: dict[str, Path] = {}
    phase1_720: dict[str, Path] = {}
    reused = 0
    generated = 0
    for row in manifest["inventory"]:
        identity = _canonical_identity(str(row["uuid"]), int(row["start"]))
        if identity in phase1_256:
            raise RuntimeError(f"duplicate Phase-1 manifest window {identity}")
        if row["reused"]:
            path_256 = Path(row["phase1_256"])
            path_720 = Path(row["phase1_720"])
            reused += 1
        else:
            name = str(row["exact_name"])
            path_256 = missing_256_root / name / "vision.mp4"
            path_720 = missing_720_root / name / "vision.mp4"
            generated += 1
        for path in (path_256, path_720):
            if not path.is_file():
                raise FileNotFoundError(path)
        phase1_256[identity] = path_256
        phase1_720[identity] = path_720
    return phase1_256, phase1_720, {"reused": reused, "generated": generated}


def _phase3_256_videos(roots: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in roots:
        for path in (root / "viz").glob("motimg2video_*_gen.mp4"):
            name = path.name.removeprefix("motimg2video_").removesuffix("_gen.mp4")
            identity = _task_identity(name)
            if identity in result:
                raise RuntimeError(
                    f"duplicate Phase-3 256 window {identity}: "
                    f"{result[identity]} and {path}"
                )
            result[identity] = path
    return result


def _phase3_720_videos(root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    summary = json.loads((root / "summary.json").read_text())
    result: dict[str, Path] = {}
    for name, artifacts in summary["artifacts"].items():
        identity = _task_identity(name)
        if identity in result:
            raise RuntimeError(f"duplicate Phase-3 720 window {identity}")
        path = Path(artifacts["generated_video"])
        if not path.is_file():
            raise FileNotFoundError(path)
        result[identity] = path
    if summary.get("n") != len(result):
        raise RuntimeError(
            f"Phase-3 720 summary count mismatch: n={summary.get('n')} "
            f"artifacts={len(result)}"
        )
    return result, summary


def _probe(path: Path) -> dict[str, Any]:
    payload = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    stream = payload["streams"][0]
    frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": Fraction(stream["avg_frame_rate"]),
        "frames": frames,
    }


def _is_valid_comparison(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        probe = _probe(path)
    except (KeyError, ValueError, subprocess.CalledProcessError):
        return False
    return (
        probe["width"] == 5 * TILE_SIZE
        and probe["height"] == TILE_SIZE
        and probe["fps"] == EXPECTED_FPS
        and probe["frames"] == EXPECTED_FRAMES
    )


def _assert_source_contract(
    path: Path,
    *,
    label: str,
    width: int,
    height: int,
) -> None:
    probe = _probe(path)
    expected = {
        "width": width,
        "height": height,
        "fps": EXPECTED_FPS,
        "frames": EXPECTED_FRAMES,
    }
    if probe != expected:
        raise RuntimeError(f"{label}: invalid source {path}: expected {expected}, got {probe}")


def _render(tiles: list[tuple[str, Path]], out: Path) -> None:
    if len(tiles) != 5:
        raise ValueError(f"expected five tiles, got {len(tiles)}")
    filters = []
    for index, (label, _path) in enumerate(tiles):
        chain = (
            f"[{index}:v]scale={TILE_SIZE}:{TILE_SIZE},"
            f"drawtext=text='{label}':x=8:y=8:fontsize=17:fontcolor=white:"
            "box=1:boxcolor=black@0.72"
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
    layout = "|".join(f"{index * TILE_SIZE}_0" for index in range(len(tiles)))
    filters.append(f"{inputs}xstack=inputs={len(tiles)}:layout={layout}[out]")
    input_args = [arg for _label, path in tiles for arg in ("-i", str(path))]
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f"{out.stem}.tmp.mp4")
    temp.unlink(missing_ok=True)
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
            "-threads",
            "4",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "20",
            str(temp),
        ],
        check=True,
    )
    if not _is_valid_comparison(temp):
        raise RuntimeError(f"invalid rendered comparison: {temp} ({_probe(temp)})")
    temp.replace(out)


def _assert_exact_set(
    expected: set[str],
    actual: dict[str, Path],
    label: str,
) -> None:
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    if missing or extra:
        raise RuntimeError(
            f"{label} identity mismatch: missing={missing[:5]} "
            f"(n={len(missing)}) extra={extra[:5]} (n={len(extra)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-input-jsonl", type=Path, required=True)
    parser.add_argument("--phase1-reuse-manifest", type=Path, required=True)
    parser.add_argument("--phase1-missing-256", type=Path, required=True)
    parser.add_argument("--phase1-missing-720", type=Path, required=True)
    parser.add_argument("--phase3-256", type=Path, nargs="+", required=True)
    parser.add_argument("--phase3-720", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=71)
    args = parser.parse_args()

    exact = _exact_records(args.exact_input_jsonl, args.expected_count)
    phase1_256, phase1_720, phase1_inventory = _phase1_videos(
        args.phase1_reuse_manifest,
        args.phase1_missing_256,
        args.phase1_missing_720,
    )
    phase3_256_all = _phase3_256_videos(args.phase3_256)
    phase3_720, phase3_summary = _phase3_720_videos(args.phase3_720)
    expected = set(exact)
    for label, artifacts in (
        ("Phase-1 256", phase1_256),
        ("Phase-1 720", phase1_720),
        ("Phase-3 720", phase3_720),
    ):
        _assert_exact_set(expected, artifacts, label)
    missing_phase3_256 = sorted(expected - set(phase3_256_all))
    if missing_phase3_256:
        raise RuntimeError(
            f"Phase-3 256 is missing {len(missing_phase3_256)} exact windows: "
            f"{missing_phase3_256[:5]}"
        )
    phase3_256 = {
        identity: phase3_256_all[identity] for identity in exact
    }

    videos: dict[str, dict[str, str]] = {}
    for index, identity in enumerate(exact):
        row = exact[identity]
        out = args.out_dir / "viz" / f"{index:02d}_{identity}.mp4"
        tiles = [
            ("GT", row["gt_video"]),
            ("P1 256 / S3", phase1_256[identity]),
            ("P1 720 / S10", phase1_720[identity]),
            ("P3 256 / S3", phase3_256[identity]),
            ("P3 720 / S10", phase3_720[identity]),
        ]
        for (label, path), (width, height) in zip(
            tiles,
            ((640, 640), (256, 256), (640, 640), (256, 256), (640, 640)),
        ):
            _assert_source_contract(
                path,
                label=f"{identity} {label}",
                width=width,
                height=height,
            )
        if not _is_valid_comparison(out):
            _render(tiles, out)
        videos[identity] = {
            "comparison": str(out),
            "gt": str(row["gt_video"]),
            "phase1_256": str(phase1_256[identity]),
            "phase1_720": str(phase1_720[identity]),
            "phase3_256": str(phase3_256[identity]),
            "phase3_720": str(phase3_720[identity]),
        }
        print(
            f"[phase1-phase3-resolution] {index + 1}/{len(exact)} {identity}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "n": len(videos),
        "exact_input_jsonl": str(args.exact_input_jsonl),
        "phase1_reuse_manifest": str(args.phase1_reuse_manifest),
        "phase1_inventory": phase1_inventory,
        "phase3_256": [str(path) for path in args.phase3_256],
        "phase3_256_ignored_non_clean_windows": len(phase3_256_all) - len(phase3_256),
        "phase3_720": str(args.phase3_720),
        "phase3_checkpoint": phase3_summary.get("checkpoint"),
        "phase3_high_tier_aggregate": phase3_summary.get("aggregate"),
        "tile_order": [
            "GT",
            "Phase-1 256 tier / shift 3",
            "Phase-1 720 tier / shift 10",
            "Phase-3 256 tier / shift 3",
            "Phase-3 720 tier / shift 10",
        ],
        "frame_provenance": {
            "green": "GT tile, or clean conditioned RGB frame 0 in a generated tile",
            "red": "generated RGB suffix frames 1-96",
        },
        "output_contract": {
            "frames": EXPECTED_FRAMES,
            "fps": int(EXPECTED_FPS),
            "tile_size": TILE_SIZE,
            "width": 5 * TILE_SIZE,
            "height": TILE_SIZE,
        },
        "videos": videos,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    (args.out_dir / "COMPLETE.json").write_text(
        json.dumps(
            {
                "summary": str(summary_path),
                "videos": len(videos),
                "validated": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[phase1-phase3-resolution] wrote and validated {len(videos)} comparisons "
        f"-> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
