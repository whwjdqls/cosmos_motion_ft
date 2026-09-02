#!/usr/bin/env python
"""Visualize all four native Phase 1 outputs from official inference."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from native_phase_training.sanitize_prefix_inference_inputs import runtime_mode_matches


VIDEO_MODES = ("forward_dynamics", "policy", "image2video")
ACTION_PLOT_MODES = ("inverse_dynamics", "policy")
ALL_MODES = ("forward_dynamics", "inverse_dynamics", "policy", "image2video")
INPUT_FILES = {
    "forward_dynamics": "fd_input.jsonl",
    "inverse_dynamics": "invdyn_input.jsonl",
    "policy": "policy_input.jsonl",
    "image2video": "i2v_input.jsonl",
}
GT_BORDER_COLOR = "lime"
GENERATED_BORDER_COLOR = "red"
MODE_SHORT_LABELS = {
    "forward_dynamics": "FD",
    "policy": "POLICY",
    "image2video": "I2V",
}


def _escape_drawtext(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")


def _video_frame_provenance(prefix_length: int, num_frames: int) -> dict[str, Any]:
    """Describe the 0-based GT-condition/generated boundary shown in video tiles."""
    prefix_length = int(prefix_length)
    num_frames = int(num_frames)
    if not 1 <= prefix_length <= num_frames:
        raise ValueError(f"prefix length must be in [1,{num_frames}], got {prefix_length}")
    generated_range: list[int] = [] if prefix_length == num_frames else [prefix_length, num_frames - 1]
    return {
        "frame_indexing": "zero_based_inclusive",
        "gt_reference_frames": [0, num_frames - 1],
        "generated_panel": {
            "gt_condition_frames": [0, prefix_length - 1],
            "generated_frames": generated_range,
        },
        "visual_encoding": {
            "gt_condition": {"border": GT_BORDER_COLOR, "label": "GT CONDITION"},
            "generated": {"border": GENERATED_BORDER_COLOR, "label": "GENERATED"},
        },
    }


def _annotated_video_filter(
    *,
    input_index: int,
    output_label: str,
    width: int,
    height: int,
    header_height: int,
    font_size: int,
    label: str,
    prefix_length: int | None,
) -> str:
    """Build one ffmpeg tile with frame-accurate provenance labels and borders."""
    escaped_label = _escape_drawtext(label)
    border_width = max(4, width // 48)
    filters = [f"[{input_index}:v]scale={width}:{height}"]
    if prefix_length is None:
        filters.extend(
            (
                f"drawbox=x=0:y=0:w=iw:h=ih:color={GT_BORDER_COLOR}:t={border_width}",
                f"pad=iw:ih+{header_height}:0:{header_height}:black",
                f"drawtext=text='{escaped_label}':x=7:y=5:fontcolor={GT_BORDER_COLOR}:fontsize={font_size}",
            )
        )
    else:
        prefix_length = int(prefix_length)
        if prefix_length < 1:
            raise ValueError(f"prefix length must be positive, got {prefix_length}")
        gt_label = _escape_drawtext(f"{label} | GT CONDITION")
        generated_label = _escape_drawtext(f"{label} | GENERATED")
        filters.extend(
            (
                f"drawbox=x=0:y=0:w=iw:h=ih:color={GT_BORDER_COLOR}:t={border_width}:"
                f"enable='lt(n,{prefix_length})'",
                f"drawbox=x=0:y=0:w=iw:h=ih:color={GENERATED_BORDER_COLOR}:t={border_width}:"
                f"enable='gte(n,{prefix_length})'",
                f"pad=iw:ih+{header_height}:0:{header_height}:black",
                f"drawtext=text='{gt_label}':x=7:y=5:fontcolor={GT_BORDER_COLOR}:fontsize={font_size}:"
                f"enable='lt(n,{prefix_length})'",
                f"drawtext=text='{generated_label}':x=7:y=5:fontcolor={GENERATED_BORDER_COLOR}:"
                f"fontsize={font_size}:enable='gte(n,{prefix_length})'",
            )
        )
    return ",".join(filters) + f"[{output_label}]"


def _load_expected_records(eval_root: Path, mode: str) -> list[dict[str, Any]]:
    input_path = eval_root / INPUT_FILES[mode]
    records = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"no evaluation records found in {input_path}")

    names: list[str] = []
    for record in records:
        if record.get("model_mode") != mode:
            raise ValueError(f"{input_path}: expected model_mode={mode!r}, got {record.get('model_mode')!r}")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{input_path}: every record must have a non-empty name")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"{input_path}: duplicate sample names are not allowed")
    return records


def _rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = value[:3]
    second = value[3:6]
    first = first / (np.linalg.norm(first) + 1.0e-8)
    second = second - np.dot(first, second) * first
    second = second / (np.linalg.norm(second) + 1.0e-8)
    return np.stack((first, second, np.cross(first, second)), axis=1)


def _integrate_actions(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = [np.eye(4, dtype=np.float64)]
    current = poses[0]
    for step in action:
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = _rot6d_to_matrix(step[3:9])
        delta[:3, 3] = step[:3]
        current = current @ delta
        poses.append(current.copy())
    stacked = np.stack(poses)
    return stacked[:, :3, 3], stacked[:, :3, :3]


def _load_gt_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        position = data["cam_world_pos"].astype(np.float64)
        rotation = data["cam_world_rot"].astype(np.float64)
    poses = np.tile(np.eye(4, dtype=np.float64), (len(position), 1, 1))
    poses[:, :3, :3] = rotation
    poses[:, :3, 3] = position
    poses = np.linalg.inv(poses[0])[None] @ poses
    return poses[:, :3, 3], poses[:, :3, :3]


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    variance = np.square(source_centered).sum() / len(source) + 1.0e-12
    scale = np.trace(np.diag(singular_values) @ sign) / variance
    translation = target_mean - scale * rotation @ source_mean
    return float(scale), rotation, translation


def _draw_camera(ax: Any, center: np.ndarray, rotation: np.ndarray, color: str, depth: float) -> None:
    right, up, forward = rotation[:, 0], -rotation[:, 1], rotation[:, 2]
    plane_center = center + forward * depth
    half_width, half_height = depth * 0.7, depth * 0.5
    corners = [
        plane_center + right * half_width + up * half_height,
        plane_center + right * half_width - up * half_height,
        plane_center - right * half_width - up * half_height,
        plane_center - right * half_width + up * half_height,
    ]
    for corner in corners:
        ax.plot(*np.asarray([center, corner]).T, color=color, linewidth=0.7, alpha=0.85)
    ax.plot(*np.asarray(corners + [corners[0]]).T, color=color, linewidth=0.8, alpha=0.85)


def _plot_camera_comparison(
    *, action: np.ndarray, gt_camera: Path, output: Path, title: str, n_cameras: int
) -> dict[str, float]:
    predicted_position, predicted_rotation = _integrate_actions(action)
    gt_position, gt_rotation = _load_gt_poses(gt_camera)
    length = min(len(predicted_position), len(gt_position))
    predicted_position = predicted_position[:length]
    predicted_rotation = predicted_rotation[:length]
    gt_position = gt_position[:length]
    gt_rotation = gt_rotation[:length]

    predicted_step = float(np.linalg.norm(np.diff(predicted_position, axis=0), axis=1).mean())
    gt_step = float(np.linalg.norm(np.diff(gt_position, axis=0), axis=1).mean())
    scale, alignment_rotation, translation = _umeyama(predicted_position, gt_position)
    aligned_position = (scale * (alignment_rotation @ predicted_position.T)).T + translation
    aligned_rotation = alignment_rotation[None] @ predicted_rotation
    ate = float(np.sqrt(np.square(aligned_position - gt_position).sum(axis=1).mean()))

    all_positions = np.concatenate((aligned_position, gt_position), axis=0)
    center = all_positions.mean(axis=0)
    radius = float(np.abs(all_positions - center).max() * 1.15 + 1.0e-6)
    camera_indexes = np.linspace(0, length - 1, min(n_cameras, length)).astype(int)

    figure = plt.figure(figsize=(6.4, 5.8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*gt_position.T, color="green", linewidth=1.8, label="GT")
    axis.plot(*aligned_position.T, color="red", linewidth=1.8, label="prediction, aligned")
    axis.scatter(*gt_position[0], color="black", s=28)
    for index in camera_indexes:
        _draw_camera(axis, gt_position[index], gt_rotation[index], "green", radius * 0.16)
        _draw_camera(axis, aligned_position[index], aligned_rotation[index], "red", radius * 0.16)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_title(
        f"{title}\nATE={ate:.3f}m, mean |delta| pred={predicted_step:.4f}, GT={gt_step:.4f}, "
        f"ratio={predicted_step / max(gt_step, 1.0e-8):.2f}x",
        fontsize=9,
    )
    axis.legend(fontsize=8, loc="upper left")
    axis.view_init(elev=24, azim=-60)
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return {"aligned_ate_m": ate, "pred_step_m": predicted_step, "gt_step_m": gt_step}


def _make_video_comparison(
    *, gt_video: Path, generated_video: Path, output: Path, label: str, prefix_length: int
) -> None:
    gt_filter = _annotated_video_filter(
        input_index=0,
        output_label="gt",
        width=384,
        height=384,
        header_height=32,
        font_size=16,
        label="GT REFERENCE",
        prefix_length=None,
    )
    generated_filter = _annotated_video_filter(
        input_index=1,
        output_label="pred",
        width=384,
        height=384,
        header_height=32,
        font_size=14,
        label=label,
        prefix_length=prefix_length,
    )
    filter_complex = f"{gt_filter};{generated_filter};[gt][pred]hstack"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(gt_video),
            "-i",
            str(generated_video),
            "-filter_complex",
            filter_complex,
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )


def _make_prefix_grid(
    *,
    gt_video: Path,
    generated_videos: list[tuple[int, Path, int]],
    output: Path,
    mode: str,
) -> None:
    """Create one GT-plus-five-prefix video for quick checkpoint comparison."""
    inputs: list[tuple[str, Path, int | None]] = [("GT REFERENCE", gt_video, None)] + [
        (f"{MODE_SHORT_LABELS[mode]} P{prefix}", path, prefix)
        for prefix, path, _num_frames in sorted(generated_videos)
    ]
    if len(inputs) > 6:
        raise ValueError(f"prefix grid supports at most six videos, got {len(inputs)}")
    filters: list[str] = []
    for index, (label, _path, prefix_length) in enumerate(inputs):
        filters.append(
            _annotated_video_filter(
                input_index=index,
                output_label=f"v{index}",
                width=256,
                height=256,
                header_height=28,
                font_size=13,
                label=label,
                prefix_length=prefix_length,
            )
        )
    layout: list[str] = []
    for index in range(len(inputs)):
        row, column = divmod(index, 3)
        x = ("0", "w0", "w0+w1")[column]
        y = "0" if row == 0 else "h0"
        layout.append(f"{x}_{y}")
    stack_inputs = "".join(f"[v{index}]" for index in range(len(inputs)))
    filters.append(
        f"{stack_inputs}xstack=inputs={len(inputs)}:layout={'|'.join(layout)}:fill=black[out]"
    )
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for _label, path, _prefix_length in inputs:
        command.extend(("-i", str(path)))
    command.extend(
        ("-filter_complex", ";".join(filters), "-map", "[out]", "-pix_fmt", "yuv420p", str(output))
    )
    subprocess.run(command, check=True)


def _load_successful_output(path: Path, expected_mode: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "success":
        raise ValueError(f"failed inference output at {path}: {payload.get('message')}")
    if not runtime_mode_matches(
        actual_mode=payload.get("args", {}).get("model_mode"),
        canonical_mode=expected_mode,
    ):
        raise ValueError(f"mode mismatch at {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--n-cameras", type=int, default=7)
    args = parser.parse_args()
    output_root = args.out or args.inference_root / "viz"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    covered_modes: set[str] = set()
    prefix_video_groups: dict[tuple[str, str], list[tuple[int, Path, int]]] = {}
    for mode in ALL_MODES:
        expected_records = _load_expected_records(args.eval_root, mode)
        if args.max_samples > 0:
            expected_records = expected_records[: args.max_samples]
        sample_dirs = [args.inference_root / record["name"] for record in expected_records]
        missing_dirs = [str(path) for path in sample_dirs if not path.is_dir()]
        if missing_dirs:
            raise RuntimeError(f"missing {mode} inference outputs: {missing_dirs}")
        for input_record, sample_dir in zip(expected_records, sample_dirs, strict=True):
            result = _load_successful_output(sample_dir / "sample_outputs.json", mode)
            suffix = f"_{mode}"
            source_name = input_record.get("source_name") or sample_dir.name[: -len(suffix)]
            prefix_length = input_record.get("rgb_prefix_length")
            num_frames = int(input_record.get("num_frames", 97))
            effective_prefix_length = int(prefix_length) if prefix_length is not None else 1
            artifact_stem = source_name
            if prefix_length is not None:
                artifact_stem += f"_p{int(prefix_length):03d}"
            gt_dir = args.eval_root / "samples" / source_name
            record: dict[str, Any] = {
                "mode": mode,
                "sample": source_name,
                "rgb_prefix_length": prefix_length,
                "latent_prefix_length": input_record.get("latent_prefix_length"),
                "artifacts": [],
            }

            if mode in VIDEO_MODES:
                comparison_path = output_root / f"{artifact_stem}_{mode}.mp4"
                _make_video_comparison(
                    gt_video=gt_dir / "gt_clip.mp4",
                    generated_video=sample_dir / "vision.mp4",
                    output=comparison_path,
                    label=(
                        mode.replace("_", " ")
                        if prefix_length is None
                        else f"{mode.replace('_', ' ')} prefix={prefix_length}"
                    ),
                    prefix_length=effective_prefix_length,
                )
                record["video_frame_provenance"] = _video_frame_provenance(
                    effective_prefix_length, num_frames
                )
                record["artifacts"].append(str(comparison_path))
                if prefix_length is not None:
                    prefix_video_groups.setdefault((mode, source_name), []).append(
                        (int(prefix_length), sample_dir / "vision.mp4", num_frames)
                    )

            if mode in ACTION_PLOT_MODES:
                action = np.asarray(result["outputs"][0]["content"].get("action"), dtype=np.float64)
                if action.shape != (96, 9):
                    raise ValueError(f"{sample_dir}: expected action [96,9], got {action.shape}")
                camera_path = output_root / f"{artifact_stem}_{mode}_camera.png"
                record["camera_metrics"] = _plot_camera_comparison(
                    action=action,
                    gt_camera=gt_dir / "gt_camera_cosmos.npz",
                    output=camera_path,
                    title=f"{source_name} {mode}" + ("" if prefix_length is None else f" prefix={prefix_length}"),
                    n_cameras=args.n_cameras,
                )
                record["artifacts"].append(str(camera_path))

            manifest.append(record)
            covered_modes.add(mode)
            print(f"[native-viz] {mode}: {artifact_stem}", flush=True)

    for (mode, source_name), generated_videos in sorted(prefix_video_groups.items()):
        grid_path = output_root / f"{source_name}_{mode}_prefix_grid.mp4"
        _make_prefix_grid(
            gt_video=args.eval_root / "samples" / source_name / "gt_clip.mp4",
            generated_videos=generated_videos,
            output=grid_path,
            mode=mode,
        )
        manifest.append(
            {
                "mode": f"{mode}_prefix_grid",
                "sample": source_name,
                "prefixes": [prefix for prefix, _path, _num_frames in sorted(generated_videos)],
                "video_frame_provenance_by_prefix": {
                    str(prefix): _video_frame_provenance(prefix, num_frames)
                    for prefix, _path, num_frames in sorted(generated_videos)
                },
                "artifacts": [str(grid_path)],
            }
        )

    missing_modes = set(ALL_MODES) - covered_modes
    if missing_modes:
        raise RuntimeError(f"no successful outputs found for modes: {sorted(missing_modes)}")
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[native-viz] wrote {len(manifest)} mode/sample records to {output_root}", flush=True)


if __name__ == "__main__":
    main()
