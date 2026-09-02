#!/usr/bin/env python3
"""Build a versioned Nymeria UniEgo corpus with camera-corrected Head rotation.

This command never edits the source corpus.  It preserves every NPZ member except
``features`` and writes one output file for each source sequence that has an exact,
timestamp-aligned upright RGB-camera sidecar.  A root-level manifest records the
coordinate contract, calibration digest, source coverage, and per-file status.

CPU-only example on the restored server::

    source restored_env.sh
    "$COSMOS_PYTHON" motion_expert_joint_attention/build_camera_head_aligned_uniego.py \
      --workers 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from camera_head_recanonicalization import (
    ARIA_Z_UP_TO_KIMODO_Y_UP,
    FEATURE_DIM,
    load_rotation_head_to_camera,
    recanonicalize_camera_aligned_head,
    rotation_angle_deg,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WEKA_ROOT = Path(os.environ.get("WEKA_ROOT", "/mnt/projects/ll/jungbinc/weka"))
DATA_ROOT = WEKA_ROOT / "nymeriaplus_kimodo_proportional"
DEFAULT_SOURCE_ROOT = DATA_ROOT / "uniego_rep"
DEFAULT_CAMERA_ROOT = DATA_ROOT / "camera_rgb"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "uniego_rep_camhead_v1"
DEFAULT_CALIBRATION = Path(__file__).with_name("head_camera_calibration_train.json")
DEFAULT_SPLIT_FILE = DATA_ROOT / "train_test_split.json"
MANIFEST_NAME = "camera_head_recanonicalization_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _validate_existing(source_path: Path, camera_path: Path, output_path: Path) -> dict[str, Any]:
    with (
        np.load(source_path, allow_pickle=False) as source,
        np.load(camera_path, allow_pickle=False) as camera,
        np.load(output_path, allow_pickle=False) as output,
    ):
        if output.files != source.files:
            raise ValueError(
                f"existing output has different keys: source={source.files}, output={output.files}"
            )
        if output["features"].shape != source["features"].shape:
            raise ValueError("existing output feature shape differs from source")
        if not np.array_equal(source["timestamps_us"], camera["timestamps_us"]):
            raise ValueError("source/camera timestamps differ")
        if not np.array_equal(output["timestamps_us"], source["timestamps_us"]):
            raise ValueError("existing output timestamps differ from source")
        for key in source.files:
            if key != "features" and not np.array_equal(output[key], source[key]):
                raise ValueError(f"existing output changed preserved NPZ member {key!r}")
        if not np.isfinite(output["features"]).all():
            raise ValueError("existing output features contain non-finite values")
        return {
            "frames": int(source["features"].shape[0]),
            "bytes": int(output_path.stat().st_size),
        }


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".tmp.npz", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _convert_one(payload: tuple[str, str, str, str, list[list[float]], bool]) -> dict[str, Any]:
    relative, source_root, camera_root, output_root, calibration_values, overwrite = payload
    started = time.time()
    source_path = Path(source_root) / relative
    camera_path = Path(camera_root) / relative
    output_path = Path(output_root) / relative
    record: dict[str, Any] = {"relative_path": relative}
    if not camera_path.is_file():
        record.update(status="missing_camera", seconds=time.time() - started)
        return record
    try:
        if output_path.is_file() and not overwrite:
            existing = _validate_existing(source_path, camera_path, output_path)
            record.update(status="validated_existing", seconds=time.time() - started, **existing)
            return record

        with np.load(source_path, allow_pickle=False) as source_archive:
            source = {key: source_archive[key] for key in source_archive.files}
        with np.load(camera_path, allow_pickle=False) as camera_archive:
            camera_rotations = camera_archive["cam_world_rot_upright"]
            camera_timestamps = camera_archive["timestamps_us"]
            camera_fps = np.asarray(camera_archive["fps"])

        features = source.get("features")
        if features is None or features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"bad source features shape: {None if features is None else features.shape}")
        if "timestamps_us" not in source:
            raise ValueError("source has no timestamps_us")
        if not np.array_equal(source["timestamps_us"], camera_timestamps):
            raise ValueError("source/camera timestamps are not exactly equal")
        if "fps" in source and not np.array_equal(np.asarray(source["fps"]), camera_fps):
            raise ValueError(f"source/camera fps differ: {source['fps']} versus {camera_fps}")

        result = recanonicalize_camera_aligned_head(
            features,
            camera_rotations,
            np.asarray(calibration_values, dtype=np.float64),
            output_dtype=np.float32,
        )
        correction = rotation_angle_deg(
            np.swapaxes(result.old_world_rotations[:, 6], -1, -2)
            @ result.corrected_head_rotations
        )
        source["features"] = result.features
        _atomic_savez(output_path, source)
        record.update(
            status="converted",
            frames=int(len(features)),
            bytes=int(output_path.stat().st_size),
            head_correction_deg=_summary(correction),
            seconds=time.time() - started,
        )
    except Exception as error:  # noqa: BLE001 - record each bad sequence without losing progress
        record.update(
            status="error",
            error=f"{type(error).__name__}: {error}",
            seconds=time.time() - started,
        )
    return record


def _load_split_coverage(split_file: Path, available: set[str]) -> dict[str, Any]:
    if not split_file.is_file():
        return {"split_file": str(split_file), "status": "missing"}
    with split_file.open() as input_file:
        split = json.load(input_file)
    coverage: dict[str, Any] = {"split_file": str(split_file), "status": "checked"}
    for name in ("train", "test"):
        uuids = split.get(name, [])
        missing = sorted(uuid for uuid in uuids if f"{uuid}.npz" not in available)
        coverage[name] = {
            "expected": len(uuids),
            "covered": len(uuids) - len(missing),
            "missing": missing,
        }
    return coverage


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--camera-root", type=Path, default=DEFAULT_CAMERA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    camera_root = args.camera_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        parser.error("--output-root must differ from --source-root; originals are immutable")
    if args.workers <= 0 or args.progress_every <= 0:
        parser.error("--workers and --progress-every must be positive")
    rotation_head_to_camera, calibration_payload = load_rotation_head_to_camera(args.calibration)

    source_files = sorted(source_root.glob("S*/*.npz"))
    if args.limit is not None:
        source_files = source_files[: args.limit]
    if not source_files:
        raise SystemExit(f"no S*/*.npz files under {source_root}")
    relative_paths = [str(path.relative_to(source_root)) for path in source_files]
    print(
        f"[camhead-v1] source={source_root} camera={camera_root} output={output_root} "
        f"files={len(relative_paths)} workers={args.workers}",
        flush=True,
    )

    work = [
        (
            relative,
            str(source_root),
            str(camera_root),
            str(output_root),
            rotation_head_to_camera.tolist(),
            args.overwrite,
        )
        for relative in relative_paths
    ]
    records: list[dict[str, Any]] = []
    started = time.time()
    if args.workers == 1:
        iterator = map(_convert_one, work)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(_convert_one, work, chunksize=1)
    try:
        for index, record in enumerate(iterator, 1):
            records.append(record)
            if index % args.progress_every == 0 or index == len(work):
                counts: dict[str, int] = {}
                for item in records:
                    counts[item["status"]] = counts.get(item["status"], 0) + 1
                print(
                    f"[camhead-v1] {index}/{len(work)} status={counts} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    finally:
        if args.workers != 1:
            executor.shutdown()

    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    available = {
        record["relative_path"]
        for record in records
        if record["status"] in {"converted", "validated_existing"}
    }
    manifest = {
        "schema_version": 1,
        "kind": "nymeria_uniego_camera_aligned_head_recanonicalization",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "camera_root": str(camera_root),
        "output_root": str(output_root),
        "source_files_considered": len(relative_paths),
        "status_counts": dict(sorted(counts.items())),
        "frames_written_or_validated": int(
            sum(record.get("frames", 0) for record in records if record["status"] != "missing_camera")
        ),
        "bytes_written_or_validated": int(
            sum(record.get("bytes", 0) for record in records if record["status"] != "missing_camera")
        ),
        "calibration": {
            "path": str(args.calibration.resolve()),
            "sha256": _sha256(args.calibration),
            "split": calibration_payload["split"],
            "rotation_head_to_upright_camera": rotation_head_to_camera.tolist(),
        },
        "coordinate_contract": {
            "camera_world_basis_change": ARIA_Z_UP_TO_KIMODO_Y_UP.tolist(),
            "rigid_relation": "T_world_camera = T_world_head @ T_head_camera",
            "corrected_rotation": "R_world_head = (B_aria_to_kimodo @ R_world_camera_upright) @ R_head_camera.T",
            "canonical_frame": "yaw of corrected Head +Z; translation=(Head.x,0,Head.z)",
            "changed_world_quantity": "SOMA Head joint 6 rotation only",
            "preserved_world_quantities": [
                "all joint positions",
                "all non-Head joint rotations",
                "foot contacts",
                "timestamps and all other source NPZ members",
            ],
        },
        "split_coverage": _load_split_coverage(args.split_file, available),
        "records": records,
    }
    _write_json_atomic(output_root / MANIFEST_NAME, manifest)
    print(json.dumps({key: manifest[key] for key in (
        "output_root", "source_files_considered", "status_counts",
        "frames_written_or_validated", "bytes_written_or_validated", "split_coverage"
    )}, indent=2), flush=True)
    if counts.get("error", 0):
        raise SystemExit(f"conversion completed with {counts['error']} error(s)")
    # A limited run is explicitly a smoke/debug subset.  Full production builds must
    # cover every sequence used by training and held-out evaluation.
    if args.limit is None:
        for split_name in ("train", "test"):
            split_record = manifest["split_coverage"].get(split_name, {})
            if split_record.get("missing"):
                raise SystemExit(f"conversion does not cover every {split_name} sequence")


if __name__ == "__main__":
    main()
