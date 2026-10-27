"""Prepare immutable floor-aware camera features for the exact Phase 3 cohort.

The original calibrated motion and native camera-action caches are never edited.
Statistics are fitted only on the existing full training cache, including the
same window multiplicities. Cache construction runs on an allocated compute node.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch

from motion_expert_phase3_edge.data import identity
from .settings import RECIPE, add_paths, atomic_json, digest_json, require_compute_node, sha256


def trajectory18(positions, rotations, ground_offset):
    from nymeria_baselines.prepare import camera_trajectory18, Y_FROM_Z, IDENTITY_DELTA
    if positions.shape != (97, 3) or rotations.shape != (97, 3, 3):
        raise ValueError("Trajectory must contain exactly 97 aligned camera poses")
    if ground_offset is None or not np.isfinite(ground_offset):
        raise ValueError("Floor-aware conditioning requires a known calibrated floor offset")
    poses = np.tile(np.eye(4, dtype=np.float64), (97, 1, 1))
    poses[:, :3, :3] = Y_FROM_Z @ rotations.astype(np.float64)
    poses[:, :3, 3] = positions.astype(np.float64) @ Y_FROM_Z.T
    poses[:, 1, 3] -= float(ground_offset)
    features = camera_trajectory18(poses)
    features[0, 9:18] = IDENTITY_DELTA
    if features.shape != (97, 18) or not np.isfinite(features).all():
        raise ValueError("Invalid floor-aware camera features")
    return features.astype(np.float32)


def _prepare_recording(task):
    uuid, rows, source, expected_identity, destination = task
    torch.set_num_threads(1)
    if identity(source) != expected_identity:
        raise RuntimeError(f"Camera source differs from validated native cache: {uuid}")
    with np.load(source, allow_pickle=False) as archive:
        positions = archive["cam_world_pos_upright"]
        rotations = archive["cam_world_rot_upright"]
    count = len(rows)
    if sorted(row["row"] for row in rows) != list(range(count)):
        raise RuntimeError(f"Non-contiguous baseline rows: {uuid}")
    features = np.empty((count, 97, 18), dtype=np.float32)
    for row in rows:
        start = int(row["start"])
        features[row["row"]] = trajectory18(positions[start:start + 97],
            rotations[start:start + 97], row["floor_offset"])
    if identity(source) != expected_identity:
        raise RuntimeError(f"Camera source changed during preparation: {uuid}")
    name = rows[0]["array"] + ".trajectory18.npy"
    target = Path(destination) / name
    np.save(target, features, allow_pickle=False)
    values = features.astype(np.float64).reshape(-1, 18)
    return uuid, {"file": name, "file_identity": identity(target), "sha256": sha256(target),
                  "rows": count, "source": str(source), "source_identity": expected_identity}, \
        values.sum(0), np.square(values).sum(0), len(values)


def prepare(args):
    require_compute_node()
    torch.set_num_threads(1)
    import config
    from dataset import humanize_caption
    from nymeria_joint_dataset import NymeriaJointDataset
    from motion_expert_phase3_edge.settings import RECIPE as BASE_RECIPE
    baseline = json.loads((args.cache / "index.json").read_text())
    native = json.loads((args.camera_cache / "index.json").read_text())
    if not baseline["full_training_index"] or native["source"]["baseline_fingerprint"] != baseline["fingerprint"]:
        raise RuntimeError("Trajectory preparation requires the unchanged full motion/native-camera cohort")
    for field, path in (("manifest", args.manifest), ("split", args.split_file),
                        ("floor", args.floor_calibration)):
        if baseline["source"][field] != sha256(path):
            raise RuntimeError(f"Current {field} differs from motion cache provenance")
    source = {"baseline_fingerprint": baseline["fingerprint"],
              "native_camera_fingerprint": native["fingerprint"],
              "manifest": baseline["source"]["manifest"], "split": baseline["source"]["split"],
              "floor": baseline["source"]["floor"], "rows": len(baseline["rows"]),
              "representation": RECIPE["camera_film"],
              "implementation": sha256(__file__),
              "camera_helper": sha256(Path(__file__).parents[1] / "nymeria_baselines" / "prepare.py")}
    if args.reference_run is not None:
        reference = json.loads((args.reference_run / "contract.json").read_text())
        without_film = lambda recipe: {k: v for k, v in recipe.items() if k not in {"experiment", "camera_film", "tasks"}}
        if without_film(reference["recipe"]) != without_film(RECIPE):
            raise RuntimeError("Recipe differs outside the declared FiLM and 50/50 sampling changes")
        if reference["dataset"] != digest_json({"base": baseline["fingerprint"], "camera": native["fingerprint"]}):
            raise RuntimeError("Training cohort differs from the active reference FiLM run")
    args.trajectory_cache.parent.mkdir(parents=True, exist_ok=True)
    with (args.trajectory_cache.parent / (args.trajectory_cache.name + ".lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (args.trajectory_cache / "index.json").is_file():
            saved = json.loads((args.trajectory_cache / "index.json").read_text())
            if saved["source"] != source:
                raise RuntimeError("Existing trajectory cache has a different contract")
            print(f"[trajectory/cache] reusing {source['rows']} windows", flush=True)
            return
        if args.trajectory_cache.exists():
            raise RuntimeError("Incomplete trajectory cache exists; refusing to overwrite")
        raw = NymeriaJointDataset(manifest_path=str(args.manifest), split_file=str(args.split_file),
            split="train", num_frames=97, aligned_num_frames=97, fps=20,
            task_weights=BASE_RECIPE["tasks"], bones_text2motion_frac=0,
            cfg_dropout=0, prefer_latents=True, latent_root=str(args.latent_root),
            train=False, uniego_root=config.NYMERIA_UNIEGO_ROOT,
            floor_calibration_json=str(args.floor_calibration), seed=0)
        offsets = {}
        for row in raw._index:
            key = (row["uuid"], int(row["s"]), humanize_caption(row["cap"]))
            if key in offsets and offsets[key] != row["off"]:
                raise RuntimeError(f"Ambiguous calibrated floor offset: {key}")
            offsets[key] = row["off"]
        groups = defaultdict(list)
        for row in baseline["rows"]:
            key = (row["uuid"], int(row["start"]), row["caption"])
            if key not in offsets or offsets[key] is None:
                raise RuntimeError(f"Cached window lacks a matched calibrated floor: {key}")
            groups[row["uuid"]].append({**row, "floor_offset": offsets[key]})
        temporary = args.trajectory_cache.with_name(args.trajectory_cache.name + f".building.{os.getpid()}")
        temporary.mkdir()
        tasks = [(uuid, rows, native["records"][uuid]["source"],
                  native["records"][uuid]["source_identity"], str(temporary))
                 for uuid, rows in sorted(groups.items())]
        records, total, squares, frames = {}, np.zeros(18), np.zeros(18), 0
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, (uuid, record, sums, sumsquares, nframes) in enumerate(pool.map(_prepare_recording, tasks), 1):
                records[uuid] = record
                total += sums
                squares += sumsquares
                frames += nframes
                print(f"[trajectory/cache] recordings={i}/{len(tasks)} frames={frames} seconds={time.monotonic()-start:.1f}", flush=True)
        if frames != 97 * source["rows"]:
            raise RuntimeError("Trajectory preparation changed the baseline cohort")
        mean = total / frames
        std = np.sqrt(np.maximum(squares / frames - mean * mean, 0))
        constant = std < 1e-6
        std[constant] = 1.0
        normalization = {"mean": mean.tolist(), "std": std.tolist(), "constant_channels": np.flatnonzero(constant).tolist(),
                         "frames": frames, "windows": source["rows"], "split": "train", "std_floor_policy": "std_below_1e-6_becomes_1"}
        content = {"source": source, "records": records, "normalization": normalization}
        atomic_json(temporary / "index.json", {**content, "fingerprint": digest_json(content), "full_training_index": True})
        os.replace(temporary, args.trajectory_cache)
        print(f"[trajectory/cache] COMPLETE {source['rows']} windows; unchanged motion and native camera caches", flush=True)


if __name__ == "__main__":
    parser = add_paths(argparse.ArgumentParser())
    parser.add_argument("--workers", type=int, default=4)
    prepare(parser.parse_args())
