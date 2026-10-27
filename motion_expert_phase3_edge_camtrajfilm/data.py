"""Add immutable camera actions without changing the baseline training cohort."""
import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from motion_expert_phase3_edge.data import (
    Phase3Dataset as BaseDataset, GlobalDrawSampler, identity, read_latents,
)
from .settings import (
    RECIPE, add_paths, atomic_json, digest_json, require_compute_node, sha256,
    has_video, effective_caption,
)


def _prepare_camera(task):
    uuid, rows, camera_root, destination = task
    from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel
    torch.set_num_threads(1)
    source = Path(camera_root) / (uuid + ".npz")
    before = identity(source)
    with np.load(source, allow_pickle=False) as archive:
        positions = archive["cam_world_pos_upright"].astype(np.float32)
        rotations = archive["cam_world_rot_upright"].astype(np.float32)
    count = max(row["row"] for row in rows) + 1
    if sorted(row["row"] for row in rows) != list(range(count)):
        raise RuntimeError(f"Non-contiguous baseline cache rows: {uuid}")
    actions = np.empty((count, 96, 9), dtype=np.float32)
    for row in rows:
        start = int(row["start"])
        pos, rot = positions[start:start + 97], rotations[start:start + 97]
        if pos.shape != (97, 3) or rot.shape != (97, 3, 3):
            raise ValueError(f"Camera window is not aligned to T97: {uuid}@{start}")
        if not np.isfinite(pos).all() or not np.isfinite(rot).all():
            raise ValueError(f"Non-finite camera condition: {uuid}@{start}")
        poses = np.tile(np.eye(4), (97, 1, 1))
        poses[:, :3, :3], poses[:, :3, 3] = rot, pos
        action = pose_abs_to_rel(poses, rotation_format="rot6d", pose_convention="backward_framewise")
        if action.shape != (96, 9) or not np.isfinite(action).all():
            raise ValueError(f"Invalid native camera action: {uuid}@{start}")
        actions[row["row"]] = action
    if identity(source) != before:
        raise RuntimeError(f"Camera source changed during preparation: {source}")
    name = rows[0]["array"] + ".camera.npy"
    target = Path(destination) / name
    np.save(target, actions, allow_pickle=False)
    return uuid, {"source": str(source), "source_identity": before,
                  "file": name, "sha256": sha256(target), "rows": count}


def prepare(args):
    require_compute_node()
    torch.set_num_threads(1)
    baseline = json.loads((args.cache / "index.json").read_text())
    if not baseline["full_training_index"]:
        raise RuntimeError("Camera preparation requires the full validated baseline cohort")
    groups = defaultdict(list)
    for row in baseline["rows"]:
        groups[row["uuid"]].append(row)
    source = {"baseline_fingerprint": baseline["fingerprint"],
              "representation": RECIPE["camera_representation"],
              "rows": len(baseline["rows"]),
              "camera_sources": {uuid: identity(args.camera_root / (uuid + ".npz")) for uuid in sorted(groups)}}
    args.camera_cache.parent.mkdir(parents=True, exist_ok=True)
    with (args.camera_cache.parent / (args.camera_cache.name + ".lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (args.camera_cache / "index.json").is_file():
            saved = json.loads((args.camera_cache / "index.json").read_text())
            if saved["source"] != source:
                raise RuntimeError("Existing camera cache has different provenance")
            print(f"[camcond/cache] reusing {source['rows']} aligned windows", flush=True)
            return
        if args.camera_cache.exists():
            raise RuntimeError("Incomplete camera cache exists; do not overwrite it")
        temporary = args.camera_cache.with_name(args.camera_cache.name + f".building.{os.getpid()}")
        temporary.mkdir()
        tasks = [(uuid, rows, str(args.camera_root), str(temporary)) for uuid, rows in sorted(groups.items())]
        records = {}
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, (uuid, record) in enumerate(pool.map(_prepare_camera, tasks), 1):
                records[uuid] = record
                print(f"[camcond/cache] recordings={i}/{len(tasks)} windows={sum(r['rows'] for r in records.values())} seconds={time.monotonic()-start:.1f}", flush=True)
        if sum(record["rows"] for record in records.values()) != len(baseline["rows"]):
            raise RuntimeError("Camera cache changed the baseline cohort")
        content = {"source": source, "records": records}
        atomic_json(temporary / "index.json", {**content, "fingerprint": digest_json(content), "full_training_index": True})
        os.replace(temporary, args.camera_cache)
        print(f"[camcond/cache] COMPLETE: {source['rows']} windows; no dropped or substituted cases", flush=True)


class Phase3Dataset(BaseDataset):
    def __init__(self, cache, camera_cache, trajectory_cache):
        super().__init__(cache)
        self.camera_root = Path(camera_cache)
        self.camera_index = json.loads((self.camera_root / "index.json").read_text())
        if self.camera_index["source"]["baseline_fingerprint"] != self.index["fingerprint"]:
            raise RuntimeError("Camera cache is not paired with this baseline cohort")
        self.index = {**self.index, "fingerprint": digest_json({
            "base": self.index["fingerprint"], "camera": self.camera_index["fingerprint"]})}
        self.camera_arrays = OrderedDict()
        self.reference_fingerprint = self.index["fingerprint"]
        self.trajectory_root = Path(trajectory_cache)
        self.trajectory_index = json.loads((self.trajectory_root / "index.json").read_text())
        source = self.trajectory_index["source"]
        if (source["baseline_fingerprint"] != self.camera_index["source"]["baseline_fingerprint"]
                or source["native_camera_fingerprint"] != self.camera_index["fingerprint"]
                or source["rows"] != len(self.rows) or not self.trajectory_index["full_training_index"]):
            raise RuntimeError("Trajectory cache does not match the unchanged training cohort")
        stats = self.trajectory_index["normalization"]
        self.trajectory_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.trajectory_std = np.asarray(stats["std"], dtype=np.float32)
        if (self.trajectory_mean.shape != (18,) or self.trajectory_std.shape != (18,)
                or not np.isfinite(self.trajectory_mean).all()
                or not np.isfinite(self.trajectory_std).all() or (self.trajectory_std <= 0).any()):
            raise RuntimeError("Invalid camera-only training statistics")
        self.index = {**self.index, "fingerprint": digest_json({
            "reference": self.reference_fingerprint, "trajectory": self.trajectory_index["fingerprint"]})}
        self.trajectory_arrays = OrderedDict()

    def __getitem__(self, key):
        return self.get_item(key)

    def get_item(self, key, force_mode=None, dropout=True):
        draw, index = key if isinstance(key, tuple) else (int(key), int(key))
        row = self.rows[index]
        rng = random.Random(RECIPE["seed"] + 1000003 * (draw + 1))
        value, cumulative = rng.random(), 0.0
        mode = None
        for candidate, probability in RECIPE["tasks"].items():
            cumulative += probability
            if value < cumulative:
                mode = candidate
                break
        mode = force_mode or mode
        if mode not in RECIPE["tasks"]:
            raise ValueError(mode)
        caption = effective_caption(mode, row["caption"])
        if dropout and caption and rng.random() < RECIPE["cfg_dropout"]:
            caption = ""
        array = row["array"]
        if array not in self.arrays:
            self.arrays[array] = (np.load(self.root / (array + ".motion.npy"), mmap_mode="r"),
                                  np.load(self.root / (array + ".neutral.npy"), mmap_mode="r"))
        self.arrays.move_to_end(array)
        while len(self.arrays) > 16:
            self.arrays.popitem(last=False)
        motion, neutral = self.arrays[array]
        record = self.camera_index["records"][row["uuid"]]
        if identity(record["source"]) != record["source_identity"]:
            raise RuntimeError("Validated camera source changed")
        if array not in self.camera_arrays:
            self.camera_arrays[array] = np.load(self.camera_root / record["file"], mmap_mode="r")
        self.camera_arrays.move_to_end(array)
        while len(self.camera_arrays) > 16:
            self.camera_arrays.popitem(last=False)
        camera = np.zeros((96, 64), dtype=np.float32)
        camera[:, :9] = self.camera_arrays[array][row["row"]]
        trajectory_record = self.trajectory_index["records"][row["uuid"]]
        if trajectory_record["source_identity"] != record["source_identity"]:
            raise RuntimeError("Direct and native camera source identities disagree")
        if array not in self.trajectory_arrays:
            path = self.trajectory_root / trajectory_record["file"]
            if identity(path) != trajectory_record["file_identity"]:
                raise RuntimeError("Validated trajectory cache changed")
            self.trajectory_arrays[array] = np.load(path, mmap_mode="r")
        self.trajectory_arrays.move_to_end(array)
        while len(self.trajectory_arrays) > 16:
            self.trajectory_arrays.popitem(last=False)
        features = (self.trajectory_arrays[array][row["row"]] - self.trajectory_mean) / self.trajectory_std
        if has_video(mode):
            if identity(row["latent"]) != row["latent_identity"]:
                raise RuntimeError("Validated video latent changed")
            video = read_latents(row["latent"]).copy()
        else:
            # Batch-shape placeholder only; never passed to the native packer.
            video = np.zeros((48, 25, 16, 16), dtype=np.float16)
        return {"motion": torch.from_numpy(motion[row["row"]].copy()),
                "neutral": torch.from_numpy(neutral.copy()), "video": torch.from_numpy(video),
                "camera_action": torch.from_numpy(camera), "camera_features": torch.from_numpy(features.copy()),
                "mode": mode, "caption": caption,
                "case": f"{row['uuid']}@{row['start']}", "draw": draw}


def collate(rows):
    return {**{key: torch.stack([row[key] for row in rows])
               for key in ("motion", "neutral", "video", "camera_action", "camera_features")},
            "modes": [row["mode"] for row in rows], "captions": [row["caption"] for row in rows],
            "cases": [row["case"] for row in rows], "draws": [row["draw"] for row in rows]}


if __name__ == "__main__":
    parser = add_paths(argparse.ArgumentParser())
    parser.add_argument("--workers", type=int, default=4)
    prepare(parser.parse_args())
