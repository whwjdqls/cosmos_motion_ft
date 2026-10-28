"""Validated, immutable phase-3 motion cache paired with existing Wan latents.

Preparation and training must run on compute nodes. There is no video decode,
VAE encode, random replacement, or bad-window retry in the training loader.
"""
from __future__ import annotations

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
from torch.utils.data import Dataset, Sampler

import config
from dataset import humanize_caption
from nymeria_joint_dataset import NymeriaJointDataset, latent_path
from uniego_layout import canonicalize_frame0, ground_features
from native_phase_training.latent_cache_contract import load_latent_cache_contract
from runtime_paths import resolve_legacy_path
from motion_expert_phase3_edge.settings import (
    RECIPE, add_paths, atomic_json, digest_json, require_compute_node, sha256,
)


def identity(path):
    st = Path(resolve_legacy_path(path)).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def read_latents(path):
    with np.load(resolve_legacy_path(path), allow_pickle=False) as archive:
        value = archive["latents"]
    if tuple(value.shape) != (48, 25, 16, 16) or value.dtype != np.float16:
        raise ValueError(f"Invalid latent shape/dtype: {path}: {value.shape}/{value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite cached video: {path}")
    return value


def _prepare_recording(task):
    rows, latent_root, destination, mean_path, std_path = task
    mean, std = np.load(mean_path), np.load(std_path)
    features_out, rows_out, rejected = [], [], []
    source = rows[0]["uni"]
    try:
        with np.load(source, allow_pickle=False) as archive:
            features = archive["features"].astype(np.float32)
            neutral = archive["neutral_joints"].astype(np.float32)
        neutral = neutral - neutral.mean(axis=0, keepdims=True)
        if neutral.shape != (30, 3) or not np.isfinite(neutral).all():
            raise ValueError("invalid neutral joints")
    except Exception as exc:
        return [], [{"uuid": row["uuid"], "start": row["s"], "error": str(exc)} for row in rows]
    checked = {}
    for row in rows:
        path = latent_path(row["uuid"], row["s"], str(latent_root))
        try:
            if path not in checked:
                read_latents(path)
                checked[path] = identity(path)
            motion = features[row["s"]:row["s"] + 97].copy()
            if motion.shape != (97, 283):
                raise ValueError("short or incompatible motion window")
            if row["off"] is not None:
                motion = ground_features(motion, row["off"])
            motion = (canonicalize_frame0(motion) - mean) / std
            if not np.isfinite(motion).all() or np.abs(motion).max() > 20:
                raise ValueError("non-finite or extreme normalized motion")
            rows_out.append({"uuid": row["uuid"], "start": row["s"],
                             "caption": humanize_caption(row["cap"]), "latent": path,
                             "latent_identity": checked[path], "row": len(features_out)})
            features_out.append(motion.astype(np.float32))
        except (OSError, ValueError, KeyError) as exc:
            rejected.append({"uuid": row["uuid"], "start": row["s"], "error": str(exc)})
    if rows_out:
        key = digest_json(rows[0]["uuid"])[:24]
        np.save(Path(destination) / (key + ".motion.npy"), np.stack(features_out), allow_pickle=False)
        np.save(Path(destination) / (key + ".neutral.npy"), neutral, allow_pickle=False)
        for row in rows_out:
            row["array"] = key
    return rows_out, rejected


def prepare(args):
    require_compute_node()
    torch.set_num_threads(1)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with (args.cache.parent / (args.cache.name + ".lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        latent_contract = load_latent_cache_contract(args.latent_root)
        if (latent_contract.num_frames != 97 or latent_contract.fps != 20
                or latent_contract.expected_latent_shape != (48, 25, 16, 16)
                or latent_contract.expected_image_hw != (256, 256)
                or latent_contract.split not in {"train", "all"}):
            raise RuntimeError("Wrong Wan cache contract")
        split = json.loads(args.split_file.read_text())
        if set(split["train"]) & set(split["test"]):
            raise RuntimeError("Recording-level train/test leakage")
        if not args.floor_calibration.is_file():
            raise FileNotFoundError(args.floor_calibration)
        dataset = NymeriaJointDataset(
            manifest_path=str(args.manifest), split_file=str(args.split_file), split="train",
            num_frames=97, aligned_num_frames=97, fps=20, task_weights=RECIPE["tasks"],
            bones_text2motion_frac=0, cfg_dropout=0, prefer_latents=True,
            latent_root=str(args.latent_root), train=False,
            uniego_root=str(config.NYMERIA_UNIEGO_ROOT),
            floor_calibration_json=str(args.floor_calibration), seed=0,
        )
        rows = dataset._index
        if args.limit:
            rows = rows[:args.limit]
        groups = defaultdict(list)
        for row in rows:
            groups[row["uni"]].append(row)
        source = {
            "recipe": RECIPE, "manifest": sha256(args.manifest), "split": sha256(args.split_file),
            "floor": sha256(args.floor_calibration), "mean": sha256(config.MOTION_STATS_MEAN),
            "std": sha256(config.MOTION_STATS_STD), "latent_contract": latent_contract.to_dict(),
            "motion_sources": {name: identity(name) for name in sorted(groups)}, "limit": args.limit,
        }
        fingerprint = digest_json(source)
        if (args.cache / "index.json").is_file():
            saved = json.loads((args.cache / "index.json").read_text())
            if saved["fingerprint"] != fingerprint:
                raise RuntimeError("Existing phase-3 cache has different provenance; choose a new --cache")
            print(f"[phase3/data] reuse validated {len(saved['rows'])} rows in {args.cache}", flush=True)
            return
        if args.cache.exists():
            raise RuntimeError("Incomplete cache destination exists; choose a new --cache")
        temporary = args.cache.with_name(args.cache.name + f".building.{os.getpid()}")
        temporary.mkdir()
        tasks = [(items, str(args.latent_root), str(temporary), str(config.MOTION_STATS_MEAN),
                  str(config.MOTION_STATS_STD)) for _, items in sorted(groups.items())]
        output, rejected = [], []
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, (kept, dropped) in enumerate(pool.map(_prepare_recording, tasks), 1):
                output.extend(kept)
                rejected.extend(dropped)
                if i % 10 == 0 or i == len(tasks):
                    print(f"[phase3/data] recordings={i}/{len(tasks)} kept={len(output)} "
                          f"rejected={len(rejected)} elapsed={time.monotonic()-start:.1f}s", flush=True)
        if not output:
            raise RuntimeError("No valid training windows")
        atomic_json(temporary / "rejected.json", rejected)
        atomic_json(temporary / "index.json", {"schema": 1, "fingerprint": fingerprint,
                    "source": source, "rows": output, "rejected_count": len(rejected),
                    "candidate_count": len(rows), "full_training_index": args.limit == 0})
        os.replace(temporary, args.cache)
        print(f"[phase3/data] complete {args.cache}: {len(output)} valid rows", flush=True)


class Phase3Dataset(Dataset):
    def __init__(self, cache):
        self.root = Path(cache)
        self.index = json.loads((self.root / "index.json").read_text())
        self.rows = self.index["rows"]
        self.arrays = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        draw, index = key if isinstance(key, tuple) else (int(key), int(key))
        row = self.rows[index]
        if identity(row["latent"]) != row["latent_identity"]:
            raise RuntimeError(f"Validated latent changed: {row['latent']}")
        array = row["array"]
        if array not in self.arrays:
            self.arrays[array] = (
                np.load(self.root / (array + ".motion.npy"), mmap_mode="r", allow_pickle=False),
                np.load(self.root / (array + ".neutral.npy"), mmap_mode="r", allow_pickle=False),
            )
        self.arrays.move_to_end(array)
        while len(self.arrays) > 16:
            self.arrays.popitem(last=False)
        motion, neutral = self.arrays[array]
        rng = random.Random(RECIPE["seed"] + 1000003 * (draw + 1))
        mode = "video2motion" if rng.random() < 0.5 else "motimg2video"
        caption = "" if mode == "video2motion" or rng.random() < 0.1 else row["caption"]
        return {"motion": torch.from_numpy(motion[row["row"]].copy()),
                "neutral": torch.from_numpy(neutral.copy()),
                "video": torch.from_numpy(read_latents(row["latent"]).copy()),
                "mode": mode, "caption": caption, "case": f"{row['uuid']}@{row['start']}",
                "draw": draw}


def collate(rows):
    return {"motion": torch.stack([r["motion"] for r in rows]),
            "neutral": torch.stack([r["neutral"] for r in rows]),
            "video": torch.stack([r["video"] for r in rows]),
            "modes": [r["mode"] for r in rows], "captions": [r["caption"] for r in rows],
            "cases": [r["case"] for r in rows], "draws": [r["draw"] for r in rows]}


class GlobalDrawSampler(Sampler):
    """Resumable global draw order independent of worker prefetch and GPU count."""
    def __init__(self, size, start_step, steps, microbatch, rank, world):
        self.size, self.start, self.steps = size, start_step, steps
        self.microbatch, self.rank, self.world = microbatch, rank, world
        if 32 % (microbatch * world):
            raise ValueError("microbatch * world size must divide global batch 32")

    def __iter__(self):
        epoch, permutation = -1, None
        for step in range(self.start, self.steps):
            for offset in range(0, 32, self.microbatch * self.world):
                batch = []
                for j in range(self.microbatch):
                    draw = step * 32 + offset + self.rank * self.microbatch + j
                    current = draw // self.size
                    if current != epoch:
                        epoch = current
                        generator = torch.Generator().manual_seed(RECIPE["seed"] + epoch)
                        permutation = torch.randperm(self.size, generator=generator).tolist()
                    batch.append((draw, permutation[draw % self.size]))
                yield batch

    def __len__(self):
        return (self.steps - self.start) * (32 // (self.microbatch * self.world))


def main():
    parser = add_paths(argparse.ArgumentParser())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Smoke preparation only; production rejects this cache")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
