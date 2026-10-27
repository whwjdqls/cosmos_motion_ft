"""Nymeria adapters for the unchanged C45 model, representation, and losses."""
from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from decode_uniego import decode_joints
from st_dataset import resample_joints_time
from uniego_layout import ground_features


def require_compute():
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or "login" in host.lower() or host == "cs-gpu-01":
        raise RuntimeError("Nymeria preparation/training/evaluation requires an owned Slurm compute node")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_torch_save(payload, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def read_rows(path):
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty Nymeria manifest: {path}")
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case IDs: {path}")
    return rows


def stats_fingerprint(path):
    root = Path(path)
    files = sorted(root.rglob("*.npy"))
    if not files or not any("mean" in p.name.lower() for p in files) or not any(
        "std" in p.name.lower() for p in files
    ):
        raise ValueError(f"expected TMR mean/std NPY files below {root}")
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def training_contract(args):
    root = Path(args.nymeria_data_dir)
    prepared = json.loads((root / "prepared.json").read_text())
    if prepared["request"]["data_fps"] != args.data_fps:
        raise ValueError("prepared source FPS differs from training data FPS")
    pools = {name: read_rows(root / f"{name}.jsonl") for name in ("train", "validation", "test")}
    uuids = {name: {r["uuid"] for r in rows} for name, rows in pools.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if uuids[a] & uuids[b]:
            raise ValueError(f"recording leakage between {a} and {b}")
    recipe_keys = (
        "batch", "num_workers", "max_steps", "warmup", "lr", "weight_decay",
        "lambda_kl_motion", "lambda_kl_text", "lambda_recon", "lambda_contrastive",
        "cross_modal_recon", "lambda_align", "freeze_text_encoder", "text_use_mean",
        "motion_use_mean", "aug_feat_noise_std", "frozen_dup_filter", "info_nce_temp",
        "queue_size", "text_dup_threshold", "aug_time_jitter_sec", "fps", "data_fps",
        "latent_dim", "ff_size", "enc_layers", "dec_layers", "num_heads", "dropout",
        "grad_clip", "seed",
    )
    recipe = {key: getattr(args, key) for key in recipe_keys}
    text_lr_multiplier = getattr(args, "text_lr_multiplier", 1.0)
    if text_lr_multiplier != 1.0:
        recipe["text_lr_multiplier"] = text_lr_multiplier
    selection = getattr(args, "nymeria_selection_metric", "plain_avg_tie_R03")
    if selection == "R32_R03_mean":
        recipe["retrieval32"] = {"group_size": 32, "seeds": [0, 1, 2],
                                 "positive": "paired_only", "drop_last": True}
    return {
        "schema": "nymeria_c45_v1",
        "dataset": "nymeria",
        "representation": prepared["request"]["representation"],
        "manifests": {name: sha256_file(root / f"{name}.jsonl") for name in pools},
        "stats_sha256": stats_fingerprint(args.stats_path),
        "text_cache_sha256": sha256_file(args.text_emb_cache),
        "eval_text_cache_sha256": sha256_file(args.eval_text_cache or args.text_emb_cache),
        "recipe": recipe,
        "selection": "validation/" + selection,
        "canonicalization": "raw decoded joints -> TMRMotionRep, once",
        "grounding": "manifest ground_offset_y + recorded floor-calibration delta",
        "augmentation": "boundary jitter clipped inside each annotated window",
    }


class NymeriaShapeTMRDataset(Dataset):
    """Immutable captioned windows; no images, CFG dropout, or generator z-scoring."""

    def __init__(self, manifest, motion_rep, *, train=False, time_jitter_sec=0.0,
                 src_fps=20.0, rep_fps=30.0):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        prepared = json.loads((self.root / "prepared.json").read_text())
        if float(prepared["request"]["data_fps"]) != float(src_fps):
            raise ValueError("manifest and requested source FPS disagree")
        self.rows = read_rows(self.manifest)
        self.motion_rep = motion_rep
        self.train = train
        self.jitter = max(0, round(time_jitter_sec * src_fps)) if train else 0
        self.src_fps = src_fps
        self.rep_fps = rep_fps
        self.min_frames = int(prepared["request"]["min_frames"])

    def __len__(self):
        return len(self.rows)

    def raw_window(self, index, augment=False):
        row = self.rows[index]
        array = np.load(self.root / row["features_file"], mmap_mode="r", allow_pickle=False)
        neutral = np.load(self.root / row["neutral_file"], allow_pickle=False)
        if array.shape != (row["num_frames"], 283) or neutral.shape != (30, 3):
            raise ValueError(f"{row['case_id']}: changed cache shape")
        start, end = 0, len(array)
        if augment and self.jitter:
            # Never extend a caption into an adjacent action or another floor segment.
            start = min(max(0, int(torch.randint(-self.jitter, self.jitter + 1, ()))),
                        end - self.min_frames)
            end = min(end, max(start + self.min_frames,
                              end + int(torch.randint(-self.jitter, self.jitter + 1, ()))))
        raw = np.array(array[start:end], dtype=np.float32, copy=True)
        neutral = np.asarray(neutral, dtype=np.float32)
        if not np.isfinite(raw).all() or not np.isfinite(neutral).all():
            raise ValueError(f"{row['case_id']}: non-finite cached input")
        raw = ground_features(raw, row["floor_offset"])
        neutral = neutral - neutral.mean(axis=0, keepdims=True)
        return raw, neutral

    def __getitem__(self, index):
        row = self.rows[index]
        raw, neutral = self.raw_window(index, augment=self.train)
        with torch.no_grad():
            joints = decode_joints(torch.from_numpy(raw).unsqueeze(0))
            joints = resample_joints_time(joints, self.src_fps, self.rep_fps)
            length = joints.shape[1]
            features = self.motion_rep(
                posed_joints=joints, lengths=torch.tensor([length]),
                to_canonicalize=True, to_normalize=True,
            )[0].float()
        if features.shape != (length, 186) or not torch.isfinite(features).all():
            raise ValueError(f"{row['case_id']}: invalid decoded TMR features")
        return {
            "features": features, "length": length, "text": row["text"],
            "neutral_joints": torch.from_numpy(neutral), "source": "nymeria",
        }
