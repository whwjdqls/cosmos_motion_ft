"""Phase 1: (text, motion_uniego, skeleton) dataset for the MotionExpert POC.

Reads `pairs_{train,val}.jsonl` (built by build_pairs.py), loads the uniego features slice
`features[start:end]`, crops/pads to a fixed window length T, applies the per-window frame-0
canon reset, normalizes with the Phase-0 stats, and loads + centers the per-actor
`neutral_joints` (size cue). Caption is dropped to "" with prob `cfg_dropout` for CFG.

__getitem__ → {
  "motion":          float32 [T, 283]  (normalized, frame-0-canonicalized),
  "neutral_joints":  float32 [30, 3]   (centered),
  "caption":         str,
  "motion_pad_mask": bool   [T]        (True = padded frame, excluded from loss),
}
The frozen reasoner is run on `caption` in the training loop (on-the-fly), not here.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from uniego_layout import FEAT_DIM, canonicalize_frame0, ground_features

HERE = os.path.dirname(os.path.abspath(__file__))


class UniegoTextMotionDataset(Dataset):
    def __init__(
        self,
        pairs_jsonl: str,
        mean_path: str = os.path.join(HERE, "stats", "uniego283_mean.npy"),
        std_path: str = os.path.join(HERE, "stats", "uniego283_std.npy"),
        T: int = 96,
        train: bool = True,
        cfg_dropout: float = 0.10,
        seed: int = 0,
    ):
        self.rows = [json.loads(l) for l in open(pairs_jsonl)]
        self.mean = np.load(mean_path).astype(np.float32)
        self.std = np.load(std_path).astype(np.float32)
        assert self.mean.shape[0] == FEAT_DIM
        self.T = int(T)
        self.train = train
        self.cfg_dropout = float(cfg_dropout)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def _crop_start(self, n: int) -> int:
        if n <= self.T:
            return 0
        return self.rng.randint(0, n - self.T) if self.train else (n - self.T) // 2

    def __getitem__(self, i: int):
        r = self.rows[i]
        feats = np.load(r["uniego_path"])["features"][r["start"]:r["end"]].astype(np.float32)  # [n,283]
        feats = ground_features(feats, r["ground_offset_y"])  # feet → room floor (y≈0)
        n = feats.shape[0]
        c = self._crop_start(n)
        feats = feats[c:c + self.T]                       # up to T frames
        feats = canonicalize_frame0(feats)               # window starts canonically
        feats = (feats - self.mean) / self.std           # normalize

        T = self.T
        motion = np.zeros((T, FEAT_DIM), dtype=np.float32)
        pad = np.ones((T,), dtype=bool)                  # True = pad
        m = feats.shape[0]
        motion[:m] = feats
        pad[:m] = False

        nj = np.load(r["uniego_path"])["neutral_joints"].astype(np.float32)  # [30,3]
        nj = nj - nj.mean(axis=0, keepdims=True)          # center (keep scale = size cue)

        caption = r["caption"]
        if self.train and self.rng.random() < self.cfg_dropout:
            caption = ""

        return {
            "motion": torch.from_numpy(motion),
            "neutral_joints": torch.from_numpy(nj),
            "caption": caption,
            "motion_pad_mask": torch.from_numpy(pad),
        }


def collate(batch):
    return {
        "motion": torch.stack([b["motion"] for b in batch]),
        "neutral_joints": torch.stack([b["neutral_joints"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "motion_pad_mask": torch.stack([b["motion_pad_mask"] for b in batch]),
    }
