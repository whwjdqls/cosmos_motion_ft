"""v2: loader for the precomputed H_R cache (cosmos env, used in training)."""
from __future__ import annotations

import json
import os

import numpy as np
import torch


class HRCache:
    def __init__(self, cache_dir="/weka/jungbin/cosmos_motion_ft_runs/hr_cache", device="cuda"):
        self.dir = cache_dir
        self.device = device
        self.captions = json.load(open(os.path.join(cache_dir, "captions.json")))
        self.cap2id = {c: i for i, c in enumerate(self.captions)}
        # infer nshards from files
        shards = sorted(int(f[5:f.index("_hr")]) for f in os.listdir(cache_dir)
                        if f.startswith("shard") and f.endswith("_hr.npy"))
        self.nshards = max(shards) + 1
        self.hr = [np.load(os.path.join(cache_dir, f"shard{k}_hr.npy"), mmap_mode="r")
                   for k in range(self.nshards)]
        self.offsets = []
        for k in range(self.nshards):
            lens = np.load(os.path.join(cache_dir, f"shard{k}_lens.npy"))
            self.offsets.append(np.concatenate([[0], np.cumsum(lens)]))

    def get(self, caption: str) -> torch.Tensor:
        """caption → H_R [Ti, 4096] float32 on device."""
        cid = self.cap2id.get(caption, 0)         # 0 = "" (null) fallback
        k, p = cid % self.nshards, cid // self.nshards
        off, nxt = self.offsets[k][p], self.offsets[k][p + 1]
        arr = np.asarray(self.hr[k][off:nxt])      # [Ti,4096] fp16
        return torch.from_numpy(arr).to(self.device, torch.float32)

    def null(self) -> torch.Tensor:
        return self.get("")

    def batch(self, captions: list[str]):
        """list[str] → (H_R [B,Tmax,4096], key_padding_mask [B,Tmax] True=pad)."""
        Hs = [self.get(c) for c in captions]
        Tmax = max(h.shape[0] for h in Hs)
        B = len(Hs)
        H = torch.zeros(B, Tmax, Hs[0].shape[1], device=self.device, dtype=torch.float32)
        pad = torch.ones(B, Tmax, dtype=torch.bool, device=self.device)
        for i, h in enumerate(Hs):
            H[i, :h.shape[0]] = h
            pad[i, :h.shape[0]] = False
        return H, pad
