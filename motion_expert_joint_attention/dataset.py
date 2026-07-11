"""Joint Nymeria + BONES-SEED 283-D uniego dataset for the joint-attention motion expert.

Both data sources share the same **283-D SOMA-30 UniEgoMotion** representation (`motion_uniego`),
which is the natural unification point: NymeriaPlus ego motion and Cosmos BONES-SEED text→motion
pairs are merged into one loader that emits the identical per-sample dict and is collated by a
shared batch-max padding collate.

Each source is a `UniegoPairsDataset` over a pairs jsonl whose rows look like::

    {"uuid": ..., "uniego_path": "/weka/.../seq.npz", "start": 0, "end": 100,
     "caption": "...", "ground_offset_y": -0.155}   # ground_offset_y optional (Nymeria only)

`__getitem__` reuses the PoC pipeline (`uniego_layout.ground_features` / `canonicalize_frame0`):
  ground (Nymeria only) → crop (random if train, center if val) → frame-0 canon → normalize →
  load + center `neutral_joints` (size cue) → CFG-drop caption to "".

Per-sample output dict::

    {
      "motion":          float32 [T, 283]  (normalized, frame-0-canonicalized),
      "neutral_joints":  float32 [30, 3]   (centered; scale kept as size cue),
      "caption":         str               ("" with prob cfg_dropout during train),
      "motion_pad_mask": bool   [T]        (all False here; collate pads with True),
      "source":          str               ("nymeria" | "bones"),
    }

Variable-length / ragged is the standard path: pass ``T=None`` so each item keeps its full window
length, and the shared `collate` pads to the batch-max T (zeros for motion, True for pad mask). A
fixed window (``T=int``) is supported per-source for ablations, but Nymeria and BONES can only
coexist on the ragged + batch-max path (their natural window lengths differ).

The trainer derives ``noisy_frame_mask`` later (all valid — i.e. non-pad — motion frames are
noised by the rectified-flow forward); this module only marks padding.

NORMALIZATION CAVEAT (read before changing stats)
-------------------------------------------------
The 283-D mean/std used to z-score motion are **stat-set dependent**. By default BOTH sources are
normalized with the SAME shared stats — Nymeria's ``motion_expert/stats/uniego283_{mean,std}.npy`` —
so the two streams land in one normalized space and the single trainable motion expert sees a
consistent distribution. This is an *approximation* for BONES, whose proportional bones-seed tree
was originally z-scored with its own ``Mean_uniego.npy`` / ``Std_uniego.npy`` (see
`motion_expert/bs_dataset.py`). `make_bones(..., proportional_stats=True)` switches BONES to those
per-source proportional stats instead — but then the two sources occupy slightly different
normalized spaces and the decoded-geometry losses / sampler de-normalization must select the right
stats per source. Pick ONE convention and apply it consistently. Default here: shared Nymeria stats
for both, documented as an approximation for BONES.
"""
from __future__ import annotations

import json
import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from uniego_layout import FEAT_DIM, canonicalize_frame0, ground_features

HERE = os.path.dirname(os.path.abspath(__file__))

# Nymeria captions refer to the camera-wearer as a bare "C" ("C is talking with his peer ...").
# Replace the standalone token with "a person" for more natural training text. No-op for BONES
# captions (motion descriptions that contain no standalone "C"). Applied to every emitted caption.
import re as _re
_C_TOKEN = _re.compile(r"\bC\b")


def humanize_caption(caption: str) -> str:
    if not caption:
        return caption
    c = _C_TOKEN.sub("a person", caption)
    return c[:1].upper() + c[1:]  # capitalize the first char ("C is ..." -> "A person is ...")

# ---------------------------------------------------------------------------------------------
# Default stat / pairs locations (shared with / mirrored from the PoC).
# ---------------------------------------------------------------------------------------------
# Shared (Nymeria) 283-D stats — the default normalization for BOTH sources.
SHARED_MEAN_PATH = os.path.join(HERE, "motion_expert", "stats", "uniego283_mean.npy")
SHARED_STD_PATH = os.path.join(HERE, "motion_expert", "stats", "uniego283_std.npy")
# Fall back to the PoC tree if a local copy under this repo does not exist yet.
_POC_STATS = os.path.join(os.path.dirname(HERE), "motion_expert", "stats")
if not os.path.exists(SHARED_MEAN_PATH) and os.path.exists(os.path.join(_POC_STATS, "uniego283_mean.npy")):
    SHARED_MEAN_PATH = os.path.join(_POC_STATS, "uniego283_mean.npy")
    SHARED_STD_PATH = os.path.join(_POC_STATS, "uniego283_std.npy")

# Per-source proportional BONES stats (only used when proportional_stats=True). These live next to
# the proportional bones-seed tree; mirror `motion_expert/bs_dataset.py:{MEAN,STD}_PATH`.
PROPORTIONAL_DATA_ROOT = "/weka/jungbin/seed/soma_proportional_uniegomotion_20fps"
PROPORTIONAL_MEAN_PATH = os.path.join(PROPORTIONAL_DATA_ROOT, "Mean_uniego.npy")
PROPORTIONAL_STD_PATH = os.path.join(PROPORTIONAL_DATA_ROOT, "Std_uniego.npy")

NYMERIA_PAIRS = os.path.join(HERE, "motion_expert", "pairs_{split}.jsonl")
if not os.path.exists(os.path.dirname(NYMERIA_PAIRS)):
    NYMERIA_PAIRS = os.path.join(os.path.dirname(HERE), "motion_expert", "pairs_{split}.jsonl")
# BONES pairs jsonl is built offline by build_bones_pairs.py (kimodo env) into RUNS_ROOT/joint_attention
# on weka (matches config.BONES_PAIRS_TRAIN/VAL + build_bones_pairs.out_dir). Must NOT point at the repo dir.
BONES_PAIRS = "/weka/jungbin/cosmos_motion_ft_runs/joint_attention/bones_pairs_{split}.jsonl"

N_NEUTRAL_JOINTS = 30


class UniegoPairsDataset(Dataset):
    """One 283-D uniego source (Nymeria OR BONES) read from a pairs jsonl.

    Args:
        pairs_jsonl:  path to the jsonl of rows (see module docstring).
        mean_path / std_path: 283-D normalization stats (numpy, shape (283,)).
        T:            fixed window length, or ``None`` for ragged / full-window (the default merge
                      path). When set, crop to T (random if train, center otherwise).
        train:        random crop + caption CFG-dropout when True; deterministic center crop / no
                      dropout when False.
        cfg_dropout:  per-sample probability of replacing the caption with "" (train only).
        source_tag:   value placed in the emitted ``"source"`` field ("nymeria" | "bones").
        ground:       apply `ground_features` using each row's ``ground_offset_y`` (Nymeria). BONES
                      is already grounded → pass False (and rows have no ``ground_offset_y``).
        seed:         RNG seed for crop / dropout reproducibility.
    """

    def __init__(
        self,
        pairs_jsonl: str,
        mean_path: str = SHARED_MEAN_PATH,
        std_path: str = SHARED_STD_PATH,
        T: Optional[int] = None,
        train: bool = True,
        cfg_dropout: float = 0.10,
        source_tag: str = "nymeria",
        ground: bool = True,
        seed: int = 0,
    ):
        with open(pairs_jsonl) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.mean = np.load(mean_path).astype(np.float32)
        self.std = np.load(std_path).astype(np.float32)
        assert self.mean.shape == (FEAT_DIM,) and self.std.shape == (FEAT_DIM,), (
            f"stats must be ({FEAT_DIM},); got mean={self.mean.shape} std={self.std.shape}"
        )
        self.T = None if T is None else int(T)
        self.train = bool(train)
        self.cfg_dropout = float(cfg_dropout)
        self.source_tag = str(source_tag)
        self.ground = bool(ground)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _crop_start(self, n: int, T: int) -> int:
        if n <= T:
            return 0
        return self.rng.randint(0, n - T) if self.train else (n - T) // 2

    def __getitem__(self, i: int):
        r = self.rows[i]
        with np.load(r["uniego_path"]) as npz:
            feats = npz["features"][r["start"]:r["end"]].astype(np.float32)  # [n, 283]
            nj = npz["neutral_joints"].astype(np.float32)                    # [30, 3]

        # Ground (Nymeria): feet → room floor (y≈0). BONES is already grounded.
        off = r.get("ground_offset_y", None)
        if self.ground and off is not None:
            feats = ground_features(feats, off)

        # Crop (random if train, center otherwise) when a fixed T is requested; else keep full.
        if self.T is not None:
            c = self._crop_start(feats.shape[0], self.T)
            feats = feats[c:c + self.T]

        feats = canonicalize_frame0(feats)        # window starts canonically
        feats = (feats - self.mean) / self.std    # normalize (shared or per-source stats)

        m = feats.shape[0]
        # pad mask is all-valid here; the collate pads the ragged batch to its max T.
        pad = np.zeros((m,), dtype=bool)

        nj = nj - nj.mean(axis=0, keepdims=True)  # center (keep scale = size cue)
        assert nj.shape == (N_NEUTRAL_JOINTS, 3), f"neutral_joints must be (30,3); got {nj.shape}"

        caption = humanize_caption(r["caption"])
        if self.train and self.rng.random() < self.cfg_dropout:
            caption = ""

        return {
            "motion": torch.from_numpy(np.ascontiguousarray(feats)),
            "neutral_joints": torch.from_numpy(np.ascontiguousarray(nj)),
            "caption": caption,
            "motion_pad_mask": torch.from_numpy(pad),
            "source": self.source_tag,
        }


# ---------------------------------------------------------------------------------------------
# Source factories.
# ---------------------------------------------------------------------------------------------
def make_nymeria(
    split: str,
    pairs_jsonl: Optional[str] = None,
    mean_path: str = SHARED_MEAN_PATH,
    std_path: str = SHARED_STD_PATH,
    T: Optional[int] = None,
    cfg_dropout: float = 0.10,
    seed: int = 0,
) -> UniegoPairsDataset:
    """Nymeria source over ``motion_expert/pairs_{split}.jsonl`` with shared stats, grounded."""
    pairs = pairs_jsonl or NYMERIA_PAIRS.format(split=split)
    return UniegoPairsDataset(
        pairs_jsonl=pairs,
        mean_path=mean_path,
        std_path=std_path,
        T=T,
        train=(split == "train"),
        cfg_dropout=cfg_dropout,
        source_tag="nymeria",
        ground=True,
        seed=seed,
    )


def make_bones(
    split: str,
    pairs_jsonl: Optional[str] = None,
    mean_path: str = SHARED_MEAN_PATH,
    std_path: str = SHARED_STD_PATH,
    T: Optional[int] = None,
    cfg_dropout: float = 0.10,
    proportional_stats: bool = False,
    seed: int = 1,
) -> UniegoPairsDataset:
    """BONES-SEED source over ``bones_pairs_{split}.jsonl`` (built offline by build_bones_pairs.py).

    Already grounded (``ground=False``). Uses the SAME shared Nymeria stats by default; set
    ``proportional_stats=True`` to normalize with the per-source proportional BONES stats instead
    (see the module-level NORMALIZATION CAVEAT).
    """
    pairs = pairs_jsonl or BONES_PAIRS.format(split=split)
    if proportional_stats:
        mean_path, std_path = PROPORTIONAL_MEAN_PATH, PROPORTIONAL_STD_PATH
    return UniegoPairsDataset(
        pairs_jsonl=pairs,
        mean_path=mean_path,
        std_path=std_path,
        T=T,
        train=(split == "train"),
        cfg_dropout=cfg_dropout,
        source_tag="bones",
        ground=False,
        seed=seed,
    )


# ---------------------------------------------------------------------------------------------
# Joint dataset + per-source sampler.
# ---------------------------------------------------------------------------------------------
def joint_dataset(
    split: str,
    ratio: float = 1.0,
    nymeria_kwargs: Optional[dict] = None,
    bones_kwargs: Optional[dict] = None,
):
    """Concat the Nymeria + BONES sources into one dataset.

    Returns ``(dataset, source_index)`` where ``source_index`` is an int8 array aligned with the
    concatenated dataset (0 = nymeria, 1 = bones), suitable for building a WeightedRandomSampler.

    Both sources keep ``T=None`` (ragged) by default so they coexist on the batch-max collate path.
    ``ratio`` is the desired nymeria:bones sampling ratio (e.g. 1.0 → 1:1); pass it through to
    `make_source_sampler` to realize the target mix.
    """
    nymeria_kwargs = dict(nymeria_kwargs or {})
    bones_kwargs = dict(bones_kwargs or {})
    nymeria = make_nymeria(split, **nymeria_kwargs)
    bones = make_bones(split, **bones_kwargs)
    ds = ConcatDataset([nymeria, bones])
    source_index = np.concatenate([
        np.zeros(len(nymeria), dtype=np.int8),
        np.ones(len(bones), dtype=np.int8),
    ])
    return ds, source_index


def build_joint_dataset(
    data_mix: str = "both",
    split: str = "train",
    T: Optional[int] = None,
    cfg_dropout: float = 0.10,
    mean=None,
    std=None,
    ratio: float = 1.0,
    return_source_index: bool = False,
):
    """train.py-facing constructor: build the dataset for a `data_mix` selection.

    `data_mix` selects the source(s):
        "nymeria" -> Nymeria only, "bones" -> BONES only, "both" -> concatenated joint dataset.

    `T`/`cfg_dropout` are threaded into each `UniegoPairsDataset`. `mean`/`std` are accepted for
    call-site symmetry with the trainer but are IGNORED here: each source normalizes with its own
    configured stat paths (shared Nymeria stats by default; see the module NORMALIZATION CAVEAT).
    train.py passes `mean=None, std=None` and applies its own loaded stats for the decoded losses.

    Returns the dataset by default; with `return_source_index=True` returns
    `(dataset, source_index)` (the latter only meaningful for `data_mix="both"`, else None).
    """
    src_kwargs = dict(split=split, T=T, cfg_dropout=cfg_dropout)
    if data_mix == "nymeria":
        ds = make_nymeria(**src_kwargs)
        source_index = None
    elif data_mix == "bones":
        ds = make_bones(**src_kwargs)
        source_index = None
    elif data_mix == "both":
        ds, source_index = joint_dataset(
            split=split,
            ratio=ratio,
            nymeria_kwargs=dict(T=T, cfg_dropout=cfg_dropout),
            bones_kwargs=dict(T=T, cfg_dropout=cfg_dropout),
        )
    else:
        raise ValueError(f"unknown data_mix={data_mix!r}; expected nymeria|bones|both")

    if return_source_index:
        return ds, source_index
    return ds


def make_source_sampler(
    source_index: np.ndarray,
    ratio: float = 1.0,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that targets a nymeria:bones mix of ``ratio:1``.

    Within each source samples are drawn uniformly; the per-source weight is set so the expected
    number of nymeria vs bones draws matches ``ratio`` regardless of the raw source sizes (e.g.
    ratio=1.0 → balanced 1:1 even if one source is far larger). ``num_samples`` defaults to the full
    dataset size (one epoch's worth of draws).
    """
    src = np.asarray(source_index)
    n0 = int((src == 0).sum())  # nymeria
    n1 = int((src == 1).sum())  # bones
    # target prob mass: nymeria = ratio/(ratio+1), bones = 1/(ratio+1); spread evenly within source.
    p0 = ratio / (ratio + 1.0)
    p1 = 1.0 / (ratio + 1.0)
    w0 = (p0 / n0) if n0 > 0 else 0.0
    w1 = (p1 / n1) if n1 > 0 else 0.0
    weights = np.where(src == 0, w0, w1).astype(np.float64)
    n = int(num_samples) if num_samples is not None else len(src)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights), num_samples=n, replacement=True
    )


# ---------------------------------------------------------------------------------------------
# Collate: ragged → batch-max padded.
# ---------------------------------------------------------------------------------------------
def collate(batch: List[dict]) -> dict:
    """Pad a ragged batch to its max T.

    Returns::
        {
          "motion":          float32 [B, Tmax, 283]  (zeros in padded frames),
          "neutral_joints":  float32 [B, 30, 3],
          "caption":         List[str]               (length B),
          "motion_pad_mask": bool    [B, Tmax]       (True = padded frame, excluded from loss),
          "source":          List[str]               (length B),
        }
    """
    B = len(batch)
    lens = [b["motion"].shape[0] for b in batch]
    Tmax = max(lens)

    motion = torch.zeros((B, Tmax, FEAT_DIM), dtype=torch.float32)
    pad_mask = torch.ones((B, Tmax), dtype=torch.bool)  # default True (pad), flip valid frames off
    for i, b in enumerate(batch):
        m = lens[i]
        motion[i, :m] = b["motion"]
        pad_mask[i, :m] = False

    neutral_joints = torch.stack([b["neutral_joints"] for b in batch], dim=0)  # [B,30,3]
    captions = [b["caption"] for b in batch]
    sources = [b["source"] for b in batch]

    return {
        "motion": motion,
        "neutral_joints": neutral_joints,
        "caption": captions,
        "motion_pad_mask": pad_mask,
        "source": sources,
    }
