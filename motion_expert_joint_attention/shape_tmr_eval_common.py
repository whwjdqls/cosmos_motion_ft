"""Shared data contracts for Phase-2 C45 shape-aware TMR evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from decode_uniego_torch import decode_joints
from uniego_layout import FEAT_DIM, FOOT_SLICE, canonicalize_frame0, ground_features


DEFAULT_BUNDLE_ROOT = Path("/weka/jungbin/shape_aware_motion_eval_c45_20260715")
DEFAULT_TESTSUITE = Path("/weka/jungbin/Kimodo-Motion-Gen-Benchmark-20fps/testsuite")
DEFAULT_BONES_UNIEGO_ROOT = Path("/weka/jungbin/seed/soma_proportional_uniegomotion_20fps")
SUITES = tuple(
    (split, group)
    for split in ("content", "repetition")
    for group in ("overview", "timeline_single", "timeline_multi")
)


def add_bundle_python_paths(bundle_root: str | os.PathLike[str] = DEFAULT_BUNDLE_ROOT) -> Path:
    """Prepend the bundle-pinned Kimodo and C45 source trees."""
    root = Path(bundle_root).resolve()
    paths = (
        root / "code" / "kimodo_open",
        root / "code" / "cosmos_motion_ft" / "shape_aware_TMR",
    )
    for path in reversed(paths):
        if not path.is_dir():
            raise FileNotFoundError(f"shape-TMR bundle source is missing: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % (2**31)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    cohort: str
    text: str
    num_frames: int
    seed: int
    motion_path: str
    gt_start: int
    gt_end: int
    source_kind: str
    floor_offset: float | None = None
    image_path: str | None = None
    image_start: int | None = None
    uuid: str | None = None

    @classmethod
    def from_dict(cls, row: dict) -> "EvalCase":
        return cls(**row)

    def to_dict(self) -> dict:
        return asdict(self)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[EvalCase]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str | os.PathLike[str]) -> list[EvalCase]:
    with open(path) as handle:
        return [EvalCase.from_dict(json.loads(line)) for line in handle if line.strip()]


def load_case_features(case: EvalCase) -> tuple[np.ndarray, np.ndarray]:
    """Load raw, grounded/canonical UniEgo and centered proportional skeleton."""
    with np.load(case.motion_path, mmap_mode="r") as data:
        features = np.asarray(data["features"][case.gt_start:case.gt_end]).astype(np.float32)
        neutral = np.asarray(data["neutral_joints"]).astype(np.float32)
    if features.ndim != 2 or features.shape[1] != FEAT_DIM:
        raise ValueError(f"{case.case_id}: invalid UniEgo shape {features.shape}")
    if case.source_kind == "nymeria" and case.floor_offset is not None:
        features = ground_features(features, float(case.floor_offset))
    features = canonicalize_frame0(features)
    neutral = neutral - neutral.mean(axis=0, keepdims=True)
    if neutral.shape != (30, 3):
        raise ValueError(f"{case.case_id}: invalid neutral_joints shape {neutral.shape}")
    if not np.isfinite(features).all() or not np.isfinite(neutral).all():
        raise ValueError(f"{case.case_id}: non-finite motion or skeleton")
    return features, neutral


def load_gt_batch(
    cases: list[EvalCase],
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded raw features, decoded joints, contacts, and GT lengths."""
    arrays = [load_case_features(case)[0] for case in cases]
    lengths = torch.tensor([len(x) for x in arrays], dtype=torch.long, device=device)
    max_t = int(lengths.max().item())
    features = torch.empty(
        len(arrays), max_t, FEAT_DIM, dtype=torch.float32, device=device
    )
    for index, array in enumerate(arrays):
        value = torch.from_numpy(array).to(device)
        features[index, : len(array)] = value
        features[index, len(array) :] = value[-1]
    joints = decode_joints(features)
    contacts = features[..., FOOT_SLICE] > 0.5
    return features, joints, contacts, lengths


def load_neutral_batch(cases: list[EvalCase], device: str | torch.device) -> torch.Tensor:
    neutral = [load_case_features(case)[1] for case in cases]
    return torch.from_numpy(np.stack(neutral)).to(device=device, dtype=torch.float32)


def seeded_initial_noise(
    cases: list[EvalCase],
    max_t: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Per-case CPU-seeded noise, invariant to rank, shard, batch, and comparison pass."""
    noise = torch.zeros(len(cases), max_t, FEAT_DIM, dtype=torch.float32)
    for index, case in enumerate(cases):
        generator = torch.Generator(device="cpu").manual_seed(int(case.seed))
        noise[index, : case.num_frames] = torch.randn(
            case.num_frames,
            FEAT_DIM,
            generator=generator,
            dtype=torch.float32,
        )
    return noise.to(device)


def chunked_farthest_indices(target_bones: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    """Exact farthest natural skeleton without materializing an N-by-N distance matrix."""
    target = np.asarray(target_bones, dtype=np.float64)
    if target.ndim != 2 or len(target) < 2 or not np.isfinite(target).all():
        raise ValueError("target_bones must be a finite [N>=2, bones] matrix")
    squared = np.einsum("ij,ij->i", target, target)
    result = np.empty(len(target), dtype=np.int64)
    for start in range(0, len(target), chunk_size):
        end = min(start + chunk_size, len(target))
        distance = squared[start:end, None] + squared[None, :] - 2.0 * target[start:end] @ target.T
        rows = np.arange(start, end)
        distance[np.arange(end - start), rows] = -np.inf
        result[start:end] = np.argmax(distance, axis=1)
    return result

