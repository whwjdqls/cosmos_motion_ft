"""Body-only SOMA subset and train-only shape PCA for both official baselines."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


# SOMA-30 joint indices. Retain wrists/hands as body endpoints; drop articulated
# finger tips, eye joints, and jaw. The same ordered subset feeds both models.
BODY_JOINTS = (0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 16, 17, 18, 19,
               22, 23, 24, 25, 26, 27, 28, 29)
PARENTS = (-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 3, 11, 12, 13,
           0, 15, 16, 17, 0, 19, 20, 21)
CONTACT_JOINTS = (17, 18, 21, 22)  # left ankle/toe, right ankle/toe in subset
MOTION_INDICES = np.array(
    [index for joint in BODY_JOINTS for index in range(joint * 9, joint * 9 + 9)]
    + list(range(270, 283)), dtype=np.int64
)
BODY_FEATURE_DIM = len(MOTION_INDICES)  # 23*9 + 9 + 4 = 220


def shape_vector(neutral: np.ndarray) -> np.ndarray:
    if neutral.shape != (30, 3):
        raise ValueError(f"expected SOMA-30 neutral joints, got {neutral.shape}")
    selected = neutral[list(BODY_JOINTS)]
    return (selected - selected[0]).reshape(-1).astype(np.float64)


def body_features(motion: np.ndarray) -> np.ndarray:
    if motion.shape[-1] != 283:
        raise ValueError(f"expected UniEgo283 source, got {motion.shape}")
    return motion[..., MOTION_INDICES]


def shape_coefficients(neutral: np.ndarray, mean: np.ndarray,
                       components: np.ndarray, scale: np.ndarray) -> np.ndarray:
    score = (shape_vector(neutral) - mean) @ components.T
    return (score / scale).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in os.uname().nodename:
        raise RuntimeError("Fit shape parameters on an owned compute node")
    index = json.loads((args.cache / "index.json").read_text())
    counts = Counter(row["array"] for row in index["rows"] if row["split"] == "train")
    keys = sorted(counts)
    if len(keys) < 17:
        raise ValueError("need at least 17 training actors for a 16-D body shape PCA")
    matrix = np.stack([
        shape_vector(np.load(args.cache / "records" / f"{key}.neutral.npy", allow_pickle=False))
        for key in keys
    ])
    weights = np.asarray([counts[key] for key in keys], dtype=np.float64)
    weights /= weights.sum()
    mean = (weights[:, None] * matrix).sum(axis=0)
    _, singular, right = np.linalg.svd((matrix - mean) * np.sqrt(weights[:, None]),
                                       full_matrices=False)
    components = right[:16]
    scale = singular[:16]
    if np.min(scale) <= 1e-8:
        raise ValueError("Nymeria body shape has fewer than 16 independent components")
    output = args.cache / "body_shape16.npz"
    if output.exists():
        raise FileExistsError(output)
    np.savez(output, mean=mean.astype(np.float32),
             components=components.astype(np.float32), scale=scale.astype(np.float32),
             body_joints=np.asarray(BODY_JOINTS), parents=np.asarray(PARENTS),
             contact_joints=np.asarray(CONTACT_JOINTS), train_recordings=len(keys))
    print(json.dumps({"stage": "body_shape_ready", "path": str(output),
                      "train_recordings": len(keys), "components": 16}), flush=True)


if __name__ == "__main__":
    main()
