"""Paired C45 retrieval in seeded 32-candidate groups; no caption-positive merging."""
from __future__ import annotations
import numpy as np


def paired_retrieval32(motion, text, seeds=(0, 1, 2)):
    motion = np.asarray(motion, dtype=np.float64)
    text = np.asarray(text, dtype=np.float64)
    if motion.ndim != 2 or motion.shape != text.shape or len(motion) < 32:
        raise ValueError("Expected aligned N x D embeddings with N >= 32")
    if not np.isfinite(motion).all() or not np.isfinite(text).all():
        raise ValueError("Nonfinite retrieval embeddings; refusing to drop cases")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Use exactly three distinct fixed grouping seeds")
    n = len(motion)
    used = n // 32 * 32
    values = {k: [] for k in (1, 2, 3)}
    result = {"R32_cases_per_seed": used, "R32_dropped_per_seed": n - used}
    for seed in seeds:
        groups = np.random.default_rng(seed).permutation(n)[:used].reshape(-1, 32)
        m, t = motion[groups], text[groups]
        distances = np.maximum(
            (t * t).sum(-1)[:, :, None] + (m * m).sum(-1)[:, None, :]
            - 2 * np.einsum("gqd,gkd->gqk", t, m), 0,
        )
        candidates = np.argsort(distances, axis=-1, kind="stable")
        ranks = np.argmax(candidates == np.arange(32)[None, :, None], axis=-1) + 1
        for k in values:
            value = float((ranks <= k).mean() * 100)
            values[k].append(value)
            result[f"R32_R{k:02d}_seed{seed}"] = value
    for k, series in values.items():
        result[f"R32_R{k:02d}_mean"] = float(np.mean(series))
        result[f"R32_R{k:02d}_std"] = float(np.std(series, ddof=1))
    return result
