"""Train-only leaf completion for common SOMA23 -> SOMA30 TMR evaluation."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from nymeria_baselines.body_shape import BODY_JOINTS


KEEP = np.asarray(BODY_JOINTS, dtype=np.int64)
MISSING = np.asarray([j for j in range(30) if j not in BODY_JOINTS], dtype=np.int64)


def rotation6d(values):
    x, y = values[..., :3].copy(), values[..., 3:6].copy()
    first = np.linalg.norm(x, axis=-1, keepdims=True)
    if (first < 1e-8).any():
        raise ValueError("Degenerate training rotation")
    x /= first
    z = np.cross(x, y)
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    if (norm < 1e-8).any():
        raise ValueError("Degenerate training rotation axes")
    z /= norm
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=-1)


def fit_template(cache: Path, parents, test_uuids):
    """Equal-record averages from one accepted T97 train window per recording.

    Raw UniEgo per-joint positions/rotations share a frame, so R_parent.T @
    (p_leaf-p_parent) is independent of floor offsets and global canonical yaw.
    This also respects the cam/head-aligned Head axes, unlike assuming that
    their rest rotation is identity.
    """
    index = json.loads((cache / "index.json").read_text())
    groups = defaultdict(list)
    for row in index["rows"]:
        if row["split"] == "train":
            groups[row["uuid"]].append(row)
    if not groups or set(groups) & set(test_uuids):
        raise RuntimeError("Template calibration requires disjoint training recordings")
    parents = np.asarray(parents, dtype=np.int64)
    parent = parents[MISSING]
    anchor = parents[parent]
    if any(int(j) not in BODY_JOINTS for j in np.concatenate((parent, anchor))):
        raise RuntimeError("Completion expects only omitted leaves with retained parent bones")
    local, rest, provenance = [], [], []
    for uuid in sorted(groups):
        rows = sorted(groups[uuid], key=lambda r: (r["start"], r["caption"]))
        row = rows[len(rows) // 2]
        base = cache / "records" / row["array"]
        neutral = np.asarray(np.load(str(base) + ".neutral.npy"), dtype=np.float64)
        motion = np.load(str(base) + ".motion.npy", mmap_mode="r")
        window = np.array(motion[row["start"]:row["start"] + 97:8, :270], dtype=np.float64).reshape(-1, 30, 9)
        if window.shape != (13, 30, 9) or neutral.shape != (30, 3):
            raise RuntimeError("Invalid accepted calibration window")
        scale = np.linalg.norm(neutral[parent] - neutral[anchor], axis=-1)
        if not np.isfinite(window).all() or not np.isfinite(neutral).all() or (scale < 1e-5).any():
            raise RuntimeError("Invalid training attachment geometry")
        rotation = rotation6d(window[:, parent, :6])
        delta = window[:, MISSING, 6:9] - window[:, parent, 6:9]
        offset = np.einsum("tnji,tnj->tni", rotation, delta)
        local.append((offset / scale[None, :, None]).mean(axis=0))
        rest.append((neutral[MISSING] - neutral[parent]) / scale[:, None])
        provenance.append(dict(uuid=uuid, array=row["array"], start=int(row["start"]),
                               frames=list(range(int(row["start"]), int(row["start"]) + 97, 8)),
                               sampled_motion_sha256=hashlib.sha256(window.tobytes()).hexdigest(),
                               neutral_sha256=hashlib.sha256(neutral.tobytes()).hexdigest()))
    return dict(
        schema="body23_leaf_completion_v1", kept=KEEP.tolist(), missing=MISSING.tolist(),
        parents=parents.tolist(), leaf_parents=parent.tolist(), scale_anchors=anchor.tolist(),
        local_ratio=np.mean(local, axis=0).tolist(), rest_ratio=np.mean(rest, axis=0).tolist(),
        train_recordings=len(groups), train_frames=13 * len(groups), records=provenance,
        policy="Equal-record train-only mean attachment offsets divided by retained parent-bone length. No held-out poses, body fitting, floor correction, or changes to retained body joints.",
    )


def complete_body(joints, rotations, neutral23, template):
    joints = np.asarray(joints, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float64)
    neutral23 = np.asarray(neutral23, dtype=np.float64)
    if joints.shape != (97, 23, 3) or rotations.shape != (97, 23, 3, 3) or neutral23.shape != (23, 3):
        raise ValueError("Expected T97 body23 geometry and a single body23 neutral skeleton")
    if not all(np.isfinite(x).all() for x in (joints, rotations, neutral23)):
        raise ValueError("Nonfinite body geometry")
    lookup = {int(j): i for i, j in enumerate(KEEP)}
    parent = np.asarray(template["leaf_parents"])
    anchor = np.asarray(template["scale_anchors"])
    p = np.asarray([lookup[int(j)] for j in parent])
    a = np.asarray([lookup[int(j)] for j in anchor])
    scale = np.linalg.norm(neutral23[p] - neutral23[a], axis=-1)
    if (scale < 1e-5).any():
        raise ValueError("Predicted shape has a degenerate attachment parent bone")
    full = np.empty((97, 30, 3), dtype=np.float32)
    full[:, KEEP] = joints
    offsets = np.asarray(template["local_ratio"]) * scale[:, None]
    full[:, MISSING] = joints[:, p] + np.einsum("tnij,nj->tni", rotations[:, p], offsets)
    neutral = np.empty((30, 3), dtype=np.float32)
    neutral[KEEP] = neutral23 - neutral23[:1]
    neutral[MISSING] = neutral[parent] + np.asarray(template["rest_ratio"]) * scale[:, None]
    if not np.array_equal(full[:, KEEP], joints):
        raise RuntimeError("Adapter altered a retained body joint")
    return full, neutral


def neutral_from_pca(coefficients, shape):
    """One clip-level neutral shape; do not alter the original moving joints."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[1] not in (10, 16):
        raise ValueError("Expected per-frame predicted SOMA PCA10 or PCA16")
    n = coefficients.shape[1]
    value = shape["mean"] + (coefficients.mean(axis=0) * shape["scale"][:n]) @ shape["components"][:n]
    neutral = value.reshape(23, 3)
    return neutral - neutral[:1]
