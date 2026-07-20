"""Shared constants + helpers for the 283-D SOMA-30 UniEgoMotion rep (`motion_uniego`).

Mirrors `kimodo/motion_rep/uniego.py` for the SOMA-30 / Nymeria device rep so the
MotionExpert POC is self-contained in the `cosmos` env (no kimodo import for train/data).
Decode-to-joints (viz) still uses kimodo's `uniego_world_joints_from_features` (kimodo env).

283-D layout (J=30, head_idx=6, n_foot=4):
    [0  :270)  local_pose     30 joints x [6D rot (cols 0,1) ++ 3D trans]  (trans = joint pos)
    [270:279)  canon_delta    residual head-canonical frame (6D rot ++ 3D trans)
                              frame 0 = absolute cM[0]; frames 1+ = cM[t-1]^-1 cM[t]
    [279:283)  foot_contacts  4 binary flags
"""
from __future__ import annotations

import numpy as np

N_JOINTS = 30
N_FOOT = 4
LOCAL = N_JOINTS * 9          # 270
DELTA = LOCAL + 9             # 279
FEAT_DIM = DELTA + N_FOOT     # 283

LOCAL_SLICE = slice(0, LOCAL)
CANON_DELTA_SLICE = slice(LOCAL, DELTA)
FOOT_SLICE = slice(DELTA, FEAT_DIM)

# Contact-channel order is [L heel/ankle, L toe, R heel/ankle, R toe].
# These are the corresponding joints and raw local-pose Y channels in SOMA-30.
FOOT_JOINT_IDX = (24, 25, 28, 29)
FOOT_Y_IDX = tuple(joint_idx * 9 + 7 for joint_idx in FOOT_JOINT_IDX)

# matrix_to_cont6d(I) = [I[:,0], I[:,1]] = [1,0,0, 0,1,0]; translation = 0
IDENTITY_DELTA9 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


# Y-translation channel index of each joint in the local_pose block (for grounding).
JOINT_Y_IDX = [j * 9 + 7 for j in range(N_JOINTS)]


def ground_features(features: np.ndarray, ground_offset_y: float) -> np.ndarray:
    """Ground a window: subtract `ground_offset_y` from every joint's Y translation.

    The uniego canonical frame is pure-yaw (no vertical component), so each joint's decoded
    height equals its local_pose Y-translation. Grounding the kimodo motion
    (`root_positions[:,1] -= ground_offset_y`) is therefore a rigid vertical shift of all
    joints → subtract from `feat[..., j*9+7]` for every joint. Feet → y≈0 (room floor).
    Accepts (T,283) or (B,T,283); returns a copy.
    """
    out = features.copy()
    out[..., JOINT_Y_IDX] -= float(ground_offset_y)
    return out


def canonicalize_frame0(features: np.ndarray) -> np.ndarray:
    """Set frame-0 `canon_delta` to identity (origin, canonical facing).

    Accepts (T, 283) or (B, T, 283); returns a copy. Drops the absolute world
    placement of the window's first frame so every training window starts
    canonically. Frames 1+ (relative residuals) are untouched. Matches
    `kimodo/motion_rep/uniego.py:canonicalize_frame0`.
    """
    out = features.copy()
    if out.ndim == 2:
        out[0, CANON_DELTA_SLICE] = IDENTITY_DELTA9
    elif out.ndim == 3:
        out[:, 0, CANON_DELTA_SLICE] = IDENTITY_DELTA9
    else:
        raise ValueError(f"expected (T,D) or (B,T,D); got {features.shape}")
    return out
