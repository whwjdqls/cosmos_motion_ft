"""Differentiable torch decode: 283-D UniEgoMotion (unnormalized) -> world joints.

Self-contained copy of `motion_expert/decode_uniego_torch.py` (bit-matches
`kimodo/motion_rep/uniego.py:uniego_world_joints_from_features`): column-convention
6D->matrix Gram-Schmidt + cumulative canon-frame compose + M = cM @ local_T.
Used to obtain RAW joint positions (T,30,3) from the on-disk uniego features —
the raw-joints input the shape-aware TMR consumes (via kimodo TMRMotionRep).
"""
from __future__ import annotations

import torch

N_JOINTS = 30


def cont6d_to_matrix(c6: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_raw, y_raw = c6[..., 0:3], c6[..., 3:6]
    x = x_raw / (x_raw.norm(dim=-1, keepdim=True) + eps)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / (z.norm(dim=-1, keepdim=True) + eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)  # columns = x,y,z


def _se3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(R.shape[:-2] + (4, 4), device=R.device, dtype=R.dtype)
    out[..., :3, :3] = R
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    return out


def decode_joints(feat: torch.Tensor, n_joints: int = N_JOINTS) -> torch.Tensor:
    """feat [B,T,283] (unnormalized uniego) -> world joints [B,T,J,3] (differentiable)."""
    B, T, _ = feat.shape
    lo = n_joints * 9
    fj = feat[..., :lo].reshape(B, T, n_joints, 9)
    local_T = _se3(cont6d_to_matrix(fj[..., :6]), fj[..., 6:9])          # [B,T,J,4,4]
    fd = feat[..., lo:lo + 9]
    delta = _se3(cont6d_to_matrix(fd[..., :6]), fd[..., 6:9])             # [B,T,4,4]
    cMs = [delta[:, 0]]
    for t in range(1, T):
        cMs.append(cMs[-1] @ delta[:, t])
    cM = torch.stack(cMs, dim=1)                                          # [B,T,4,4]
    M = cM[:, :, None] @ local_T                                         # [B,T,J,4,4]
    return M[..., :3, 3]


if __name__ == "__main__":
    # parity check vs kimodo on a real proportional clip
    import glob
    import sys
    import numpy as np
    sys.path.insert(0, "/home/jungbin_cho/kimodo_open")
    from kimodo.motion_rep.uniego import uniego_world_joints_from_features

    f = sorted(glob.glob("/mnt/shared/jungbin_cho/seed/soma_proportional_uniegomotion_20fps/*/*.npz"))[0]
    feat = np.load(f)["features"][:96].astype(np.float32)
    ft = torch.from_numpy(feat).float()
    mine = decode_joints(ft.unsqueeze(0))[0]
    ref = uniego_world_joints_from_features(ft, n_joints=30)
    err = (mine - ref).abs().max().item()
    print(f"torch decode vs kimodo: max abs err = {err:.2e}  shape={tuple(mine.shape)}")
    print("MATCH" if err < 1e-4 else "MISMATCH")
