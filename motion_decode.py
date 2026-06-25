"""Self-contained, pure-torch port of kimodo's BONES-SEED motion decode + FK.

NO kimodo import. Works in any torch version (verified in both the kimodo env,
torch 2.4, and the cosmos env, torch 2.13+). Batched and differentiable, so the
FK path can be used as a training loss.

Mirrors:
  - kimodo.skeleton.kinematics.fk / forward_kinematics (canonical, neutral_joints=None)
  - kimodo.geometry.cont6d_to_matrix
  - kimodo.skeleton.transforms.global_rots_to_local_rots
  - kimodo.motion_rep.reps.kimodo_motionrep.KimodoMotionRep.inverse(posed_joints_from="rotations")
  - kimodo.motion_rep.stats.Stats.normalize/unnormalize  ( (x-mean)/sqrt(std**2+eps) )
  - kimodo.scripts.train.KimodoLoss FK term (gamma_7)

369-d / frame layout (SOMASkeleton30, J=30, fps=20):
    [0:3]    smooth_root_pos          (3)
    [3:5]    global_root_heading      (cos, sin) (2)
    [5:95]   local_joints_positions   (30 x 3)   (90)
    [95:275] global_rot_data          (30 x rot6d)  GLOBAL rotations (180)
    [275:365]velocities               (30 x 3)   (90)
    [365:369]foot_contacts            (4)

Stats: cat([global_root(5), body(364)]) -> (369,); normalize = (x-mean)/sqrt(std^2+1e-5).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (mirror the 369-d layout above)
# ---------------------------------------------------------------------------
NJOINTS = 30
FEAT_DIM = 369
STATS_EPS = 1e-5
SMOOTH_L1_BETA = 1.0

SLICES = {
    "smooth_root_pos": slice(0, 3),
    "global_root_heading": slice(3, 5),
    "local_joints_positions": slice(5, 95),
    "global_rot_data": slice(95, 275),
    "velocities": slice(275, 365),
    "foot_contacts": slice(365, 369),
}
# (start, length) used to slice the global (369,) stats into per-block stats.
BLOCK_DIMS = {
    "smooth_root_pos": 3,
    "global_root_heading": 2,
    "local_joints_positions": NJOINTS * 3,
    "global_rot_data": NJOINTS * 6,
    "velocities": NJOINTS * 3,
    "foot_contacts": 4,
}

# Default kimodo bones_seed_full.yaml block weights (see configs + KimodoLoss).
DEFAULT_LOSS_WEIGHTS = {
    "smooth_root_pos": 10.0,
    "global_root_heading": 2.0,
    "local_joints_positions": 10.0,
    "global_rot_data": 10.0,
    "velocities": 3.0,
    "foot_contacts": 4.0,
    "fk": 5.0,
}


# ---------------------------------------------------------------------------
# Skeleton I/O
# ---------------------------------------------------------------------------
def load_skeleton(npz_path: str) -> Dict[str, torch.Tensor]:
    """Load canonical SOMASkeleton30 FK constants.

    Returns a dict with:
      parents : long  (J,)   parent index per joint, root = -1
      offsets : float (J,3)  ABSOLUTE rest joint positions (kimodo's neutral_joints;
                             NOT parent-relative). fk() recenters / differences these.
      root_idx: int
      joint_names, fps (passthrough)
    """
    d = np.load(npz_path, allow_pickle=True)
    out = {
        "parents": torch.from_numpy(d["parents"].astype(np.int64)),
        "offsets": torch.from_numpy(d["offsets"].astype(np.float32)),
        "root_idx": int(d["root_idx"]),
        "joint_names": list(d["joint_names"]) if "joint_names" in d else None,
        "fps": int(d["fps"]) if "fps" in d else 20,
    }
    return out


def _to(skeleton: Dict, device, dtype) -> Dict:
    """Return a shallow copy of skeleton with tensors moved to device/dtype."""
    return {
        "parents": skeleton["parents"].to(device=device),
        "offsets": skeleton["offsets"].to(device=device, dtype=dtype),
        "root_idx": int(skeleton["root_idx"]),
    }


# ---------------------------------------------------------------------------
# Feature unpacking
# ---------------------------------------------------------------------------
def unpack_369(feat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split a [..., 369] tensor into named blocks (any leading batch/time dims).

    Returns blocks reshaped so joint blocks have a trailing joint axis:
      smooth_root_pos        [..., 3]
      global_root_heading    [..., 2]
      local_joints_positions [..., J, 3]
      global_rot_data        [..., J, 6]
      velocities             [..., J, 3]
      foot_contacts          [..., 4]
    """
    assert feat.shape[-1] == FEAT_DIM, f"expected last dim {FEAT_DIM}, got {feat.shape[-1]}"
    lead = feat.shape[:-1]
    out = {
        "smooth_root_pos": feat[..., SLICES["smooth_root_pos"]],
        "global_root_heading": feat[..., SLICES["global_root_heading"]],
        "local_joints_positions": feat[..., SLICES["local_joints_positions"]].reshape(*lead, NJOINTS, 3),
        "global_rot_data": feat[..., SLICES["global_rot_data"]].reshape(*lead, NJOINTS, 6),
        "velocities": feat[..., SLICES["velocities"]].reshape(*lead, NJOINTS, 3),
        "foot_contacts": feat[..., SLICES["foot_contacts"]],
    }
    return out


# ---------------------------------------------------------------------------
# Rotation conversions (mirror kimodo.geometry / transforms)
# ---------------------------------------------------------------------------
def rot6d_to_matrix(x6d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """6D continuous rotation -> (..., 3, 3). Gram-Schmidt on the two columns.

    Exact mirror of kimodo.geometry.cont6d_to_matrix: columns are stacked as
    [x, y, z] where x=normalize(x_raw), z=normalize(cross(x,y_raw)), y=cross(z,x).
    Matches matrix_to_cont6d, which concatenates matrix[...,0] and matrix[...,1]
    (the FIRST TWO COLUMNS).
    """
    assert x6d.shape[-1] == 6, "last dim must be 6"
    x_raw = x6d[..., 0:3]
    y_raw = x6d[..., 3:6]
    x = x_raw / (torch.norm(x_raw, dim=-1, keepdim=True) + eps)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / (torch.norm(z, dim=-1, keepdim=True) + eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)  # columns -> (..., 3, 3)


def global_to_local_rots(global_mats: torch.Tensor, parents: torch.Tensor, root_idx: int) -> torch.Tensor:
    """Convert global rotations (..., J, 3, 3) to local (parent-relative) rotations.

    Mirror of kimodo.skeleton.transforms.global_rots_to_local_rots:
        local[j] = parent_global[j]^T @ global[j], with parent_global[root] = I.
    """
    lead = global_mats.shape[:-3]
    J = global_mats.shape[-3]
    flat = global_mats.reshape(-1, J, 3, 3)  # (N, J, 3, 3)
    parent_rot = flat[:, parents]  # (N, J, 3, 3); parents[root] indexes some joint
    # root's parent rot := identity (override the bogus gather at root)
    eye = torch.eye(3, device=flat.device, dtype=flat.dtype)
    parent_rot = parent_rot.clone()
    parent_rot[:, root_idx] = eye
    parent_inv = parent_rot.transpose(-1, -2)
    # local[j] = parent_global[j]^T @ global[j]  (batched matmul over (N,J,3,3))
    local = torch.matmul(parent_inv, flat)
    return local.reshape(*lead, J, 3, 3)


# ---------------------------------------------------------------------------
# Forward kinematics (mirror kimodo.skeleton.kinematics.fk, neutral_joints=None)
# ---------------------------------------------------------------------------
def fk(
    local_rot_mats: torch.Tensor,
    root_pos: torch.Tensor,
    skeleton: Dict,
    root_positions_is_global: bool = True,
) -> torch.Tensor:
    """Forward kinematics -> world joint positions (..., J, 3). Differentiable.

    Mirrors kimodo's fk + forward_kinematics for the canonical (neutral_joints=None)
    path on a skeleton WITHOUT baked rest_local_rots (somaskel30):
      * rest positions = skeleton.offsets (absolute), with the pelvis offset
        subtracted when root_positions_is_global (so root translation is world-space
        and does not depend on the pelvis offset).
      * accumulate transforms down the hierarchy using parent-relative bone vectors
        (rel_joints = joints - joints[parents]).
      * add root_pos to every joint at the end.

    Args:
        local_rot_mats : (..., J, 3, 3) local (parent-relative) rotations.
        root_pos       : (..., 3) root translation.
        skeleton       : dict from load_skeleton (parents, offsets, root_idx).
    """
    device, dtype = local_rot_mats.device, local_rot_mats.dtype
    sk = _to(skeleton, device, dtype)
    parents = sk["parents"]
    offsets = sk["offsets"]  # (J, 3) absolute rest positions
    root_idx = sk["root_idx"]
    J = offsets.shape[0]

    lead = local_rot_mats.shape[:-3]
    R = local_rot_mats.reshape(-1, J, 3, 3)  # (N, J, 3, 3)
    N = R.shape[0]
    root = root_pos.reshape(-1, 3)  # (N, 3)

    joints = offsets.clone()  # (J, 3)
    if root_positions_is_global:
        joints = joints - joints[root_idx]
    joints = joints.unsqueeze(0).expand(N, J, 3)  # (N, J, 3)

    # parent-relative bone vectors (root keeps its absolute position, which is 0
    # after recentering above).
    rel = joints.clone()
    mask = torch.ones(J, dtype=torch.bool, device=device)
    mask[root_idx] = False
    rel[:, mask] = joints[:, mask] - joints[:, parents[mask]]

    # Per-joint local 4x4 transforms.
    T = torch.zeros(N, J, 4, 4, device=device, dtype=dtype)
    T[:, :, :3, :3] = R
    T[:, :, :3, 3] = rel
    T[:, :, 3, 3] = 1.0

    # Accumulate down the hierarchy level by level (matches forward_kinematics).
    idx_levs = _compute_idx_levels(parents, root_idx)
    glob = torch.zeros_like(T)
    glob[:, root_idx] = T[:, root_idx]
    for indices in idx_levs:
        if indices.numel() == 0:
            continue
        glob[:, indices] = torch.matmul(glob[:, parents[indices]], T[:, indices])

    posed = glob[:, :, :3, 3]  # (N, J, 3) joints, root at origin
    posed = posed + root[:, None, :]  # add world root translation
    return posed.reshape(*lead, J, 3)


def _compute_idx_levels(parents: torch.Tensor, root_idx: int):
    """Group joint indices by hierarchy depth (mirror compute_idx_levels).

    Returns a list of long tensors. Level 0 = direct children of root, etc.
    Assumes joints are topologically ordered (parent index < child index), which
    holds for SOMASkeleton30.
    """
    parents_l = parents.tolist()
    lev = {root_idx: -1}
    levels: list[list[int]] = []
    for i in range(len(parents_l)):
        if i == root_idx:
            continue
        p = parents_l[i]
        depth = lev[p] + 1
        lev[i] = depth
        while depth + 1 > len(levels):
            levels.append([])
        levels[depth].append(i)
    return [torch.tensor(x, dtype=torch.long, device=parents.device) for x in levels]


# ---------------------------------------------------------------------------
# Stats (mirror kimodo.motion_rep.stats.Stats + base assembly)
# ---------------------------------------------------------------------------
def load_stats(stats_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble the (369,) mean/std from the split-layout stats dir.

    mean = cat([global_root/mean (5,), body/mean (364,)]); same for std.
    (local_root stats exist on disk but are NOT part of the 369-d feature.)
    """
    gr_mean = np.load(os.path.join(stats_dir, "global_root", "mean.npy"))
    gr_std = np.load(os.path.join(stats_dir, "global_root", "std.npy"))
    body_mean = np.load(os.path.join(stats_dir, "body", "mean.npy"))
    body_std = np.load(os.path.join(stats_dir, "body", "std.npy"))
    mean = np.concatenate([gr_mean, body_mean]).astype(np.float32)
    std = np.concatenate([gr_std, body_std]).astype(np.float32)
    assert mean.shape == (FEAT_DIM,), f"mean shape {mean.shape} != ({FEAT_DIM},)"
    assert std.shape == (FEAT_DIM,)
    return torch.from_numpy(mean), torch.from_numpy(std)


def normalize(feat: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = STATS_EPS) -> torch.Tensor:
    """(x - mean) / sqrt(std^2 + eps)."""
    mean = mean.to(device=feat.device, dtype=feat.dtype)
    std = std.to(device=feat.device, dtype=feat.dtype)
    return (feat - mean) / torch.sqrt(std**2 + eps)


def unnormalize(feat: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = STATS_EPS) -> torch.Tensor:
    """x * sqrt(std^2 + eps) + mean."""
    mean = mean.to(device=feat.device, dtype=feat.dtype)
    std = std.to(device=feat.device, dtype=feat.dtype)
    return feat * torch.sqrt(std**2 + eps) + mean


# ---------------------------------------------------------------------------
# Full decode (mirror KimodoMotionRep.inverse, posed_joints_from="rotations")
# ---------------------------------------------------------------------------
def reconstruct_root(blocks: Dict[str, torch.Tensor], root_idx: int) -> torch.Tensor:
    """World root position from features (mirror inverse, kimodo_motionrep.py:214-217).

    root = [ local_jp[root].x + smooth_root.x,
             local_jp[root].y,
             local_jp[root].z + smooth_root.z ].
    """
    smooth = blocks["smooth_root_pos"]              # (..., 3)
    local_jp = blocks["local_joints_positions"]     # (..., J, 3)
    rj = local_jp[..., root_idx, :]                 # (..., 3)
    return torch.stack(
        [rj[..., 0] + smooth[..., 0], rj[..., 1], rj[..., 2] + smooth[..., 2]],
        dim=-1,
    )


def decode_features_to_joints(
    feat: torch.Tensor,
    skeleton: Dict,
    is_normalized: bool,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Decode 369-d features to world joint positions [..., J, 3].

    Mirrors KimodoMotionRep.inverse(features, is_normalized, posed_joints_from="rotations"):
      unnormalize -> unpack -> rot6d->global mats -> global->local rots ->
      reconstruct world root -> FK.
    """
    if is_normalized:
        assert stats is not None, "stats=(mean,std) required when is_normalized=True"
        feat = unnormalize(feat, stats[0], stats[1])
    blocks = unpack_369(feat)
    global_mats = rot6d_to_matrix(blocks["global_rot_data"])  # (..., J, 3, 3)
    local_mats = global_to_local_rots(global_mats, skeleton["parents"].to(feat.device), int(skeleton["root_idx"]))
    root_pos = reconstruct_root(blocks, int(skeleton["root_idx"]))  # (..., 3)
    return fk(local_mats, root_pos, skeleton, root_positions_is_global=True)


# ---------------------------------------------------------------------------
# FK consistency loss (mirror KimodoLoss._fk_positions + _fk_term, fk_target="gt")
# ---------------------------------------------------------------------------
def _fk_world_from_feat_rotations(feat_un: torch.Tensor, skeleton: Dict) -> torch.Tensor:
    """FK(predicted rotations) -> world joints. Root reconstructed from features.

    Mirror of KimodoLoss._fk_world_from_pred (fk_kind='standard'): rot6d->global,
    global->local, root from smooth_root + local_jp[root], then skeleton.fk.
    Input feat_un is UNNORMALIZED.
    """
    blocks = unpack_369(feat_un)
    global_mats = rot6d_to_matrix(blocks["global_rot_data"])
    local_mats = global_to_local_rots(global_mats, skeleton["parents"].to(feat_un.device), int(skeleton["root_idx"]))
    root_pos = reconstruct_root(blocks, int(skeleton["root_idx"]))
    return fk(local_mats, root_pos, skeleton, root_positions_is_global=True)


def _gt_world_from_positions_block(feat_un: torch.Tensor, root_idx: int) -> torch.Tensor:
    """Target world joints from the GT local_joints_positions block (fk_target='gt').

    Mirror of KimodoLoss._fk_positions GT branch: world = local_jp with
    smooth_root x/z added to ALL joints. Input feat_un is UNNORMALIZED.
    """
    blocks = unpack_369(feat_un)
    smooth = blocks["smooth_root_pos"]              # (..., 3)
    world = blocks["local_joints_positions"].clone()  # (..., J, 3)
    world[..., 0] = world[..., 0] + smooth[..., None, 0]
    world[..., 2] = world[..., 2] + smooth[..., None, 2]
    return world


def fk_consistency_loss(
    pred_feat: torch.Tensor,
    gt_feat: torch.Tensor,
    skeleton: Dict,
    stats: Tuple[torch.Tensor, torch.Tensor],
    pad_mask: Optional[torch.Tensor] = None,
    beta: float = SMOOTH_L1_BETA,  # accepted for API parity; the FK term uses L1, not smooth-L1
) -> torch.Tensor:
    """kimodo's gamma_7 FK term: mean L1 |FK(pred_rot) - GT_positions| over valid frames.

    Inputs are NORMALIZED 369-d features (pred_feat, gt_feat), [..., T, 369] with a
    leading batch dim, i.e. shape (B, T, 369). pad_mask is (B, T) bool (True = valid);
    if None, all frames are valid.

    Reduction EXACTLY matches KimodoLoss._fk_term:
        num = sum_over(valid frames, J, 3) |pred_posed - tgt_world|
        den = (#valid frames) * J * 3 + 1e-8
        loss = num / den
    Note kimodo's FK term is plain L1 (`.abs()`), NOT smooth-L1; `beta` is ignored
    here and only kept so this matches the per-block loss signature.
    """
    mean, std = stats
    pred_un = unnormalize(pred_feat, mean, std)
    gt_un = unnormalize(gt_feat, mean, std)
    root_idx = int(skeleton["root_idx"])

    pred_posed = _fk_world_from_feat_rotations(pred_un, skeleton)  # (B, T, J, 3)
    tgt_world = _gt_world_from_positions_block(gt_un, root_idx)    # (B, T, J, 3)

    diff = (pred_posed - tgt_world).abs()  # (B, T, J, 3)
    J = pred_posed.shape[-2]
    if pad_mask is None:
        return diff.mean()
    m = pad_mask.to(diff.dtype)[..., None, None]  # (B, T, 1, 1)
    num = (diff * m).sum()
    den = m.sum() * float(J * 3) + 1e-8
    return num / den


def block_smooth_l1_losses(
    pred_feat: torch.Tensor,
    gt_feat: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
    beta: float = SMOOTH_L1_BETA,
) -> Dict[str, torch.Tensor]:
    """Per-block smooth-L1 (Huber) losses, each averaged over valid frames * block_dim.

    Mirror of KimodoLoss.forward per-block loop:
        diff = smooth_l1(pred, target, beta, reduction='none')  over the WHOLE 369-d
        for each block:
            num = sum(diff[block] * mask);  den = mask.sum() * block_dim + 1e-8
            block_loss = num / den
    Returns a dict keyed by block name (no fk term). pred/gt are NORMALIZED features
    of shape (B, T, 369); pad_mask (B, T) bool.
    """
    diff = torch.nn.functional.smooth_l1_loss(pred_feat, gt_feat, reduction="none", beta=beta)  # (B,T,369)
    if pad_mask is None:
        mask = torch.ones(*diff.shape[:-1], 1, device=diff.device, dtype=diff.dtype)
    else:
        mask = pad_mask.to(diff.dtype)[..., None]  # (B, T, 1)
    out: Dict[str, torch.Tensor] = {}
    for name, sl in SLICES.items():
        d_block = diff[..., sl]
        block_n = float(d_block.shape[-1])
        num = (d_block * mask).sum()
        den = mask.sum() * block_n + 1e-8
        out[name] = num / den
    return out


def kimodo_weighted_loss(
    pred_feat: torch.Tensor,
    gt_feat: torch.Tensor,
    skeleton: Dict,
    stats: Tuple[torch.Tensor, torch.Tensor],
    pad_mask: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
    beta: float = SMOOTH_L1_BETA,
) -> Dict[str, torch.Tensor]:
    """Full kimodo-weighted loss on RECONSTRUCTED-x0 normalized features.

    total = ( sum_b w_b * block_loss_b + w_fk * fk_loss ) / sum(all used weights)

    This exactly mirrors KimodoLoss.forward (the denominator is the sum of the
    weights actually used, i.e. weighted-average semantics). Returns a dict with
    every component (detached except the scalar "loss").
    """
    if weights is None:
        weights = DEFAULT_LOSS_WEIGHTS
    blocks = block_smooth_l1_losses(pred_feat, gt_feat, pad_mask=pad_mask, beta=beta)

    total = pred_feat.new_zeros(())
    denom = 0.0
    out: Dict[str, torch.Tensor] = {}
    for name, bl in blocks.items():
        w = float(weights.get(name, 0.0))
        if w == 0.0:
            continue
        out[f"l_{name}"] = bl.detach()
        total = total + w * bl
        denom += w

    fk_w = float(weights.get("fk", 0.0))
    if fk_w > 0.0:
        fk_l = fk_consistency_loss(pred_feat, gt_feat, skeleton, stats, pad_mask=pad_mask, beta=beta)
        out["l_fk"] = fk_l.detach()
        total = total + fk_w * fk_l
        denom += fk_w

    out["loss"] = total / max(denom, 1e-8)
    return out
