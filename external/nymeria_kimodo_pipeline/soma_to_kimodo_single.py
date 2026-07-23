"""Convert a NymeriaPlus per-sequence xdata_soma.npz into the kimodo-compatible
motion NPZ format (local_rot_mats + root_positions + timestamps_us @ 20 fps).

Input file (per sequence, from process_nymeriaplus.py):
  body/xdata_soma.npz with keys:
    poses          (T, 77, 3)   rotvec, T-pose-relative
    transl         (T, 3)       meters (world frame)
    identity_coeffs (1, 10)
    joint_names    (77,)        SOMA 77-joint order
    joint_orient   (77, 3, 3)   SOMA T-pose joint orientations
    rotation_repr  "rotvec"
    absolute_pose  False
    unit           "meters"
    keep_root      False
    timestamps_us  (T,)         int64 native (240 Hz)

Output file (kimodo motion NPZ):
  local_rot_mats   (T_out, 77, 3, 3)  float32  joint-parent-local rotmats
  root_positions   (T_out, 3)          float32  world meters
  timestamps_us    (T_out,)            int64    chosen at target_fps
  fps              ()                  int64    target fps (20)

Math:
  R_rel = rotvec_to_matrix(poses)                              # (T, 77, 3, 3)
  orient_parent_T = joint_orient[parent_ids].T                 # (77, 3, 3)
  local_rot_mats = orient_parent_T[None] @ R_rel @ joint_orient[None]
  (equiv. to soma.geometry.rig_utils.apply_joint_orient_local)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

SOMA_ROOT = Path("/home/jungbin_cho/SOMA-X")
if str(SOMA_ROOT) not in sys.path:
    sys.path.insert(0, str(SOMA_ROOT))

from soma.soma import SOMALayer  # noqa: E402
from soma.geometry.rig_utils import apply_joint_orient_local  # noqa: E402
from soma.geometry.transforms import rotvec_to_matrix  # noqa: E402


def pick_indices_at_fps(timestamps_us: np.ndarray, target_fps: float) -> np.ndarray:
    """Choose frame indices closest to a uniform target_fps grid (timestamp-aware).

    Robust to dropped frames; falls back to stride if timestamps are exactly uniform.
    """
    t0, t1 = int(timestamps_us[0]), int(timestamps_us[-1])
    n_out = max(1, int((t1 - t0) / 1e6 * target_fps))
    query = np.linspace(t0, t1, n_out).astype(np.int64)
    idx = np.searchsorted(timestamps_us, query)
    idx = np.clip(idx, 0, len(timestamps_us) - 1)
    return np.unique(idx)


def convert_one(soma_npz: Path, out_path: Path, target_fps: float, soma_layer: SOMALayer,
                save_intermediate: bool = False) -> dict:
    """Convert one xdata_soma.npz -> kimodo NPZ. Returns metadata dict."""
    z = np.load(soma_npz)
    poses = z["poses"].astype(np.float32)             # (T, 77, 3) rotvec T-pose-rel
    transl = z["transl"].astype(np.float32)           # (T, 3) m
    timestamps_us = z["timestamps_us"].astype(np.int64)  # (T,)
    joint_names_npz = list(z["joint_names"])
    joint_orient_npz = z["joint_orient"].astype(np.float32)  # (77, 3, 3) saved orient

    # Sanity: npz drops the SOMA "Root" joint (keep_root=False) so it has J=77,
    # whereas SOMALayer.rig_data["joint_names"] has J=78 with Root at index 0.
    # Verify the rest aligns to kimodo's SOMASkeleton77 convention (which also
    # starts at Hips with parent=None).
    soma_joint_names = list(soma_layer.rig_data["joint_names"])
    assert joint_names_npz == soma_joint_names[1:], (
        f"joint order mismatch:\n  npz = {joint_names_npz[:3]}...\n  layer[1:] = {soma_joint_names[1:4]}..."
    )

    # Slice layer's orient + parent_T to match the npz 77-joint convention by
    # dropping the Root entry.  In the 78-joint space, joint 1 (Hips) had its
    # parent_T pointing at joint 0 (Root); after slicing index 0, the new index
    # 0 (Hips) keeps a parent_T that still references the original Root T-pose
    # orient -- which is exactly what we need for the Hips world->local math.
    orient_78 = soma_layer._t_pose_orient.detach().cpu().numpy().astype(np.float32)        # (78, 3, 3)
    orient_parent_T_78 = soma_layer._t_pose_orient_parent_T.detach().cpu().numpy().astype(np.float32)
    orient = orient_78[1:]            # (77, 3, 3)
    orient_parent_T = orient_parent_T_78[1:]  # (77, 3, 3)

    # joint_orient in the npz is the full 78-joint table; cross-check
    err = float(np.max(np.abs(joint_orient_npz - orient_78)))
    if err > 1e-5:
        print(f"  WARNING: joint_orient in npz differs from layer by max={err:.2e}")

    # Subsample to target fps
    idx = pick_indices_at_fps(timestamps_us, target_fps)
    poses_sub = poses[idx]            # (T_out, 77, 3)
    transl_sub = transl[idx]          # (T_out, 3)
    ts_sub = timestamps_us[idx]       # (T_out,)
    T_out = poses_sub.shape[0]

    # rotvec -> rotmat (T_pose-relative)
    R_rel_flat = rotvec_to_matrix(torch.from_numpy(poses_sub).reshape(-1, 3))  # (T_out*77, 3, 3)
    R_rel = R_rel_flat.reshape(T_out, 77, 3, 3).cpu().numpy().astype(np.float32)

    # Bake T-pose orient back -> joint-parent-local rotmats
    R_rel_t = torch.from_numpy(R_rel)
    orient_t = torch.from_numpy(orient)
    orient_parent_T_t = torch.from_numpy(orient_parent_T)
    local_rot_mats = apply_joint_orient_local(R_rel_t, orient_t, orient_parent_T_t).numpy().astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        local_rot_mats=local_rot_mats,
        root_positions=transl_sub.astype(np.float32),
        timestamps_us=ts_sub.astype(np.int64),
        fps=np.int64(target_fps),
        source_seq=str(soma_npz.parent.parent.name),
        source_subject=str(soma_npz.parent.parent.parent.name),
    )
    return {
        "in_frames": int(poses.shape[0]),
        "out_frames": int(T_out),
        "target_fps": float(target_fps),
        "source_fps_inferred": round(float(poses.shape[0]) /
                                     ((timestamps_us[-1] - timestamps_us[0]) / 1e6), 3),
        "duration_sec": round(float((ts_sub[-1] - ts_sub[0]) / 1e6), 3),
        "orient_match_err": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, required=True,
                    help="NymeriaPlus sequence dir (containing body/xdata_soma.npz)")
    ap.add_argument("--out", type=Path, required=True, help="Output kimodo motion NPZ")
    ap.add_argument("--target-fps", type=float, default=20.0)
    ap.add_argument("--device", default="cpu",
                    help="Device for SOMALayer (cpu is fine for orient-only use)")
    args = ap.parse_args()

    print(f"[load] SOMALayer assets ({args.device})")
    soma_layer = SOMALayer(SOMA_ROOT / "assets", identity_model_type="smpl",
                           device=args.device, mode="warp")
    print(f"[load] joint_names[:5] = {list(soma_layer.rig_data['joint_names'])[:5]}")
    soma_npz = args.seq_dir / "body" / "xdata_soma.npz"
    t0 = time.perf_counter()
    meta = convert_one(soma_npz, args.out, args.target_fps, soma_layer)
    dt = time.perf_counter() - t0
    print(f"[done] {meta}  ({dt:.2f}s)")
    print(f"[done] wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
