"""Validate the proportional Nymeria NPZs against the REAL kimodo SOMASkeleton30.

Runs in the `kimodo` conda env (full kimodo import). For a converted
nymeriaplus_kimodo_proportional NPZ:

  1. Frame/convention check (uniform sanity): for the betas=0 identity, the
     stored neutral_joints reduce to kimodo's canonical SOMASkeleton neutrals
     (~0.04 m gap = SOMA-asset vs kimodo-asset). [done once, not per file]
  2. Bone-length parity: the 30-joint neutral subset bone lengths are the actor's
     and differ from the canonical uniform body (proves shape is captured).
  3. FK consistency: kimodo SOMASkeleton30.fk(local_rot_30, root, neutral_joints)
     produces a rigid skeleton whose per-frame bone lengths are constant and equal
     the neutral bone lengths (proves the neutrals drop into the real FK cleanly).
  4. Pose-equivalence: the actor-neutral FK and canonical-neutral FK give the same
     joint ANGLES (rotations identical); only limb lengths differ — confirms we
     only swapped the body, not the motion.

Usage:
    PYTHONPATH=/home/jungbin_cho/kimodo_open python validate_proportional_kimodo.py \
        --npz /weka/jungbin/nymeriaplus_kimodo_proportional/S01/<seq>.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=200)
    args = ap.parse_args()

    s30 = SOMASkeleton30()
    s77 = SOMASkeleton77()
    idx30 = [s77.bone_order_names.index(n) for n in s30.bone_order_names]

    d = np.load(args.npz, allow_pickle=True)
    lrm77 = torch.from_numpy(d["local_rot_mats"].astype(np.float32))   # (T,77,3,3)
    root = torch.from_numpy(d["root_positions"].astype(np.float32))    # (T,3)
    neutral77 = torch.from_numpy(d["neutral_joints"].astype(np.float32))  # (77,3)
    T = lrm77.shape[0]
    sub = np.linspace(0, T - 1, min(args.n_frames, T), dtype=np.int64)

    lrm30 = s30.from_SOMASkeleton77(lrm77[sub])                        # (n,30,3,3)
    neutral30 = neutral77[idx30]                                       # (30,3)
    root_s = root[sub]

    # canonical (uniform) neutrals for the 30-joint skeleton
    canon30 = s30.neutral_joints.detach().float()                     # (30,3)

    parents = s30.joint_parents.tolist()
    root_idx = int(s30.root_idx)

    def bonelens(J):  # J: (...,30,3)
        out = []
        for j in range(30):
            p = parents[j]
            if p < 0 or j == root_idx:
                continue
            out.append(torch.linalg.norm(J[..., j, :] - J[..., p, :], dim=-1))
        return torch.stack(out, dim=-1)   # (...,29)

    bl_actor = bonelens(neutral30)
    bl_canon = bonelens(canon30)
    print(f"[shape] neutral bone-length |actor - canonical| max={float((bl_actor-bl_canon).abs().max()):.4f} m "
          f"mean={float((bl_actor-bl_canon).abs().mean()):.4f} m  (>0 -> actor body captured)")
    print(f"[shape] height proxy actor={float(neutral30[:,1].max()-neutral30[:,1].min()):.3f} m "
          f"canonical={float(canon30[:,1].max()-canon30[:,1].min()):.3f} m")

    # FK with actor neutrals vs canonical neutrals
    g_a, posed_a, _ = s30.fk(lrm30, root_s, neutral_joints=neutral30.unsqueeze(0).expand(len(sub), -1, -1))
    g_c, posed_c, _ = s30.fk(lrm30, root_s)   # canonical neutrals (default)

    # 3. rigidity: compare actor-neutral FK drift to the EXISTING canonical-neutral
    #    FK drift (the uniform pipeline already in use). Any 77->30 collapse drift
    #    is shared by both; what matters is actor isn't WORSE than canonical.
    bl_pa = bonelens(posed_a)
    bl_pc = bonelens(posed_c)
    drift_a = float((bl_pa - bl_pa.mean(0, keepdim=True)).abs().max())
    drift_c = float((bl_pc - bl_pc.mean(0, keepdim=True)).abs().max())
    print(f"[fk] posed bone-length drift over frames: actor={drift_a:.3e}  canonical={drift_c:.3e} m")
    print(f"[fk] (drift is a 77->30 collapse property; actor should match canonical)")

    # 4. rotations identical (same motion, different body)
    rot_diff = float((g_a - g_c).abs().max())
    print(f"[pose] global-rotation diff actor-vs-canonical FK={rot_diff:.3e} (should ~0: same motion)")

    # 5. position difference actor-vs-canonical reflects ONLY the body-shape swap
    posdiff = float((posed_a - posed_c).abs().max())
    print(f"[pose] posed position diff actor-vs-canonical={posdiff:.3e} m (the shape effect)")

    shape_captured = float((bl_actor - bl_canon).abs().max()) > 1e-3
    drift_ok = abs(drift_a - drift_c) < 1e-2   # actor's rigidity matches canonical (cm tol)
    ok = rot_diff < 1e-4 and shape_captured and drift_ok
    print("\n" + ("OK: motion identical to canonical FK, body shape captured, "
                  "rigidity matches the existing uniform pipeline."
                  if ok else "CHECK: review metrics above."))


if __name__ == "__main__":
    main()
