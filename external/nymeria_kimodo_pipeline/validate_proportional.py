"""Shape-aware FK *position* round-trip for the proportional Nymeria converter.

The existing validate_motion.py checks rotations only (with zero neutrals).
This script checks that the per-subject rest joints derived from identity_coeffs,
used as kimodo neutral_joints, reproduce SOMA-X's posed joint *world positions*.

For one sequence:
  1. neutral_joints (77,3) = SOMALayer.prepare_identity(identity_coeffs) bind-pose
     world joint translations, Root@0 dropped.
  2. SOMA-X ground truth posed joints: SOMALayer.pose(poses_rotvec, transl)['joints'].
  3. kimodo posed joints: kinematics.fk(local_rot_mats, root_positions, neutral_joints).
  4. Compare (3) vs (2) at matching frames. Both are world meters.

A pass means the SOMA bind-pose rest joints drop straight into kimodo FK as
shape-aware neutrals — no extra frame conversion needed. The interesting metric
is the *non-root* joint error: that's what proves the actor's bone lengths are
applied correctly (root is trivially aligned by construction).

Usage (soma env; CPU is fine):
    /home/jungbin_cho/miniforge3/envs/soma/bin/python validate_proportional.py \
        --seq-dir /weka/jungbin/nymeriaplus/S01/<seq> \
        --kimodo  /weka/jungbin/nymeriaplus_kimodo/motions/S01/<seq>.npz \
        --n-frames 200
"""
from __future__ import annotations
import argparse, importlib.util, sys, types, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

SOMA_ROOT = Path("/home/jungbin_cho/SOMA-X")
KIMODO_ROOT = Path("/home/jungbin_cho/kimodo_open")
if str(SOMA_ROOT) not in sys.path:
    sys.path.insert(0, str(SOMA_ROOT))

from soma.soma import SOMALayer  # noqa: E402
from soma.geometry.transforms import rotvec_to_matrix  # noqa: E402


def load_kimodo_kinematics():
    """Import kimodo/skeleton/kinematics.py standalone (no kimodo.__init__)."""
    pkg_kimodo = types.ModuleType("kimodo")
    pkg_skel = types.ModuleType("kimodo.skeleton")
    pkg_tools = types.ModuleType("kimodo.tools")

    def ensure_batched(**kwargs):
        def _wrap(fn):
            def _inner(*a, **kw):
                return fn(*a, **kw)
            return _inner
        return _wrap

    pkg_tools.ensure_batched = ensure_batched
    sys.modules["kimodo"] = pkg_kimodo
    sys.modules["kimodo.skeleton"] = pkg_skel
    sys.modules["kimodo.tools"] = pkg_tools
    spec = importlib.util.spec_from_file_location(
        "kimodo.skeleton.kinematics",
        KIMODO_ROOT / "kimodo" / "skeleton" / "kinematics.py",
    )
    kin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kin)
    return kin


def soma_tpose_joints(layer: SOMALayer, identity_coeffs: np.ndarray) -> np.ndarray:
    """(77,3) world joint positions at the T-pose (poses=0, transl=0) for this identity."""
    layer.prepare_identity(torch.from_numpy(identity_coeffs).float())
    zeros = torch.zeros(1, 77, 3)
    with torch.no_grad():
        out = layer.pose(zeros, transl=torch.zeros(1, 3), pose2rot=True, absolute_pose=False)
    return out["joints"][0].detach().cpu().numpy().astype(np.float32)  # (77,3)


def neutral_candidate_A_bind(layer: SOMALayer, identity_coeffs: np.ndarray) -> np.ndarray:
    """Raw bind-pose world joint translations (Root@0 dropped). World-frame rest layout."""
    layer.prepare_identity(torch.from_numpy(identity_coeffs).float())
    bind = layer._cached_bind_transforms_world.detach()
    return bind[0, 1:, :3, 3].cpu().numpy().astype(np.float32)


def neutral_candidate_B_orient_walk(
    tpose_joints: np.ndarray, orient_77: np.ndarray, parents_77: list[int],
) -> np.ndarray:
    """Build kimodo neutrals so that kimodo FK reproduces the T-pose joints.

    kimodo FK gives  posed[j]-posed[parent] = global_rot_rest[parent] @ (N[j]-N[parent]),
    and at the T-pose kimodo's global rest rotation == SOMA orient[j] (validate_motion
    TEST 3). So the rest offset must be PRE-rotated by orient[parent]^T:

        N[j] = N[parent] + orient[parent]^T @ (Tpose[j] - Tpose[parent])

    Root keeps its world T-pose position (kimodo recenters by pelvis at FK time).
    """
    J = tpose_joints.shape[0]
    N = np.zeros((J, 3), dtype=np.float32)
    N[0] = tpose_joints[0]
    # parents_77 is a valid topological order (parent index < child for SOMA rig).
    for j in range(1, J):
        p = parents_77[j]
        bone_world = tpose_joints[j] - tpose_joints[p]
        N[j] = N[p] + orient_77[p].T @ bone_world
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, required=True)
    ap.add_argument("--kimodo", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--floor", type=float, default=1e-3,
                    help="acceptance floor in meters for non-root joint position error")
    args = ap.parse_args()

    layer = SOMALayer(SOMA_ROOT / "assets", identity_model_type="smpl",
                      device=args.device, mode="warp")

    soma_npz = args.seq_dir / "body" / "xdata_soma.npz"
    z = np.load(soma_npz, allow_pickle=True)
    poses = z["poses"].astype(np.float32)          # (T,77,3) rotvec, T-pose-rel
    transl = z["transl"].astype(np.float32)        # (T,3) world m
    ts_orig = z["timestamps_us"].astype(np.int64)
    identity_coeffs = z["identity_coeffs"].astype(np.float32)

    conv = np.load(args.kimodo)
    local_rot_mats = conv["local_rot_mats"].astype(np.float32)   # (T_out,77,3,3)
    root_positions = conv["root_positions"].astype(np.float32)   # (T_out,3)
    ts_out = conv["timestamps_us"].astype(np.int64)

    # parents
    parents_78 = layer.rig_data["joint_parent_ids"]
    parents_78 = parents_78.tolist() if hasattr(parents_78, "tolist") else list(parents_78)
    parents_77 = [p - 1 for p in parents_78[1:]]
    orient_77 = layer._t_pose_orient.detach().cpu().numpy()[1:].astype(np.float32)  # (77,3,3)

    # candidate neutrals
    tpose = soma_tpose_joints(layer, identity_coeffs)               # (77,3)
    span_y = float(tpose[:, 1].max() - tpose[:, 1].min())
    print(f"identity height proxy (T-pose span_y) = {span_y:.3f} m")
    cand = {
        "A_bind_world": neutral_candidate_A_bind(layer, identity_coeffs),
        "B_orient_walk": neutral_candidate_B_orient_walk(tpose, orient_77, parents_77),
    }

    # frame correspondence orig<->conv
    orig_idx_for_conv = np.clip(np.searchsorted(ts_orig, ts_out), 0, len(ts_orig) - 1)
    n = min(args.n_frames, len(ts_out))
    sub_conv = np.linspace(0, len(ts_out) - 1, n, dtype=np.int64)
    sub_orig = orig_idx_for_conv[sub_conv]

    # ---- SOMA-X ground truth posed joints (world) ----
    poses_t = torch.from_numpy(poses[sub_orig])                  # (n,77,3) rotvec
    transl_t = torch.from_numpy(transl[sub_orig])                # (n,3)
    layer.prepare_identity(torch.from_numpy(identity_coeffs).float())
    with torch.no_grad():
        out = layer.pose(poses_t, transl=transl_t, pose2rot=True, absolute_pose=False)
    soma_joints = out["joints"].detach().cpu().numpy()            # (n,77,3) world

    kin = load_kimodo_kinematics()

    class AdHocSkel:
        pass

    lrm = torch.from_numpy(local_rot_mats[sub_conv])
    rp = torch.from_numpy(root_positions[sub_conv])
    jn = list(layer.rig_data["joint_names"])[1:]

    best = None
    for name, neutral in cand.items():
        skel = AdHocSkel()
        skel.joint_parents = torch.tensor(parents_77, dtype=torch.long)
        skel.neutral_joints = torch.from_numpy(neutral.astype(np.float32))
        skel.root_idx = 0
        _, posed_kimodo, _ = kin.fk(lrm, rp, skel, root_positions_is_global=True)
        posed_kimodo = posed_kimodo.numpy()
        diff = np.abs(posed_kimodo - soma_joints)
        per_joint = diff.reshape(n, 77, 3).max(axis=(0, 2))
        root_err = float(diff[:, 0].max())
        nonroot_err = float(diff[:, 1:].max())
        mean_err = float(diff.mean())
        worst = np.argsort(-per_joint)[:5]
        print(f"\n[{name}] root={root_err:.3e}  non-root max={nonroot_err:.3e}  mean={mean_err:.3e} m")
        print("   worst-5:", [(jn[i], round(float(per_joint[i]), 4)) for i in worst])
        if best is None or nonroot_err < best[1]:
            best = (name, nonroot_err)

    print(f"\nbest candidate: {best[0]} (non-root max {best[1]:.3e} m)")
    if best[1] > args.floor:
        print(f"FAIL: best non-root error {best[1]:.3e} m > floor {args.floor:.0e} m")
        sys.exit(1)
    print(f"OK: '{best[0]}' neutrals reproduce SOMA-X posed joints within {args.floor:.0e} m.")


if __name__ == "__main__":
    main()
