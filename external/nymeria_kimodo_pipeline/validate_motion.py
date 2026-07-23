"""FK round-trip validation for soma->kimodo motion converter.

Three tests:
  1. apply_joint_orient_local should reproduce the stored local_rot_mats from
     the original rotvec poses (bit-exact since we used the same math at
     conversion time).
  2. remove_joint_orient_local should invert it exactly (fp32 floor).
  3. kimodo's standard FK on the stored local_rot_mats should produce the
     same per-joint GLOBAL rotation matrices as SOMA-X's joint_local_to_world
     using the same parent chain (rotations are bone-length invariant; we
     don't need kimodo's neutral_joints for this check).
"""
from __future__ import annotations
import argparse, importlib.util, sys, types
from pathlib import Path
import numpy as np
import torch

SOMA_ROOT = Path("/home/jungbin_cho/SOMA-X")
KIMODO_ROOT = Path("/home/jungbin_cho/kimodo_open")
if str(SOMA_ROOT) not in sys.path: sys.path.insert(0, str(SOMA_ROOT))

from soma.soma import SOMALayer
from soma.geometry.rig_utils import (
    apply_joint_orient_local, remove_joint_orient_local,
    joint_local_to_world,
)
from soma.geometry.transforms import rotvec_to_matrix


def load_kimodo_kinematics():
    """Import kimodo/skeleton/kinematics.py without triggering kimodo.__init__
    (which pulls in llm2vec/peft that aren't installed in the soma env)."""
    # Stub kimodo.tools.ensure_batched so the relative import in kinematics.py works
    pkg_kimodo = types.ModuleType("kimodo")
    pkg_skel = types.ModuleType("kimodo.skeleton")
    pkg_tools = types.ModuleType("kimodo.tools")
    def ensure_batched(**kwargs):
        def _wrap(fn):
            def _inner(*a, **kw): return fn(*a, **kw)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", type=Path, required=True)
    ap.add_argument("--kimodo", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=200)
    args = ap.parse_args()

    soma = SOMALayer(SOMA_ROOT / "assets", identity_model_type="smpl",
                     device="cuda", mode="warp")

    orig = np.load(args.orig)
    poses_orig = orig["poses"].astype(np.float32)
    ts_orig = orig["timestamps_us"].astype(np.int64)

    conv = np.load(args.kimodo)
    local_rot_mats = conv["local_rot_mats"].astype(np.float32)
    ts_out = conv["timestamps_us"].astype(np.int64)
    print(f"orig frames={poses_orig.shape[0]}  conv frames={local_rot_mats.shape[0]}")

    orig_idx_for_conv = np.clip(np.searchsorted(ts_orig, ts_out), 0, len(ts_orig) - 1)

    n = min(args.n_frames, len(ts_out))
    sub_conv_idx = np.linspace(0, len(ts_out) - 1, n, dtype=np.int64)
    sub = orig_idx_for_conv[sub_conv_idx]

    R_rel = rotvec_to_matrix(torch.from_numpy(poses_orig[sub]).reshape(-1, 3)).reshape(
        n, 77, 3, 3
    )
    orient_77 = soma._t_pose_orient.detach().cpu()[1:]
    opt_77 = soma._t_pose_orient_parent_T.detach().cpu()[1:]

    # ---- TEST 1
    local_rot_repro = apply_joint_orient_local(R_rel, orient_77, opt_77).numpy()
    stored = torch.from_numpy(local_rot_mats[sub_conv_idx])
    d1 = np.abs(local_rot_repro - stored.numpy())
    print(f"[TEST 1] apply_joint_orient_local reproduces stored: "
          f"max={d1.max():.3e}  mean={d1.mean():.3e}")

    # ---- TEST 2
    rel_back = remove_joint_orient_local(stored, orient_77, opt_77).numpy()
    d2 = np.abs(rel_back - R_rel.numpy())
    print(f"[TEST 2] orient bake round-trip (rel == rel_back): "
          f"max={d2.max():.3e}  mean={d2.mean():.3e}")

    # ---- TEST 3
    parents_78 = soma.rig_data["joint_parent_ids"]
    parents_78 = parents_78.tolist() if hasattr(parents_78, "tolist") else list(parents_78)
    parents_77 = [p - 1 for p in parents_78[1:]]

    stored_world = joint_local_to_world(stored, parents_77).numpy()  # (n, 77, 3, 3)

    kin = load_kimodo_kinematics()

    class AdHocSkel:
        pass
    skel = AdHocSkel()
    skel.joint_parents = torch.tensor(parents_77, dtype=torch.long)
    skel.neutral_joints = torch.zeros(77, 3)  # rotations don't depend on this
    skel.root_idx = 0
    root_pos = torch.from_numpy(conv["root_positions"].astype(np.float32))[sub_conv_idx]
    global_rot, posed_joints, _ = kin.fk(stored, root_pos, skel,
                                         root_positions_is_global=True)
    d3 = np.abs(global_rot.numpy() - stored_world)
    print(f"[TEST 3] kimodo kinematics.fk vs SOMA-X joint_local_to_world: "
          f"max={d3.max():.3e}  mean={d3.mean():.3e}")

    # acceptance gate
    fp32_floor = 5e-5
    fails = []
    if d1.max() > fp32_floor: fails.append(f"TEST1 ({d1.max():.2e})")
    if d2.max() > fp32_floor: fails.append(f"TEST2 ({d2.max():.2e})")
    if d3.max() > fp32_floor: fails.append(f"TEST3 ({d3.max():.2e})")
    if fails:
        print(f"\nFAIL: tests above floor {fp32_floor:.0e}: {fails}")
        sys.exit(1)
    print(f"\nOK: all 3 tests within fp32 floor {fp32_floor:.0e}")


if __name__ == "__main__":
    main()
