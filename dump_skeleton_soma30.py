"""Dump canonical SOMASkeleton30 FK constants to an npz (run in the kimodo env).

The constants are exactly what kimodo's fk() consumes when neutral_joints=None:
the skeleton.neutral_joints buffer (absolute rest joint positions, J,3) and the
joint_parents tensor. somaskel30 has NO rest_pose_local_rot.p and NO
standard_t_pose_global_offsets_rots.p, so fk's `rest_local_rots` is None and the
neutral_joints are used directly as rest positions.
"""
import numpy as np
import torch

from kimodo.skeleton.definitions import SOMASkeleton30

OUT = "/home/jungbin_cho/cosmos_motion_ft/skeleton_soma30.npz"


def main():
    skel = SOMASkeleton30()
    J = skel.nbjoints
    parents = skel.joint_parents.cpu().numpy().astype(np.int64)  # (J,)
    root_idx = int(skel.root_idx)
    joint_names = list(skel.bone_order_names)
    neutral_joints = skel.neutral_joints.detach().cpu().numpy().astype(np.float64)  # (J,3) absolute rest positions

    # rel offsets: per-joint offset from its parent (root offset = its own pos).
    # forward_kinematics() reconstructs exactly this internally; we store it for
    # convenience / inspection but the port's fk uses neutral_joints directly.
    offsets_rel = neutral_joints.copy()
    mask = np.ones(J, dtype=bool)
    mask[root_idx] = False
    offsets_rel[mask] = neutral_joints[mask] - neutral_joints[parents[mask]]

    # Sanity: somaskel30 must have no baked rest local / tpose offsets.
    assert getattr(skel, "rest_local_rots", None) is None, "unexpected rest_local_rots"
    assert getattr(skel, "rest_pose_local_rot", None) is None, "unexpected rest_pose_local_rot"
    assert getattr(skel, "global_rot_offsets", None) is None, "unexpected global_rot_offsets"
    assert (neutral_joints[root_idx] == 0).all(), "root neutral joint must be 0"

    np.savez(
        OUT,
        parents=parents,
        # `offsets` = absolute rest joint positions (what kimodo fk consumes as
        # `joints` / `neutral_joints`). NOT parent-relative.
        offsets=neutral_joints.astype(np.float32),
        offsets_rel=offsets_rel.astype(np.float32),
        root_idx=np.int64(root_idx),
        joint_names=np.array(joint_names, dtype=object),
        fps=np.int64(20),
    )
    print(f"Wrote {OUT}")
    print(f"  J={J} root_idx={root_idx}")
    print(f"  parents={parents.tolist()}")
    print(f"  neutral_joints[:3]=\n{neutral_joints[:3]}")

    # Verify round-trip load.
    d = np.load(OUT, allow_pickle=True)
    assert d["parents"].shape == (J,)
    assert d["offsets"].shape == (J, 3)
    assert int(d["root_idx"]) == root_idx
    assert int(d["fps"]) == 20
    assert list(d["joint_names"]) == joint_names
    print("Round-trip load OK.")
    print(f"  joint_names={list(d['joint_names'])}")


if __name__ == "__main__":
    main()
