"""Visualization for the BONES-SEED POC — GT|gen side-by-side, identical to kimodo's.

Wraps ``kimodo.scripts.render_soma.render_sidebyside`` — the SAME renderer
``kimodo/scripts/train.py:viz_step`` uses — so our viz matches kimodo's: GT (left,
blue) vs generated (right, red), one stick-figure segment per (parent, child) from
SOMASkeleton30's ``joint_parents``, with the fingertip-end joints skipped (kimodo's
``_VIZ_DROP_JOINT_NAMES``). ``gt_joints=None`` renders a blank left panel.

The ONE difference from kimodo: our motion is the 283-D UniEgoMotion rep and is
**shape-aware**, so joints are decoded via ``decode_uniego_torch.decode_joints`` (not
``motion_rep.inverse``) and each clip carries its actor's own bone lengths. The skeleton
TOPOLOGY (``joint_parents``) is the shared SOMASkeleton30 — per-actor shape lives in the
decoded joint positions, so the same renderer shows tall/short bodies correctly.

Runs in the ``kimodo`` env (``PYTHONPATH=/home/jungbin_cho/kimodo_open``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_expert_joint_attention.render_motion import render_sidebyside, write_mp4

SKEL = str(Path(__file__).resolve().parents[1] / "skeleton_soma30.npz")
# Same fingertip-end joints kimodo drops from the SOMA stick figure.
_DROP_NAMES = ("LeftHandThumbEnd", "LeftHandMiddleEnd", "RightHandThumbEnd", "RightHandMiddleEnd")


def load_skeleton(skel_path: str = SKEL):
    """-> (joint_parents: list[int], skip_joints: list[int]) for SOMASkeleton30."""
    d = np.load(skel_path, allow_pickle=True)
    parents = [int(p) for p in d["parents"]]
    names = [str(n) for n in d["joint_names"]]
    skip = [i for i, n in enumerate(names) if n in _DROP_NAMES]
    return parents, skip


def render_pair(left_joints, gen_joints, joint_parents, out_mp4, caption="",
                skip_joints=None, fps=20, camera="follow", *,
                gen_head_rotation0=None, gt_head_rotation0=None):
    """left|right side-by-side -> mp4 (h264/pyav), exactly as kimodo viz_step writes it.

    left_joints: (T,J,3) world joints (Y-up, +Z-fwd) or None (blank left panel). Conventionally
                 GT (in-training viz) or the conditioned input skeleton (eval) — see caption.
    gen_joints : (T,J,3). render_sidebyside does its own display remap + viewport.
    camera     : "follow" (per-frame root tracking) or "fixed" (one static viewport for the whole
                 clip — use when the left panel is a static input pose so it stays in frame).
    """
    if camera == "fixed":
        # Preserve explicit legacy static-input-pose callers. Training uses follow.
        from kimodo.scripts.render_soma import render_sidebyside as render_fixed
        frames = render_fixed(
            left_joints, gen_joints, joint_parents, caption=caption,
            skip_joints=skip_joints, camera="fixed",
        )
    elif camera == "follow":
        frames = render_sidebyside(
            left_joints, gen_joints, joint_parents, caption=caption,
            skip_joints=skip_joints, gen_head_rotation0=gen_head_rotation0,
            gt_head_rotation0=gt_head_rotation0,
        )
    else:
        raise ValueError(f"unsupported visualization camera: {camera}")
    write_mp4(frames, str(out_mp4), fps=fps)
