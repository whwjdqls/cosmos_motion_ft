"""Visualize NymeriaPlus proportional (shape-aware) kimodo motions, using the
SAME renderer the training viz uses (kimodo.scripts.render_soma).

For each input NPZ:
  - FK the stored local_rot_mats with the actor's stored neutral_joints
    (shape-aware) on SOMASkeleton30 -> posed world joints.
  - Also FK with the canonical (uniform) neutrals for a side-by-side that makes
    the body-shape difference visible (left = uniform body, right = actor body;
    identical motion).
  - Render to mp4 with the training viz's skip-joints (fingertip ends) and
    follow camera.

Run (env `kimodo`, on a compute node):
    PYTHONPATH=/home/jungbin_cho/kimodo_open python viz_proportional.py \
        --npz <a.npz> <b.npz> ... --out-dir /home/jungbin_cho/_nymeria_prop_viz \
        --max-frames 200 --fps 20
If no --npz given, picks a few sequences from distinct subjects automatically.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import torch

from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
from kimodo.scripts.render_soma import render_single, render_sidebyside

# Same fingertip-end joints the training viz drops (in SOMASkeleton30 order).
VIZ_DROP_JOINT_NAMES = (
    "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightHandThumbEnd", "RightHandMiddleEnd",
)
DEFAULT_OUT = Path("/home/jungbin_cho/_nymeria_prop_viz")
PROP_ROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")


def posed_joints(s30, lrm30, root, neutral30):
    """(n,30,3) world posed joints via kimodo FK with the given (30,3) neutrals."""
    n = lrm30.shape[0]
    nj = neutral30.unsqueeze(0).expand(n, -1, -1)
    _, posed, _ = s30.fk(lrm30, root, neutral_joints=nj)
    return posed.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, nargs="*", default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--n-auto", type=int, default=4, help="how many to auto-pick if --npz omitted")
    ap.add_argument("--sidebyside", action="store_true", default=True,
                    help="also render uniform-vs-actor body side-by-side")
    args = ap.parse_args()

    import imageio.v3 as iio

    s30, s77 = SOMASkeleton30(), SOMASkeleton77()
    idx30 = [s77.bone_order_names.index(n) for n in s30.bone_order_names]
    skip = [s30.bone_index[n] for n in VIZ_DROP_JOINT_NAMES if n in s30.bone_index]
    parents = s30.joint_parents.tolist()
    canon30 = s30.neutral_joints.detach().float()

    if args.npz:
        files = [Path(p) for p in args.npz]
    else:
        # one sequence per distinct subject for variety
        files = []
        for subj in sorted(p.name for p in PROP_ROOT.iterdir() if p.is_dir()):
            hits = sorted(glob.glob(str(PROP_ROOT / subj / "*.npz")))
            if hits:
                files.append(Path(hits[0]))
            if len(files) >= args.n_auto:
                break
    print(f"visualizing {len(files)} sequence(s) -> {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        d = np.load(f, allow_pickle=True)
        lrm77 = torch.from_numpy(d["local_rot_mats"].astype(np.float32))
        root = torch.from_numpy(d["root_positions"].astype(np.float32))
        neutral30 = torch.from_numpy(d["neutral_joints"].astype(np.float32))[idx30]
        lrm30 = s30.from_SOMASkeleton77(lrm77)
        T = lrm30.shape[0]
        h = float(neutral30[:, 1].max() - neutral30[:, 1].min())
        cap = f"{f.parent.name}/{f.stem}  (actor h~{h:.2f}m, T={T})"

        actor = posed_joints(s30, lrm30, root, neutral30)
        base = args.out_dir / f"{f.parent.name}__{f.stem}"

        # single (shape-aware actor body), exactly like the training per-sample viz
        frames = render_single(actor, parents, caption=cap, color="tab:red",
                               max_frames=args.max_frames, frame_stride=args.frame_stride,
                               skip_joints=skip, camera="follow")
        iio.imwrite(str(base.with_suffix(".mp4")), frames, fps=args.fps, codec="libx264")
        print(f"  wrote {base.with_suffix('.mp4').name}  ({frames.shape[0]} frames)")

        if args.sidebyside:
            uniform = posed_joints(s30, lrm30, root, canon30)
            frames2 = render_sidebyside(
                uniform, actor, parents,
                caption=f"{f.stem}  | LEFT uniform body  RIGHT actor body (h~{h:.2f}m) | same motion",
                max_frames=args.max_frames, frame_stride=args.frame_stride,
                skip_joints=skip, camera="follow",
            )
            sxs = base.with_name(base.name + "_uniform-vs-actor").with_suffix(".mp4")
            iio.imwrite(str(sxs), frames2, fps=args.fps, codec="libx264")
            print(f"  wrote {sxs.name}  ({frames2.shape[0]} frames)")


if __name__ == "__main__":
    main()
