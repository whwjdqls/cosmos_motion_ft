#!/usr/bin/env python
"""Render decoded SOMASkeleton30 joints (T,30,3) to mp4 using kimodo's render_soma
AS-IS (follow camera, floor/grid). RUN IN THE KIMODO ENV:
  /home/jungbin_cho/miniforge3/envs/kimodo/bin/python render_kimodo.py --dir <viz_step_dir>
  ... --joints a_joints.npy --out a.mp4 --caption "..."
Mirrors kimodo/scripts/train.py viz: render_single(...camera='follow') -> frames ->
imageio.v3.imwrite(codec='h264', plugin='pyav')."""
import argparse
import glob
import os
import sys

sys.path.insert(0, "/home/jungbin_cho/kimodo_open")
import numpy as np
import imageio.v3 as iio
from kimodo.scripts.render_soma import render_single
from kimodo.skeleton.definitions import SOMASkeleton30

_PARENTS = SOMASkeleton30().joint_parents.cpu().numpy().tolist()


def render_one(joints_npy, out_mp4, caption, fps=20):
    joints = np.load(joints_npy)  # (T, 30, 3) world joint positions
    frames = render_single(joints, _PARENTS, caption=caption, camera="follow")
    iio.imwrite(out_mp4, frames, fps=float(fps), codec="h264", plugin="pyav")
    return out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="render every *_joints.npy in this dir -> <name>.mp4")
    ap.add_argument("--joints", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--caption", default="")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    if args.dir:
        for jp in sorted(glob.glob(os.path.join(args.dir, "*_joints.npy"))):
            name = os.path.basename(jp).replace("_joints.npy", "")
            out = os.path.join(args.dir, name + ".mp4")
            render_one(jp, out, caption=name, fps=args.fps)
            print(f"[render_kimodo] {jp} -> {out}")
    else:
        render_one(args.joints, args.out, args.caption, args.fps)
        print(f"[render_kimodo] {args.joints} -> {args.out}")


if __name__ == "__main__":
    main()
