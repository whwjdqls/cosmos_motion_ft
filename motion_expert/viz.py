"""Phase 3: decode sampled UniEgoMotion → joints → render (kimodo env).

Reads samples_manifest.json + the unnormalized 283-D .npy from sample.py, decodes to world
joints with kimodo's `uniego_world_joints_from_features` (translation block = positions; no
FK), and renders a stick-figure animation per sample. For the POC ablation it stacks the
text-conditioned ("cond") clip next to the unconditional ("null") clip per prompt.

Run (kimodo env):
  ssh a3ultravis-a3ultranodeset-1 'unset LD_LIBRARY_PATH; \
    /home/jungbin_cho/miniforge3/envs/kimodo/bin/python motion_expert/viz.py --dir <samples_dir>'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, "/home/jungbin_cho/kimodo_open")
from kimodo.motion_rep.uniego import uniego_world_joints_from_features  # noqa: E402

SKEL = "/home/jungbin_cho/cosmos_motion_ft/skeleton_soma30.npz"


def decode(feat_unnorm: np.ndarray) -> np.ndarray:
    """(T,283) unnormalized → (T,30,3) world joints."""
    j = uniego_world_joints_from_features(torch.from_numpy(feat_unnorm).float(), n_joints=30)
    return j.numpy()


def render(joints: np.ndarray, parents: np.ndarray, out_mp4: str, fps: int = 20, title: str = ""):
    """joints (T,30,3) world **Y-up, +Z forward** (uniego/HML3D convention) → stick-figure mp4.

    matplotlib mplot3d is Z-up, so we remap world (x,y,z) -> plot (x, z, y) with up = world Y,
    floor at y=0, and negate world X (character-right on screen-right) — mirroring
    kimodo/scripts/render_hml3d.py. (Plotting (x,y,z) directly would lay the skeleton on its side.)
    """
    T = joints.shape[0]
    dx = -joints[..., 0]          # negate X for display
    dz = joints[..., 2]           # forward → plot depth
    dy = joints[..., 1]           # up → plot vertical
    # stable bounds; floor at 0
    cx = (dx.min() + dx.max()) / 2
    cz = (dz.min() + dz.max()) / 2
    half = max(dx.max() - dx.min(), dz.max() - dz.min()) / 2 * 1.1 + 1e-3
    y_top = max(float(dy.max()) * 1.05, 1.0)
    y_floor = min(0.0, float(dy.min()))
    edges = [(j, int(p)) for j, p in enumerate(parents) if 0 <= int(p) < len(parents) and int(p) != j]

    frames = []
    step = max(1, T // 120)
    for t in range(0, T, step):
        fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111, projection="3d")
        for a, b in edges:
            ax.plot([dx[t, a], dx[t, b]], [dz[t, a], dz[t, b]], [dy[t, a], dy[t, b]],
                    "-", color="tab:blue", lw=2)
        ax.scatter(dx[t], dz[t], dy[t], c="k", s=8)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cz - half, cz + half)
        ax.set_zlim(y_floor, y_top)
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
        ax.set_xlabel("x"); ax.set_ylabel("z (fwd)"); ax.set_zlabel("y (up)")
        ax.view_init(elev=20, azim=-60)
        ax.set_title(f"{title}\nframe {t}/{T}", fontsize=8)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames.append(buf.copy()); plt.close(fig)
    iio.imwrite(out_mp4, np.stack(frames), fps=fps, codec="libx264")
    return out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="samples dir written by sample.py")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    sk = np.load(SKEL, allow_pickle=True)
    parents = sk["parents"]
    man = json.load(open(os.path.join(args.dir, "samples_manifest.json")))

    # group by prompt to make cond|null side-by-side
    by_prompt: dict[str, dict] = {}
    for s in man["samples"]:
        by_prompt.setdefault(s["prompt"], {})[s["mode"]] = s["file"]

    for prompt, modes in by_prompt.items():
        slug = "".join(c if c.isalnum() else "_" for c in prompt)[:40]
        clips = {}
        for mode, fn in modes.items():
            feat = np.load(os.path.join(args.dir, fn))
            joints = decode(feat)
            out = os.path.join(args.dir, fn.replace(".npy", ".mp4"))
            render(joints, parents, out, fps=args.fps, title=f"{mode}: {prompt[:34]}")
            rstep = np.linalg.norm(np.diff(joints[:, 0], axis=0), axis=1).mean()
            clips[mode] = out
            print(f"  {mode:4s} '{prompt[:34]}' root-step={rstep:.3f} -> {os.path.basename(out)}")
        # side-by-side cond|null if both present (the ablation)
        if "cond" in clips and "null" in clips:
            import subprocess
            sbs = os.path.join(args.dir, f"{slug}__ABLATION.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", clips["cond"], "-i", clips["null"],
                "-filter_complex",
                "[0:v]drawtext=text=TEXT-conditioned:x=8:y=6:fontcolor=yellow:fontsize=16[a];"
                "[1:v]drawtext=text=NULL (no text):x=8:y=6:fontcolor=yellow:fontsize=16[b];[a][b]hstack",
                sbs], check=False)
            print(f"  ABLATION -> {os.path.basename(sbs)}")
    print(f"[viz] done: {args.dir}")


if __name__ == "__main__":
    main()
