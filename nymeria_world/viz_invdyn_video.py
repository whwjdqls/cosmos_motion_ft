"""Animated inverse-dynamics visualization (video -> camera), corrected-frame aware.

Reconstructs the predicted camera trajectory and the GT trajectory FROM IDENTITY and
overlays them with **no rotation alignment** (unlike the Umeyama plot, which hides the
frame). Both are unit-path-normalized so the 7x scale gap doesn't dominate the shape.
Shows:
  red    = predicted camera path (+ optical-axis arrow)
  green  = GT in corrected frame  (rgb-optical + Rz(-90), = Cosmos OpenCV)
  gray-- = GT in raw device frame  (what we used before; rotated away if the fix matters)
If the corrected frame is right, RED tracks GREEN and diverges from GRAY.
Per sample -> <invdyn_out>/<name>/invdyn_traj.mp4
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import imageio.v3 as iio

import argparse
ROOT = argparse.ArgumentParser()
ROOT.add_argument("--root", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline")
ROOT = ROOT.parse_args().root


def rot6d_to_R(v6):
    a0, a1 = v6[:3], v6[3:6]
    b0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1p = a1 - (b0 @ a1) * b0
    b1 = a1p / (np.linalg.norm(a1p) + 1e-8)
    return np.stack([b0, b1, np.cross(b0, b1)], axis=1)


def rel9_to_abs(a9, init=None):
    P = [np.eye(4) if init is None else init.copy()]; cur = P[0]
    for i in range(len(a9)):
        dT = np.eye(4); dT[:3, :3] = rot6d_to_R(a9[i, 3:9]); dT[:3, 3] = a9[i, :3]
        cur = cur @ dT; P.append(cur.copy())
    return np.stack(P)


def gt_abs(npz):
    d = np.load(npz); pos, rot = d["cam_world_pos"].astype(np.float64), d["cam_world_rot"].astype(np.float64)
    T = len(pos); P = np.tile(np.eye(4), (T, 1, 1)); P[:, :3, :3] = rot; P[:, :3, 3] = pos
    return np.einsum("ij,tjk->tik", np.linalg.inv(P[0]), P)  # start at identity


def unit_path(P):
    xyz = P[:, :3, 3].copy()
    L = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    return xyz / (L + 1e-9), P[:, :3, :3]


def main():
    names = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(ROOT, "invdyn_out", "*", "sample_outputs.json")))
    for name in names:
        sd = os.path.join(ROOT, "samples", name)
        pred9 = np.array(json.load(open(os.path.join(ROOT, "invdyn_out", name, "sample_outputs.json")))
                         ["outputs"][0]["content"]["action"], dtype=np.float64)
        predP = rel9_to_abs(pred9)
        gtcP = gt_abs(os.path.join(sd, "gt_camera_cosmos.npz"))
        gtdP = gt_abs(os.path.join(sd, "gt_camera.npz"))
        Tn = min(len(predP), len(gtcP), len(gtdP))
        (pxyz, pR) = unit_path(predP[:Tn]); (cxyz, cR) = unit_path(gtcP[:Tn]); (dxyz, _) = unit_path(gtdP[:Tn])

        allp = np.concatenate([pxyz, cxyz, dxyz]); ctr = allp.mean(0)
        rad = np.abs(allp - ctr).max() * 1.1 + 1e-6
        meta = json.load(open(os.path.join(sd, "meta.json")))
        arrow = rad * 0.25

        frames = []
        step = max(1, Tn // 80)  # cap ~80 frames
        for t in range(2, Tn + 1, step):
            fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
            ax.plot(*dxyz[:t].T, color="0.6", ls="--", lw=1.2, label="GT device frame (old)")
            ax.plot(*cxyz[:t].T, "g-", lw=2.2, label="GT corrected (OpenCV)")
            ax.plot(*pxyz[:t].T, "r-", lw=2.2, label="predicted")
            # optical-axis (forward) arrows at current pose
            pf = pR[t - 1][:, 2] * arrow; cf = cR[t - 1][:, 2] * arrow
            ax.quiver(*pxyz[t - 1], *pf, color="r", lw=1.5)
            ax.quiver(*cxyz[t - 1], *cf, color="g", lw=1.5)
            ax.scatter(*pxyz[t - 1], c="r", s=25); ax.scatter(*cxyz[t - 1], c="g", s=25)
            for setlim, c in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
                setlim(ctr[c] - rad, ctr[c] + rad)
            ax.set_title(f"inverse dynamics: predicted vs GT camera path\n{name[:24]} (unit-norm, no rotation-align)",
                         fontsize=9)
            ax.legend(fontsize=7, loc="upper left"); ax.view_init(elev=22, azim=-60)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
            frames.append(buf.copy()); plt.close(fig)
        out = os.path.join(ROOT, "invdyn_out", name, "invdyn_traj.mp4")
        iio.imwrite(out, np.stack(frames), fps=12, codec="libx264")
        # directional cosine in each frame for the caption
        def relt(P): return np.stack([(np.linalg.inv(P[i]) @ P[i + 1])[:3, 3] for i in range(len(P) - 1)])
        def cos(a, b):
            an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9); bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
            return float((an * bn).sum(1).mean())
        pt, ct, dt = relt(predP[:Tn]), relt(gtcP[:Tn]), relt(gtdP[:Tn])
        print(f"  {name}: dir-cos pred·GTcorrected={cos(pt,ct):+.2f}  pred·GTdevice={cos(pt,dt):+.2f}  -> {out}")


if __name__ == "__main__":
    main()
