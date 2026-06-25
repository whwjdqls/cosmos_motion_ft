"""Visualize sampled 3-task eval for a checkpoint:
  - inverse_dynamics: predicted vs GT camera TRAJECTORY with CAMERA FRUSTA (shows orientation),
    Umeyama-aligned (pred->GT) -> <samples>/viz/invdyn_camera.png
  - forward_dynamics / policy: GT | generated video side-by-side -> <samples>/viz/<subj>_<task>.mp4

Run in the `kimodo` env (numpy/matplotlib + ffmpeg). Sampling itself is run_infer_merged.sh.

Usage:
  python viz_eval_samples.py --samples <ckpt>/samples --eval_root <nymeria_eval[_33]> --tag 97f_iter2500
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def rot6d_to_R(v6):
    a0, a1 = v6[:3], v6[3:6]
    b0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1p = a1 - (b0 @ a1) * b0; b1 = a1p / (np.linalg.norm(a1p) + 1e-8)
    return np.stack([b0, b1, np.cross(b0, b1)], axis=1)


def pred_poses(a9):
    """rel 9D actions -> (T,3) positions + (T,3,3) rotations accumulated from identity."""
    P = [np.eye(4)]; c = P[0]
    for i in range(len(a9)):
        d = np.eye(4); d[:3, :3] = rot6d_to_R(a9[i, 3:9]); d[:3, 3] = a9[i, :3]
        c = c @ d; P.append(c.copy())
    P = np.stack(P)
    return P[:, :3, 3], P[:, :3, :3]


def gt_poses(npz):
    d = np.load(npz); pos, rot = d["cam_world_pos"].astype(float), d["cam_world_rot"].astype(float)
    T = len(pos); P = np.tile(np.eye(4), (T, 1, 1)); P[:, :3, :3] = rot; P[:, :3, 3] = pos
    P0 = np.linalg.inv(P[0])
    P = np.einsum("ij,tjk->tik", P0, P)
    return P[:, :3, 3], P[:, :3, :3]


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); s0, d0 = src - ms, dst - md
    cov = d0.T @ s0 / len(src); U, D, Vt = np.linalg.svd(cov); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; var = (s0 ** 2).sum() / len(src) + 1e-12; s = np.trace(np.diag(D) @ S) / var
    t = md - s * R @ ms; return s, R, t


def draw_cam(ax, C, R, color, d):
    """Frustum: apex at C, image plane d ahead along optical axis. R cols = cam axes (OpenCV)."""
    right, up, fwd = R[:, 0], -R[:, 1], R[:, 2]
    w, h = d * 0.7, d * 0.5; c = C + fwd * d
    cor = [c + right * w + up * h, c + right * w - up * h, c - right * w - up * h, c - right * w + up * h]
    for k in cor:
        ax.plot(*np.array([C, k]).T, color=color, lw=0.7, alpha=0.85)
    ax.plot(*np.array(cor + [cor[0]]).T, color=color, lw=0.8, alpha=0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="<ckpt>/samples dir")
    ap.add_argument("--eval_root", required=True, help="nymeria_eval[_33] with samples/<name>/{gt_clip.mp4,gt_camera_cosmos.npz}")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ncam", type=int, default=6)
    args = ap.parse_args()
    viz = os.path.join(args.samples, "viz"); os.makedirs(viz, exist_ok=True)

    # ---- inverse_dynamics: camera frusta ----
    names = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(args.samples, "invdyn_out", "*", "sample_outputs.json")))
    fig = plt.figure(figsize=(5.4 * len(names), 5.2))
    for i, n in enumerate(names):
        pr = np.array(json.load(open(os.path.join(args.samples, "invdyn_out", n, "sample_outputs.json")))
                      ["outputs"][0]["content"]["action"], float)
        ppos, pR = pred_poses(pr)
        gpos, gR = gt_poses(os.path.join(args.eval_root, "samples", n, "gt_camera_cosmos.npz"))
        k = min(len(ppos), len(gpos)); ppos, pR, gpos, gR = ppos[:k], pR[:k], gpos[:k], gR[:k]
        pmag = np.linalg.norm(np.diff(ppos, axis=0), axis=1).mean()
        gmag = np.linalg.norm(np.diff(gpos, axis=0), axis=1).mean()
        s, R, t = umeyama(ppos, gpos)
        ppos = (s * (R @ ppos.T)).T + t; pR = R[None] @ pR
        ate = np.sqrt(((ppos - gpos) ** 2).sum(1).mean())
        allp = np.concatenate([ppos, gpos]); ctr = allp.mean(0); rad = np.abs(allp - ctr).max() * 1.15 + 1e-6
        ts = np.linspace(0, k - 1, args.ncam).astype(int)
        ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
        ax.plot(*gpos.T, "g-", lw=1.8, label="GT"); ax.plot(*ppos.T, "r-", lw=1.8, label="pred")
        ax.scatter(*gpos[0], c="k", s=25)
        for tt in ts:
            draw_cam(ax, gpos[tt], gR[tt], "green", rad * 0.16)
            draw_cam(ax, ppos[tt], pR[tt], "red", rad * 0.16)
        for setl, c in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setl(ctr[c] - rad, ctr[c] + rad)
        ax.set_title(f"{n.split('_', 2)[1]}  ATE={ate:.3f}m\npred|Δ|={pmag:.3f} GT|Δ|={gmag:.3f} (ratio {pmag/max(gmag,1e-6):.1f}x)",
                     fontsize=8)
        ax.legend(fontsize=7, loc="upper left"); ax.view_init(elev=24, azim=-60)
    fig.suptitle(f"inverse-dynamics: predicted vs GT camera (frusta = orientation @ {args.ncam} steps)  {args.tag}", fontsize=11)
    fig.tight_layout(); out = os.path.join(viz, "invdyn_camera.png"); fig.savefig(out, dpi=115); plt.close(fig)
    print("saved", out)

    # ---- forward_dynamics / policy: GT | generated video ----
    LBL = {"fd": "fwd-dyn (img+text+cam)", "policy": "policy (img+text)"}
    for task in ("fd", "policy"):
        for n in names:
            gen = os.path.join(args.samples, f"{task}_out", n, "vision.mp4")
            gt = os.path.join(args.eval_root, "samples", n, "gt_clip.mp4")
            if not (os.path.isfile(gen) and os.path.isfile(gt)):
                continue
            # unique per-sample tag (t#_S##) so duplicate subjects don't collide
            tag = "_".join(n.split("_")[:2])
            out = os.path.join(viz, f"{tag}_{task}.mp4")
            fc = (f"[0:v]scale=380:380,pad=iw:ih+26:0:26:black,drawtext=text=GT:x=6:y=3:fontcolor=yellow:fontsize=18[a];"
                  f"[1:v]scale=380:380,pad=iw:ih+26:0:26:black,drawtext=text={LBL[task]}:x=6:y=3:fontcolor=yellow:fontsize=14[b];[a][b]hstack")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", gt, "-i", gen,
                            "-filter_complex", fc, out], check=False)
            print("saved", out)


if __name__ == "__main__":
    main()
