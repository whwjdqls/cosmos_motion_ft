#!/usr/bin/env python
"""Visualization for the joint-model CAMERA eval (reads eval_camera.py's output layout).

Reuses the nymeria_world viz LOGIC verbatim (imported, not reinvented):
  * inverse-dynamics camera FRUSTA (pred vs GT, Umeyama-aligned) -- viz_eval_samples.py
  * inverse-dynamics METRIC-SCALE trajectory montage (rows=windows) -- montage_invdyn_metric math
  * forward_dynamics / policy GT|generated video side-by-side -- viz_fd.py logic

Since eval_camera.py writes the EXACT nymeria_world schema
(``<eval>/samples/<seq>/{gt_camera_cosmos.npz,gt_clip.mp4}``, ``<eval>/invdyn_out/<seq>/
sample_outputs.json``, ``<eval>/{fd_out,policy_out}/<seq>/vision.mp4``), we point those tools
at ``--samples <eval> --eval_root <eval>``. This wrapper drives them from a single
``--eval_dir`` and additionally builds a self-contained metric-scale trajectory PNG.

Run in the kimodo OR cosmos env (numpy + matplotlib + ffmpeg)::

    python viz_camera.py --eval_dir <ckpt_dir>/camera_eval
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_NYMERIA_WORLD = "/home/jungbin_cho/cosmos_motion_ft/nymeria_world"
if _NYMERIA_WORLD not in sys.path:
    sys.path.insert(0, _NYMERIA_WORLD)
# Reuse the proven pose helpers (rot6d->R, rel-action integration, GT pose reader, Umeyama).
from viz_eval_samples import rot6d_to_R, pred_poses, gt_poses, umeyama, draw_cam  # noqa: E402


def _seq_names(eval_dir: str):
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(eval_dir, "invdyn_out", "*",
                                                   "sample_outputs.json")))


# ---------------------------------------------------------------------------
# 1. inverse-dynamics camera FRUSTA (identical to viz_eval_samples.main's first panel).
# ---------------------------------------------------------------------------
def viz_frusta(eval_dir: str, tag: str, ncam: int = 6, max_seqs: int = 0):
    viz = os.path.join(eval_dir, "viz"); os.makedirs(viz, exist_ok=True)
    names = _seq_names(eval_dir)
    if max_seqs and len(names) > max_seqs:
        names = names[:max_seqs]
    if not names:
        print("[viz_camera] no invdyn_out windows; skipping frusta"); return
    fig = plt.figure(figsize=(5.4 * len(names), 5.2))
    for i, n in enumerate(names):
        pr = np.array(json.load(open(os.path.join(eval_dir, "invdyn_out", n,
                      "sample_outputs.json")))["outputs"][0]["content"]["action"], float)
        ppos, pR = pred_poses(pr)
        gpos, gR = gt_poses(os.path.join(eval_dir, "samples", n, "gt_camera_cosmos.npz"))
        k = min(len(ppos), len(gpos)); ppos, pR, gpos, gR = ppos[:k], pR[:k], gpos[:k], gR[:k]
        pmag = np.linalg.norm(np.diff(ppos, axis=0), axis=1).mean()
        gmag = np.linalg.norm(np.diff(gpos, axis=0), axis=1).mean()
        s, R, t = umeyama(ppos, gpos)
        ppos = (s * (R @ ppos.T)).T + t; pR = R[None] @ pR
        ate = np.sqrt(((ppos - gpos) ** 2).sum(1).mean())
        allp = np.concatenate([ppos, gpos]); ctr = allp.mean(0); rad = np.abs(allp - ctr).max() * 1.15 + 1e-6
        ts = np.linspace(0, k - 1, ncam).astype(int)
        ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
        ax.plot(*gpos.T, "g-", lw=1.8, label="GT"); ax.plot(*ppos.T, "r-", lw=1.8, label="pred")
        ax.scatter(*gpos[0], c="k", s=25)
        for tt in ts:
            draw_cam(ax, gpos[tt], gR[tt], "green", rad * 0.16)
            draw_cam(ax, ppos[tt], pR[tt], "red", rad * 0.16)
        for setl, c in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setl(ctr[c] - rad, ctr[c] + rad)
        ax.set_title(f"{n[:22]}  ATE={ate:.3f}m\npred|Δ|={pmag:.3f} GT|Δ|={gmag:.3f} "
                     f"({pmag/max(gmag,1e-6):.1f}x)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left"); ax.view_init(elev=24, azim=-60)
    fig.suptitle(f"inverse-dynamics: pred vs GT camera (frusta @ {ncam} steps)  {tag}", fontsize=11)
    fig.tight_layout(); out = os.path.join(viz, "invdyn_camera.png"); fig.savefig(out, dpi=115); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# 2. metric-scale trajectory montage (montage_invdyn_metric logic, single-model,
#    self-contained: rows = windows, 2D PCA projection onto GT plane, 1 m scale bar).
# ---------------------------------------------------------------------------
def _pred_pos(a9):
    a9 = np.asarray(a9, float); P = [np.eye(4)]; c = P[0]
    for i in range(len(a9)):
        d = np.eye(4); d[:3, :3] = rot6d_to_R_row(a9[i, 3:9]); d[:3, 3] = a9[i, :3]
        c = c @ d; P.append(c.copy())
    return np.stack(P)[:, :3, 3]


def rot6d_to_R_row(v6):  # (6,) -> (3,3) (montage_invdyn_metric convention)
    a0, a1 = v6[:3], v6[3:6]
    b0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1p = a1 - (b0 @ a1) * b0; b1 = a1p / (np.linalg.norm(a1p) + 1e-8)
    return np.stack([b0, b1, np.cross(b0, b1)], 1)


def _gt_pos(npz):
    d = np.load(npz); pos, rot = d["cam_world_pos"].astype(float), d["cam_world_rot"].astype(float)
    T = len(pos); P = np.tile(np.eye(4), (T, 1, 1)); P[:, :3, :3] = rot; P[:, :3, 3] = pos
    return (np.linalg.inv(P[0]) @ P)[:, :3, 3]


def viz_metric_montage(eval_dir: str, tag: str, max_seqs: int = 8):
    viz = os.path.join(eval_dir, "viz"); os.makedirs(viz, exist_ok=True)
    names = _seq_names(eval_dir)[:max_seqs]
    if not names:
        print("[viz_camera] no invdyn_out windows; skipping metric montage"); return
    nR = len(names)
    fig, axes = plt.subplots(nR, 1, figsize=(4.2, 3.4 * nR), squeeze=False)
    for r, n in enumerate(names):
        gp = _gt_pos(os.path.join(eval_dir, "samples", n, "gt_camera_cosmos.npz"))
        a = np.array(json.load(open(os.path.join(eval_dir, "invdyn_out", n,
                     "sample_outputs.json")))["outputs"][0]["content"]["action"], float)
        pp = _pred_pos(a)
        c0 = gp.mean(0); U, Sg, Vt = np.linalg.svd(gp - c0); B = Vt[:2]
        proj = lambda X: (X - c0) @ B.T
        g2, p2 = proj(gp), proj(pp)
        glen = np.linalg.norm(np.diff(gp, axis=0), axis=1).sum()
        plen = np.linalg.norm(np.diff(pp, axis=0), axis=1).sum()
        cp = np.concatenate([g2, p2]); ctr = cp.mean(0); rad = np.abs(cp - ctr).max() * 1.15 + 1e-3
        ax = axes[r, 0]
        ax.plot(*g2.T, "g-", lw=2, label="GT"); ax.plot(*p2.T, "r-", lw=1.6, label="pred")
        ax.scatter(*g2[0], c="k", s=18, zorder=5)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        x0 = ctr[0] - rad * 0.9; y0 = ctr[1] - rad * 0.85
        ax.plot([x0, x0 + 1.0], [y0, y0], "k-", lw=2.5); ax.text(x0, y0 + rad * 0.04, "1 m", fontsize=7)
        ax.set_title(f"{n[:26]}\nGT {glen:.1f}m | pred {plen:.1f}m ({plen/max(glen,1e-6):.1f}x)",
                     fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"inverse-dynamics METRIC scale: pred (red) vs GT (green)  {tag}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(viz, "invdyn_metric_montage.png"); fig.savefig(out, dpi=120); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# 3. forward_dynamics / policy GT|generated side-by-side (viz_fd.py logic).
# ---------------------------------------------------------------------------
def viz_video(eval_dir: str, tag: str):
    viz = os.path.join(eval_dir, "viz"); os.makedirs(viz, exist_ok=True)
    LBL = {"fd_out": "fwd-dyn (img+text+cam)", "policy_out": "policy (img+text)"}
    for sub, lbl in LBL.items():
        names = sorted(os.path.basename(os.path.dirname(p))
                       for p in glob.glob(os.path.join(eval_dir, sub, "*", "vision.mp4")))
        for n in names:
            gen = os.path.join(eval_dir, sub, n, "vision.mp4")
            gt = os.path.join(eval_dir, "samples", n, "gt_clip.mp4")
            if not (os.path.isfile(gen) and os.path.isfile(gt)):
                continue
            key = sub.replace("_out", "")
            out = os.path.join(viz, f"{n[:26]}_{key}.mp4")
            fc = (f"[0:v]scale=380:380,pad=iw:ih+26:0:26:black,"
                  f"drawtext=text=GT:x=6:y=3:fontcolor=yellow:fontsize=18[a];"
                  f"[1:v]scale=380:380,pad=iw:ih+26:0:26:black,"
                  f"drawtext=text={lbl} {tag}:x=6:y=3:fontcolor=yellow:fontsize=13[b];[a][b]hstack")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", gt, "-i", gen,
                            "-filter_complex", fc, out], check=False)
            print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="<ckpt_dir>/camera_eval (eval_camera.py output)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ncam", type=int, default=6)
    ap.add_argument("--max_seqs", type=int, default=8)
    ap.add_argument("--no_video", action="store_true", help="skip the FD/policy video side-by-sides")
    args = ap.parse_args()

    viz_frusta(args.eval_dir, args.tag, ncam=args.ncam, max_seqs=args.max_seqs)
    viz_metric_montage(args.eval_dir, args.tag, max_seqs=args.max_seqs)
    if not args.no_video:
        viz_video(args.eval_dir, args.tag)
    print(f"\n[viz_camera] all outputs under {os.path.join(args.eval_dir, 'viz')}")


if __name__ == "__main__":
    main()
