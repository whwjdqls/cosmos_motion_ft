"""Compare camera trajectories: GT vs Cosmos-predicted (inverse dynamics) vs VGGT-Omega.

Per sample, one image: 3 colored camera paths (Umeyama-aligned to GT metric space) with
a small CAMERA FRUSTUM (pyramid) drawn at sampled timestamps, so position + viewing
direction + roll are unambiguous (vs a single arrow).

  green  = GT (corrected OpenCV frame)
  blue   = VGGT-Omega (up-to-scale, similarity-aligned to GT)
  red    = Cosmos-3 inverse-dynamics (its own units, similarity-aligned to GT)
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from PIL import Image


def rot6d_to_R(v6):
    a0, a1 = v6[:3], v6[3:6]
    b0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1p = a1 - (b0 @ a1) * b0; b1 = a1p / (np.linalg.norm(a1p) + 1e-8)
    return np.stack([b0, b1, np.cross(b0, b1)], axis=1)


def cosmos_traj(pred9):
    """rel 9D -> (pos[N,3], R_world_cam[N,3,3]) accumulated from identity."""
    P = [np.eye(4)]; cur = P[0]
    for i in range(len(pred9)):
        dT = np.eye(4); dT[:3, :3] = rot6d_to_R(pred9[i, 3:9]); dT[:3, 3] = pred9[i, :3]
        cur = cur @ dT; P.append(cur.copy())
    P = np.stack(P)
    return P[:, :3, 3], P[:, :3, :3]


def umeyama(src, dst):
    """dst ≈ s R src + t. Returns s,R,t."""
    mu_s, mu_d = src.mean(0), dst.mean(0); s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src); U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; var = (s0 ** 2).sum() / len(src) + 1e-12
    s = np.trace(np.diag(D) @ S) / var; t = mu_d - s * R @ mu_s
    return s, R, t


def draw_camera(ax, C, R, color, d):
    """Frustum: apex at camera center C, image plane d in front along optical axis.
    R columns = camera axes in world (right, down, forward) [OpenCV]."""
    right, up, fwd = R[:, 0], -R[:, 1], R[:, 2]   # up = -y (OpenCV y is down)
    w, h = d * 0.7, d * 0.5
    c = C + fwd * d
    corners = [c + right * w + up * h, c + right * w - up * h,
               c - right * w - up * h, c - right * w + up * h]
    for cor in corners:  # apex -> corners
        ax.plot(*np.array([C, cor]).T, color=color, lw=0.8, alpha=0.85)
    rect = corners + [corners[0]]
    ax.plot(*np.array(rect).T, color=color, lw=0.9, alpha=0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline_batch")
    ap.add_argument("--vggt_root", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_vggt")
    ap.add_argument("--ncam", type=int, default=5, help="number of frusta per trajectory")
    args = ap.parse_args()
    names = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(args.vggt_root, "*", "vggt_cameras.npz")))
    pngs = []
    for name in names:
        sd = os.path.join(args.src_root, "samples", name)
        gt = np.load(os.path.join(sd, "gt_camera_cosmos.npz"))
        gpos = gt["cam_world_pos"].astype(np.float64); gpos = gpos - gpos[0]
        gR = gt["cam_world_rot"].astype(np.float64)                      # R_world_cam
        v = np.load(os.path.join(args.vggt_root, name, "vggt_cameras.npz"))
        vpos = v["cam_pos"].astype(np.float64)
        vR = v["extrinsics"].astype(np.float64)[:, :3, :3].transpose(0, 2, 1)  # c2w
        pred = np.array(json.load(open(os.path.join(args.src_root, "invdyn_out", name, "sample_outputs.json")))
                        ["outputs"][0]["content"]["action"], dtype=np.float64)
        cpos, cR = cosmos_traj(pred)
        n = min(len(gpos), len(vpos), len(cpos))
        gpos, gR, vpos, vR, cpos, cR = gpos[:n], gR[:n], vpos[:n], vR[:n], cpos[:n], cR[:n]

        gscale = np.linalg.norm(np.diff(gpos, axis=0), axis=1).sum()
        if gscale > 1e-3:
            sv, Rv, tv = umeyama(vpos, gpos); vpos = (sv * (Rv @ vpos.T)).T + tv; vR = Rv[None] @ vR
            sc, Rc, tc = umeyama(cpos, gpos); cpos = (sc * (Rc @ cpos.T)).T + tc; cR = Rc[None] @ cR
        ts = np.linspace(0, n - 1, args.ncam).astype(int)
        allp = np.concatenate([gpos, vpos, cpos]); ctr = allp.mean(0); rad = np.abs(allp - ctr).max() * 1.15 + 1e-6
        d = rad * 0.17

        fig = plt.figure(figsize=(6.4, 6.4)); ax = fig.add_subplot(111, projection="3d")
        for pos, Rm, col, lab in [(gpos, gR, "green", "GT"), (vpos, vR, "tab:blue", "VGGT-Omega"),
                                  (cpos, cR, "red", "Cosmos invdyn")]:
            ax.plot(*pos.T, color=col, lw=1.8, label=lab, alpha=0.9)
            ax.scatter(*pos[0], color=col, s=22)
            for t in ts:
                draw_camera(ax, pos[t], Rm[t], col, d)
        for setl, c in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setl(ctr[c] - rad, ctr[c] + rad)
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
        subj = name.split("_", 1)[1][:16]
        ax.set_title(f"{subj}: camera frusta @ {args.ncam} timesteps\nGT vs VGGT vs Cosmos (aligned to GT)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left"); ax.view_init(elev=38, azim=-66)
        out = os.path.join(args.vggt_root, name, "traj_compare.png")
        fig.tight_layout(); fig.savefig(out, dpi=115); plt.close(fig); pngs.append(out)
        print(f"  {name}: GT-pathlen={gscale:.2f}m -> {out}")

    if pngs:
        ims = [Image.open(p) for p in pngs]
        w, h = ims[0].size; cols, rows = 3, (len(ims) + 2) // 3
        sheet = Image.new("RGB", (cols * w, rows * h), "white")
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        sp = os.path.join(args.vggt_root, "contact_sheet.png"); sheet.save(sp)
        print("contact sheet ->", sp)


if __name__ == "__main__":
    main()
