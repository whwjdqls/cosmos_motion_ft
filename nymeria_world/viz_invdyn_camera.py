"""Visualize pretrained-Nano inverse_dynamics (video->camera) vs GT camera.

Self-contained (numpy + matplotlib); reconstructs absolute trajectories from the 9D
relative pseudo-actions the same way Cosmos does (backward_framewise: T_{i+1}=T_i @ dT,
rot6d->R by Gram-Schmidt). Because the pretrained camera head predicts in Cosmos's own
camera convention while GT is the Aria device frame, the 3D overlay is shown after a
Umeyama similarity alignment (R,s,t), and we ALSO report convention-invariant per-step
scalars: translation-delta magnitude and rotation-delta angle (these need no alignment).

Per sample -> one PNG: [3D trajectory overlay | per-step translation mag | per-step rot angle].
"""
from __future__ import annotations

import argparse, glob, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def rot6d_to_R(v6: np.ndarray) -> np.ndarray:
    """rot6d = [col0(3), col1(3)] -> R (3x3) via Gram-Schmidt (Zhou et al. 2019)."""
    a0, a1 = v6[:3], v6[3:6]
    b0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1p = a1 - (b0 @ a1) * b0
    b1 = a1p / (np.linalg.norm(a1p) + 1e-8)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=1)  # columns


def rel9_to_abs(actions9: np.ndarray, init=None) -> np.ndarray:
    """(T,9) backward_framewise deltas -> (T+1,4,4) absolute poses."""
    T = actions9.shape[0]
    poses = [np.eye(4) if init is None else init.copy()]
    cur = poses[0]
    for i in range(T):
        dT = np.eye(4)
        dT[:3, :3] = rot6d_to_R(actions9[i, 3:9])
        dT[:3, 3] = actions9[i, :3]
        cur = cur @ dT
        poses.append(cur.copy())
    return np.stack(poses)


def abs_to_rel9(poses: np.ndarray) -> np.ndarray:
    """(T,4,4) -> (T-1,9) backward_framewise deltas (matches camera_to_action)."""
    out = []
    for i in range(len(poses) - 1):
        dT = np.linalg.inv(poses[i]) @ poses[i + 1]
        R = dT[:3, :3]
        out.append(np.concatenate([dT[:3, 3], R[:, 0], R[:, 1]]))
    return np.stack(out).astype(np.float32)


def gt_abs_poses(npz_path: str) -> np.ndarray:
    d = np.load(npz_path)
    pos, rot = d["cam_world_pos"].astype(np.float64), d["cam_world_rot"].astype(np.float64)
    T = len(pos)
    P = np.tile(np.eye(4), (T, 1, 1))
    P[:, :3, :3] = rot
    P[:, :3, 3] = pos
    # express in first-frame frame so it starts at identity (comparable to pred)
    P0inv = np.linalg.inv(P[0])
    return np.einsum("ij,tjk->tik", P0inv, P)


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (R,s,t) mapping src->dst (Nx3). Returns aligned src."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / (var + 1e-12)
    t = mu_d - s * R @ mu_s
    return (s * (R @ src.T)).T + t


def rot_angles(rel9: np.ndarray) -> np.ndarray:
    ang = []
    for v in rel9:
        R = rot6d_to_R(v[3:9])
        ang.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return np.array(ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline")
    ap.add_argument("--gt_name", default="gt_camera.npz",
                    help="GT camera npz in each sample dir (e.g. gt_camera_cosmos.npz)")
    args = ap.parse_args()
    invdyn_out = os.path.join(args.root, "invdyn_out")
    samples = os.path.join(args.root, "samples")

    names = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(invdyn_out, "*", "sample_outputs.json")))
    print(f"found {len(names)} inverse_dynamics outputs")
    for name in names:
        out = json.load(open(os.path.join(invdyn_out, name, "sample_outputs.json")))
        pred9 = np.array(out["outputs"][0]["content"]["action"], dtype=np.float64)  # (T,9)
        gtP = gt_abs_poses(os.path.join(samples, name, args.gt_name))                # (Tg,4,4)
        gt9 = abs_to_rel9(gtP)                                                       # (Tg-1,9)
        Tn = min(len(pred9), len(gt9))
        pred9, gt9 = pred9[:Tn], gt9[:Tn]

        predP = rel9_to_abs(pred9)            # (Tn+1,4,4) from identity
        pred_xyz, gt_xyz = predP[:, :3, 3], gtP[: Tn + 1, :3, 3]
        pred_aligned = umeyama(pred_xyz, gt_xyz)

        gt_tmag, pred_tmag = np.linalg.norm(gt9[:, :3], axis=1), np.linalg.norm(pred9[:, :3], axis=1)
        gt_ang, pred_ang = rot_angles(gt9), rot_angles(pred9)
        # convention-invariant correlations
        c_t = np.corrcoef(gt_tmag, pred_tmag)[0, 1]
        c_r = np.corrcoef(gt_ang, pred_ang)[0, 1]
        # directional agreement (no alignment): mean cosine of per-step translation directions
        pn = pred9[:, :3] / (np.linalg.norm(pred9[:, :3], axis=1, keepdims=True) + 1e-9)
        gn = gt9[:, :3] / (np.linalg.norm(gt9[:, :3], axis=1, keepdims=True) + 1e-9)
        dir_cos = float((pn * gn).sum(1).mean())
        ate = np.sqrt(((pred_aligned - gt_xyz) ** 2).sum(1).mean())  # aligned trajectory RMSE

        meta = json.load(open(os.path.join(samples, name, "meta.json")))
        fig = plt.figure(figsize=(16, 5))
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        ax.plot(*gt_xyz.T, "g-", lw=2, label="GT")
        ax.plot(*pred_aligned.T, "r--", lw=2, label="pred (Umeyama-aligned)")
        ax.scatter(*gt_xyz[0], c="k", s=40); ax.set_title(f"camera path (ATE={ate:.3f} m)")
        ax.legend(); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(gt_tmag, "g-", label="GT"); ax2.plot(pred_tmag, "r--", label="pred")
        ax2.set_title(f"per-step translation |Δ| (corr={c_t:.2f})"); ax2.set_xlabel("frame"); ax2.set_ylabel("m"); ax2.legend()

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(gt_ang, "g-", label="GT"); ax3.plot(pred_ang, "r--", label="pred")
        ax3.set_title(f"per-step rotation angle (corr={c_r:.2f})"); ax3.set_xlabel("frame"); ax3.set_ylabel("deg"); ax3.legend()

        fig.suptitle(f"{name}   [dir-cosine pred·GT = {dir_cos:+.2f}, frame={args.gt_name}]\n"
                     f"{meta['caption'][:110]}", fontsize=9)
        fig.tight_layout()
        outpng = os.path.join(invdyn_out, name, "camera_vs_gt.png")
        fig.savefig(outpng, dpi=110); plt.close(fig)
        print(f"  {name}: ATE={ate:.3f}m  trans-corr={c_t:.2f}  rot-corr={c_r:.2f}  "
              f"dir-cos={dir_cos:+.2f}  GT|Δ|={gt_tmag.mean():.3f} pred|Δ|={pred_tmag.mean():.3f}  -> {outpng}")


if __name__ == "__main__":
    main()
