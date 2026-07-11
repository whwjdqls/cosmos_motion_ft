"""Quantitative inverse-dynamics eval on the test set: predicted camera action vs GT.

Reads sampled inverse_dynamics outputs (``<samples>/invdyn_out/<seq>/sample_outputs.json``,
the model's predicted [T-1,9] per-step camera action) and the GT camera poses
(``<eval_root>/samples/<seq>/gt_camera_cosmos.npz``), and reports per-step + trajectory metrics.

The 9D action is Cosmos ``backward_framewise`` rel pose: ΔT_t = T_{t-1}^{-1} T_t, 9D = [Δt(3), rot6d(6)].
GT is computed the SAME way directly from the GT poses, so the comparison is convention-exact and
needs no cosmos imports (runs in the `kimodo` env).

Metrics (per sequence, then aggregated mean/median over the test set):
  rot_deg        : per-step rotation geodesic error  (pred ΔR vs GT ΔR)         [lower=better]
  trans_dir_cos  : per-step translation direction cosine (scale-free)           [1=best]
  scale_ratio    : mean|Δt_pred| / mean|Δt_gt|   (1.0 = correct metric scale)
  trans_err_norm : per-step ||Δt|| L2 error AFTER matching pred scale to GT (shape, scale-free) [m]
  ate_m          : integrated-trajectory ATE, Umeyama Sim(3)-aligned RMSE        [m]
  len_ratio      : pred path length / GT path length

Usage:
  python eval_inverse_dynamics.py --samples <ckpt>/samples --eval_root nymeria_eval5 [--out metrics.json]
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def rot6d_to_R(v6):
    a0, a1 = v6[..., :3], v6[..., 3:6]
    b0 = a0 / (np.linalg.norm(a0, axis=-1, keepdims=True) + 1e-8)
    a1p = a1 - (np.sum(b0 * a1, -1, keepdims=True)) * b0
    b1 = a1p / (np.linalg.norm(a1p, axis=-1, keepdims=True) + 1e-8)
    return np.stack([b0, b1, np.cross(b0, b1)], axis=-1)  # cols = axes


def gt_abs(npz):
    d = np.load(npz); pos, rot = d["cam_world_pos"].astype(float), d["cam_world_rot"].astype(float)
    T = len(pos); P = np.tile(np.eye(4), (T, 1, 1)); P[:, :3, :3] = rot; P[:, :3, 3] = pos
    return P


def rel_steps(P):
    """abs poses (T,4,4) -> per-step ΔT = T_{t-1}^{-1} T_t : (Δt (T-1,3), ΔR (T-1,3,3))."""
    inv = np.linalg.inv(P[:-1]); dT = inv @ P[1:]
    return dT[:, :3, 3], dT[:, :3, :3]


def pred_abs(dt, dR):
    """integrate per-step rel transforms from identity -> abs positions (N+1,3)."""
    P = [np.eye(4)]; c = P[0]
    for i in range(len(dt)):
        d = np.eye(4); d[:3, :3] = dR[i]; d[:3, 3] = dt[i]; c = c @ d; P.append(c.copy())
    return np.stack(P)[:, :3, 3]


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); s0, d0 = src - ms, dst - md
    cov = d0.T @ s0 / len(src); U, D, Vt = np.linalg.svd(cov); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; var = (s0 ** 2).sum() / len(src) + 1e-12; s = np.trace(np.diag(D) @ S) / var
    return s, R, md - s * R @ ms


def geodesic_deg(Ra, Rb):
    Rerr = np.einsum("nji,njk->nik", Ra, Rb)  # Ra^T @ Rb
    tr = np.clip((np.trace(Rerr, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def eval_seq(pred9, P_gt):
    pred9 = np.asarray(pred9, float)
    dt_p, dR_p = pred9[:, :3], rot6d_to_R(pred9[:, 3:9])
    dt_g, dR_g = rel_steps(P_gt)
    k = min(len(dt_p), len(dt_g)); dt_p, dR_p, dt_g, dR_g = dt_p[:k], dR_p[:k], dt_g[:k], dR_g[:k]
    np_, ng = np.linalg.norm(dt_p, axis=1), np.linalg.norm(dt_g, axis=1)
    scale = np_.mean() / max(ng.mean(), 1e-9)
    dir_cos = float(np.mean(np.sum(dt_p * dt_g, 1) / (np_ * ng + 1e-9)))
    trans_err_norm = float(np.linalg.norm(dt_p / max(scale, 1e-9) - dt_g, axis=1).mean())  # scale-matched
    rot_deg = float(geodesic_deg(dR_p, dR_g).mean())
    pp, gp = pred_abs(dt_p, dR_p), P_gt[:k + 1, :3, 3]
    s, R, t = umeyama(pp, gp); ppa = (s * (R @ pp.T)).T + t
    ate = float(np.sqrt(((ppa - gp) ** 2).sum(1).mean()))
    len_ratio = float(np_.sum() / max(ng.sum(), 1e-9))
    return dict(rot_deg=rot_deg, trans_dir_cos=dir_cos, scale_ratio=float(scale),
                trans_err_norm=trans_err_norm, ate_m=ate, len_ratio=len_ratio, frames=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="<ckpt>/samples dir (with invdyn_out/)")
    ap.add_argument("--eval_root", required=True, help="dir with samples/<seq>/gt_camera_cosmos.npz")
    ap.add_argument("--out", default=None, help="optional path for the metrics JSON")
    args = ap.parse_args()

    seqs = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(args.samples, "invdyn_out", "*", "sample_outputs.json")))
    if not seqs:
        raise SystemExit(f"no invdyn_out/*/sample_outputs.json under {args.samples}")
    rows = {}
    for n in seqs:
        pred = json.load(open(os.path.join(args.samples, "invdyn_out", n, "sample_outputs.json")))
        pred9 = pred["outputs"][0]["content"]["action"]
        P_gt = gt_abs(os.path.join(args.eval_root, "samples", n, "gt_camera_cosmos.npz"))
        rows[n] = eval_seq(pred9, P_gt)

    keys = ["rot_deg", "trans_dir_cos", "scale_ratio", "trans_err_norm", "ate_m", "len_ratio"]
    agg = {k: {"mean": float(np.mean([rows[n][k] for n in seqs])),
               "median": float(np.median([rows[n][k] for n in seqs]))} for k in keys}

    print(f"\n=== inverse-dynamics eval | {len(seqs)} test sequences | {args.samples} ===")
    hdr = f"{'sequence':28s} " + " ".join(f"{k:>13s}" for k in keys)
    print(hdr); print("-" * len(hdr))
    for n in seqs:
        print(f"{n[:28]:28s} " + " ".join(f"{rows[n][k]:13.4f}" for k in keys))
    print("-" * len(hdr))
    print(f"{'MEAN':28s} " + " ".join(f"{agg[k]['mean']:13.4f}" for k in keys))
    print(f"{'MEDIAN':28s} " + " ".join(f"{agg[k]['median']:13.4f}" for k in keys))
    print("\nguide: rot_deg↓ trans_dir_cos→1 scale_ratio→1 trans_err_norm↓(m) ate_m↓ len_ratio→1")

    out = args.out or os.path.join(args.samples, "invdyn_metrics.json")
    json.dump({"n": len(seqs), "aggregate": agg, "per_sequence": rows}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
