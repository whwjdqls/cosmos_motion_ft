#!/usr/bin/env python
"""Motion-reconstruction metrics for the MOTION-target tasks of the joint-attention model.

The analog of ``eval_inverse_dynamics`` (camera) for MOTION. Given predicted vs GT motion in
the 283-d uniego representation, this module computes a small suite of reconstruction metrics by
decoding BOTH to world joints ``[T,30,3]`` (via ``decode_uniego_torch.decode_joints``, the
bit-exact torch FK) and comparing:

    MPJPE        mean per-joint position error (meters), per-frame then averaged.
    PA-MPJPE     Procrustes-aligned MPJPE -- a rigid (scale+rot+trans) Umeyama fit of the WHOLE
                 clip's pred joints onto GT removes global drift/orientation before MPJPE.
    feat_mse     mean-squared error in the 283-d Z-SCORED feature space (the training loss space).
    accel_err    mean L2 of the 2nd time-difference of joints (pred vs GT) -- the JITTER metric
                 the project tracks (text->motion samples show ~13x excess accel vs GT).
    root_err     global root-joint (joint 0) trajectory error (meters), per-frame then averaged.
    accel_pred / accel_gt   raw mean |2nd-diff| of each (so excess-jitter ratio is inspectable).

CRITICAL normalization contract (mirrors viz / train EXACTLY): the samplers + dataset both work
in Z-SCORED 283-d space (``(feat - mean) / std``). ``decode_uniego_torch.decode_joints`` needs
UN-normalized 283-d motion, so ``feat_mse`` is computed on the z-scored arrays as given, while
MPJPE / PA-MPJPE / accel / root are computed after ``feat * std + mean`` -> decode_joints. Use
the SAME stats file the dataset uses (``config.MOTION_STATS_MEAN/STD``).

Pure numpy for the metric math (Umeyama etc.); torch only for the decode (so it runs in the
``cosmos`` env alongside sampling, or CPU for the dry-run self-test). Schema of the written
``motion_recon_metrics.json`` mirrors ``invdyn_metrics.json``: {"n", "aggregate":{k:{mean,median}},
"per_sequence":{seq:{k:v}}}.

Self-test (CPU, no GPU / no ckpt needed) -- proves the metric path + shapes end to end::

    python eval_motion_recon.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Metric keys (order = the summary-table column order).
METRIC_KEYS = ["mpjpe_m", "pa_mpjpe_m", "feat_mse", "accel_err", "accel_pred", "accel_gt",
               "root_err_m"]
# For generative (not-reconstruction) tasks we still report these but flag them.
GENERATIVE_TASKS = {"text2motion", "textimg2motion"}


# ---------------------------------------------------------------------------
# Stats (z-score) -- SAME file the dataset + trainer use.
# ---------------------------------------------------------------------------
def load_stats(mean_path: Optional[str] = None, std_path: Optional[str] = None):
    """Return (mean[283], std[283]) float32. Defaults to config.MOTION_STATS_MEAN/STD."""
    if mean_path is None or std_path is None:
        import config as C
        mean_path = mean_path or C.MOTION_STATS_MEAN
        std_path = std_path or C.MOTION_STATS_STD
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    return mean, std


def unnormalize(feat_z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """z-scored 283-d motion -> raw 283-d (the inverse of the dataset's ``(f-mean)/std``)."""
    return feat_z.astype(np.float32) * std + mean


# ---------------------------------------------------------------------------
# Decode 283-d (unnormalized) -> world joints [T,30,3] via the bit-exact torch FK.
# ---------------------------------------------------------------------------
def decode_to_joints(feat_raw: np.ndarray) -> np.ndarray:
    """feat_raw [T,283] UNNORMALIZED -> joints [T,30,3] (numpy). Uses decode_uniego_torch on CPU."""
    import torch
    from decode_uniego_torch import decode_joints
    ft = torch.from_numpy(np.ascontiguousarray(feat_raw)).float().unsqueeze(0)  # [1,T,283]
    with torch.no_grad():
        J = decode_joints(ft)[0]                                                # [T,30,3]
    return J.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Rigid (Umeyama) alignment for PA-MPJPE. Aligns the whole clip's pred point cloud
# (T*J,3) onto GT with scale+rotation+translation (same math as EID.umeyama).
# ---------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray):
    """src,dst : (N,3). Returns (s, R, t) minimizing ||dst - (s R src + t)||^2 (with reflection fix)."""
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    cov = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src) + 1e-12
    s = np.trace(np.diag(D) @ S) / var
    t = md - s * R @ ms
    return s, R, t


def apply_rigid(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (R @ pts.T)).T + t


# ---------------------------------------------------------------------------
# The metric over one (pred, gt) pair of joints [T,30,3].
# ---------------------------------------------------------------------------
def joint_metrics(pred_j: np.ndarray, gt_j: np.ndarray) -> dict:
    """pred_j, gt_j : [T,30,3] world joints. Returns MPJPE / PA-MPJPE / accel / root metrics."""
    k = min(len(pred_j), len(gt_j))
    p, g = pred_j[:k], gt_j[:k]

    # MPJPE: per-frame per-joint L2, then mean.
    mpjpe = float(np.linalg.norm(p - g, axis=-1).mean())

    # PA-MPJPE: whole-clip rigid Umeyama align pred->gt, then MPJPE.
    pflat, gflat = p.reshape(-1, 3), g.reshape(-1, 3)
    s, R, t = umeyama(pflat, gflat)
    p_al = apply_rigid(pflat, s, R, t).reshape(k, -1, 3)
    pa_mpjpe = float(np.linalg.norm(p_al - g, axis=-1).mean())

    # Acceleration error / jitter: 2nd time-difference of joints.
    if k >= 3:
        ap = np.diff(p, n=2, axis=0)           # [k-2,J,3]
        ag = np.diff(g, n=2, axis=0)
        accel_err = float(np.linalg.norm(ap - ag, axis=-1).mean())
        accel_pred = float(np.linalg.norm(ap, axis=-1).mean())
        accel_gt = float(np.linalg.norm(ag, axis=-1).mean())
    else:
        accel_err = accel_pred = accel_gt = float("nan")

    # Global root-joint (joint 0) trajectory error.
    root_err = float(np.linalg.norm(p[:, 0] - g[:, 0], axis=-1).mean())

    return dict(mpjpe_m=mpjpe, pa_mpjpe_m=pa_mpjpe, accel_err=accel_err,
                accel_pred=accel_pred, accel_gt=accel_gt, root_err_m=root_err, frames=int(k))


def recon_metrics(pred_z: np.ndarray, gt_z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    """pred_z, gt_z : [T,283] Z-SCORED motion. Full recon metric row (feat_mse + joint metrics).

    feat_mse is computed in the z-scored space over the OVERLAPPING frames (the training loss
    space); joint metrics decode the UN-normalized motion to world joints.
    """
    k = min(len(pred_z), len(gt_z))
    pz, gz = pred_z[:k], gt_z[:k]
    feat_mse = float(((pz - gz) ** 2).mean())

    pred_j = decode_to_joints(unnormalize(pz, mean, std))
    gt_j = decode_to_joints(unnormalize(gz, mean, std))
    row = joint_metrics(pred_j, gt_j)
    row["feat_mse"] = feat_mse
    return row


# ---------------------------------------------------------------------------
# Aggregate + write (invdyn_metrics.json schema) + print a table.
# ---------------------------------------------------------------------------
def aggregate_and_write(rows: dict, out_path: str, *, tag: str = "", generative: bool = False):
    """rows: {seq_name: metric_dict}. Writes JSON in the invdyn_metrics schema + prints a table."""
    seqs = sorted(rows)
    if not seqs:
        raise SystemExit("[eval_motion_recon] no sequences to aggregate")
    keys = [k for k in METRIC_KEYS if k in rows[seqs[0]]]

    def _safe(vals):
        v = [x for x in vals if x == x]  # drop NaN
        return v or [float("nan")]

    agg = {k: {"mean": float(np.mean(_safe([rows[n][k] for n in seqs]))),
               "median": float(np.median(_safe([rows[n][k] for n in seqs])))} for k in keys}

    hdr = f"{'sequence':32s} " + " ".join(f"{k:>12s}" for k in keys)
    print(f"\n=== motion-recon eval | {len(seqs)} windows | {tag} ===")
    if generative:
        print("  NOTE: this is a GENERATIVE task (text->motion); recon metrics vs GT are NOT a "
              "reconstruction score (there is no single correct motion). Reported for reference.")
    print(hdr); print("-" * len(hdr))
    for n in seqs:
        print(f"{n[:32]:32s} " + " ".join(f"{rows[n][k]:12.4f}" for k in keys))
    print("-" * len(hdr))
    print(f"{'MEAN':32s} " + " ".join(f"{agg[k]['mean']:12.4f}" for k in keys))
    print(f"{'MEDIAN':32s} " + " ".join(f"{agg[k]['median']:12.4f}" for k in keys))
    print("guide: mpjpe_m/pa_mpjpe_m/root_err_m/accel_err all lower-better (m); "
          "accel_pred vs accel_gt = jitter ratio; feat_mse in z-scored space")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump({"n": len(seqs), "generative": generative, "aggregate": agg, "per_sequence": rows},
              open(out_path, "w"), indent=2)
    return out_path


# ---------------------------------------------------------------------------
# CPU self-test on synthetic arrays (no GPU, no ckpt).
# ---------------------------------------------------------------------------
def _selftest():
    print("[eval_motion_recon] SELFTEST on synthetic 283-d motion (CPU)")
    rng = np.random.default_rng(0)
    T, D = 20, 283
    try:
        mean, std = load_stats()
        print(f"  loaded stats mean{mean.shape} std{std.shape}")
    except Exception as e:  # noqa: BLE001
        print(f"  (config stats unavailable: {e}); using unit stats")
        mean, std = np.zeros(D, np.float32), np.ones(D, np.float32)

    gt_z = rng.standard_normal((T, D)).astype(np.float32) * 0.5
    # perfect reconstruction -> all errors ~0
    row_perfect = recon_metrics(gt_z.copy(), gt_z.copy(), mean, std)
    print("  perfect-recon row:", {k: round(v, 6) for k, v in row_perfect.items()})
    assert row_perfect["feat_mse"] < 1e-10, "perfect feat_mse should be ~0"
    assert row_perfect["mpjpe_m"] < 1e-4, "perfect MPJPE should be ~0"
    assert row_perfect["pa_mpjpe_m"] < 1e-3, "perfect PA-MPJPE should be ~0"

    # noised prediction -> positive errors
    pred_z = gt_z + rng.standard_normal((T, D)).astype(np.float32) * 0.3
    row_noisy = recon_metrics(pred_z, gt_z, mean, std)
    print("  noisy-recon row:  ", {k: round(v, 6) for k, v in row_noisy.items()})
    assert row_noisy["feat_mse"] > row_perfect["feat_mse"], "noise must raise feat_mse"
    assert row_noisy["mpjpe_m"] > row_perfect["mpjpe_m"], "noise must raise MPJPE"
    assert np.isfinite(row_noisy["accel_err"]), "accel_err must be finite for T>=3"

    # aggregate + write to a temp path
    import tempfile
    out = os.path.join(tempfile.gettempdir(), "motion_recon_selftest.json")
    aggregate_and_write({"seq0": row_perfect, "seq1": row_noisy}, out, tag="selftest")
    d = json.load(open(out))
    assert set(d.keys()) == {"n", "generative", "aggregate", "per_sequence"}
    assert d["n"] == 2
    print(f"  wrote + reloaded {out}  (n={d['n']})")
    print("[eval_motion_recon] SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run the CPU synthetic-array self-test (no GPU / no ckpt) and exit")
    ap.add_argument("--pred_dir", default=None,
                    help="dir of <seq>.npy z-scored predicted motion [T,283] (paired with --gt_dir)")
    ap.add_argument("--gt_dir", default=None, help="dir of <seq>.npy z-scored GT motion [T,283]")
    ap.add_argument("--out", default=None, help="output metrics JSON path")
    ap.add_argument("--mean", default=None)
    ap.add_argument("--std", default=None)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not (args.pred_dir and args.gt_dir and args.out):
        raise SystemExit("provide --selftest OR (--pred_dir --gt_dir --out)")
    import glob
    mean, std = load_stats(args.mean, args.std)
    rows = {}
    for p in sorted(glob.glob(os.path.join(args.pred_dir, "*.npy"))):
        n = os.path.splitext(os.path.basename(p))[0]
        g = os.path.join(args.gt_dir, n + ".npy")
        if not os.path.isfile(g):
            continue
        rows[n] = recon_metrics(np.load(p), np.load(g), mean, std)
    aggregate_and_write(rows, args.out, tag=args.pred_dir)


if __name__ == "__main__":
    main()
