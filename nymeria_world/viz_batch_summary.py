"""Aggregate inverse-dynamics findings across a batch of samples.

For each sample: model per-action |Δt|, GT 1-frame |Δt|, the best-matching k (which
multi-frame GT delta matches the model's magnitude), and directional cosine of the
predicted vs GT translation in the device frame vs the corrected OpenCV frame.

Three panels:
  (1) scale ratio model/GT(k=1)         -> the "7-17x" spread
  (2) best-matching k (GT k-frame delta) -> clusters ~6-8 = temporal-step mismatch
  (3) dir-cosine device vs corrected     -> corrected frame helps direction
Saves <root>/batch_summary.png and prints a table.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def relt(P):
    return np.stack([(np.linalg.inv(P[i]) @ P[i + 1])[:3, 3] for i in range(len(P) - 1)])


def abs_from(npz):
    d = np.load(npz); pos, rot = d["cam_world_pos"].astype(np.float64), d["cam_world_rot"].astype(np.float64)
    T = len(pos); P = np.tile(np.eye(4), (T, 1, 1)); P[:, :3, :3] = rot; P[:, :3, 3] = pos
    return P


def cos(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9); bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return float((an * bn).sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline_batch"])
    ap.add_argument("--out", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline_batch/batch_summary.png")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        for p in sorted(glob.glob(os.path.join(root, "invdyn_out", "*", "sample_outputs.json"))):
            name = os.path.basename(os.path.dirname(p))
            sd = os.path.join(root, "samples", name)
            pred = np.array(json.load(open(p))["outputs"][0]["content"]["action"], dtype=np.float64)
            pmag = np.linalg.norm(pred[:, :3], axis=1).mean()
            pos = np.load(os.path.join(sd, "gt_camera.npz"))["cam_world_pos"].astype(np.float64)
            gtk = {k: np.linalg.norm(pos[k:] - pos[:-k], axis=1).mean() for k in range(1, 16) if k < len(pos)}
            gt1 = gtk[1]
            if gt1 < 1e-4:  # degenerate static clip
                continue
            bestk = min(gtk, key=lambda k: abs(gtk[k] - pmag))
            Pd, Pc = abs_from(os.path.join(sd, "gt_camera.npz")), abs_from(os.path.join(sd, "gt_camera_cosmos.npz"))
            Tn = min(len(pred), len(Pd) - 1)
            cdev, ccos = cos(pred[:Tn, :3], relt(Pd)[:Tn]), cos(pred[:Tn, :3], relt(Pc)[:Tn])
            rows.append(dict(name=name, pmag=pmag, gt1=gt1, ratio=pmag / gt1, bestk=bestk,
                             cdev=cdev, ccos=ccos))
    rows.sort(key=lambda r: r["ratio"])
    labels = [r["name"].split("_", 1)[1][:10] for r in rows]
    print(f"{'sample':16s} {'GT|Δt|':>8} {'pred|Δt|':>8} {'ratio':>6} {'best-k':>6} {'dir(dev)':>8} {'dir(cosmos)':>11}")
    for r in rows:
        print(f"{r['name'][:16]:16s} {r['gt1']:8.4f} {r['pmag']:8.4f} {r['ratio']:6.1f} {r['bestk']:6d} {r['cdev']:8.2f} {r['ccos']:11.2f}")
    if rows:
        print(f"\nmedian ratio={np.median([r['ratio'] for r in rows]):.1f}x  median best-k={np.median([r['bestk'] for r in rows]):.0f}  "
              f"mean dir(dev)={np.mean([r['cdev'] for r in rows]):.2f}  mean dir(cosmos)={np.mean([r['ccos'] for r in rows]):.2f}")

    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].bar(x, [r["ratio"] for r in rows], color="indianred"); ax[0].axhline(1, ls="--", c="k", alpha=.5)
    ax[0].set_title("scale ratio  model / GT(1 frame)"); ax[0].set_ylabel("×"); ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=70, fontsize=7)
    ax[1].bar(x, [r["bestk"] for r in rows], color="steelblue")
    med = np.median([r["bestk"] for r in rows]); ax[1].axhline(med, ls="--", c="k", alpha=.6, label=f"median k={med:.0f}")
    ax[1].set_title("best-matching k: model |Δt| ≈ GT over k frames"); ax[1].set_ylabel("frames"); ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=70, fontsize=7); ax[1].legend()
    w = 0.4
    ax[2].bar(x - w/2, [r["cdev"] for r in rows], w, label="device frame", color="0.6")
    ax[2].bar(x + w/2, [r["ccos"] for r in rows], w, label="corrected (OpenCV)", color="seagreen")
    ax[2].set_title("direction cosine(pred, GT): device vs corrected"); ax[2].set_ylabel("cosine"); ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=70, fontsize=7); ax[2].legend()
    fig.suptitle(f"Zero-shot inverse-dynamics across {len(rows)} NymeriaPlus clips", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=120); print("saved", args.out)


if __name__ == "__main__":
    main()
