"""Metric-scale inverse-dynamics montage: rows = test samples, cols = models.
Each cell plots GT (green) and predicted (red) camera trajectory in the SHARED first-frame
coordinate frame, NO scale alignment -> the metric over-prediction is visible. Axes are shared
per ROW (same limits across models for a sample), so the red prediction visibly shrinks toward GT
as finetuning improves the metric scale. 2D projection onto the GT trajectory's principal plane."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world"
GT_ROOT = "/weka/jungbin/cosmos_motion_ft_runs/nymeria_eval_full/samples"
MODELS = [
    ("zero-shot", "/weka/jungbin/cosmos_motion_ft_runs/zeroshot_eval/eval_full71/invdyn_out"),
    ("97f_b8 i13500", R + "/world_camera_nymeria_97f_b8/checkpoints/iter_000013500/eval_full71/invdyn_out"),
    ("97f-hung i7000", R + "/world_camera_nymeria_97f_hung_iter6000/checkpoints/iter_000007000/eval_full71/invdyn_out"),
    ("33f-hung i7000", R + "/world_camera_nymeria_33f_hung_iter7500/checkpoints/iter_000007000/eval_full71/invdyn_out"),
]
# same 5 sequences as the FD montage
SEQS = ["t0_S07_20231013_s0_shelley_jones_act3_w23p1b", "t1_S05_20230817_s1_rebecca_ward_act2_39a7o2",
        "t2_S19_20230926_s1_megan_mejia_act4_qfzkks", "t3_S07_20231004_s1_ashley_reyes_act4_mhhxmg",
        "t4_S19_20231020_s0_anthony_chen_act1_vhzag9"]
import sys
PERCELL = "percell" in sys.argv
OUT = "/weka/jungbin/cosmos_motion_ft_runs/zeroshot_eval/invdyn_metric_montage" + ("_percell" if PERCELL else "") + ".png"


def rot6d_to_R(v6):  # single (6,) -> (3,3), proven version from eval_inverse_dynamics
    a0, a1 = v6[:3], v6[3:6]
    b0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1p = a1 - (b0 @ a1) * b0; b1 = a1p / (np.linalg.norm(a1p) + 1e-8)
    return np.stack([b0, b1, np.cross(b0, b1)], 1)


def pred_pos(a9):  # per-step integration (proven)
    a9 = np.asarray(a9, float); P = [np.eye(4)]; c = P[0]
    for i in range(len(a9)):
        d = np.eye(4); d[:3, :3] = rot6d_to_R(a9[i, 3:9]); d[:3, 3] = a9[i, :3]
        c = c @ d; P.append(c.copy())
    return np.stack(P)[:, :3, 3]


def gt_pos(npz):
    d = np.load(npz); pos, rot = d["cam_world_pos"].astype(float), d["cam_world_rot"].astype(float)
    T = len(pos); P = np.tile(np.eye(4), (T, 1, 1)); P[:, :3, :3] = rot; P[:, :3, 3] = pos
    return (np.linalg.inv(P[0]) @ P)[:, :3, 3]


nR, nC = len(SEQS), len(MODELS)
fig, axes = plt.subplots(nR, nC, figsize=(3.1 * nC, 3.1 * nR))
for r, seq in enumerate(SEQS):
    gp = gt_pos(os.path.join(GT_ROOT, seq, "gt_camera_cosmos.npz"))
    # PCA plane from GT
    c0 = gp.mean(0); U, S, Vt = np.linalg.svd(gp - c0); B = Vt[:2]  # 2x3 basis
    def proj(X): return (X - c0) @ B.T
    g2 = proj(gp); glen = np.linalg.norm(np.diff(gp, axis=0), axis=1).sum()
    preds, plens = {}, {}
    for name, root in MODELS:
        a = np.array(json.load(open(os.path.join(root, seq, "sample_outputs.json")))["outputs"][0]["content"]["action"], float)
        p3 = pred_pos(a); preds[name] = proj(p3); plens[name] = np.linalg.norm(np.diff(p3, axis=0), axis=1).sum()
    allpts = np.concatenate([g2] + list(preds.values()))
    row_ctr = allpts.mean(0); row_rad = np.abs(allpts - row_ctr).max() * 1.1 + 1e-3
    for c, (name, _) in enumerate(MODELS):
        ax = axes[r, c]; p2 = preds[name]; plen = plens[name]
        if PERCELL:  # each cell auto-zoomed to its own GT+pred (+ 1m scale bar to keep it metric)
            cp = np.concatenate([g2, p2]); ctr = cp.mean(0); rad = np.abs(cp - ctr).max() * 1.15 + 1e-3
        else:        # per-row shared axes -> the metric scale collapse is visually dramatic
            ctr, rad = row_ctr, row_rad
        ax.plot(*g2.T, "g-", lw=2, label="GT"); ax.plot(*p2.T, "r-", lw=1.6, label="pred")
        ax.scatter(*g2[0], c="k", s=18, zorder=5)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\nGT {glen:.1f}m | pred {plen:.1f}m ({plen/max(glen,1e-6):.1f}x)", fontsize=8)
        if PERCELL or c == 0:  # scale bar (every cell in per-cell mode; GT col otherwise)
            x0 = ctr[0] - rad * 0.9; y0 = ctr[1] - rad * 0.85
            ax.plot([x0, x0 + 1.0], [y0, y0], "k-", lw=2.5); ax.text(x0, y0 + rad * 0.04, "1 m", fontsize=7)
        if c == 0:
            ax.text(0.03, 0.04, "_".join(seq.split("_")[:2]), transform=ax.transAxes, fontsize=9, color="navy",
                    weight="bold")
            ax.legend(fontsize=7, loc="upper right")
_sub = ("each cell auto-zoomed + 1m scale bar -> shape match; ratios in titles give the metric scale"
        if PERCELL else "per-row shared axes, NO scale alignment -> pred shrinks toward GT as finetuning fixes scale")
fig.suptitle(f"Inverse-dynamics in METRIC scale: predicted (red) vs GT (green) camera trajectory  ({_sub})", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.985]); fig.savefig(OUT, dpi=120)
print("saved", OUT)
