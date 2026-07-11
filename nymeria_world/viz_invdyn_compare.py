"""Compare inverse-dynamics metrics of several checkpoints over a test set (per-sequence distribution).
Reads each run's eval_full71/invdyn_metrics.json -> box+scatter per metric. Usage: edit RUNS or pass via args."""
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world"
RUNS = [
    ("ZERO-SHOT (base)", "/weka/jungbin/cosmos_motion_ft_runs/zeroshot_eval/eval_full71/invdyn_metrics.json"),
    ("97f_b8 iter13500", f"{R}/world_camera_nymeria_97f_b8/checkpoints/iter_000013500/eval_full71/invdyn_metrics.json"),
    ("97f-hung iter7000", f"{R}/world_camera_nymeria_97f_hung_iter6000/checkpoints/iter_000007000/eval_full71/invdyn_metrics.json"),
    ("33f-hung iter7000", f"{R}/world_camera_nymeria_33f_hung_iter7500/checkpoints/iter_000007000/eval_full71/invdyn_metrics.json"),
]
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{R}/world_camera_nymeria_97f_b8/checkpoints/iter_000013500/eval_full71/viz/invdyn_compare_71.png"

data = [(n, json.load(open(p))["per_sequence"]) for n, p in RUNS]
mets = [("rot_deg", "rotation err deg (lower)"), ("trans_dir_cos", "dir cosine (->1)"),
        ("scale_ratio", "scale ratio (->1)"), ("ate_m", "ATE m (lower)")]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4)); cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for ax, (mk, ml) in zip(axes, mets):
    series = [[s[mk] for s in d.values()] for _, d in data]
    bp = ax.boxplot(series, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, sv in enumerate(series): ax.scatter([i + 1] * len(sv), sv, s=6, c="k", alpha=0.22, zorder=3)
    means = [np.mean(s) for s in series]
    for i, m in enumerate(means): ax.text(i + 1, m, f"  {m:.3f}", fontsize=8, va="center", color="darkred")
    ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels([n for n, _ in data], rotation=20, ha="right", fontsize=8)
    ax.set_title(ml, fontsize=11); ax.grid(axis="y", alpha=0.3)
    if mk in ("scale_ratio", "trans_dir_cos"): ax.axhline(1.0, ls="--", c="r", lw=1)
    if mk == "scale_ratio": ax.set_yscale("log"); ax.set_ylabel("log scale")
fig.suptitle("Inverse-dynamics on 71 held-out test sequences  (box = IQR across sequences, dots = per-seq, red = mean)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT, dpi=120); print("saved", OUT)
