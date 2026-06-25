"""Decode REAL GT uniego windows through the exact train pipeline to isolate
decode/normalization correctness from model quality. (kimodo env)

Renders: gt_raw (decode stored features directly) and gt_roundtrip (canon0→normalize→
unnormalize→decode). Both should be clean human motion if the pipeline is correct.
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/motion_expert")
sys.path.insert(0, "/home/jungbin_cho/kimodo_open")
from uniego_layout import canonicalize_frame0, CANON_DELTA_SLICE
from kimodo.motion_rep.uniego import uniego_world_joints_from_features
from viz import render

HERE = "/home/jungbin_cho/cosmos_motion_ft/motion_expert"
OUT = "/weka/jungbin/cosmos_motion_ft_runs/motionexpert_poc_v1/gt_check"
os.makedirs(OUT, exist_ok=True)
mean = np.load(f"{HERE}/stats/uniego283_mean.npy"); std = np.load(f"{HERE}/stats/uniego283_std.npy")
sk = np.load("/home/jungbin_cho/cosmos_motion_ft/skeleton_soma30.npz", allow_pickle=True)
parents = sk["parents"]

rows = [json.loads(l) for l in open(f"{HERE}/pairs_val.jsonl")][:3]
for i, r in enumerate(rows):
    feat = np.load(r["uniego_path"])["features"][r["start"]:r["end"]].astype(np.float32)
    feat = feat[:96]
    feat0 = canonicalize_frame0(feat)

    # (a) decode stored GT directly
    j_raw = uniego_world_joints_from_features(torch.from_numpy(feat0).float(), n_joints=30).numpy()
    rs_raw = np.linalg.norm(np.diff(j_raw[:, 0], axis=0), axis=1).mean()
    render(j_raw, parents, f"{OUT}/gt{i}_raw.mp4", title=f"GT raw: {r['caption'][:30]}")

    # (b) normalize then unnormalize then decode (the model's I/O path)
    rt = ((feat0 - mean) / std) * std + mean
    j_rt = uniego_world_joints_from_features(torch.from_numpy(rt.astype(np.float32)).float(), n_joints=30).numpy()
    err = np.abs(j_rt - j_raw).max()

    # canon_delta rotation magnitude (per-frame residual) — "spinning" check
    cd = feat0[:, CANON_DELTA_SLICE]
    print(f"gt{i} '{r['caption'][:34]}' root-step={rs_raw:.4f} m  roundtrip_max_err={err:.2e}")
    print(f"     canon_delta[1:] rot6d range [{cd[1:,:6].min():.3f},{cd[1:,:6].max():.3f}] "
          f"trans range [{cd[1:,6:9].min():.3f},{cd[1:,6:9].max():.3f}]")
print(f"[gt_check] rendered to {OUT}")
