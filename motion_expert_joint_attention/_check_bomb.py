"""Verify whether a flagged 'bomb' window is real: compare raw y-channels vs DECODED joints."""
import json
import os

import numpy as np
import torch

import config
from decode_uniego_torch import decode_joints
from uniego_layout import JOINT_Y_IDX, canonicalize_frame0, ground_features

bombs = json.load(open("/weka/jungbin/cosmos_motion_ft_runs/joint_attention/bomb_windows.json"))
cal = json.load(open(config.FLOOR_CALIBRATION_JSON))
deltas = cal["deltas"]

for b in bombs[:3] + bombs[len(bombs) // 2 : len(bombs) // 2 + 2]:
    u, ws, we = b["uuid"], b["ws"], b["we"]
    p = f"/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep/{u}.npz"
    with np.load(p) as z:
        feats = z["features"][ws:we].astype(np.float32)
    # manifest offset for this window
    off = None
    with open(config.NYMERIA_MANIFEST) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("uuid") != u:
                continue
            for w in rec.get("t2w_windows", []):
                if int(w["start_frame"]) == ws:
                    off = w.get("ground_offset_y")
            break
    total = off + deltas[u]
    g = ground_features(feats, total)
    raw_max_y = float(np.abs(g[:, JOINT_Y_IDX]).max())
    # DECODED joints (the ground truth): mirror the dataset (ground -> canon -> decode)
    c = canonicalize_frame0(g)
    j = decode_joints(torch.from_numpy(c).unsqueeze(0))[0].numpy()  # (T,30,3)
    dec_max_y = float(j[..., 1].max())
    dec_min_foot = float(j[:, [24, 25, 28, 29], 1].min())
    height = float((j[..., 1].max(-1) - j[..., 1].min(-1)).mean())
    print(f"{u}@{ws}: raw_max|y|={raw_max_y:.2f}m  DECODED max_y={dec_max_y:.2f}m "
          f"min_foot={dec_min_foot:+.3f}m body_height={height:.2f}m  claimed={b['maxy']}m")
