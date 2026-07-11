#!/usr/bin/env python
"""CPU verification of the floor calibration + window-drop dataset integration (Part C check).

Instantiates ``NymeriaJointDataset`` in BOTH training configs -- T=200 native text2motion and
T=97 aligned 7-task -- with calibration ON vs OFF (``floor_calibration_json=None``), reports
index sizes before/after drops, then decodes real samples through the REAL ``__getitem__`` /
``_load_motion`` path (z-scored -> unnormalize -> decode_uniego_torch) and measures each
window's min-foot-y. Also verifies BONES samples are bit-identical with calibration on/off.

Run (CPU, cosmos env, on a node):
    ssh <node> 'cd .../motion_expert_joint_attention && LD_LIBRARY_PATH= \
        ~/miniforge3/envs/cosmos/bin/python _verify_floor_calibration.py'
"""
from __future__ import annotations

import numpy as np
import torch

import config
from nymeria_joint_dataset import NymeriaJointDataset
from decode_uniego_torch import decode_joints

FOOT_J = [24, 25, 28, 29]


def minfoot_stats(vals, tag):
    a = np.asarray(vals)
    print(f"  [{tag}] n={a.size} min-foot-y: median={np.median(a)*100:+.2f}cm "
          f"p10={np.percentile(a,10)*100:+.1f} p90={np.percentile(a,90)*100:+.1f} | "
          f"pen>5cm {(a < -0.05).mean():.1%}  pen>20cm {(a < -0.20).mean():.1%}", flush=True)
    return a


def decode_minfoot(ds, n, tag):
    """Draw n samples through the REAL path; decode to joints; min foot y over valid frames."""
    mean = torch.from_numpy(ds._mean)
    std = torch.from_numpy(ds._std)
    vals, caps = [], []
    step = max(1, len(ds._t2m_index) // n)
    got = i = 0
    while got < n and i < len(ds):
        item = ds[i * step]
        i += 1
        if item["source"] != "nymeria" or item.get("motion") is None:
            continue
        m = item["motion"]                    # [T,283] z-scored (padded)
        pad = item["motion_pad_mask"]
        feat = m[~pad].float() * std + mean   # valid frames, unnormalized
        j = decode_joints(feat.unsqueeze(0))[0]        # [k,30,3]
        vals.append(float(j[:, FOOT_J, 1].min()))
        caps.append(item["caption"][:40])
        got += 1
    return minfoot_stats(vals, tag)


def main():
    torch.manual_seed(0)
    common = dict(split="train", prefer_latents=False, force_on_the_fly=True, seed=0)

    # ---------- T=200 native text2motion config ----------
    print("== T=200 text2motion config ==", flush=True)
    t2m_w = {"text2motion": 1.0}
    ds200_off = NymeriaJointDataset(num_frames=200, task_weights=t2m_w,
                                    bones_text2motion_frac=0.0,
                                    floor_calibration_json=None, **common)
    print(f"  UNCALIBRATED: _index={len(ds200_off._index)} "
          f"_t2m_index={len(ds200_off._t2m_index)}", flush=True)
    ds200_on = NymeriaJointDataset(num_frames=200, task_weights=t2m_w,
                                   bones_text2motion_frac=0.0, **common)
    print(f"  CALIBRATED  : _index={len(ds200_on._index)} "
          f"_t2m_index={len(ds200_on._t2m_index)}", flush=True)

    # entry bookkeeping: off must equal off_gt + delta
    e = ds200_on._t2m_index[0]
    assert abs(e["off"] - (e["off_gt"] + e["delta"])) < 1e-9, e
    print(f"  entry[0]: off={e['off']:+.4f} = off_gt={e['off_gt']:+.4f} + delta={e['delta']:+.4f}")

    # ---------- decode ~20 samples through the REAL path ----------
    print("\n== decoded min-foot-y through the REAL __getitem__ path (T=200 t2m) ==", flush=True)
    decode_minfoot(ds200_off, 20, "BEFORE (uncalibrated)")
    decode_minfoot(ds200_on, 20, "AFTER  (calibrated)  ")

    # ---------- T=97 aligned 7-task config ----------
    print("\n== T=97 aligned 7-task config ==", flush=True)
    ds97_off = NymeriaJointDataset(num_frames=97, task_weights=dict(config.TASK_WEIGHTS),
                                   bones_text2motion_frac=0.0,
                                   floor_calibration_json=None, **common)
    print(f"  UNCALIBRATED: _index={len(ds97_off._index)} "
          f"_t2m_index={len(ds97_off._t2m_index)}", flush=True)
    ds97_on = NymeriaJointDataset(num_frames=97, task_weights=dict(config.TASK_WEIGHTS),
                                  bones_text2motion_frac=0.0, **common)
    print(f"  CALIBRATED  : _index={len(ds97_on._index)} "
          f"_t2m_index={len(ds97_on._t2m_index)}", flush=True)

    # aligned-index motion path (no video decode -- call _load_motion directly, the real seam)
    print("\n== aligned _index motion windows (_load_motion, T=97) ==", flush=True)
    for ds, tag in ((ds97_off, "BEFORE (uncalibrated)"), (ds97_on, "AFTER  (calibrated)  ")):
        mean = torch.from_numpy(ds._mean)
        std = torch.from_numpy(ds._std)
        vals = []
        step = max(1, len(ds._index) // 20)
        for k in range(20):
            it = ds._index[(k * step) % len(ds._index)]
            feats, nj, pad = ds._load_motion(it["uni"], it["s"], it["off"])
            feat = torch.from_numpy(feats[~pad]).float() * std + mean
            j = decode_joints(feat.unsqueeze(0))[0]
            vals.append(float(j[:, FOOT_J, 1].min()))
        minfoot_stats(vals, tag)

    # ---------- BONES unchanged ----------
    print("\n== BONES samples unchanged by calibration ==", flush=True)
    b_on = NymeriaJointDataset(num_frames=200, task_weights=t2m_w,
                               bones_text2motion_frac=1.0, train=False, split="test",
                               prefer_latents=False, force_on_the_fly=True, seed=0)
    b_off = NymeriaJointDataset(num_frames=200, task_weights=t2m_w,
                                bones_text2motion_frac=1.0, train=False, split="test",
                                prefer_latents=False, force_on_the_fly=True, seed=0,
                                floor_calibration_json=None)
    if b_on._bones is None or b_off._bones is None:
        print("  BONES stream unavailable -- skipped", flush=True)
    else:
        same = True
        vals = []
        mean = torch.from_numpy(b_on._mean)
        std = torch.from_numpy(b_on._std)
        for i in range(8):
            x = b_on._bones[i]["motion"]
            y = b_off._bones[i]["motion"]
            same = same and torch.equal(x, y)
            feat = x.float() * std + mean
            j = decode_joints(feat.unsqueeze(0))[0]
            vals.append(float(j[:, FOOT_J, 1].min()))
        print(f"  8 BONES items bit-identical calib-on vs calib-off: {same}")
        minfoot_stats(vals, "BONES (reference)")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
