"""Verify motion_decode.py (the pure-torch port) against kimodo's real decoder.

Run in the kimodo env. Decodes a few real (normalized) samples from the subset
export with BOTH the port and KimodoMotionRep.inverse, and reports max abs diff
of the resulting world joint positions.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/jungbin_cho/kimodo_open")
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft")

import motion_decode as md  # the port

STATS = "/weka/jungbin/seed/stats/soma_uniform_motions_20fps/"
SUBSET = "/weka/jungbin/seed/cosmos_text_motion_subset"
SKEL = "/home/jungbin_cho/cosmos_motion_ft/skeleton_soma30.npz"

# ---- load a few real normalized samples ----
ix = json.load(open(SUBSET + "/index.json"))
feats = np.load(SUBSET + "/features.npy", mmap_mode="r")
off = ix["offsets"]
samples = []
for i in [0, 1, 7, 42, 100]:
    s = np.asarray(feats[off[i]:off[i + 1]], dtype=np.float32)  # (T,369) normalized
    samples.append(torch.from_numpy(s))

# ---- port decode ----
skel = md.load_skeleton(SKEL)
mean, std = md.load_stats(STATS)
port_joints = [md.decode_features_to_joints(s, skel, is_normalized=True, stats=(mean, std)) for s in samples]

# ---- kimodo decode ----
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.definitions import SOMASkeleton30

mr = KimodoMotionRep(skeleton=SOMASkeleton30(), fps=20, stats_path=STATS)
kimodo_joints = []
for s in samples:
    out = mr.inverse(s, is_normalized=True, posed_joints_from="rotations")
    kimodo_joints.append(out["posed_joints"])

# ---- compare ----
print(f"{'sample':>8} {'T':>5} {'port_shape':>16} {'kimodo_shape':>16} {'max_abs_diff':>14} {'mean_abs_diff':>14}")
ok = True
for i, (p, k) in enumerate(zip(port_joints, kimodo_joints)):
    p = p.detach().float()
    k = k.detach().float()
    if p.shape != k.shape:
        print(f"  sample {i}: SHAPE MISMATCH port={tuple(p.shape)} kimodo={tuple(k.shape)}")
        ok = False
        continue
    d = (p - k).abs()
    print(f"{i:>8} {p.shape[0]:>5} {str(tuple(p.shape)):>16} {str(tuple(k.shape)):>16} {d.max().item():>14.6e} {d.mean().item():>14.6e}")
    if d.max().item() > 1e-3:
        ok = False

print("\nRESULT:", "PASS (max diff < 1e-3)" if ok else "FAIL — investigate FK/offset/root convention")
