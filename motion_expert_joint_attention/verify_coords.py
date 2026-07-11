"""Verify NymeriaPlus vs BONES-SEED share the SAME uniego coordinate convention before trusting
the stats comparison: (1) raw neutral_joints up-axis + scale, (2) decoded motion-clip up-axis/ground."""
import glob, random, sys
import numpy as np
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention")
from uniego_layout import canonicalize_frame0
try:
    from decode_uniego_torch import decode_joints
    import torch
    HAVE_DECODE = True
except Exception as e:
    HAVE_DECODE = False; print("decode unavailable:", e)

def neutral_report(files, name, n=200, seed=0):
    rng = random.Random(seed); J = []
    for f in rng.sample(files, min(n, len(files))):
        try:
            with np.load(f) as d: nj = d["neutral_joints"].astype(np.float64)
        except Exception: continue
        J.append(nj)
    J = np.stack(J)  # [n,30,3]
    rng_axis = (J.max(1) - J.min(1)).mean(0)         # mean per-axis extent of the skeleton
    up = int(np.argmax(rng_axis))                     # axis with biggest extent = up (standing)
    # head-vs-pelvis sign on the up axis: joint 0 is usually pelvis/root; top = max on up axis
    top_minus_root = (J[:, :, up].max(1) - J[:, 0, up]).mean()
    print(f"  [{name}] neutral_joints per-axis extent (x,y,z) = "
          f"({rng_axis[0]:.3f},{rng_axis[1]:.3f},{rng_axis[2]:.3f})  -> UP-AXIS={'xyz'[up]}  "
          f"top-above-root on up = {top_minus_root:+.3f}")
    return up, rng_axis

def decoded_report(files, name, n=40, T=60, seed=1):
    if not HAVE_DECODE: return
    rng = random.Random(seed); ups = []; heights = []
    for f in rng.sample(files, min(n, len(files))):
        try:
            with np.load(f) as d: x = d["features"].astype(np.float32)
        except Exception: continue
        if x.shape[0] < 2: continue
        s = rng.randint(0, max(0, x.shape[0]-T))
        feat = canonicalize_frame0(x[s:s+T])
        j = decode_joints(torch.from_numpy(feat).unsqueeze(0))[0].numpy()  # [T,30,3]
        ext = (j.reshape(-1,3).max(0) - j.reshape(-1,3).min(0))
        ups.append(int(np.argmax(ext))); heights.append(ext)
    ups = np.array(ups); H = np.stack(heights).mean(0)
    from collections import Counter
    print(f"  [{name}] decoded clip per-axis extent (x,y,z) = ({H[0]:.2f},{H[1]:.2f},{H[2]:.2f})  "
          f"up-axis vote = {dict(Counter('xyz'[u] for u in ups))}")

nym = sorted(glob.glob("/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep/*/*.npz"))
bon = sorted(glob.glob("/weka/jungbin/seed/soma_proportional_uniegomotion_20fps/*/*.npz"))
print("=== 1) raw neutral_joints (skeleton frame + scale) ===")
nu, _ = neutral_report(nym, "nymeria"); bu, _ = neutral_report(bon, "bones")
print("=== 2) decoded motion clip (motion frame + ground) ===")
decoded_report(nym, "nymeria"); decoded_report(bon, "bones")
print(f"\nVERDICT: neutral up-axis match = {nu==bu} ({'xyz'[nu]} vs {'xyz'[bu]})")
