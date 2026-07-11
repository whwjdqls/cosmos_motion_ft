"""Compare the 283-d uniego mean/std of NymeriaPlus vs BONES-SEED (20fps proportional).
Both processed identically (canonicalize_frame0) so it's apples-to-apples. Reports per-block
stats AND the key 'mismatch' = what BONES looks like AFTER z-scoring with Nymeria's stats
(residual mean should be ~0, residual std ~1 if the shared stats fit BONES)."""
import glob, random, sys
import numpy as np
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention")
from uniego_layout import canonicalize_frame0

BLOCKS = [("joints SE3 [0:270]", 0, 270), ("canon_delta [270:279]", 270, 279), ("foot [279:283]", 279, 283)]

def collect(files, n_clips, clip_len=100, seed=0):
    rng = random.Random(seed); feats = []
    for f in rng.sample(files, min(n_clips, len(files))):
        try:
            with np.load(f) as d: x = d["features"].astype(np.float32)
        except Exception: continue
        if x.shape[0] < 2: continue
        s = rng.randint(0, max(0, x.shape[0]-clip_len))
        feats.append(canonicalize_frame0(x[s:s+clip_len]))
    X = np.concatenate(feats, 0)
    return X.mean(0), X.std(0), X.shape[0]

nym = sorted(glob.glob("/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep/*/*.npz"))
bon = sorted(glob.glob("/weka/jungbin/seed/soma_proportional_uniegomotion_20fps/*/*.npz"))
print(f"nymeria files: {len(nym)} | bones files: {len(bon)}")
nm_mean, nm_std, nm_n = collect(nym, 400)
bs_mean, bs_std, bs_n = collect(bon, 400)
print(f"frames pooled: nymeria={nm_n} bones={bs_n}\n")

# Key: BONES z-scored with Nymeria stats -> residual mean & std (target 0 and 1).
resid_mean = (bs_mean - nm_mean) / (nm_std + 1e-6)
resid_std  = bs_std / (nm_std + 1e-6)
print("=== Per-block summary (283-d) ===")
print(f"{'block':22s} {'|mean| nym':>11s} {'|mean| bon':>11s} {'mean std nym':>13s} {'mean std bon':>13s} "
      f"{'|resid mean|':>12s} {'resid std':>10s}")
for name, a, b in BLOCKS:
    print(f"{name:22s} {np.abs(nm_mean[a:b]).mean():11.3f} {np.abs(bs_mean[a:b]).mean():11.3f} "
          f"{nm_std[a:b].mean():13.3f} {bs_std[a:b].mean():13.3f} "
          f"{np.abs(resid_mean[a:b]).mean():12.3f} {resid_std[a:b].mean():10.3f}")

print("\n=== Overall (all 283 dims) ===")
print(f"  mean L2 distance |nm_mean - bs_mean|     = {np.linalg.norm(nm_mean-bs_mean):.3f}")
print(f"  mean cosine(nm_mean, bs_mean)            = {np.dot(nm_mean,bs_mean)/(np.linalg.norm(nm_mean)*np.linalg.norm(bs_mean)+1e-9):.4f}")
print(f"  |residual mean| (BONES under nym z): mean={np.abs(resid_mean).mean():.3f}  median={np.median(np.abs(resid_mean)):.3f}  max={np.abs(resid_mean).max():.2f}")
print(f"  residual std    (BONES under nym z): mean={resid_std.mean():.3f}  median={np.median(resid_std):.3f}  min={resid_std.min():.2f} max={resid_std.max():.2f}")
bad_m = int((np.abs(resid_mean) > 0.5).sum()); bad_s = int(((resid_std<0.5)|(resid_std>2.0)).sum())
print(f"  dims with |resid mean|>0.5: {bad_m}/283   dims with resid std outside [0.5,2]: {bad_s}/283")
print("\n  worst-5 dims by |resid mean|:", np.argsort(-np.abs(resid_mean))[:5].tolist())
