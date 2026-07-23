"""Final verification over a random sample of the corrected proportional NPZs.
Checks: schema keys, finite, neutral Y-up + plausible height, FK feet grounded
(min foot Y near 0), FK body upright when the actor is upright (max head-foot Y).
"""
import numpy as np, torch, glob, random
from kimodo.skeleton import SOMASkeleton77

s = SOMASkeleton77(); par = s.joint_parents.tolist(); names = list(s.bone_order_names)
H, LF, RF = names.index("Head"), names.index("LeftFoot"), names.index("RightFoot")
files = sorted(glob.glob("/weka/jungbin/nymeriaplus_kimodo_proportional/**/*.npz", recursive=True))
print(f"total files on disk: {len(files)}")
random.seed(0)
sample = random.sample(files, min(40, len(files)))

bad = []
foot_mins, heights, maxhf = [], [], []
for f in sample:
    d = np.load(f, allow_pickle=True)
    need = {"local_rot_mats", "root_positions", "neutral_joints", "identity_coeffs", "timestamps_us", "fps"}
    if not need.issubset(set(d.files)): bad.append((f, "missing keys")); continue
    lrm = torch.from_numpy(d["local_rot_mats"].astype(np.float32))
    root = torch.from_numpy(d["root_positions"].astype(np.float32))
    nj = torch.from_numpy(d["neutral_joints"].astype(np.float32))
    if not (np.isfinite(d["local_rot_mats"]).all() and np.isfinite(d["root_positions"]).all()
            and np.isfinite(d["neutral_joints"]).all()):
        bad.append((f, "non-finite")); continue
    h = float(nj[:, 1].max() - nj[:, 1].min())
    if not (1.4 < h < 2.1): bad.append((f, f"neutral height {h:.2f}")); continue
    B = lrm.shape[0]; sel = torch.linspace(0, B - 1, min(150, B)).long()
    _, posed, _ = s.fk(lrm[sel], root[sel], neutral_joints=nj.unsqueeze(0).expand(len(sel), -1, -1))
    p = posed.numpy()
    fmin = float(p[:, [LF, RF], 1].min())
    hf_y_max = float((p[:, H, 1] - 0.5 * (p[:, LF, 1] + p[:, RF, 1])).max())
    foot_mins.append(fmin); heights.append(h); maxhf.append(hf_y_max)
    if abs(fmin) > 0.35: bad.append((f, f"feet not grounded min={fmin:.2f}")); continue

print(f"sampled {len(sample)}  | clean {len(sample)-len(bad)}  | flagged {len(bad)}")
print(f"neutral height: mean={np.mean(heights):.2f} range[{np.min(heights):.2f},{np.max(heights):.2f}]")
print(f"foot Y min (grounding): mean={np.mean(foot_mins):.3f} range[{np.min(foot_mins):.2f},{np.max(foot_mins):.2f}]")
print(f"max head-foot Y (upright moments): mean={np.mean(maxhf):.2f} min={np.min(maxhf):.2f}")
for f, why in bad:
    print("  FLAG:", "/".join(f.split("/")[-2:]), "->", why)
print("OK" if not bad else "REVIEW FLAGS ABOVE")
