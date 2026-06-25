"""Test whether the OpenCV (rgb optical) camera frame matches Cosmos better than the
raw device frame, using the existing zero-shot inverse_dynamics predictions.

(A) Confirms |Δt| and rotation-angle are IDENTICAL across frames (consistency check).
(B) Shows device-vs-camera per-step direction differs by ~the 39deg optical tilt.
(C) Discriminator: mean cosine between the MODEL's predicted Δt direction and GT Δt
    direction, with NO alignment, for candidate frames {device, rgb, rgb+Rz(90/180/270)}.
    The pretrained head predicts in Cosmos's camera convention, so the frame whose GT
    directions best agree with the prediction is the convention Cosmos expects.
"""
import glob, json, os
import numpy as np

ROOT = "/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline"


def abs_from_npz(p):
    d = np.load(p); pos, rot = d["cam_world_pos"].astype(np.float64), d["cam_world_rot"].astype(np.float64)
    T = len(pos); P = np.tile(np.eye(4), (T, 1, 1)); P[:, :3, :3] = rot; P[:, :3, 3] = pos
    return P


def rel_dt(P):  # (T-1,3) translation deltas in the previous body frame
    return np.stack([(np.linalg.inv(P[i]) @ P[i + 1])[:3, 3] for i in range(len(P) - 1)])


def rel_ang(P):
    out = []
    for i in range(len(P) - 1):
        R = (np.linalg.inv(P[i]) @ P[i + 1])[:3, :3]
        out.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return np.array(out)


def rotz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def apply_zrot(P, deg):  # rotate camera frame about its optical axis
    Q = P.copy(); Q[:, :3, :3] = np.einsum("tij,jk->tik", P[:, :3, :3], rotz(deg)); return Q


def cosine_dir(a, b):  # mean cosine of row-wise directions
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return float((an * bn).sum(1).mean())


names = sorted(os.path.basename(os.path.dirname(p))
               for p in glob.glob(os.path.join(ROOT, "invdyn_out", "*", "sample_outputs.json")))
print(f"{len(names)} samples\n")
agg = {k: [] for k in ["device", "rgb", "rgb+Rz90", "rgb+Rz180", "rgb+Rz270"]}
for name in names:
    pred = np.array(json.load(open(os.path.join(ROOT, "invdyn_out", name, "sample_outputs.json")))
                    ["outputs"][0]["content"]["action"], dtype=np.float64)
    pred_dt = pred[:, :3]
    Pdev = abs_from_npz(os.path.join(ROOT, "samples", name, "gt_camera.npz"))
    Prgb = abs_from_npz(os.path.join(ROOT, "samples", name, "gt_camera_opencv.npz"))
    Tn = min(len(pred_dt), len(Pdev) - 1)

    dev_dt, rgb_dt = rel_dt(Pdev)[:Tn], rel_dt(Prgb)[:Tn]
    # (A) magnitude/angle invariance
    inv_t = np.allclose(np.linalg.norm(dev_dt, axis=1), np.linalg.norm(rgb_dt, axis=1), atol=1e-6)
    inv_a = np.allclose(rel_ang(Pdev)[:Tn], rel_ang(Prgb)[:Tn], atol=1e-4)
    # (B) device-vs-camera direction tilt
    tilt = np.degrees(np.arccos(np.clip(cosine_dir(dev_dt, rgb_dt), -1, 1)))
    # (C) cosine vs model prediction for each candidate frame
    cands = {"device": dev_dt, "rgb": rgb_dt}
    for z in (90, 180, 270):
        cands[f"rgb+Rz{z}"] = rel_dt(apply_zrot(Prgb, z))[:Tn]
    cos = {k: cosine_dir(pred_dt[:Tn], v) for k, v in cands.items()}
    for k in agg:
        agg[k].append(cos[k])
    best = max(cos, key=cos.get)
    print(f"{name}")
    print(f"  (A) |Δt|&angle invariant device==rgb: {inv_t and inv_a}")
    print(f"  (B) device->rgb per-step direction tilt: {tilt:.1f} deg")
    print(f"  (C) cosine(pred, GT) by frame: " + "  ".join(f"{k}={cos[k]:+.3f}" for k in cands)
          + f"   -> best: {best}")
print("\n=== mean cosine(pred, GT) across samples ===")
for k in agg:
    print(f"  {k:10s}: {np.mean(agg[k]):+.3f}")
print("\nReading: (A/B) frame change is a consistent rotation -> magnitudes unchanged, only "
      "direction rotates. (C) higher cosine = closer to Cosmos's expected camera convention.")
