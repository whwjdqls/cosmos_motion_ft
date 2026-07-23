"""Camera-motion preprocessing: T_world_device -> T_world_upright_rgb -> relative camera actions.

Implements the documented pipeline (see CALIBRATION_CONSISTENCY.md / CAMERA_RGB_PREPROCESS.md):

  1. Load raw Nymeria SLAM device trajectory  T_world_device[t]   (already sampled 1:1 at the
     20fps video frame times by extract_camera_trajectory.py -> camera/{Sxx}/{seq}.npz).
  2. Read the per-sequence static RGB extrinsic  T_device_rgb  from VRS calibration
     (get_transform_device_sensor("camera-rgb")). Verified static within a sequence and ~globally
     constant (CALIBRATION_CONSISTENCY.md) -> no per-frame online calibration needed.
  3. T_world_rgb[t]          = T_world_device[t] · T_device_rgb
  4. T_world_upright_rgb[t]  = T_world_rgb[t] · T_rgb_upright        (T_rgb_upright = Rz(-90deg),
     matching the -90deg upright rotation applied to the training video; validated as the frame
     that best matches the pretrained model's predicted camera direction).
  5/6. Slice = whole sequence here (training windows are sliced downstream from these per-frame poses).
  7. Relative camera actions  A_t = inv(T_world_upright_rgb[t]) · T_world_upright_rgb[t+k]
     encoded Cosmos-exact: [translation(3), rot6d(6)] via pose_abs_to_rel (backward_framewise, scales=1).
  8. Intrinsics K_rgb_raw and K_rgb_upright (after the -90deg image rotation) stored for record.

The model should consume  video_rgb_upright + cam_action_upright  (NOT the device-frame action).

Output (parallel to raw camera dir):
  /weka/jungbin/nymeriaplus_kimodo_proportional/camera_rgb/{Sxx}/{seq}.npz
Newly generated files also carry per-step translation/rotation continuity diagnostics and a
conservative implausible-step mask. These expose upstream jumps; they do not repair them.
Run in the `nymeria_plus` env.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")
sys.path.insert(0, "/home/jungbin_cho/cosmos-framework")
import argparse, glob, json, os
import numpy as np
from projectaria_tools.core import data_provider
from cosmos_framework.data.vfm.action.pose_utils import pose_abs_to_rel

MROOT = "/weka/jungbin/nymeriaplus_kimodo_proportional"
NROOT = "/weka/jungbin/nymeriaplus"
CAM_RAW = os.path.join(MROOT, "camera")          # T_world_device per-frame (input)
CAM_OUT = os.path.join(MROOT, "camera_rgb")      # processed (output)

# -90 deg about the optical (z) axis = the upright video rotation, in the CAMERA frame (right-multiply).
RZ_NEG90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
T_RGB_UPRIGHT = np.eye(4); T_RGB_UPRIGHT[:3, :3] = RZ_NEG90
MAX_PLAUSIBLE_STEP_TRANSLATION_M = 0.25
MAX_PLAUSIBLE_STEP_ROTATION_DEG = 30.0
PREPROCESS_VERSION = 2


def find_vrs(rec_dir: str):
    for rel in ("data/data.vrs", "data/motion.vrs"):
        p = os.path.join(rec_dir, rel)
        if os.path.isfile(p):
            return p
    return None


def output_is_current(subj: str, seq: str) -> bool:
    """A sidecar is current only when it postdates and timestamp-matches its raw input."""
    raw_path = os.path.join(CAM_RAW, subj, f"{seq}.npz")
    out_path = os.path.join(CAM_OUT, subj, f"{seq}.npz")
    if not (os.path.isfile(raw_path) and os.path.isfile(out_path)):
        return False
    try:
        if os.path.getmtime(out_path) < os.path.getmtime(raw_path):
            return False
        with np.load(raw_path) as raw, np.load(out_path) as out:
            version = int(out["preprocess_version"]) if "preprocess_version" in out else 0
            return version >= PREPROCESS_VERSION and np.array_equal(
                raw["timestamps_us"], out["timestamps_us"]
            )
    except Exception:
        return False


def rgb_calib(dc):
    cc = dc.get_camera_calib("camera-rgb")
    f = np.asarray(cc.get_focal_lengths(), float)          # (fx, fy)
    pp = np.asarray(cc.get_principal_point(), float)        # (cx, cy)
    wh = np.asarray(cc.get_image_size(), int)               # (W, H)
    K_raw = np.array([[f[0], 0, pp[0]], [0, f[1], pp[1]], [0, 0, 1.0]])
    # Pillow ``rotate(-90, expand=True)`` is clockwise:
    # (u_new, v_new) = (H - 1 - v_old, u_old), output (W_new,H_new)=(H,W).
    W, H = int(wh[0]), int(wh[1])
    K_up = np.array(
        [[f[1], 0, H - 1 - pp[1]], [0, f[0], pp[0]], [0, 0, 1.0]]
    )
    image_size_hw = np.asarray([W, H], dtype=int)  # new H=W_old, new W=H_old
    return K_raw, K_up, image_size_hw, str(cc.get_model_name())


def process_one(subj: str, seq: str) -> dict:
    raw_npz = os.path.join(CAM_RAW, subj, f"{seq}.npz")
    if not os.path.isfile(raw_npz):
        return {"seq": seq, "status": "no_raw_camera"}
    d = np.load(raw_npz)
    pos_dev = d["cam_world_pos"].astype(np.float64)         # (T,3)  T_world_device translation
    rot_dev = d["cam_world_rot"].astype(np.float64)         # (T,3,3) R_world_device
    timestamps_us = d["timestamps_us"].astype(np.int64)
    T = len(pos_dev)
    if T < 2 or len(timestamps_us) != T or rot_dev.shape != (T, 3, 3) or not (
        np.isfinite(pos_dev).all() and np.isfinite(rot_dev).all()
    ) or np.any(np.diff(timestamps_us) <= 0):
        return {"seq": seq, "status": "invalid_raw_camera", "frames": T}
    vrs = find_vrs(os.path.join(NROOT, subj, seq, "recording_head"))
    if vrs is None:
        return {"seq": seq, "status": "no_vrs"}
    dp = data_provider.create_vrs_data_provider(vrs)
    dc = dp.get_device_calibration()
    T_device_rgb = dc.get_transform_device_sensor("camera-rgb").to_matrix().astype(np.float64)
    K_raw, K_up, img_hw, model = rgb_calib(dc)

    # T_world_device (T,4,4)
    Twd = np.tile(np.eye(4), (T, 1, 1)); Twd[:, :3, :3] = rot_dev; Twd[:, :3, 3] = pos_dev
    # T_world_rgb = T_world_device · T_device_rgb ;  T_world_upright = · T_rgb_upright
    Twr = Twd @ T_device_rgb
    Twu = Twr @ T_RGB_UPRIGHT
    upos = Twu[:, :3, 3].astype(np.float32)
    urot = Twu[:, :3, :3].astype(np.float32)

    # Relative camera actions (Cosmos-exact: rot6d, backward_framewise, scales=1), k=1 per frame.
    act_k1 = pose_abs_to_rel(Twu.astype(np.float64), rotation_format="rot6d",
                             pose_convention="backward_framewise").astype(np.float32)  # (T-1,9)
    step_translation = np.linalg.norm(act_k1[:, :3], axis=1)
    step_rotation_matrix = np.transpose(Twu[:-1, :3, :3], (0, 2, 1)) @ Twu[1:, :3, :3]
    step_rotation_deg = np.degrees(np.arccos(np.clip(
        (np.trace(step_rotation_matrix, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)))
    implausible_step = (
        (step_translation >= MAX_PLAUSIBLE_STEP_TRANSLATION_M)
        | (step_rotation_deg >= MAX_PLAUSIBLE_STEP_ROTATION_DEG)
    )

    out_dir = os.path.join(CAM_OUT, subj); os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq}.npz")
    np.savez(
        out_path,
        # --- final outputs the model should use ---
        cam_world_pos_upright=upos,           # (T,3)  T_world_upright_rgb translation
        cam_world_rot_upright=urot,           # (T,3,3) R_world_upright_rgb (OpenCV optical, upright)
        cam_action_upright_k1=act_k1,         # (T-1,9) [tx,ty,tz, rot6d] per-frame relative action
        # --- source-quality diagnostics (do not train through flagged transitions) ---
        camera_step_translation_m=step_translation.astype(np.float32),
        camera_step_rotation_deg=step_rotation_deg.astype(np.float32),
        camera_step_implausible=implausible_step,
        preprocess_version=np.int64(PREPROCESS_VERSION),
        # --- static transforms (reproducibility) ---
        T_device_rgb=T_device_rgb.astype(np.float32),
        T_rgb_upright=T_RGB_UPRIGHT.astype(np.float32),
        # --- intrinsics ---
        K_rgb_raw=K_raw.astype(np.float32),
        K_rgb_upright=K_up.astype(np.float32),
        image_size_hw=np.asarray(img_hw, np.int32),
        camera_model=np.array(model),
        # --- raw device trajectory (reference) ---
        cam_world_pos_device=pos_dev.astype(np.float32),
        cam_world_rot_device=rot_dev.astype(np.float32),
        # --- sync ---
        timestamps_us=timestamps_us, fps=d["fps"] if "fps" in d else np.int64(20),
    )
    return {"seq": seq, "status": "ok", "frames": T,
            "k1_trans_mean_m": round(float(step_translation.mean()), 5),
            "k1_trans_max_m": round(float(step_translation.max()), 5),
            "k1_rot_max_deg": round(float(step_rotation_deg.max()), 3),
            "n_implausible_steps": int(implausible_step.sum()), "out": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*", default=None, help="subj:seq (default: all with raw camera npz)")
    ap.add_argument("--subjects", type=str, default="", help="comma list to restrict, e.g. S01,S02")
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip timestamp-matching camera_rgb sidecars at the current preprocess version",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.seqs:
        pairs = [tuple(s.split(":", 1)) for s in args.seqs]
    else:
        subj_filter = set(s for s in args.subjects.split(",") if s)
        pairs = []
        for p in sorted(glob.glob(os.path.join(CAM_RAW, "S*", "*.npz"))):
            subj = os.path.basename(os.path.dirname(p)); seq = os.path.basename(p)[:-4]
            if subj_filter and subj not in subj_filter:
                continue
            if args.skip_existing and output_is_current(subj, seq):
                continue
            pairs.append((subj, seq))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"processing {len(pairs)} sequences -> {CAM_OUT}")
    n_ok = n_bad = n_quality_bad = 0; tall = []
    for i, (subj, seq) in enumerate(pairs):
        try:
            r = process_one(subj, seq)
        except Exception as e:
            r = {"seq": seq, "status": f"error:{type(e).__name__}:{str(e)[:60]}"}
        if r["status"] == "ok":
            n_ok += 1; tall.append(r["k1_trans_mean_m"])
            if r["n_implausible_steps"]:
                n_quality_bad += 1
                print(f"  [QUALITY] {subj}/{seq}: {r['n_implausible_steps']} implausible "
                      f"step(s), max={r['k1_trans_max_m']:.3f}m/"
                      f"{r['k1_rot_max_deg']:.1f}deg")
        else:
            n_bad += 1; print(f"  [{r['status']}] {subj}/{seq}")
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(pairs)}  ok={n_ok} bad={n_bad}")
    print(f"\ndone: ok={n_ok} bad={n_bad} quality_flagged_sequences={n_quality_bad}")
    if tall:
        t = np.array(tall)
        print(f"per-frame (k=1) action |Δt| over sequences: mean {t.mean():.4f}m  median {np.median(t):.4f}  "
              f"min {t.min():.4f}  max {t.max():.4f}")
        json.dump({"n_ok": n_ok, "n_bad": n_bad,
                   "n_quality_flagged_sequences": n_quality_bad,
                   "quality_thresholds": {
                       "max_step_translation_m": MAX_PLAUSIBLE_STEP_TRANSLATION_M,
                       "max_step_rotation_deg": MAX_PLAUSIBLE_STEP_ROTATION_DEG,
                   },
                   "k1_trans_mean_per_seq": tall},
                  open(os.path.join(CAM_OUT, "_summary.json"), "w"))


if __name__ == "__main__":
    main()
