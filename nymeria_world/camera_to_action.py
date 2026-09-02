"""NymeriaPlus ego-camera trajectory -> Cosmos-3 9D camera pseudo-action.

Matches the ORIGINAL Cosmos-3 camera/action configuration exactly (verified against
`cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py` and `pose_utils.py`):

  - relative-pose pseudo-action:  delta_T = T_{t-1}^{-1} @ T_t   (pose_convention="backward_framewise")
  - rotation block = 6D (Zhou et al. 2019)                       (rotation_format="rot6d")
  - NO translation_scale / rotation_scale (both = 1.0)           -> raw delta, like every camera/pose caller
  - NO mean/std/quantile normalization                          (action_normalization is None for camera)
  - axis remap of the pose ROTATION block into OpenCV camera convention, mirroring DROID's
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _DEVICE_TO_OPENCV
    (translation block left untouched, exactly as DROID does)
  - domain = "camera_pose"  -> domain_id 2,  raw_action_dim 9,  zero-pad to max_action_dim (64)

The model consumes the result AS-IS: `omni_mot_model._normalize_action_databatch` only
densifies the list; it applies no scaling. So this representation *is* the normalization.

Output for T video frames is (T-1, 9): action[t] is the transition v_{t-1} -> v_t, which is
why packing uses `action_start_frame_offset = 1` (action[0] aligns with vision frame 1).

Camera npz schema (from nymeria_kimodo_pipeline/camera/extract_camera_trajectory.py):
    cam_world_pos (T,3) float32   # device translation in Aria SLAM world (Z-up)
    cam_world_rot (T,3,3) float32 # R_world_device
World frame cancels in delta_T = T^{-1}T, so Aria's Z-up world needs no global remap.
"""
from __future__ import annotations

import numpy as np

try:  # Cosmos-3 Nano framework layout
    from cosmos_framework.data.vfm.action.domain_utils import (
        EMBODIMENT_TO_DOMAIN_ID,
        EMBODIMENT_TO_RAW_ACTION_DIM,
    )
    from cosmos_framework.data.vfm.action.pose_utils import pose_abs_to_rel, pose_rel_to_abs
except ModuleNotFoundError:  # Cosmos-3 Edge framework layout
    from cosmos_framework.data.generator.action.utils.domain_utils import (
        EMBODIMENT_TO_DOMAIN_ID,
        EMBODIMENT_TO_RAW_ACTION_DIM,
    )
    from cosmos_framework.data.generator.action.utils.pose_utils import (
        pose_abs_to_rel,
        pose_rel_to_abs,
    )

EMBODIMENT = "camera_pose"
DOMAIN_ID = EMBODIMENT_TO_DOMAIN_ID[EMBODIMENT]          # 2
RAW_ACTION_DIM = EMBODIMENT_TO_RAW_ACTION_DIM[EMBODIMENT]  # 9 (3 pos + 6 rot)

# Aria device frame -> OpenCV camera convention (x-right, y-down, z-forward/optical axis).
# Default identity: since the Cosmos camera action head is re-initialized from scratch for
# this finetune (DROID recipe puts action2llm/llm2action/action_modality_embed in
# keys_to_skip_loading), the model learns whatever *consistent* convention we feed; the
# remap only fixes the *semantic* meaning of the 9D axes. Set this to the true Aria
# device->OpenCV rotation to align semantics with Cosmos's pretrained camera space.
_DEVICE_TO_OPENCV: np.ndarray = np.eye(3, dtype=np.float32)


def build_abs_poses(cam_world_pos: np.ndarray, cam_world_rot: np.ndarray,
                    device_to_opencv: np.ndarray = _DEVICE_TO_OPENCV) -> np.ndarray:
    """(T,3)+(T,3,3) -> (T,4,4) SE(3), with the DROID-style rotation-block axis remap."""
    T = cam_world_pos.shape[0]
    poses = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    poses[:, :3, :3] = cam_world_rot.astype(np.float32)
    poses[:, :3, 3] = cam_world_pos.astype(np.float32)
    # mirror droid_lerobot_dataset.py:403  (rotation block only)
    poses[:, :3, :3] = poses[:, :3, :3] @ device_to_opencv
    return poses


def camera_poses_to_action(cam_world_pos: np.ndarray, cam_world_rot: np.ndarray,
                           device_to_opencv: np.ndarray = _DEVICE_TO_OPENCV) -> np.ndarray:
    """Full trajectory -> (T-1, 9) camera pseudo-action, matching Cosmos exactly."""
    poses_abs = build_abs_poses(cam_world_pos, cam_world_rot, device_to_opencv)
    # identical call to droid_lerobot_dataset.py:406 (scales default to 1.0 -> raw delta)
    poses_rel = pose_abs_to_rel(
        poses_abs, rotation_format="rot6d", pose_convention="backward_framewise"
    )
    assert poses_rel.shape[1] == RAW_ACTION_DIM, poses_rel.shape
    return poses_rel.astype(np.float32)


def pad_action_to_max_dim(action: np.ndarray, max_action_dim: int = 64) -> np.ndarray:
    """Zero-pad (T,9) -> (T,max_action_dim); trailing channels masked in loss via raw_action_dim."""
    T, D = action.shape
    if D == max_action_dim:
        return action
    out = np.zeros((T, max_action_dim), dtype=np.float32)
    out[:, :D] = action
    return out


# --------------------------------------------------------------------------------------
# Self-test: round-trip + stats on real NymeriaPlus camera npz
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, glob, os

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str,
                    default="/weka/jungbin/nymeriaplus_kimodo_proportional/camera/S01/"
                            "20230607_s0_james_johnson_act1_7xwm28.npz")
    ap.add_argument("--n", type=int, default=200, help="frames to test (window-sized)")
    ap.add_argument("--scan", action="store_true", help="scan many seqs for range/NaN audit")
    args = ap.parse_args()

    def load(npz):
        d = np.load(npz)
        return d["cam_world_pos"].astype(np.float32), d["cam_world_rot"].astype(np.float32)

    def roundtrip(npz, n):
        pos, rot = load(npz)
        pos, rot = pos[:n], rot[:n]
        poses_abs = build_abs_poses(pos, rot)
        act = camera_poses_to_action(pos, rot)                # (n-1, 9)
        # invert: recover absolute poses from the rel action + the true initial pose
        rec = pose_rel_to_abs(act, rotation_format="rot6d",
                              pose_convention="backward_framewise",
                              initial_pose=poses_abs[0].copy())
        rot_err = np.abs(rec[:, :3, :3] - poses_abs[:, :3, :3]).max()
        trn_err = np.abs(rec[:, :3, 3] - poses_abs[:, :3, 3]).max()
        return act, rot_err, trn_err

    if args.scan:
        files = sorted(glob.glob("/weka/jungbin/nymeriaplus_kimodo_proportional/camera/S*/*.npz"))
        print(f"scanning {len(files)} seqs (first 200 frames each)...")
        gmin = np.full(9, np.inf); gmax = np.full(9, -np.inf)
        worst_rt = 0.0; nbad = 0
        for f in files:
            try:
                act, re_, te_ = roundtrip(f, 200)
            except Exception as e:
                nbad += 1; continue
            if not np.isfinite(act).all():
                nbad += 1
            gmin = np.minimum(gmin, act.min(0)); gmax = np.maximum(gmax, act.max(0))
            worst_rt = max(worst_rt, re_, te_)
        print(f"non-finite/err seqs: {nbad}")
        print(f"worst round-trip abs err over all seqs: {worst_rt:.2e}")
        np.set_printoptions(precision=4, suppress=True)
        print("per-channel min (3 trans + 6 rot6d):", gmin)
        print("per-channel max (3 trans + 6 rot6d):", gmax)
    else:
        print(f"npz: {os.path.basename(args.npz)}  (first {args.n} frames)")
        act, re_, te_ = roundtrip(args.npz, args.n)
        np.set_printoptions(precision=4, suppress=True)
        print("action shape:", act.shape, "(expect (n-1, 9))")
        print("rotation round-trip max abs err:", f"{re_:.2e}")
        print("translation round-trip max abs err:", f"{te_:.2e}")
        print("per-frame translation-delta norm  mean/max (m):",
              f"{np.linalg.norm(act[:, :3], axis=1).mean():.4f} / "
              f"{np.linalg.norm(act[:, :3], axis=1).max():.4f}")
        print("rot6d block min/max:", act[:, 3:].min(), act[:, 3:].max())
        print("padded ->", pad_action_to_max_dim(act).shape)
