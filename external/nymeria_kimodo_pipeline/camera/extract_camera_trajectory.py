"""Extract the egocentric (head Aria) camera trajectory, sampled at the body's
20-fps frame times, as a sidecar NPZ aligned 1:1 with the kimodo motion NPZ.

The head MPS closed-loop trajectory lives in the **Aria SLAM world** frame, which is
**Z-up** (gravity = (0,0,-9.81)) and normally matches the world used by the body
SOMA/SMPL fit and `objects/boxy` floors (timesync < 1 ms). A 2026-07-21 full audit
also found sparse upstream MPS/body discontinuities, so shared-world identity must
be continuity-gated rather than assumed for every frame. The camera can be put into
kimodo coords with the same Z-up->Y-up rotation
(`kimodo (x,y,z) = (world_x, world_z, -world_y)`) and grounded by the same per-slice
`ground_offset_y`.

We sample `T_world_device` at each body frame's TIME_CODE timestamp (the loader
converts TIME_CODE -> device time internally), so camera frame i == motion frame i ==
egocentric image `frame_{i:06d}.webp`.

Output: `/weka/jungbin/nymeriaplus_kimodo_proportional/camera/{Sxx}/{seq}.npz`
  cam_world_pos (T,3) f32   # device origin in Aria world (Z-up)
  cam_world_rot (T,3,3) f32 # R_world_device (columns = device axes in world)
  tdiff_ns (T,) i64         # |sampled pose time - requested time|; ~0 means good sync
  timestamps_us (T,) i64    # copied from the motion NPZ (TIME_CODE us)
  camera_step_translation_m (T-1,) f32
  camera_step_rotation_deg (T-1,) f32
  camera_step_implausible (T-1,) bool

Env: `nymeria_plus` (needs projectaria_tools + the local `nymeriaplus` package).
"""
from __future__ import annotations
import sys; sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")
import argparse, json
from pathlib import Path
import numpy as np
from nymeriaplus.loaders.recording import RecordingLoader

MROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")
NROOT = Path("/weka/jungbin/nymeriaplus")
MAX_PLAUSIBLE_STEP_TRANSLATION_M = 0.25
MAX_PLAUSIBLE_STEP_ROTATION_DEG = 30.0

# default demo sequences (those with cached egocentric images + diverse motion)
DEFAULT_SEQS = [
    ("S02", "20231006_s1_kirk_flowers_act0_hfjvo9"),
]


def extract_one(
    subj: str,
    seq: str,
    *,
    motion_root: Path = MROOT,
    source_root: Path = NROOT,
    output_root: Path | None = None,
) -> dict:
    motion_root = Path(motion_root)
    source_root = Path(source_root)
    output_root = motion_root / "camera" if output_root is None else Path(output_root)
    motion_npz = motion_root / subj / f"{seq}.npz"
    if not motion_npz.is_file():
        return {"seq": seq, "status": "no_motion"}
    ts_us = np.load(motion_npz, allow_pickle=True)["timestamps_us"].astype(np.int64)
    if len(ts_us) < 2:
        return {"seq": seq, "status": "short_motion", "frames": int(len(ts_us))}
    rec = RecordingLoader(source_root / subj / seq / "recording_head")
    if not rec.has_pose:
        return {"seq": seq, "status": "no_pose"}
    poses, tdiffs = rec.sample_trajectory_at_timecodes(ts_us * 1000)   # (T,4,4), (T,)
    cam_world_pos = poses[:, :3, 3].astype(np.float32)
    cam_world_rot = poses[:, :3, :3].astype(np.float32)
    step_translation = np.linalg.norm(np.diff(cam_world_pos.astype(np.float64), axis=0), axis=1)
    step_rotation = (
        np.transpose(cam_world_rot[:-1].astype(np.float64), (0, 2, 1))
        @ cam_world_rot[1:].astype(np.float64)
    )
    step_rotation_deg = np.degrees(np.arccos(np.clip(
        (np.trace(step_rotation, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)))
    implausible_step = (
        (step_translation >= MAX_PLAUSIBLE_STEP_TRANSLATION_M)
        | (step_rotation_deg >= MAX_PLAUSIBLE_STEP_ROTATION_DEG)
    )
    out_dir = output_root / subj
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq}.npz"
    np.savez(out_path,
             cam_world_pos=cam_world_pos, cam_world_rot=cam_world_rot,
             tdiff_ns=tdiffs.astype(np.int64), timestamps_us=ts_us,
             camera_step_translation_m=step_translation.astype(np.float32),
             camera_step_rotation_deg=step_rotation_deg.astype(np.float32),
             camera_step_implausible=implausible_step)
    return {"seq": seq, "status": "ok", "frames": int(len(ts_us)),
            "median_tdiff_ms": round(float(np.median(np.abs(tdiffs))) / 1e6, 4),
            "n_implausible_steps": int(implausible_step.sum()),
            "max_step_translation_m": round(float(step_translation.max()), 4),
            "max_step_rotation_deg": round(float(step_rotation_deg.max()), 3),
            "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*", default=None,
                    help="subj:seq pairs, e.g. S02:20231006_s1_kirk_flowers_act0_hfjvo9")
    args = ap.parse_args()
    seqs = DEFAULT_SEQS if not args.seqs else [tuple(s.split(":", 1)) for s in args.seqs]
    for subj, seq in seqs:
        r = extract_one(subj, seq)
        print(json.dumps(r))


if __name__ == "__main__":
    main()
