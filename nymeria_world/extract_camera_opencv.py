"""Re-express the baseline camera windows in the RGB optical (OpenCV) frame.

T_world_rgb[i] = T_world_device[i] @ T_device_rgb   (T_device_rgb = fixed VRS extrinsic)

Aria's camera-rgb frame is x-right, y-down, z-forward(optical) == OpenCV, which is what
Cosmos-3 expects for camera ego-pose. World frame cancels in the relative pose delta, so
only the body-frame change (device -> rgb optical, a 39deg tilt) matters.

Reuses the per-frame device poses already saved in gt_camera.npz, so we only need the VRS
once (for the constant extrinsic). Run in the `nymeria_plus` env.
Writes <sample>/gt_camera_opencv.npz with cam_world_pos / cam_world_rot in the rgb frame.
"""
from __future__ import annotations
import sys; sys.path.insert(0, "/home/jungbin_cho/nymeria_dataset")
import json, os
from pathlib import Path

import numpy as np
from nymeriaplus.loaders.recording import RecordingLoader

import argparse
NROOT = Path("/weka/jungbin/nymeriaplus")


def T_device_rgb(subj: str, seq: str) -> np.ndarray:
    rec = RecordingLoader(NROOT / subj / seq / "recording_head")
    dc = rec.vrs.get_device_calibration()
    return dc.get_transform_device_sensor("camera-rgb").to_matrix().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline")
    ROOT = Path(ap.parse_args().root) / "samples"
    for name in sorted(os.listdir(ROOT)):
        sdir = ROOT / name
        meta = json.load(open(sdir / "meta.json"))
        subj, seq = meta["uuid"].split("/", 1)
        Tdr = T_device_rgb(subj, seq)

        d = np.load(sdir / "gt_camera.npz")
        pos, rot = d["cam_world_pos"].astype(np.float64), d["cam_world_rot"].astype(np.float64)
        T = len(pos)
        Twd = np.tile(np.eye(4), (T, 1, 1)); Twd[:, :3, :3] = rot; Twd[:, :3, 3] = pos
        Twr = np.einsum("tij,jk->tik", Twd, Tdr)   # T_world_rgb
        np.savez(sdir / "gt_camera_opencv.npz",
                 cam_world_pos=Twr[:, :3, 3].astype(np.float32),
                 cam_world_rot=Twr[:, :3, :3].astype(np.float32),
                 T_device_rgb=Tdr.astype(np.float32))
        tilt = np.degrees(np.arccos(np.clip(Tdr[2, 2], -1, 1)))
        print(f"  {name}: subj={subj} optical-tilt={tilt:.1f}deg -> gt_camera_opencv.npz")


if __name__ == "__main__":
    main()
