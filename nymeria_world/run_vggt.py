"""Predict camera trajectories with VGGT-Omega on the GT clips, save per-sample npz.

For each sample: decode gt_clip.mp4 frames -> VGGT-Omega -> per-frame camera-from-world
extrinsics (OpenCV) -> camera centers C=-R^T T and forward axes f=R^T[0,0,1].
Saves <out>/<name>/vggt_cameras.npz {extrinsics(N,3,4), intrinsics, cam_pos(N,3), cam_fwd(N,3)}.
Run in the `cosmos` env (torch 2.10, vggt_omega installed).
"""
from __future__ import annotations
import argparse, glob, os, tempfile
import numpy as np
import torch
import imageio.v3 as iio

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_baseline_batch")
    ap.add_argument("--out", default="/weka/jungbin/cosmos_motion_ft_runs/nymeria_vggt")
    ap.add_argument("--ckpt", default="/weka/jungbin/vggt_omega_ckpt/vggt_omega_1b_512.pt")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    print("model loaded")

    names = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(args.src_root, "samples", "*", "gt_clip.mp4")))
    for name in names:
        clip = os.path.join(args.src_root, "samples", name, "gt_clip.mp4")
        frames = iio.imread(clip, plugin="pyav")[:: args.stride]  # (T,H,W,3) uint8
        odir = os.path.join(args.out, name); os.makedirs(odir, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, fr in enumerate(frames):
                p = os.path.join(td, f"{i:04d}.png"); iio.imwrite(p, fr); paths.append(p)
            images = load_and_preprocess_images(paths, image_resolution=args.res).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(images)
            extr, intr = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
        extr = extr[0].float().cpu().numpy()           # (N,3,4) camera-from-world
        intr = intr[0].float().cpu().numpy()
        R, T = extr[:, :3, :3], extr[:, :3, 3]
        C = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), T)   # camera center in world
        fwd = R[:, 2, :]                                         # optical axis in world (R^T[:,2])
        np.savez(os.path.join(odir, "vggt_cameras.npz"),
                 extrinsics=extr, intrinsics=intr, cam_pos=C.astype(np.float32), cam_fwd=fwd.astype(np.float32))
        step = np.linalg.norm(np.diff(C, axis=0), axis=1).mean()
        print(f"  {name}: {len(frames)} frames  cam-center |Δ| mean={step:.4f} (VGGT up-to-scale) -> {odir}/vggt_cameras.npz")


if __name__ == "__main__":
    main()
