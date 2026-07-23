#!/usr/bin/env python
"""Convert a NymeriaPlus sequence's SMPL motion -> SOMA pose params.

Mirrors SOMA-X tools/smpl2soma.py, but loads the NymeriaPlus
body/xdata_smpl_neutral.npz (per-frame axis-angle SMPL params) instead of the
demo animation. SOMA identity is fit to the person via the native SMPL identity
model: with identity_model_type="smpl", inv.prepare_identity(betas) maps the
subject's SMPL shape onto the SOMA rest shape.

Run in the `soma` conda env on a GPU node:

    conda activate soma
    SEQ=/weka/jungbin/nymeriaplus/S11/20230710_s0_barbara_norman_act4_ebaqa8
    python smpl2soma_nymeria.py -i "$SEQ" \
        --smpl-model-path /home/jungbin_cho/SMPL_NEUTRAL.pkl \
        --out-npz out/barbara_soma.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

SOMA_ROOT = Path("/home/jungbin_cho/SOMA-X")
if str(SOMA_ROOT) not in sys.path:
    sys.path.insert(0, str(SOMA_ROOT))

import smplx  # noqa: E402
from soma.geometry.rig_utils import remove_joint_orient_local  # noqa: E402
from soma.geometry.transforms import matrix_to_rotvec  # noqa: E402
from soma.io import save_soma_npz  # noqa: E402
from soma.pose_inversion import PoseInversion  # noqa: E402
from soma.soma import SOMALayer  # noqa: E402
from soma.units import Unit  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="NymeriaPlus SMPL -> SOMA converter.")
    p.add_argument("-i", "--sequence-dir", type=Path, required=True)
    p.add_argument("--smpl-model-path", type=Path,
                   default=Path("/home/jungbin_cho/SMPL_NEUTRAL.pkl"))
    p.add_argument("--out-npz", type=Path, default=Path("out/soma.npz"))
    p.add_argument("--body-iters", type=int, default=2)
    p.add_argument("--finger-iters", type=int, default=0)
    p.add_argument("--full-iters", type=int, default=1)
    p.add_argument("--autograd-iters", type=int, default=0)
    p.add_argument("--autograd-lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--target-fps", type=float, default=None,
                   help="Resample to this fps via timestamps (default: native rate).")
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames (debug).")
    p.add_argument("--bake-verts", type=Path, default=None,
                   help="Save world-frame SOMA vertices+faces+timestamps (for scene viz).")
    p.add_argument("--bake-frames", type=int, default=100, help="Frames to bake (from start).")
    p.add_argument("--render", action="store_true", help="Render SMPL-vs-SOMA comparison mp4.")
    p.add_argument("--render-frames", type=int, default=300, help="Frames to render (from start).")
    p.add_argument("--render-out", type=Path, default=Path("out/smpl_vs_soma.mp4"))
    p.add_argument("--soma-assets", type=Path, default=SOMA_ROOT / "assets")
    p.add_argument("--output-unit", default="meters")
    p.add_argument("--keep-root", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA unavailable; running on CPU (slow).")

    # --- load NymeriaPlus SMPL params ---
    npz = np.load(args.sequence_dir / "body" / "xdata_smpl_neutral.npz")
    betas_all = npz["betas"].astype(np.float32)
    body_pose = npz["body_pose"].astype(np.float32)
    global_orient = npz["global_orient"].astype(np.float32)
    transl = npz["transl"].astype(np.float32)
    ts_us = npz["timestamps"].astype(np.int64)
    N = body_pose.shape[0]

    if args.target_fps is not None:
        n_out = max(1, int((ts_us[-1] - ts_us[0]) / 1e6 * args.target_fps))
        query = np.linspace(ts_us[0], ts_us[-1], n_out).astype(np.int64)
        idx = np.searchsorted(ts_us, query).clip(0, N - 1)
    else:
        idx = np.arange(N)
    if args.max_frames:
        idx = idx[: args.max_frames]
    body_pose = body_pose[idx]
    global_orient = global_orient[idx]
    transl = transl[idx]
    ts_sel = ts_us[idx]
    num_frames = len(idx)

    # constant identity = median betas (shape is physically constant; npz stores
    # 5 piecewise-constant windows -> median removes that artifact)
    betas_med = np.median(betas_all, axis=0).astype(np.float32)[None]  # (1,10)
    print(f"[load] {N} frames -> {num_frames} used; betas(median)[:3]={betas_med[0,:3].round(3)}")

    # --- SMPL model (vertex source) ---
    smpl_model = smplx.create(
        model_type="smpl", model_path=str(args.smpl_model_path),
        use_pca=False, flat_hand_mean=True, batch_size=1,
    ).to(device)

    # --- SOMA + PoseInversion; identity = the subject's SMPL shape ---
    soma = SOMALayer(args.soma_assets, identity_model_type="smpl", device=device, mode="warp")
    inv = PoseInversion(soma, low_lod=True)
    betas_id = torch.from_numpy(betas_med).to(device)
    inv.prepare_identity(betas_id)

    betas_t = betas_id  # (1,10) reused for every forward (constant shape)

    def fit_chunk(s, e):
        with torch.no_grad():
            out = smpl_model(
                betas=betas_t.expand(e - s, -1),
                body_pose=torch.from_numpy(body_pose[s:e]).to(device),
                global_orient=torch.from_numpy(global_orient[s:e]).to(device),
                transl=torch.from_numpy(transl[s:e]).to(device),
            )
        return inv.fit(
            out.vertices, body_iters=args.body_iters, finger_iters=args.finger_iters,
            full_iters=args.full_iters, autograd_iters=args.autograd_iters,
            autograd_lr=args.autograd_lr,
        )

    # warmup (jit/kernel cache)
    _ = fit_chunk(0, 1)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    rot, root_t, errs = [], [], []
    bs = args.batch_size
    for s in range(0, num_frames, bs):
        e = min(s + bs, num_frames)
        r = fit_chunk(s, e)
        rot.append(r["rotations"].cpu())
        root_t.append(r["root_translation"].cpu())
        errs.append(r["per_vertex_error"].cpu())
        print(f"\r[fit] {e}/{num_frames}", end="", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    rotations = torch.cat(rot, 0)
    root_transl = torch.cat(root_t, 0)
    err = torch.cat(errs, 0)
    print(f"\n[fit] {num_frames} frames in {dt:.1f}s ({num_frames/dt:.0f} fps)")
    print(f"[err] per-vertex  mean={err.mean()*100:.3f}cm  "
          f"median={err.median()*100:.3f}cm  max={err.max()*100:.3f}cm")

    # --- optional: bake world-frame SOMA vertices for scene visualization ---
    if args.bake_verts:
        _s = inv.soma
        bs_sk = _s.batched_skinning
        bind = _s._cached_bind_transforms_world
        rest = _s._cached_rest_shape
        bf = min(args.bake_frames, num_frames)
        soma_v = []
        for s in range(0, bf, args.batch_size):
            e = min(s + args.batch_size, bf)
            bs_sk.rebind(bind.expand(e - s, -1, -1, -1), rest.expand(e - s, -1, -1))
            with torch.no_grad():
                sv, _ = bs_sk.pose(
                    rotations[s:e].to(device), root_transl[s:e].to(device),
                    absolute_pose=True, return_transforms=True,
                )
            soma_v.append(sv.detach().cpu().numpy())
        verts = np.concatenate(soma_v, 0).astype(np.float32)  # world frame
        args.bake_verts.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.bake_verts, verts=verts,
            faces=_s.faces.cpu().numpy().astype(np.int32),
            timestamps_us=ts_sel[:bf].astype(np.int64),
        )
        print(f"[bake] {verts.shape} world verts -> {args.bake_verts}")

    # --- optional SMPL-vs-SOMA comparison render ---
    if args.render:
        from tools.vis_pyrender import (
            default_pyopengl_platform, render_comparison_video, set_pyopengl_platform,
        )
        set_pyopengl_platform(default_pyopengl_platform())
        _s = inv.soma
        bs_sk = _s.batched_skinning
        bind = _s._cached_bind_transforms_world
        rest = _s._cached_rest_shape
        rf = min(args.render_frames, num_frames)
        smpl_v, soma_v = [], []
        for s in range(0, rf, args.batch_size):
            e = min(s + args.batch_size, rf)
            with torch.no_grad():
                out = smpl_model(
                    betas=betas_t.expand(e - s, -1),
                    body_pose=torch.from_numpy(body_pose[s:e]).to(device),
                    global_orient=torch.from_numpy(global_orient[s:e]).to(device),
                    transl=torch.from_numpy(transl[s:e]).to(device),
                )
            smpl_v.append(out.vertices.cpu().numpy())
            bs_sk.rebind(bind.expand(e - s, -1, -1, -1), rest.expand(e - s, -1, -1))
            with torch.no_grad():
                sv, _ = bs_sk.pose(
                    rotations[s:e].to(device), root_transl[s:e].to(device),
                    absolute_pose=True, return_transforms=True,
                )
            soma_v.append(sv.detach().cpu().numpy())
        smpl_arr = np.concatenate(smpl_v, 0)
        soma_arr = np.concatenate(soma_v, 0)
        # NymeriaPlus world is Z-up; SOMA-X renderer camera is Y-up -> rotate
        # both meshes -90deg about X so the body stands upright: (x,y,z)->(x,z,-y)
        Rzy = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        smpl_arr = smpl_arr @ Rzy.T
        soma_arr = soma_arr @ Rzy.T
        args.render_out.parent.mkdir(parents=True, exist_ok=True)
        render_comparison_video(
            str(args.render_out),
            smpl_arr, smpl_model.faces,
            soma_arr, _s.faces.cpu().numpy(),
            label_source="SMPL", cam_dist_scale=3.0, center=True,
        )
        print(f"[render] {rf} frames -> {args.render_out}")

    # --- save SOMA npz (absolute -> T-pose-relative rotvec) ---
    _soma = inv.soma
    odev = _soma._t_pose_orient.device
    rel = remove_joint_orient_local(
        rotations.to(odev), _soma._t_pose_orient, _soma._t_pose_orient_parent_T
    )
    poses_rotvec = matrix_to_rotvec(rel.reshape(-1, 3, 3)).reshape(
        rotations.shape[0], rotations.shape[1], 3
    ).cpu()

    save_transl = root_transl.clone()
    tgt = Unit.from_name(args.output_unit)
    scale = _soma.output_unit.meters_per_unit / tgt.meters_per_unit
    if scale != 1.0:
        save_transl = save_transl * scale

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_soma_npz(
        args.out_npz, poses_rotvec, save_transl,
        joint_names=list(_soma.rig_data["joint_names"]),
        identity_model_type=_soma.identity_model_type,
        identity_coeffs=betas_id.cpu(),
        joint_orient=_soma._t_pose_orient,
        unit=args.output_unit, keep_root=args.keep_root,
    )
    # store timestamps alongside (save_soma_npz doesn't carry them)
    np.save(str(args.out_npz.with_suffix(".timestamps_us.npy")), ts_sel)
    print(f"[done] {args.out_npz}  poses={tuple(poses_rotvec.shape)}  transl={tuple(save_transl.shape)}")


if __name__ == "__main__":
    main()
