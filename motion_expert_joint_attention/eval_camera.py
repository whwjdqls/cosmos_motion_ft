#!/usr/bin/env python
"""Evaluation + output writer for the CAMERA tasks of the joint-attention model.

Mirrors what ``nymeria_world`` does for the native camera world-model, but drives the
JointMotionModel through *its own* ``sample.py`` camera samplers (NOT the native Cosmos
inference). This is the eval seam for "Phase 1" checkpoints (gen-LoRA finetune on the
camera tasks with ``--freeze_motion``): the motion expert is frozen / never exercised, so
only the frozen-reasoner + (LoRA) generator pathway runs.

For each of N test-split NymeriaPlus windows this script samples the three camera tasks
through ``sample.sample_task`` and writes, PER WINDOW, an output layout that the
nymeria_world metric + viz tooling reads VERBATIM:

    <ckpt_dir>/camera_eval/
        samples/<seq>/gt_camera_cosmos.npz   {cam_world_pos, cam_world_rot}  (GT abs poses)
        samples/<seq>/gt_clip.mp4            GT pixel window (decoded, for the FD side-by-side)
        invdyn_out/<seq>/sample_outputs.json {"outputs":[{"content":{"action": [T-1,9]}}]}
        fd_out/<seq>/vision.mp4              forward_dynamics generated video (VAE-decoded)
        policy_out/<seq>/vision.mp4          policy generated video (VAE-decoded)
        policy_out/<seq>/sample_outputs.json policy predicted camera action [T-1,9]
        window_meta.json                     per-seq window bookkeeping + config

The ``sample_outputs.json`` / ``gt_camera_cosmos.npz`` / ``vision.mp4`` / ``gt_clip.mp4``
schema is EXACTLY the one ``eval_inverse_dynamics.py`` / ``viz_eval_samples.py`` /
``viz_fd.py`` / ``montage_*`` expect, so those tools run against this dir unchanged with
``--samples <ckpt_dir>/camera_eval --eval_root <ckpt_dir>/camera_eval``.

The camera action is the un-normalized 9-d Cosmos pseudo-action (motion stats do NOT
apply); video is generated as Wan-VAE latents and decoded to pixels with the SAME
``Wan2pt2VAEInterface`` the precompute used.

Run (cosmos env, single GPU on a node) via ``run_camera_eval.sh``::

    bash run_camera_eval.sh <ckpt.pt> <n_windows>
    # under the hood:
    bash run.sh eval_camera.py --ckpt <ckpt.pt> --n 8 --steps 50 --cfg 2.5

Then metrics + viz (kimodo OR cosmos env; both are numpy/matplotlib/ffmpeg)::

    python eval_camera.py --ckpt <ckpt.pt> --n 8            # (writes outputs above)
    # metrics (reuses nymeria_world math, imported):
    #   already computed here -> camera_eval/invdyn_metrics.json
    # viz:
    #   python viz_camera.py --eval_dir <ckpt_dir>/camera_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from runtime_paths import COSMOS_FRAMEWORK_ROOT, REPO_ROOT, WAN_VAE_PATH, resolve_legacy_path

_NYMERIA_WORLD = str(REPO_ROOT / "nymeria_world")
for _p in (HERE, _NYMERIA_WORLD, str(COSMOS_FRAMEWORK_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C  # noqa: E402
import sample as S  # noqa: E402  (the joint-model camera samplers + load_joint_model)

# nymeria_world source-of-truth loaders (verbatim; same ones the dataset/precompute reuse).
from nymeria_camera_rgb_dataset import (  # noqa: E402
    _load_rgb_cam,
    _rgb_path,
    rel_action_from_window,
)
from nymeria_camera_dataset import decode_window_pyav  # noqa: E402
# Reuse the metric math EXACTLY (geodesic rot, dir cosine, scale, Umeyama ATE).
import eval_inverse_dynamics as EID  # noqa: E402
from nymeria_joint_dataset import latent_path, _load_latents  # noqa: E402


# ============================================================================
# Test-window index: exactly the (uuid, start) windows the trainer's test split
# draws camera tasks from, but keeping the on-disk keys so we can read the
# precomputed latent + the GT camera + decode the GT pixel clip. Mirrors
# precompute_latents.build_index / NymeriaJointDataset's sub-window slicing.
# ============================================================================
def build_test_index(
    manifest_path: str,
    split_file: str,
    split: str,
    num_frames: int,
    latent_root: str,
    require_latents: bool = True,
    windows_json: str = None,
):
    """Build the eval window index.

    windows_json: optional path to a JSON list of {"uuid", "start"} — when given, the
    index contains EXACTLY those windows (in that order), matched against the manifest
    for paths. This is how we evaluate on the SAME held-out 71-window set nymeria_world
    used (one window per test sequence, full71_windows.json), making the metrics
    directly comparable across models. Without it: manifest order, first-N (which can
    degenerate to many windows of one sequence)."""
    want = None  # {(uuid, start) -> order} when windows_json is given
    if windows_json:
        rows = json.load(open(windows_json))
        want = {(r["uuid"], int(r["start"])): i for i, r in enumerate(rows)}

    keep_uuids = None
    if split not in ("all", None):
        sp = json.load(open(split_file))
        assert split in sp, f"split '{split}' not in {split_file}"
        keep_uuids = set(sp[split])

    index = []
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            uuid = rec.get("uuid")
            cam = resolve_legacy_path(rec.get("camera_path"))
            vis = resolve_legacy_path(rec.get("vision_path"))
            nb = int(rec.get("nb_frames", 0))
            if not uuid or not cam or not vis:
                continue
            if want is not None:
                # explicit window list: take the requested starts for this uuid directly
                # (no usable/caption re-filtering — the list IS the eval set).
                rgb = _rgb_path(cam)
                if not os.path.isfile(rgb):
                    continue
                caps = {int(w["start_frame"]): w.get("caption", "")
                        for w in rec.get("t2w_windows", [])}
                for (u, s), order in want.items():
                    if u == uuid and s + num_frames <= nb:
                        lp = latent_path(uuid, s, latent_root)
                        if (not require_latents) or os.path.isfile(lp):
                            index.append({
                                "uuid": uuid, "vis": vis, "rgb": rgb, "s": int(s),
                                "cap": caps.get(s, ""), "lp": lp, "_order": order,
                            })
                continue
            if keep_uuids is not None and uuid not in keep_uuids:
                continue
            rgb = _rgb_path(cam)
            if not os.path.isfile(rgb):
                continue
            for w in rec.get("t2w_windows", []):
                if not w.get("usable", False) or not w.get("caption"):
                    continue
                ws, we = int(w["start_frame"]), int(w["end_frame"])
                hi = min(we, nb)
                s = ws
                while s + num_frames <= hi:
                    lp = latent_path(uuid, s, latent_root)
                    if (not require_latents) or os.path.isfile(lp):
                        index.append({
                            "uuid": uuid, "vis": vis, "rgb": rgb, "s": int(s),
                            "cap": w["caption"], "lp": lp,
                        })
                    s += num_frames
    if want is not None:
        index.sort(key=lambda it: it["_order"])
        print(f"[eval] windows_json: matched {len(index)}/{len(want)} requested windows")
    return index


def seq_name(t: int, uuid: str, start: int) -> str:
    """A montage-friendly unique per-window name: ``t{i}_{uuid_safe}_{start}`` (the
    nymeria_world montage/viz tools split on '_' and take the first two tokens for a label)."""
    return f"t{t}_{uuid.replace('/', '_')}_{start}"


# ============================================================================
# GT writers (produce the exact files the nymeria_world readers consume).
# ============================================================================
def write_gt_camera(dst_npz: str, pos: np.ndarray, rot: np.ndarray) -> None:
    """Write GT ABSOLUTE camera poses in the ``gt_camera_cosmos.npz`` schema
    (``cam_world_pos`` (T,3) + ``cam_world_rot`` (T,3,3)) that ``eval_inverse_dynamics.gt_abs``
    / ``viz_eval_samples.gt_poses`` read. We use the SAME upright RGB poses the dataset /
    ``rel_action_from_window`` use, so the rel-action integration is convention-exact."""
    os.makedirs(os.path.dirname(dst_npz), exist_ok=True)
    np.savez(dst_npz,
             cam_world_pos=pos.astype(np.float32),
             cam_world_rot=rot.astype(np.float32))


def frames_to_mp4(frames_uint8: np.ndarray, dst_mp4: str, fps: float) -> None:
    """(T,H,W,3) uint8 -> mp4 via imageio-ffmpeg (falls back to per-frame PNGs + ffmpeg)."""
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(dst_mp4, list(frames_uint8), fps=fps, quality=8,
                         macro_block_size=None)
        return
    except Exception as e:  # noqa: BLE001
        print(f"[eval_camera] imageio mp4 write failed ({e}); trying ffmpeg pipe", flush=True)
    import subprocess
    T, H, W, _ = frames_uint8.shape
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", dst_mp4],
        stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames_uint8).tobytes())


def latent_to_frames(vae, latents: np.ndarray, device: str) -> np.ndarray:
    """Wan-VAE latents (C,T_lat,h,w) -> (T,H,W,3) uint8 via ``Wan2pt2VAEInterface.decode``.

    Decode returns [-1,1] pixels [1,3,T,H,W]; we map to uint8 and (T,H,W,3)."""
    z = torch.from_numpy(np.ascontiguousarray(latents)).float().to(device).unsqueeze(0)  # [1,C,T_lat,h,w]
    with torch.no_grad():
        px = vae.decode(z.to(vae.dtype))                     # [1,3,T,H,W] in [-1,1]
    px = px[0].float().clamp(-1, 1)                          # [3,T,H,W]
    px = ((px + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return px.permute(1, 2, 3, 0).cpu().numpy()             # (T,H,W,3)


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None,
                    help="Phase-1 train.py checkpoint (.pt). If omitted / missing, evals the "
                         "BASE model (no gen delta) -- smoke path to prove the pipeline runs.")
    ap.add_argument("--out_dir", default=None,
                    help="output root (default <ckpt_dir>/camera_eval, or ./camera_eval_base)")
    ap.add_argument("--n", type=int, default=8, help="number of test windows to eval")
    ap.add_argument("--tasks", nargs="*",
                    default=["inverse_dynamics", "forward_dynamics", "policy"],
                    help="camera tasks to sample")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--manifest", default=C.NYMERIA_MANIFEST)
    ap.add_argument("--split_file", default=C.NYMERIA_SPLIT_FILE)
    ap.add_argument("--latent_root", default=None,
                    help="precomputed-latent root; default = the per-T root the CHECKPOINT was "
                         "trained at (config.VIDEO_LATENT_ROOT if T==config.VIDEO_NUM_FRAMES "
                         "else <root>_T{T}, exactly like train.py). Explicit value overrides.")
    ap.add_argument("--num_frames", type=int, default=None,
                    help="pixel-window length T; default = the CHECKPOINT's --T "
                         "(ckpt['args']['T']), falling back to config.VIDEO_NUM_FRAMES for "
                         "old/absent ckpts. Explicit value overrides.")
    ap.add_argument("--fps", type=float, default=float(C.FPS))
    ap.add_argument("--resolution", default="256")
    ap.add_argument("--vae_path",
                    default=os.environ.get("WAN_VAE_PATH", str(WAN_VAE_PATH)))
    ap.add_argument("--no_video", action="store_true",
                    help="skip VAE decode of generated/GT video (inverse_dynamics metrics only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--windows_json", default=None,
                    help="JSON list of {uuid,start}: evaluate EXACTLY these windows (e.g. the "
                         "held-out full71 set shared with nymeria_world) instead of first-N.")
    args = ap.parse_args()

    dev = args.device

    # ---- resolve output root -------------------------------------------------------------
    if args.out_dir:
        out_root = args.out_dir
    elif args.ckpt and os.path.isfile(args.ckpt):
        out_root = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "camera_eval")
    else:
        out_root = os.path.join(C.RUNS_ROOT, "camera_eval_base")
    os.makedirs(out_root, exist_ok=True)
    print(f"[eval_camera] out_root={out_root}", flush=True)

    # ---- model: real ckpt overlay, or base model for the smoke path ----------------------
    ck_args = {}
    if args.ckpt and os.path.isfile(args.ckpt):
        model, cosmos, ck = S.load_joint_model(args.ckpt, device=dev)
        step = ck.get("step")
        ck_args = (ck.get("args", {}) or {}) if isinstance(ck, dict) else {}
    else:
        print(f"[eval_camera] no ckpt ({args.ckpt!r}); evaluating BASE model "
              f"(no generator LoRA overlay)", flush=True)
        from cosmos_loader import FrozenCosmos
        from joint_motion_model import JointMotionModel
        from uniego_layout import FEAT_DIM
        cosmos = FrozenCosmos(dtype=torch.bfloat16, device=dev)
        model = JointMotionModel(cosmos, objective="velocity", motion_dim=FEAT_DIM,
                                 gen_lora=False, freeze_motion=True).to(dev)
        model.eval()
        step = None

    # ---- T + latent root: default from the CHECKPOINT's args; explicit CLI overrides ------
    # Mirrors train.py's per-T cache convention so the eval reads the SAME latents the run
    # trained on (a T97 ckpt evals T=97 windows from joint_latents_T97, not the T=33 cache).
    if args.num_frames is not None:
        T = int(args.num_frames)
    elif "T" in ck_args:
        T = int(ck_args["T"])
        print(f"[eval_camera] --num_frames not given; using the checkpoint's T={T}", flush=True)
    else:
        T = int(C.VIDEO_NUM_FRAMES)
        if args.ckpt and os.path.isfile(args.ckpt):
            print(f"[eval_camera] WARNING: ckpt args carry no 'T' (old checkpoint?); "
                  f"defaulting to config.VIDEO_NUM_FRAMES={T}", flush=True)
    latent_root = args.latent_root or (
        C.VIDEO_LATENT_ROOT if T == C.VIDEO_NUM_FRAMES else f"{C.VIDEO_LATENT_ROOT}_T{T}")
    print(f"[eval_camera] T={T}  latent_root={latent_root}", flush=True)

    need_video = (not args.no_video) and any(t in ("forward_dynamics", "policy") for t in args.tasks)

    # ---- Wan2.2-VAE (only if we decode video) --------------------------------------------
    vae = None
    if need_video:
        from precompute_latents import load_vae
        vae = load_vae(args.vae_path, args.resolution, T, dev)
        print(f"[eval_camera] Wan2.2-VAE loaded for latent->pixel decode", flush=True)

    # ---- test window index ----------------------------------------------------------------
    index = build_test_index(args.manifest, args.split_file, args.split, T,
                             latent_root, require_latents=True,
                             windows_json=args.windows_json)
    if not index:
        raise SystemExit(f"[eval_camera] no test windows with precomputed latents under "
                         f"{latent_root} (split={args.split})")
    index = index[: args.n]
    print(f"[eval_camera] {len(index)} test windows (split={args.split}, tasks={args.tasks})",
          flush=True)

    meta = {"ckpt": args.ckpt, "step": step, "tasks": args.tasks, "n": len(index),
            "steps": args.steps, "cfg": args.cfg, "seed": args.seed, "T": T, "fps": args.fps,
            "windows": []}

    for i, it in enumerate(index):
        uuid, s = it["uuid"], it["s"]
        name = seq_name(i, uuid, s)
        print(f"\n[eval_camera] [{i+1}/{len(index)}] {name}", flush=True)

        # --- load precomputed latents + GT camera (upright RGB poses, Cosmos-exact) --------
        latents = _load_latents(it["lp"])                       # (C, T_lat, h, w) fp32
        vlat = torch.from_numpy(np.ascontiguousarray(latents)).float().to(dev)
        C_, T_lat, h, w = vlat.shape

        pos, rot = _load_rgb_cam(it["rgb"])                     # (Nframes,3), (Nframes,3,3)
        pos_w, rot_w = pos[s:s + T], rot[s:s + T]               # window abs poses (T,·)
        gt_action = rel_action_from_window(pos_w, rot_w)        # (T-1,9) Cosmos-exact

        # --- write GT camera (abs poses) in the nymeria_world eval schema ------------------
        write_gt_camera(os.path.join(out_root, "samples", name, "gt_camera_cosmos.npz"),
                        pos_w, rot_w)

        # --- GT pixel clip (for the FD/policy side-by-side viz) ----------------------------
        if need_video:
            try:
                gt_frames = decode_window_pyav(it["vis"], s, T, args.fps)   # (T,H,W,3) uint8
                frames_to_mp4(gt_frames, os.path.join(out_root, "samples", name, "gt_clip.mp4"),
                              args.fps)
            except Exception as e:  # noqa: BLE001
                print(f"  [gt_clip] decode failed ({e}); skipping GT clip", flush=True)

        win = {"name": name, "uuid": uuid, "start": s, "T_lat": T_lat,
               "cam_frames": int(gt_action.shape[0]), "latent_hw": [h, w]}

        # ================= inverse_dynamics : video -> camera action =====================
        if "inverse_dynamics" in args.tasks:
            pred9 = S.sample_inverse_dynamics(
                model, video_latents=vlat, camera_T=T - 1,
                steps=args.steps, guidance=args.cfg, seed=args.seed, device=dev,
            )  # (T-1, 9)
            d = os.path.join(out_root, "invdyn_out", name)
            os.makedirs(d, exist_ok=True)
            json.dump({"outputs": [{"content": {"action": pred9.tolist()}}]},
                      open(os.path.join(d, "sample_outputs.json"), "w"))
            print(f"  [inverse_dynamics] pred action {pred9.shape} "
                  f"|Δt|~{np.linalg.norm(pred9[:, :3], axis=1).mean():.4f} "
                  f"(GT {np.linalg.norm(gt_action[:, :3], axis=1).mean():.4f})", flush=True)

        # ================= forward_dynamics : cam+text+image -> video ====================
        if "forward_dynamics" in args.tasks:
            gt_cam_t = torch.from_numpy(gt_action).float().to(dev)   # clean camera condition
            vid = S.sample_forward_dynamics(
                model, caption=it["cap"], image_latent=vlat[:, 0],
                camera_action=gt_cam_t, T_lat=T_lat,
                steps=args.steps, guidance=args.cfg, seed=args.seed, device=dev,
            )  # (C, T_lat, h, w) latents
            d = os.path.join(out_root, "fd_out", name)
            os.makedirs(d, exist_ok=True)
            np.savez(os.path.join(d, "gen_latents.npz"), latents=vid.astype(np.float16))
            if need_video:
                frames = latent_to_frames(vae, vid, dev)
                frames_to_mp4(frames, os.path.join(d, "vision.mp4"), args.fps)
                print(f"  [forward_dynamics] video {frames.shape} -> {d}/vision.mp4", flush=True)

        # ================= policy : text+image -> camera + video =========================
        if "policy" in args.tasks:
            outp = S.sample_policy(
                model, caption=it["cap"], image_latent=vlat[:, 0], T_lat=T_lat,
                camera_T=T - 1, steps=args.steps, guidance=args.cfg, seed=args.seed, device=dev,
            )  # {"video": (C,T_lat,h,w), "camera": (T-1,9)}
            d = os.path.join(out_root, "policy_out", name)
            os.makedirs(d, exist_ok=True)
            json.dump({"outputs": [{"content": {"action": outp["camera"].tolist()}}]},
                      open(os.path.join(d, "sample_outputs.json"), "w"))
            np.savez(os.path.join(d, "gen_latents.npz"), latents=outp["video"].astype(np.float16))
            if need_video:
                frames = latent_to_frames(vae, outp["video"], dev)
                frames_to_mp4(frames, os.path.join(d, "vision.mp4"), args.fps)
                print(f"  [policy] camera {outp['camera'].shape} + video {frames.shape} -> {d}",
                      flush=True)

        meta["windows"].append(win)

    json.dump(meta, open(os.path.join(out_root, "window_meta.json"), "w"), indent=2)
    print(f"\n[eval_camera] wrote {len(index)} windows to {out_root}", flush=True)

    # ---- inverse-dynamics metrics (reuse eval_inverse_dynamics's exact math) --------------
    if "inverse_dynamics" in args.tasks:
        metrics_path = compute_invdyn_metrics(out_root)
        print(f"[eval_camera] invdyn metrics -> {metrics_path}", flush=True)


def compute_invdyn_metrics(eval_root: str) -> str:
    """Compute the inverse-dynamics metrics over ``<eval_root>/invdyn_out/*`` vs
    ``<eval_root>/samples/*/gt_camera_cosmos.npz`` using ``eval_inverse_dynamics``'s
    functions VERBATIM, writing ``<eval_root>/invdyn_metrics.json`` in its exact schema
    (so nymeria_world's viz/montage tools also read it)."""
    import glob
    seqs = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(eval_root, "invdyn_out", "*",
                                                   "sample_outputs.json")))
    if not seqs:
        raise SystemExit(f"[eval_camera] no invdyn_out/*/sample_outputs.json under {eval_root}")
    rows = {}
    for n in seqs:
        pred = json.load(open(os.path.join(eval_root, "invdyn_out", n, "sample_outputs.json")))
        pred9 = pred["outputs"][0]["content"]["action"]
        P_gt = EID.gt_abs(os.path.join(eval_root, "samples", n, "gt_camera_cosmos.npz"))
        rows[n] = EID.eval_seq(pred9, P_gt)

    keys = ["rot_deg", "trans_dir_cos", "scale_ratio", "trans_err_norm", "ate_m", "len_ratio"]
    agg = {k: {"mean": float(np.mean([rows[n][k] for n in seqs])),
               "median": float(np.median([rows[n][k] for n in seqs]))} for k in keys}

    hdr = f"{'sequence':32s} " + " ".join(f"{k:>13s}" for k in keys)
    print(f"\n=== inverse-dynamics eval | {len(seqs)} windows | {eval_root} ===")
    print(hdr); print("-" * len(hdr))
    for n in seqs:
        print(f"{n[:32]:32s} " + " ".join(f"{rows[n][k]:13.4f}" for k in keys))
    print("-" * len(hdr))
    print(f"{'MEAN':32s} " + " ".join(f"{agg[k]['mean']:13.4f}" for k in keys))
    print(f"{'MEDIAN':32s} " + " ".join(f"{agg[k]['median']:13.4f}" for k in keys))
    print("guide: rot_deg↓ trans_dir_cos→1 scale_ratio→1 trans_err_norm↓(m) ate_m↓ len_ratio→1")

    out = os.path.join(eval_root, "invdyn_metrics.json")
    json.dump({"n": len(seqs), "aggregate": agg, "per_sequence": rows}, open(out, "w"), indent=2)
    return out


if __name__ == "__main__":
    main()
