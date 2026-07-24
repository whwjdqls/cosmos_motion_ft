#!/usr/bin/env python
"""Unified 7-task evaluation + visualization harness for the joint-attention model (cosmos env).

ONE dispatcher over all 7 tasks. It reuses the existing, working seams verbatim -- it does NOT
re-implement them:

  * CAMERA tasks (inverse_dynamics / forward_dynamics / policy):
        -> hand off to ``eval_camera.main`` (its exact nymeria_world-schema writers + the
           inverse-dynamics metric via ``eval_camera.compute_invdyn_metrics``). Viz via
           ``viz_camera``.
  * MOTION-RECON task (video2motion, and reusably textimg2motion/text2motion but FLAGGED
        generative): ``sample.sample_*`` -> z-scored motion, metric via ``eval_motion_recon``,
        GT|pred skeleton viz via ``render_motion.render_motion_mp4``.
  * MOTION-GEN viz (text2motion / textimg2motion): sample + render the predicted skeleton mp4
        (recon metric still computed vs the aligned GT window but flagged not-a-recon-score).
  * VIDEO-GEN viz (motimg2video): sample video latents -> VAE decode -> GT|generated mp4
        (same side-by-side treatment eval_camera gives forward_dynamics).

T, latent_root and the MOTION objective are resolved FROM the checkpoint's saved args (the same
logic eval_camera / sample.load_joint_model use) -- no need to pass them for a non-default-T or
x0-motion run.

Output layout under ``<out_root>`` (default ``<ckpt_dir>/eval_all``)::

    camera_eval/                         (delegated to eval_camera; only if a camera task ran)
        invdyn_out/... fd_out/... policy_out/... samples/... invdyn_metrics.json  viz/...
    motion_recon/<task>/
        pred/<seq>.npy  gt/<seq>.npy     z-scored [T,283] pred + GT motion
        motion_recon_metrics.json        (invdyn_metrics.json schema, per task)
        head_camera_alignment_metrics.json (V2M relative action error, head-camera runs)
    viz/
        <task>_<seq>.mp4                 skeleton mp4 (motion tasks; GT|pred for video2motion)
        motimg2video_<seq>.mp4           GT|generated video side-by-side
    summary.json                         everything, machine-readable
    (a one-screen SUMMARY TABLE is printed to stdout)

Run (single GPU on a node, cosmos env) via ``run_eval.sh``::

    bash run_eval.sh <ckpt.pt> <n_windows> [task ...]     # default tasks = all 7
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
_NYMERIA_WORLD = "/home/jungbin_cho/cosmos_motion_ft/nymeria_world"
# Insert HERE LAST so it lands at sys.path[0] and shadows motion_expert/sample.py (same-named
# module on PYTHONPATH). Without this, `import sample` can bind the PoC's sample.py which lacks
# the 7-task sample_* fns. (When run as a script sys.path[0] is already the script dir, but this
# makes the harness robust to being imported/launched any other way too.)
for _p in ("/home/jungbin_cho/cosmos-framework", _NYMERIA_WORLD, HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import config as C            # noqa: E402
import sample as S            # noqa: E402
import task_plan as TP        # noqa: E402
import eval_motion_recon as EMR   # noqa: E402
from head_camera_alignment import (  # noqa: E402
    DEFAULT_CALIBRATION,
    HEAD_CAMERA_GLOBAL_METRIC_KEYS,
    HEAD_CAMERA_ORACLE_ACTOR_METRIC_KEYS,
    actor_id_from_uuid,
    head_camera_errors,
    load_head_camera_calibration,
    load_oracle_actor_head_camera_calibrations,
    motion_to_camera_action,
)
from uniego_layout import FEAT_DIM, ground_features, canonicalize_frame0  # noqa: E402
from nymeria_joint_dataset import (  # noqa: E402
    _load_latents,
    _load_rgb_cam,
    latent_path,
    rel_action_from_window,
)

ALL_TASKS = list(TP.TASKS)
CAMERA_TASKS = {"inverse_dynamics", "forward_dynamics", "policy"}
MOTION_TASKS = {"text2motion", "textimg2motion", "video2motion"}
VIDEO_GEN_TASKS = {"motimg2video"}


# ============================================================================
# Test-window index carrying EVERY modality (mirrors NymeriaJointDataset's ALIGNED
# index build at lines 264-301: uni + off + latents + rgb + caption at one (uuid,start)).
# ============================================================================
def build_full_index(manifest_path, split_file, split, num_frames, latent_root, uniego_root,
                     require_latents=True, windows_json=None):
    from nymeria_camera_rgb_dataset import _rgb_path
    want = None
    want_by_uuid = {}
    if windows_json:
        rows = json.load(open(windows_json))
        want = {(row["uuid"], int(row["start"])): order for order, row in enumerate(rows)}
        if len(want) != len(rows):
            raise ValueError(f"{windows_json}: duplicate (uuid,start) rows")
        for (uuid, start), order in want.items():
            want_by_uuid.setdefault(uuid, []).append((start, order))

    keep = None
    if split not in ("all", None):
        sp = json.load(open(split_file))
        assert split in sp, f"split '{split}' not in {split_file}"
        keep = set(sp[split])
    # Floor calibration (mirrors NymeriaJointDataset): skip dropped windows + fold the per-seq
    # delta into "off" so eval GT lives in the SAME calibrated space the model trained in.
    deltas, gdelta, dropmap, calibrated = {}, 0.0, {}, False
    if os.path.isfile(C.FLOOR_CALIBRATION_JSON):
        _fc = json.load(open(C.FLOOR_CALIBRATION_JSON))
        deltas = {u: float(v) for u, v in _fc.get("deltas", {}).items()}
        gdelta = float(_fc.get("global_delta", 0.0))
        dropmap = {u: {(int(e[0]), int(e[1])) for e in lst}
                   for u, lst in _fc.get("dropped_windows", {}).items()}
        calibrated = True
    else:
        print(f"[eval_all] WARNING: floor calibration missing ({C.FLOOR_CALIBRATION_JSON}); "
              f"GT motion will be UNCALIBRATED (offset vs a calibrated-trained model)", flush=True)
    index = []
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            uuid = rec.get("uuid")
            cam, vis = rec.get("camera_path"), rec.get("vision_path")
            nb = int(rec.get("nb_frames", 0))
            if not uuid or not cam or not vis:
                continue
            if keep is not None and uuid not in keep:
                continue
            rgb = _rgb_path(cam)
            if not os.path.isfile(rgb):
                continue
            uni = os.path.join(uniego_root, f"{uuid}.npz")
            if not os.path.isfile(uni):
                continue
            if want is not None:
                windows = {int(w["start_frame"]): w for w in rec.get("t2w_windows", [])}
                for s, order in want_by_uuid.get(uuid, []):
                    if s + num_frames > nb:
                        continue
                    lp = latent_path(uuid, s, latent_root)
                    if require_latents and not os.path.isfile(lp):
                        continue
                    window = windows.get(s)
                    if window is None:
                        continue
                    off = window.get("ground_offset_y", None)
                    if calibrated and off is not None:
                        off = float(off) + deltas.get(uuid, gdelta)
                    drop_entry = next(
                        (
                            entry for entry in _fc.get("dropped_windows", {}).get(uuid, [])
                            if int(entry[0]) == int(window["start_frame"])
                            and int(entry[1]) == int(window["end_frame"])
                        ),
                        None,
                    ) if calibrated else None
                    index.append({
                        "uuid": uuid,
                        "vis": vis,
                        "rgb": rgb,
                        "uni": uni,
                        "s": int(s),
                        "cap": window.get("caption", ""),
                        "off": off,
                        "lp": lp,
                        "_order": order,
                        "floor_drop_reason": (
                            str(drop_entry[2]) if drop_entry is not None and len(drop_entry) > 2
                            else None
                        ),
                    })
                continue
            for w in rec.get("t2w_windows", []):
                if not w.get("usable", False) or not w.get("caption"):
                    continue
                ws, we = int(w["start_frame"]), int(w["end_frame"])
                if (ws, we) in dropmap.get(uuid, ()):
                    continue
                off = w.get("ground_offset_y", None)
                if calibrated and off is not None:
                    off = float(off) + deltas.get(uuid, gdelta)
                hi = min(we, nb)
                s = ws
                while s + num_frames <= hi:
                    lp = latent_path(uuid, s, latent_root)
                    if (not require_latents) or os.path.isfile(lp):
                        index.append({"uuid": uuid, "vis": vis, "rgb": rgb, "uni": uni,
                                      "s": int(s), "cap": w["caption"], "off": off, "lp": lp})
                    s += num_frames
    if want is not None:
        index.sort(key=lambda item: item["_order"])
        found = {(item["uuid"], item["s"]) for item in index}
        missing = [key for key, _order in sorted(want.items(), key=lambda pair: pair[1])
                   if key not in found]
        print(
            f"[eval_all] windows_json: matched {len(index)}/{len(want)} requested windows",
            flush=True,
        )
        if missing:
            print(f"[eval_all] windows_json missing: {missing}", flush=True)
    return index


def seq_name(i, uuid, start):
    return f"t{i}_{uuid.replace('/', '_')}_{start}"


def load_gt_motion(uni_path, s, off, T, mean, std):
    """Load GT motion window [T,283] z-scored, EXACTLY as NymeriaJointDataset._load_motion
    (ground -> canonicalize frame0 -> z-score). Returns (feat_z[k,283], neutral_joints[30,3])."""
    with np.load(uni_path) as npz:
        feats = npz["features"][s:s + T].astype(np.float32)
        nj = npz["neutral_joints"].astype(np.float32)
    if off is not None:
        feats = ground_features(feats, off)
    feats = canonicalize_frame0(feats)
    feats_z = (feats - mean) / std
    nj = nj - nj.mean(axis=0, keepdims=True)
    return feats_z, nj


# ============================================================================
# Video (Wan-VAE latent) helpers -- reuse eval_camera's exact decode/write.
# ============================================================================
def _video_side_by_side(gt_frames, gen_frames, dst_mp4, fps):
    import eval_camera as EC
    gt_path = dst_mp4.replace(".mp4", "_gt.mp4")
    gen_path = dst_mp4.replace(".mp4", "_gen.mp4")
    if gt_frames is not None:
        EC.frames_to_mp4(gt_frames, gt_path, fps)
    EC.frames_to_mp4(gen_frames, gen_path, fps)
    # ffmpeg hstack (GT | generated); falls back to the two separate clips if ffmpeg fails.
    if gt_frames is not None:
        import subprocess
        fc = (
            "[0:v]scale=380:380,pad=iw:ih+26:0:26:black,"
            "drawbox=x=0:y=26:w=iw:h=ih-26:color=green:t=6,"
            "drawtext=text=GT:x=6:y=3:fontcolor=green:fontsize=18[a];"
            "[1:v]scale=380:380,pad=iw:ih+26:0:26:black,"
            "drawbox=x=0:y=26:w=iw:h=ih-26:color=green:t=6:enable='eq(n,0)',"
            "drawbox=x=0:y=26:w=iw:h=ih-26:color=red:t=6:enable='gte(n,1)',"
            "drawtext=text='GEN (GT prefix)':x=6:y=3:fontcolor=green:fontsize=18:"
            "enable='eq(n,0)',"
            "drawtext=text=GEN:x=6:y=3:fontcolor=red:fontsize=18:enable='gte(n,1)'[b];"
            "[a][b]hstack"
        )
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", gt_path, "-i", gen_path,
                            "-filter_complex", fc, dst_mp4], check=False)
        if r.returncode == 0:
            return dst_mp4
    return gen_path


# ============================================================================
# Motion viz: render predicted (and, for video2motion, GT|pred) skeleton mp4.
# ============================================================================
def _render_motion(feat_z, mean, std, dst_mp4, caption, fps, gt_feat_z=None):
    """kimodo-style skeleton mp4 (tracking camera + world-grid floor, render_motion module):
    GT (blue, left) | pred (red, right) NATIVE side-by-side when ``gt_feat_z`` is given
    (video2motion); single tracking panel otherwise (generative text/textimg2motion)."""
    from render_motion import render_motion_mp4
    pred_j = EMR.decode_to_joints(EMR.unnormalize(feat_z, mean, std))       # [T,30,3]
    gt_j = None
    if gt_feat_z is not None:
        gt_j = EMR.decode_to_joints(EMR.unnormalize(gt_feat_z, mean, std))
    return render_motion_mp4(pred_j, dst_mp4, caption=caption, fps=int(round(fps)),
                             gt_joints=gt_j)


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None,
                    help="train.py checkpoint (.pt). If omitted -> BASE model smoke path; an "
                         "explicit missing path is an error.")
    ap.add_argument("--out_dir", default=None, help="output root (default <ckpt_dir>/eval_all)")
    ap.add_argument("--n", type=int, default=8, help="number of test windows per task")
    ap.add_argument("--tasks", nargs="*", default=ALL_TASKS, help="tasks to eval (default = all 7)")
    ap.add_argument("--windows_json", default=None,
                    help="JSON list of {uuid,start}: evaluate EXACTLY these windows, in order, "
                         "for camera, motion, and video tasks (e.g. one per held-out sequence)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--motion_native_solver",
        choices=["euler", "unipc"],
        default=None,
        help="optional inference-only override for native-schedule x0 motion checkpoints; "
             "use unipc to evaluate historical Euler-recorded checkpoints with NVIDIA's "
             "official solver",
    )
    ap.add_argument(
        "--gen_shift_override",
        type=float,
        default=None,
        help=(
            "inference-only native generator scheduler shift override. Use 10 with the "
            "480-pixel VAE bucket to reproduce the repository's high-tier sampling contract; "
            "the checkpointed shift remains recorded separately"
        ),
    )
    ap.add_argument(
        "--eval_head_camera_alignment",
        action="store_true",
        help=(
            "compute V2M relative head-camera action errors even when the checkpoint was "
            "trained without head-camera alignment; this is evaluation-only and does not "
            "change model inputs or sampling"
        ),
    )
    ap.add_argument(
        "--head_camera_calibration",
        default=None,
        help=(
            "train-split head-camera calibration JSON used by the evaluation-only metric "
            f"(default: checkpoint value, then {DEFAULT_CALIBRATION})"
        ),
    )
    ap.add_argument(
        "--oracle_test_actor_calibration",
        default=None,
        help=(
            "optional per-test-actor calibration fitted from GT test motion and GT camera; "
            "adds explicitly leaky oracle diagnostics and never changes model inputs/sampling"
        ),
    )
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--manifest", default=C.NYMERIA_MANIFEST)
    ap.add_argument("--split_file", default=C.NYMERIA_SPLIT_FILE)
    ap.add_argument("--uniego_root",
                    default="/weka/jungbin/nymeriaplus_kimodo_proportional/uniego_rep")
    ap.add_argument("--latent_root", default=None)
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=float(C.FPS))
    ap.add_argument("--resolution", default="256")
    ap.add_argument(
        "--expected_m2v_latent_hw",
        type=int,
        default=None,
        help=(
            "optional strict M2V latent spatial-size assertion. The cached 256-tier path is "
            "16x16; this repository's square 480-tier path emits 640x640 pixels and 40x40 "
            "Wan latents"
        ),
    )
    ap.add_argument("--vae_path",
                    default=os.environ.get("WAN_VAE_PATH", "/weka/jungbin/wan22_vae/Wan2.2_VAE.pth"))
    ap.add_argument("--no_video", action="store_true", help="skip all VAE latent->pixel decode")
    ap.add_argument("--motion_viz_limit", type=int, default=-1,
                    help="maximum number of motion-task mp4s to render per task; negative renders all. "
                         "Metrics and .npy outputs are still written for every evaluated window.")
    ap.add_argument("--lpips_batch_size", type=int, default=16,
                    help="frame batch size for motimg2video LPIPS-Alex evaluation")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.ckpt is not None and not os.path.isfile(args.ckpt):
        ap.error(f"--ckpt does not exist or is not a file: {args.ckpt}")
    if args.gen_shift_override is not None and args.gen_shift_override <= 0:
        ap.error("--gen_shift_override must be positive")
    if args.expected_m2v_latent_hw is not None and args.expected_m2v_latent_hw <= 0:
        ap.error("--expected_m2v_latent_hw must be positive")

    tasks = [t for t in args.tasks if t in ALL_TASKS]
    bad = [t for t in args.tasks if t not in ALL_TASKS]
    if bad:
        print(f"[eval_all] ignoring unknown tasks {bad}; valid = {ALL_TASKS}")
    dev = args.device

    # ---- output root ----
    if args.out_dir:
        out_root = args.out_dir
    elif args.ckpt and os.path.isfile(args.ckpt):
        out_root = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "eval_all")
    else:
        out_root = os.path.join(C.RUNS_ROOT, "eval_all_base")
    os.makedirs(out_root, exist_ok=True)
    print(f"[eval_all] out_root={out_root}  tasks={tasks}", flush=True)

    summary = {"ckpt": args.ckpt, "tasks": tasks, "n": args.n, "out_root": out_root,
               "windows_json": args.windows_json, "camera": None, "motion": {},
               "head_camera": None, "video": {}, "video_metrics": {}, "viz": []}

    # ---- 1. CAMERA tasks -> delegate to eval_camera (its schema + metric + viz) ---------------
    cam_tasks = [t for t in tasks if t in CAMERA_TASKS]
    if cam_tasks:
        import eval_camera as EC
        cam_out = os.path.join(out_root, "camera_eval")
        argv = ["eval_camera.py", "--out_dir", cam_out, "--n", str(args.n),
                "--tasks", *cam_tasks, "--steps", str(args.steps), "--cfg", str(args.cfg),
                "--seed", str(args.seed), "--split", args.split, "--manifest", args.manifest,
                "--split_file", args.split_file, "--fps", str(args.fps),
                "--resolution", args.resolution, "--vae_path", args.vae_path, "--device", dev]
        if args.ckpt and os.path.isfile(args.ckpt):
            argv += ["--ckpt", args.ckpt]
        if args.num_frames is not None:
            argv += ["--num_frames", str(args.num_frames)]
        if args.latent_root:
            argv += ["--latent_root", args.latent_root]
        if args.no_video:
            argv += ["--no_video"]
        if args.windows_json:
            argv += ["--windows_json", args.windows_json]
        print(f"[eval_all] delegating camera tasks -> eval_camera {cam_tasks}", flush=True)
        old = sys.argv
        try:
            sys.argv = argv
            EC.main()
        finally:
            sys.argv = old
        summary["camera"] = {"eval_dir": cam_out,
                             "metrics_json": os.path.join(cam_out, "invdyn_metrics.json")
                             if "inverse_dynamics" in cam_tasks else None}
        # camera viz
        try:
            import viz_camera as VC
            VC.viz_frusta(cam_out, tag=os.path.basename(out_root), max_seqs=args.n)
            VC.viz_metric_montage(cam_out, tag=os.path.basename(out_root), max_seqs=args.n)
            if not args.no_video:
                VC.viz_video(cam_out, tag=os.path.basename(out_root))
        except Exception as e:  # noqa: BLE001
            print(f"[eval_all] camera viz failed ({e})", flush=True)

    # ---- motion / video-gen tasks need our own model + index ----------------------------------
    mv_tasks = [t for t in tasks if t in (MOTION_TASKS | VIDEO_GEN_TASKS)]
    if not mv_tasks:
        _write_summary(out_root, summary)
        _print_summary(summary)
        return

    # ---- model (real ckpt overlay, or BASE model smoke) ---------------------------------------
    ck_args = {}
    if args.ckpt:
        model, cosmos, ck = S.load_joint_model(
            args.ckpt,
            device=dev,
            motion_native_solver_cli=args.motion_native_solver,
        )
        ck_args = (ck.get("args", {}) or {}) if isinstance(ck, dict) else {}
        objective = model.objective
    else:
        print("[eval_all] --ckpt omitted; BASE model (x0-motion default, smoke only)", flush=True)
        from cosmos_loader import FrozenCosmos
        from joint_motion_model import JointMotionModel
        cosmos = FrozenCosmos(dtype=torch.bfloat16, device=dev)
        model = JointMotionModel(cosmos, objective="x0", motion_dim=FEAT_DIM).to(dev)
        model.eval()
        objective = model.objective

    checkpoint_gen_shift = float(model.gen_shift)
    if args.gen_shift_override is not None:
        if model.gen_schedule != "native":
            raise ValueError(
                "--gen_shift_override is only valid for native generator schedules, got "
                f"{model.gen_schedule!r}"
            )
        model.gen_shift = float(args.gen_shift_override)
        print(
            f"[eval_all] inference-only generator shift override: "
            f"{checkpoint_gen_shift:g} -> {model.gen_shift:g}",
            flush=True,
        )

    summary["sampling"] = {
        "steps": int(args.steps),
        "cfg": float(args.cfg),
        "seed": int(args.seed),
        "motion_schedule": model.motion_schedule,
        "motion_shift": float(model.motion_shift),
        "motion_native_solver": model.motion_native_solver,
        "gen_schedule": model.gen_schedule,
        "gen_shift": float(model.gen_shift),
        "checkpoint_gen_shift": checkpoint_gen_shift,
        "gen_shift_overridden": args.gen_shift_override is not None,
        "gen_native_solver": model.gen_native_solver,
        "vae_resolution": str(args.resolution),
        "expected_m2v_latent_hw": args.expected_m2v_latent_hw,
    }

    # ---- T + latent root: default from ckpt args (mirrors eval_camera) -------------------------
    if args.num_frames is not None:
        T = int(args.num_frames)
    elif "T" in ck_args:
        T = int(ck_args["T"])
    else:
        T = int(C.VIDEO_NUM_FRAMES)
    ti2m_T = int(ck_args.get("ti2m_frames") or T)
    reasoner_ti2m = (
        "textimg2motion" in mv_tasks
        and getattr(model, "textimg_condition", "generator") == "reasoner"
    )
    needs_latents = (
        "video2motion" in mv_tasks
        or bool(set(mv_tasks) & VIDEO_GEN_TASKS)
        or ("textimg2motion" in mv_tasks and not reasoner_ti2m)
    )
    # A Phase-2 checkpoint can use output T=200 for T2M but 97 aligned valid frames for TI2M.
    # Build the held-out index at 97 when no selected task needs generator latents/full video.
    index_T = ti2m_T if reasoner_ti2m and not needs_latents else T
    latent_root = args.latent_root or (
        C.VIDEO_LATENT_ROOT
        if index_T == C.VIDEO_NUM_FRAMES
        else f"{C.VIDEO_LATENT_ROOT}_T{index_T}"
    )
    print(
        f"[eval_all] output_T={T} ti2m_T={ti2m_T} index_T={index_T} "
        f"needs_latents={needs_latents} latent_root={latent_root} "
        f"motion_objective={objective}",
        flush=True,
    )

    mean = np.load(C.MOTION_STATS_MEAN).astype(np.float32)
    std = np.load(C.MOTION_STATS_STD).astype(np.float32)

    evaluate_head_camera = (
        "video2motion" in mv_tasks
        and (
            args.eval_head_camera_alignment
            or bool(args.oracle_test_actor_calibration)
            or getattr(model, "head_camera_alignment", False)
        )
    )
    head_camera_rotation = None
    camera_origin_in_head = None
    motion_mean_t = None
    motion_std_t = None
    head_camera_calibration = None
    oracle_actor_calibrations = None
    oracle_actor_calibration_payload = None
    if evaluate_head_camera:
        head_camera_calibration = (
            args.head_camera_calibration
            or ck_args.get("head_camera_calibration")
            or DEFAULT_CALIBRATION
        )
        if not os.path.isfile(head_camera_calibration):
            raise FileNotFoundError(
                f"head-camera calibration does not exist: {head_camera_calibration}"
            )
        head_camera_rotation, camera_origin_in_head, _ = load_head_camera_calibration(
            head_camera_calibration
        )
        head_camera_rotation = head_camera_rotation.to(dev)
        camera_origin_in_head = camera_origin_in_head.to(dev)
        motion_mean_t = torch.from_numpy(mean).to(dev)
        motion_std_t = torch.from_numpy(std).to(dev)
        print(
            "[eval_all] V2M head-camera metric enabled "
            f"(checkpoint_alignment={getattr(model, 'head_camera_alignment', False)}, "
            f"calibration={head_camera_calibration})",
            flush=True,
        )
        if args.oracle_test_actor_calibration:
            if args.split != "test":
                raise ValueError(
                    "--oracle_test_actor_calibration is test-label-derived and requires --split test"
                )
            oracle_actor_calibrations, oracle_actor_calibration_payload = (
                load_oracle_actor_head_camera_calibrations(
                    args.oracle_test_actor_calibration
                )
            )
            print(
                "[eval_all] WARNING: test-derived per-actor oracle enabled; these metrics "
                "use GT test motion+camera and are diagnostic only "
                f"({args.oracle_test_actor_calibration})",
                flush=True,
            )

    # ---- test-window index (carries every modality) -------------------------------------------
    index = build_full_index(
        args.manifest,
        args.split_file,
        args.split,
        index_T,
        latent_root,
        args.uniego_root,
        require_latents=needs_latents,
        windows_json=args.windows_json,
    )
    if not index:
        requirement = f"latents under {latent_root} + uniego" if needs_latents else "uniego"
        raise SystemExit(
            f"[eval_all] no test windows w/ {requirement} (split={args.split}, T={index_T})"
        )
    if args.windows_json:
        requested_count = len(json.load(open(args.windows_json)))
        if len(index) != requested_count:
            raise RuntimeError(
                f"explicit held-out set incomplete: matched {len(index)}/{requested_count} windows"
            )
    index = index[: args.n]
    print(f"[eval_all] {len(index)} test windows for motion/video tasks", flush=True)
    floor_flagged = {
        seq_name(i, item["uuid"], item["s"]): item["floor_drop_reason"]
        for i, item in enumerate(index)
        if item.get("floor_drop_reason")
    }
    summary["floor_flagged_motion_gt"] = floor_flagged
    if floor_flagged:
        print(
            f"[eval_all] WARNING: {len(floor_flagged)} explicit windows are flagged by floor "
            "calibration; full-set metrics include them and a floor-valid subset is also written",
            flush=True,
        )

    # ---- VAE only if a video-gen task with pixel output -----------------------------------
    need_video = (not args.no_video) and any(t in VIDEO_GEN_TASKS for t in mv_tasks)
    vae = None
    lpips_metric = None
    if need_video:
        from precompute_latents import load_vae
        from native_phase_training.evaluate_inverse_forward import LPIPSAlex

        vae = load_vae(args.vae_path, args.resolution, T, dev)
        lpips_metric = LPIPSAlex(dev, args.lpips_batch_size)
        print("[eval_all] Wan2.2-VAE loaded", flush=True)

    viz_dir = os.path.join(out_root, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # per-task collected metric rows
    recon_rows = {t: {} for t in mv_tasks if t in MOTION_TASKS}
    motion_viz_counts = {t: 0 for t in mv_tasks if t in MOTION_TASKS}
    video_metric_rows = {}
    head_camera_rows = {} if evaluate_head_camera else None

    for i, it in enumerate(index):
        uuid, s = it["uuid"], it["s"]
        name = seq_name(i, uuid, s)
        print(f"\n[eval_all] [{i+1}/{len(index)}] {name}", flush=True)

        base_gt_z, nj = load_gt_motion(
            it["uni"], s, it["off"], index_T, mean, std
        )
        nj_t = torch.from_numpy(nj).float().to(dev)[None]
        vlat = None
        T_lat = None
        if needs_latents:
            vlat = torch.from_numpy(
                np.ascontiguousarray(_load_latents(it["lp"]))
            ).float().to(dev)
            _C, T_lat, _h, _w = vlat.shape
            if (
                "motimg2video" in mv_tasks
                and args.expected_m2v_latent_hw is not None
                and (_h != args.expected_m2v_latent_hw or _w != args.expected_m2v_latent_hw)
            ):
                raise RuntimeError(
                    f"{name}: M2V latent grid is {_h}x{_w}, expected "
                    f"{args.expected_m2v_latent_hw}x{args.expected_m2v_latent_hw}; "
                    f"latent_root={latent_root}"
                )
        cap = it["cap"]

        # ---- MOTION tasks: sample -> recon metric -> render ------------------------------------
        for t in [x for x in mv_tasks if x in MOTION_TASKS]:
            task_T = ti2m_T if t == "textimg2motion" else T
            if task_T == index_T:
                gt_task_z = base_gt_z
            else:
                gt_task_z, _ = load_gt_motion(
                    it["uni"], s, it["off"], task_T, mean, std
                )
            if t == "text2motion":
                pred_z = S.sample_text2motion(model, caption=cap, neutral_joints=nj_t, T=task_T,
                                              steps=args.steps, guidance=args.cfg,
                                              objective=objective, device=dev, seed=args.seed)
            elif t == "textimg2motion":
                reasoner_image = None
                if getattr(model, "textimg_condition", "generator") == "reasoner":
                    from nymeria_camera_dataset import decode_window_pyav
                    # Corrected reasoner-side TI2M needs only frame 0. Decoding the full T=97
                    # clip here was pure I/O/CPU overhead and did not change conditioning.
                    frames0 = decode_window_pyav(it["vis"], s, 1, args.fps)
                    reasoner_image = torch.from_numpy(
                        np.ascontiguousarray(frames0[0])
                    ).permute(2, 0, 1).contiguous()
                pred_z = S.sample_textimg2motion(model, caption=cap, neutral_joints=nj_t, T=task_T,
                                                 image_latent=(vlat[:, 0] if vlat is not None else None),
                                                 reasoner_image=reasoner_image,
                                                 steps=args.steps,
                                                 guidance=args.cfg, objective=objective,
                                                 device=dev, seed=args.seed)
            else:  # video2motion
                pred_z = S.sample_video2motion(model, neutral_joints=nj_t, T=task_T, video_latents=vlat,
                                               steps=args.steps, guidance=args.cfg,
                                               objective=objective, device=dev, seed=args.seed)
            # save pred + gt z-scored motion
            pd = os.path.join(out_root, "motion_recon", t, "pred"); os.makedirs(pd, exist_ok=True)
            gd = os.path.join(out_root, "motion_recon", t, "gt"); os.makedirs(gd, exist_ok=True)
            np.save(os.path.join(pd, name + ".npy"), pred_z)
            np.save(os.path.join(gd, name + ".npy"), gt_task_z)
            # recon metric (vs the aligned GT window)
            recon_rows[t][name] = EMR.recon_metrics(pred_z, gt_task_z, mean, std)
            print(f"  [{t}] MPJPE={recon_rows[t][name]['mpjpe_m']:.3f} "
                  f"accel_err={recon_rows[t][name]['accel_err']:.3f}", flush=True)
            if t == "video2motion" and head_camera_rows is not None:
                camera_position, camera_rotation = _load_rgb_cam(it["rgb"])
                camera_target = torch.from_numpy(rel_action_from_window(
                    camera_position[s:s + task_T],
                    camera_rotation[s:s + task_T],
                )).float().to(dev).unsqueeze(0)
                transition_mask = torch.ones(
                    camera_target.shape[:2], dtype=torch.bool, device=dev
                )
                with torch.no_grad():
                    pred_motion = (
                        torch.from_numpy(np.ascontiguousarray(pred_z))
                        .float()
                        .to(dev)
                        .unsqueeze(0)
                    )
                    gt_motion = (
                        torch.from_numpy(np.ascontiguousarray(gt_task_z))
                        .float()
                        .to(dev)
                        .unsqueeze(0)
                    )
                    pred_action = motion_to_camera_action(
                        pred_motion * motion_std_t + motion_mean_t,
                        head_camera_rotation,
                        camera_origin_in_head,
                    )
                    gt_action = motion_to_camera_action(
                        gt_motion * motion_std_t + motion_mean_t,
                        head_camera_rotation,
                        camera_origin_in_head,
                    )
                    pred_trans, pred_rot = head_camera_errors(
                        pred_action, camera_target, transition_mask
                    )
                    gt_trans, gt_rot = head_camera_errors(
                        gt_action, camera_target, transition_mask
                    )
                actor_id = actor_id_from_uuid(uuid)
                head_camera_rows[name] = {
                    "actor_id": actor_id,
                    "translation_m": float(pred_trans),
                    "rotation_deg": float(pred_rot),
                    "gt_calibration_translation_m": float(gt_trans),
                    "gt_calibration_rotation_deg": float(gt_rot),
                }
                oracle_message = ""
                if oracle_actor_calibrations is not None:
                    if actor_id not in oracle_actor_calibrations:
                        raise KeyError(
                            f"oracle test-actor calibration has no entry for {actor_id} ({uuid})"
                        )
                    oracle_rotation, oracle_lever = oracle_actor_calibrations[actor_id]
                    with torch.no_grad():
                        oracle_pred_action = motion_to_camera_action(
                            pred_motion * motion_std_t + motion_mean_t,
                            oracle_rotation.to(dev),
                            oracle_lever.to(dev),
                        )
                        oracle_gt_action = motion_to_camera_action(
                            gt_motion * motion_std_t + motion_mean_t,
                            oracle_rotation.to(dev),
                            oracle_lever.to(dev),
                        )
                        oracle_pred_trans, oracle_pred_rot = head_camera_errors(
                            oracle_pred_action, camera_target, transition_mask
                        )
                        oracle_gt_trans, oracle_gt_rot = head_camera_errors(
                            oracle_gt_action, camera_target, transition_mask
                        )
                    head_camera_rows[name].update({
                        "oracle_actor_translation_m": float(oracle_pred_trans),
                        "oracle_actor_rotation_deg": float(oracle_pred_rot),
                        "gt_oracle_actor_translation_m": float(oracle_gt_trans),
                        "gt_oracle_actor_rotation_deg": float(oracle_gt_rot),
                    })
                    oracle_message = (
                        f" oracle={float(oracle_pred_trans):.4f}m/"
                        f"{float(oracle_pred_rot):.3f}deg "
                        f"GT-oracle={float(oracle_gt_trans):.4f}m/"
                        f"{float(oracle_gt_rot):.3f}deg"
                    )
                print(
                    f"  [video2motion head-camera] trans={float(pred_trans):.4f}m "
                    f"rot={float(pred_rot):.3f}deg "
                    f"(GT calibration floor={float(gt_trans):.4f}m/{float(gt_rot):.3f}deg)"
                    f"{oracle_message}",
                    flush=True,
                )
            # viz: video2motion -> GT|pred side-by-side; generative tasks -> pred only
            if args.motion_viz_limit < 0 or motion_viz_counts[t] < args.motion_viz_limit:
                dst = os.path.join(viz_dir, f"{t}_{name}.mp4")
                try:
                    viz_caption = (
                        f"metadata only (V2M text disabled): {cap[:40]}"
                        if t == "video2motion" else cap[:40]
                    )
                    p = _render_motion(pred_z, mean, std, dst, viz_caption, args.fps,
                                       gt_feat_z=(gt_task_z if t == "video2motion" else None))
                    summary["viz"].append(p)
                    motion_viz_counts[t] += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  [{t}] render failed ({e})", flush=True)
            elif motion_viz_counts[t] == args.motion_viz_limit:
                print(f"  [{t}] motion_viz_limit={args.motion_viz_limit}; "
                      f"skipping remaining renders (metrics continue)", flush=True)
                motion_viz_counts[t] += 1

        # ---- motimg2video: sample video -> VAE decode -> GT|gen side-by-side -------------------
        if "motimg2video" in mv_tasks:
            motion_clean = torch.from_numpy(base_gt_z).float().to(dev)[None]   # [1,k,283] clean cond
            gen = S.sample_motimg2video(model, caption=cap, image_latent=vlat[:, 0],
                                        motion=motion_clean, neutral_joints=nj_t, T_lat=T_lat,
                                        steps=args.steps, guidance=args.cfg, device=dev,
                                        seed=args.seed)  # [C,T_lat,h,w] latents
            vd = os.path.join(out_root, "video", "motimg2video"); os.makedirs(vd, exist_ok=True)
            np.savez(os.path.join(vd, name + ".npz"), latents=gen.astype(np.float16))
            summary["video"].setdefault("motimg2video", []).append(name)
            if need_video:
                import eval_camera as EC
                from native_phase_training.evaluate_inverse_forward import (
                    _frame_metrics,
                    _resize_gt_like_native,
                    _summarize_frame_metrics,
                )
                from nymeria_camera_dataset import decode_window_pyav

                gt_frames = decode_window_pyav(it["vis"], s, T, args.fps)
                gen_frames = EC.latent_to_frames(vae, gen, dev)
                if len(gt_frames) != T or len(gen_frames) != T:
                    raise RuntimeError(
                        f"{name}: expected {T} GT/generated frames, got "
                        f"{len(gt_frames)}/{len(gen_frames)}"
                    )
                gt_metric_frames = _resize_gt_like_native(
                    gt_frames, gen_frames.shape[1], gen_frames.shape[2]
                )
                frame_values = _frame_metrics(
                    gt_metric_frames[1:], gen_frames[1:], lpips_metric
                )
                video_metric_rows[name] = _summarize_frame_metrics(frame_values)
                metric_row = video_metric_rows[name]
                print(
                    f"  [motimg2video] PSNR={metric_row['psnr_db']:.3f} "
                    f"SSIM={metric_row['ssim']:.4f} LPIPS={metric_row['lpips_alex']:.4f}",
                    flush=True,
                )
                dst = os.path.join(viz_dir, f"motimg2video_{name}.mp4")
                p = _video_side_by_side(gt_frames, gen_frames, dst, args.fps)
                summary["viz"].append(p)
                print(f"  [motimg2video] video -> {p}", flush=True)

    # ---- write motion recon metrics per task --------------------------------------------------
    for t, rows in recon_rows.items():
        if not rows:
            continue
        outp = os.path.join(out_root, "motion_recon", t, "motion_recon_metrics.json")
        EMR.aggregate_and_write(rows, outp, tag=f"{t} | {out_root}",
                                generative=(t in EMR.GENERATIVE_TASKS))
        summary["motion"][t] = {"metrics_json": outp,
                                "generative": t in EMR.GENERATIVE_TASKS,
                                "aggregate": json.load(open(outp))["aggregate"]}
        valid_rows = {name: row for name, row in rows.items() if name not in floor_flagged}
        if len(valid_rows) != len(rows):
            valid_out = os.path.join(
                out_root, "motion_recon", t, "motion_recon_metrics_floor_valid.json"
            )
            EMR.aggregate_and_write(
                valid_rows,
                valid_out,
                tag=f"{t} floor-valid | {out_root}",
                generative=(t in EMR.GENERATIVE_TASKS),
            )
            summary["motion"][t]["floor_valid_metrics_json"] = valid_out
            summary["motion"][t]["floor_valid_n"] = len(valid_rows)

    if head_camera_rows:
        metrics_path = os.path.join(
            out_root, "motion_recon", "video2motion", "head_camera_alignment_metrics.json"
        )
        payload = _aggregate_head_camera_metrics(head_camera_rows)
        payload["train_global_calibration"] = head_camera_calibration
        if oracle_actor_calibrations is not None:
            payload["oracle_test_actor_calibration"] = {
                "path": args.oracle_test_actor_calibration,
                "kind": oracle_actor_calibration_payload["kind"],
                "diagnostic_only": True,
                "uses_test_gt_motion": True,
                "uses_test_gt_camera": True,
                "fit_and_evaluation_windows_are_identical": bool(
                    oracle_actor_calibration_payload["leakage_contract"].get(
                        "fit_and_evaluation_windows_are_identical", False
                    )
                ),
                "fit_windows": int(
                    oracle_actor_calibration_payload.get("counts", {}).get("windows", 0)
                ),
            }
        json.dump(payload, open(metrics_path, "w"), indent=2)
        summary["head_camera"] = {
            "metrics_json": metrics_path,
            "aggregate": payload["aggregate"],
            "calibration": head_camera_calibration,
            "checkpoint_alignment_enabled": bool(
                getattr(model, "head_camera_alignment", False)
            ),
            "evaluation_only_for_checkpoint": not bool(
                getattr(model, "head_camera_alignment", False)
            ),
            "oracle_test_actor_calibration": payload.get(
                "oracle_test_actor_calibration"
            ),
            "oracle_test_actor_diagnostic_only": bool(oracle_actor_calibrations),
        }
        valid_rows = {
            name: row for name, row in head_camera_rows.items() if name not in floor_flagged
        }
        if len(valid_rows) != len(head_camera_rows):
            valid_path = os.path.join(
                out_root,
                "motion_recon",
                "video2motion",
                "head_camera_alignment_metrics_floor_valid.json",
            )
            valid_payload = _aggregate_head_camera_metrics(valid_rows)
            valid_payload["train_global_calibration"] = head_camera_calibration
            if oracle_actor_calibrations is not None:
                valid_payload["oracle_test_actor_calibration"] = payload[
                    "oracle_test_actor_calibration"
                ]
            json.dump(valid_payload, open(valid_path, "w"), indent=2)
            summary["head_camera"].update({
                "floor_valid_metrics_json": valid_path,
                "floor_valid_n": len(valid_rows),
                "floor_valid_aggregate": valid_payload["aggregate"],
            })

    if video_metric_rows:
        metrics_path = os.path.join(out_root, "video", "motimg2video_metrics.json")
        video_payload = _aggregate_video_metrics(video_metric_rows)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        json.dump(video_payload, open(metrics_path, "w"), indent=2)
        summary["video_metrics"]["motimg2video"] = {
            "metrics_json": metrics_path,
            "aggregate": video_payload["aggregate"],
        }
        valid_video_rows = {
            name: row for name, row in video_metric_rows.items() if name not in floor_flagged
        }
        if len(valid_video_rows) != len(video_metric_rows):
            valid_metrics_path = os.path.join(
                out_root, "video", "motimg2video_metrics_floor_valid.json"
            )
            valid_video_payload = _aggregate_video_metrics(valid_video_rows)
            json.dump(valid_video_payload, open(valid_metrics_path, "w"), indent=2)
            summary["video_metrics"]["motimg2video"].update({
                "floor_valid_metrics_json": valid_metrics_path,
                "floor_valid_n": len(valid_video_rows),
                "floor_valid_aggregate": valid_video_payload["aggregate"],
            })

    _write_summary(out_root, summary)
    _print_summary(summary)


def _aggregate_video_metrics(rows):
    """Aggregate per-sequence M2V pixel metrics using the native Phase-1 schema."""
    from native_phase_training.evaluate_inverse_forward import (
        HORIZONS,
        METRIC_KEYS_FORWARD,
        _aggregate,
    )

    scalar_rows = {
        name: {key: float(row[key]) for key in METRIC_KEYS_FORWARD}
        for name, row in rows.items()
    }
    horizon_aggregate = {
        horizon: {
            key: {
                "mean": float(np.mean([
                    row["horizons"][horizon][key] for row in rows.values()
                ])),
                "median": float(np.median([
                    row["horizons"][horizon][key] for row in rows.values()
                ])),
            }
            for key in METRIC_KEYS_FORWARD
        }
        for horizon in HORIZONS
    }
    evaluated_frames = int(next(iter(rows.values()))["evaluated_frames"])
    return {
        "n": len(rows),
        "task": "motimg2video",
        "conditioned_frame_excluded": True,
        "evaluated_frame_range": [1, evaluated_frames],
        "gt_preprocessing": (
            "native aspect-preserving bicubic antialias resize plus right/bottom reflection pad"
        ),
        "aggregate": _aggregate(scalar_rows, METRIC_KEYS_FORWARD),
        "horizon_aggregate": horizon_aggregate,
        "per_sequence": rows,
    }


def _aggregate_head_camera_metrics(rows):
    """Aggregate per-window V2M relative head-camera action errors."""
    if not rows:
        raise ValueError("cannot aggregate empty head-camera rows")
    keys = list(HEAD_CAMERA_GLOBAL_METRIC_KEYS)
    oracle_presence = {
        key: [key in row for row in rows.values()]
        for key in HEAD_CAMERA_ORACLE_ACTOR_METRIC_KEYS
    }
    if any(any(presence) and not all(presence) for presence in oracle_presence.values()):
        raise ValueError("partial oracle test-actor metrics across head-camera rows")
    if all(all(presence) for presence in oracle_presence.values()):
        keys.extend(HEAD_CAMERA_ORACLE_ACTOR_METRIC_KEYS)

    def aggregate_subset(subset):
        aggregate = {}
        for key in keys:
            values = np.asarray([row[key] for row in subset.values()], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"head-camera metric {key} contains non-finite values")
            aggregate[key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
            }
        return aggregate

    aggregate = aggregate_subset(rows)
    per_actor = {}
    actor_ids = {row.get("actor_id") for row in rows.values()}
    if None not in actor_ids:
        for actor_id in sorted(actor_ids):
            actor_rows = {
                name: row for name, row in rows.items() if row["actor_id"] == actor_id
            }
            per_actor[actor_id] = {
                "n": len(actor_rows),
                "aggregate": aggregate_subset(actor_rows),
            }

    metric_definitions = {
        "translation_m": (
            "predicted V2M motion mapped with the train-global calibration versus GT camera; "
            "mean relative-translation L2 error per window"
        ),
        "rotation_deg": (
            "predicted V2M motion mapped with the train-global calibration versus GT camera; "
            "mean SO(3) geodesic error per window"
        ),
        "gt_calibration_translation_m": (
            "GT motion mapped with the train-global calibration versus GT camera"
        ),
        "gt_calibration_rotation_deg": (
            "GT motion mapped with the train-global calibration versus GT camera"
        ),
    }
    if len(keys) > len(HEAD_CAMERA_GLOBAL_METRIC_KEYS):
        metric_definitions.update({
            "oracle_actor_translation_m": (
                "predicted V2M motion mapped with a test-GT-fitted actor calibration versus "
                "GT camera; diagnostic only"
            ),
            "oracle_actor_rotation_deg": (
                "predicted V2M motion mapped with a test-GT-fitted actor calibration versus "
                "GT camera; diagnostic only"
            ),
            "gt_oracle_actor_translation_m": (
                "GT motion mapped with its test-GT-fitted actor calibration versus GT camera; "
                "test-label-derived oracle floor (in-sample when fit/eval windows match)"
            ),
            "gt_oracle_actor_rotation_deg": (
                "GT motion mapped with its test-GT-fitted actor calibration versus GT camera; "
                "test-label-derived oracle floor (in-sample when fit/eval windows match)"
            ),
        })
    return {
        "n": len(rows),
        "task": "video2motion",
        "representation": "relative upright-RGB camera action (translation + SO(3))",
        "absolute_pose_used": False,
        "aggregate": aggregate,
        "per_actor_aggregate": per_actor,
        "metric_definitions": metric_definitions,
        "per_sequence": rows,
    }


def _write_summary(out_root, summary):
    json.dump(summary, open(os.path.join(out_root, "summary.json"), "w"), indent=2, default=str)


def _print_summary(summary):
    print("\n" + "=" * 78)
    print(f"EVAL SUMMARY  |  {summary['out_root']}")
    print("=" * 78)
    # camera
    cam = summary.get("camera")
    if cam and cam.get("metrics_json") and os.path.isfile(cam["metrics_json"]):
        m = json.load(open(cam["metrics_json"]))["aggregate"]
        print("inverse_dynamics  "
              f"rot={m['rot_deg']['mean']:.2f}deg  dir_cos={m['trans_dir_cos']['mean']:.3f}  "
              f"scale={m['scale_ratio']['mean']:.2f}  ATE={m['ate_m']['mean']:.3f}m")
    elif cam:
        print(f"camera tasks -> {cam['eval_dir']} (video/metric files written)")
    # motion recon
    for t, info in summary.get("motion", {}).items():
        a = info["aggregate"]
        flag = "  [GENERATIVE - not a recon score]" if info["generative"] else ""
        print(f"{t:16s}  MPJPE={a['mpjpe_m']['mean']:.3f}m  "
              f"PA-MPJPE={a['pa_mpjpe_m']['mean']:.3f}m  "
              f"accel_err={a['accel_err']['mean']:.3f}  "
              f"(pred/gt jitter={a['accel_pred']['mean']/max(a['accel_gt']['mean'],1e-9):.1f}x)"
              f"{flag}")
    head_camera = summary.get("head_camera")
    if head_camera:
        aggregate = head_camera["aggregate"]
        print(
            "v2m head-camera   "
            f"trans={aggregate['translation_m']['mean']:.4f}m  "
            f"rot={aggregate['rotation_deg']['mean']:.3f}deg  "
            f"GT-floor={aggregate['gt_calibration_translation_m']['mean']:.4f}m/"
            f"{aggregate['gt_calibration_rotation_deg']['mean']:.3f}deg"
        )
        if "gt_oracle_actor_translation_m" in aggregate:
            print(
                "v2m actor oracle  "
                f"pred={aggregate['oracle_actor_translation_m']['mean']:.4f}m/"
                f"{aggregate['oracle_actor_rotation_deg']['mean']:.3f}deg  "
                f"GT-floor={aggregate['gt_oracle_actor_translation_m']['mean']:.4f}m/"
                f"{aggregate['gt_oracle_actor_rotation_deg']['mean']:.3f}deg  "
                "[TEST-GT-DERIVED, DIAGNOSTIC ONLY]"
            )
    # video gen
    for t, names in summary.get("video", {}).items():
        print(f"{t:16s}  {len(names)} clips generated")
    for t, info in summary.get("video_metrics", {}).items():
        aggregate = info["aggregate"]
        print(
            f"{t:16s}  PSNR={aggregate['psnr_db']['mean']:.3f}dB  "
            f"SSIM={aggregate['ssim']['mean']:.4f}  "
            f"LPIPS={aggregate['lpips_alex']['mean']:.4f}"
        )
    # viz
    print(f"\nviz files written ({len(summary.get('viz', []))}):")
    for p in summary.get("viz", [])[:40]:
        print(f"  {p}")
    print("=" * 78)


if __name__ == "__main__":
    main()
