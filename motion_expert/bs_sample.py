"""Sample motion from text (+ skeleton) with a trained MotionExpertInContext (kimodo env).

text -> llm2vec cache -> MotionExpertInContext (flow x0 sampling, CFG) -> 283-D motion ->
unnormalize -> decode -> kimodo-style side-by-side mp4 (`bs_viz`) + .npy.

POC tests (all rendered as LEFT|RIGHT pairs, like kimodo's GT|gen viz):
  --ablation both   text-conditioned (left) vs "" null (right) — does the text matter?
  --shape_swap      same prompt, TALL (left) vs SHORT (right) actor skeleton — does shape matter?
                    (prints decoded bone-length MAE vs the conditioned neutral_joints)
  --sanity <npz>    decode + render a REAL uniego clip (no model) — validates the decode/render path.

There is NO live text encoder, so prompts must be REAL captions in the cache (default: pulled from
--prompts_split via the natural-description CSV); free-form strings KeyError.

Run (kimodo env, 1 GPU via srun):
  PYTHONPATH=/home/jungbin_cho/kimodo_open python bs_sample.py --ckpt <run>/ckpt_stepNNNNNN.pt \
      --out <run>/eval --ablation both
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import random

import numpy as np
import torch

import flow
import bs_native_flow
from bs_dataset import DATA_ROOT, MEAN_PATH, STD_PATH, NATURAL_CSV, SPLIT_DIR
from bs_model import MotionExpertInContext
from bs_text_cache import LLM2VecCache, DEFAULT_CACHE
from bs_viz import load_skeleton, render_pair
from decode_uniego_torch import decode_joints
from uniego_layout import FEAT_DIM, N_JOINTS, canonicalize_frame0


def load_prompts_from_split(split_path, cache, n):
    """Real bones-seed captions (natural desc) that are in the llm2vec cache."""
    import csv
    desc = {}
    with open(NATURAL_CSV, newline="") as f:
        for row in csv.DictReader(f):
            d = (row.get("content_natural_desc_1") or "").strip()
            if d:
                desc[row["filename"]] = d
    out = []
    with open(split_path) as f:
        for line in f:
            rel = line.strip()
            if not rel or rel.endswith("_M"):
                continue
            cap = desc.get(os.path.basename(rel))
            if cap and cap in cache and cap not in out:
                out.append(cap)
            if len(out) >= n:
                break
    return out


def load_centered_nj(npz_path: str) -> np.ndarray:
    nj = np.load(npz_path)["neutral_joints"].astype(np.float32)
    return nj - nj.mean(axis=0, keepdims=True)


def _stature(npz_path: str) -> float:
    try:
        nj = np.load(npz_path, mmap_mode="r")["neutral_joints"]
        return float(np.ptp(np.asarray(nj)[:, 1]))   # vertical extent of the rest pose
    except Exception:
        return float("nan")


def pick_tall_short(n_scan: int = 1500) -> tuple[str, str]:
    files = glob.glob(os.path.join(DATA_ROOT, "*", "*.npz"))
    random.Random(0).shuffle(files)
    scored = [(s, f) for f in files[:n_scan] if np.isfinite(s := _stature(f))]
    scored.sort()
    return scored[-1][1], scored[0][1]   # (tall, short)


def bone_lengths(joints_T: np.ndarray, parents) -> np.ndarray:
    """Mean per-bone length over frames. joints_T [T,J,3] -> [J] (root bone = 0)."""
    edges = [(j, int(p)) for j, p in enumerate(parents) if 0 <= int(p) < len(parents) and int(p) != j]
    out = np.zeros(len(parents), dtype=np.float32)
    for j, p in edges:
        out[j] = np.linalg.norm(joints_T[:, j] - joints_T[:, p], axis=-1).mean()
    return out


# SOMASkeleton30 limb groups (the joint whose INCOMING bone belongs to the group).
LIMB_GROUPS = {
    "arms": [11, 12, 13, 14, 15, 17, 18, 19, 20, 21],   # upper-arm, forearm, hand, fingertips (both)
    "legs": [22, 23, 24, 25, 26, 27, 28, 29],           # thigh, shin, foot, toe (both)
    "torso": [1, 2, 3],                                 # spine chain
    "neck": [4, 5, 6],                                  # neck + head
}
# Non-uniform "different skeleton" presets: per-group bone-length multipliers.
MORPHS = {
    "normal":    {},
    "long_legs": {"legs": 1.9},
    "long_arms": {"arms": 1.9},
    "gibbon":    {"arms": 2.0, "legs": 0.65},   # long arms + short legs
    "stilts":    {"legs": 2.3, "arms": 0.7},    # very long legs + short arms
}


def rescale_limbs(nj: np.ndarray, parents, group_factors: dict) -> np.ndarray:
    """Rebuild a rest pose [30,3] with named limb groups scaled, propagated to children.

    Scales each bone vector (joint - parent) by its group's factor, then walks the hierarchy
    (parents precede children in SOMASkeleton30) so e.g. lengthening the thigh carries the shin
    and foot down with it.
    """
    factor = np.ones(len(parents), dtype=np.float32)
    for g, f in group_factors.items():
        for j in LIMB_GROUPS[g]:
            factor[j] = f
    out = nj.copy().astype(np.float32)
    for j in range(len(parents)):
        p = int(parents[j])
        if p >= 0:
            out[j] = out[p] + (nj[j] - nj[p]) * factor[j]
    return out


def group_len(joints_T: np.ndarray, parents, group: str) -> float:
    """Mean bone length (m) over the joints of a limb group."""
    bl = bone_lengths(joints_T, parents)
    vals = [bl[j] for j in LIMB_GROUPS[group] if bl[j] > 0]
    return float(np.mean(vals)) if vals else 0.0


def rest_pose_motion(nj: np.ndarray, T: int) -> np.ndarray:
    """The conditioned rest pose as a static T-frame 'motion', grounded (lowest joint at y=0).

    Rendered on the LEFT next to the generated motion (right) so the conditioned skeleton's
    proportions can be compared directly against what the model produced — i.e. a visual check
    that the input skeleton is actually being used.
    """
    g = nj.copy().astype(np.float32)
    g[:, 1] -= g[:, 1].min()
    return np.repeat(g[None], int(T), axis=0)


def main(argv=None, parser_defaults=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompts", nargs="*", default=None, help="real cache captions; default = pull from --prompts_split")
    ap.add_argument("--prompts_split", default=os.path.join(SPLIT_DIR, "test_content_split_paths_small.txt"))
    ap.add_argument("--n_prompts", type=int, default=5)
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--sampler", choices=["auto", "legacy", "native"], default="auto",
                    help="auto follows the checkpoint schedule; legacy uses the original linear "
                         "1->0 sampler; native uses the shifted Cosmos 0.999->0 ladder.")
    ap.add_argument("--native_shift", type=float, default=None,
                    help="override the native shift recorded in the checkpoint")
    ap.add_argument("--native_num_train_timesteps", type=int, default=None,
                    help="override the native timestep range recorded in the checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mean", default=MEAN_PATH)
    ap.add_argument("--std", default=STD_PATH)
    ap.add_argument("--cache_path", default=DEFAULT_CACHE)
    ap.add_argument("--ablation", choices=["cond", "null", "both"], default="both")
    ap.add_argument("--skeleton_npz", default=None, help="uniego npz to take neutral_joints from")
    ap.add_argument("--shape_swap", action="store_true", help="sample each prompt with tall vs short actor")
    ap.add_argument("--tall_npz", default=None)
    ap.add_argument("--short_npz", default=None)
    ap.add_argument("--shape_scales", nargs="*", type=float, default=None,
                    help="EXTREME-skeleton sweep: scale a reference neutral_joints by each factor "
                         "(e.g. 0.5 0.75 1.0 1.5 2.0) — tests shape extrapolation beyond the data")
    ap.add_argument("--shape_morph", action="store_true",
                    help="NON-uniform skeletons (long_legs / long_arms / gibbon / stilts) — tests per-limb shape")
    ap.add_argument("--ref_npz", default=None, help="reference skeleton npz for --shape_scales/--shape_morph")
    ap.add_argument("--sanity", default=None, help="decode+render a REAL uniego npz (no model)")
    if parser_defaults:
        ap.set_defaults(**parser_defaults)
    args = ap.parse_args(argv)

    parents, skip = load_skeleton()

    # ---- sanity: decode + render a real clip (CPU OK) ----
    if args.sanity:
        out = args.out or os.path.join(os.path.dirname(args.sanity), "sanity")
        os.makedirs(out, exist_ok=True)
        feat = np.load(args.sanity)["features"].astype(np.float32)[:args.T]
        feat = canonicalize_frame0(feat)
        joints = decode_joints(torch.from_numpy(feat).unsqueeze(0))[0].numpy()
        render_pair(joints, None, parents, os.path.join(out, "sanity.mp4"),
                    caption=f"GT {os.path.basename(args.sanity)}", skip_joints=skip)
        print(f"[sanity] {args.sanity} -> {out}/sanity.mp4  joints {joints.shape} "
              f"y[{joints[...,1].min():.2f},{joints[...,1].max():.2f}]")
        return

    assert args.ckpt and args.out, "--ckpt and --out required (unless --sanity)"
    dev = "cuda"
    mean = torch.from_numpy(np.load(args.mean)).float().to(dev)
    std = torch.from_numpy(np.load(args.std)).float().to(dev)
    cache = LLM2VecCache(args.cache_path, device=dev)

    prompts = args.prompts or load_prompts_from_split(args.prompts_split, cache, args.n_prompts)
    if not prompts:
        raise SystemExit("no prompts: pass real cache captions via --prompts, or a --prompts_split with natural descriptions")
    print(f"[sample] {len(prompts)} prompts (real captions from cache): "
          + " | ".join(p[:32] for p in prompts), flush=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck.get("args", {})
    model = MotionExpertInContext(d=a.get("d", 512), n_layers=a.get("layers", 8),
                                  heads=a.get("heads", 8), ffn=a.get("ffn", 2048),
                                  text_dim=cache.dim, motion_dim=FEAT_DIM).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    pred = a.get("pred", "x0")
    schedule = a.get("schedule", "legacy")
    sampler_name = args.sampler
    if sampler_name == "auto":
        sampler_name = "native" if schedule == "native" else "legacy"
    native_shift = (
        args.native_shift if args.native_shift is not None
        else float(a.get("native_shift", bs_native_flow.DEFAULT_SHIFT))
    )
    native_num_train_timesteps = (
        args.native_num_train_timesteps if args.native_num_train_timesteps is not None
        else int(a.get("native_num_train_timesteps", bs_native_flow.DEFAULT_NUM_TRAIN_TIMESTEPS))
    )
    if sampler_name == "native":
        if pred != "x0":
            raise SystemExit("the native BONES sampler requires an x0-prediction checkpoint")
        sampler = functools.partial(
            bs_native_flow.sample_x0,
            native_shift=native_shift,
            native_num_train_timesteps=native_num_train_timesteps,
        )
    else:
        sampler = flow.sample_v if pred == "v" else flow.sample_x0
    sample_meta = {
        "training_schedule": schedule,
        "sampler": sampler_name,
        "native_shift": native_shift if sampler_name == "native" else None,
        "native_num_train_timesteps": (
            native_num_train_timesteps if sampler_name == "native" else None
        ),
    }
    print(
        f"[sample] loaded {args.ckpt} (step {ck.get('step')}, pred={pred}, "
        f"training_schedule={schedule}, sampler={sampler_name}, shift={native_shift:g})",
        flush=True,
    )

    os.makedirs(args.out, exist_ok=True)
    null_H = cache.null(1)

    def sample_one(prompt, nj_np, mode):
        nj_t = torch.from_numpy(nj_np).float().to(dev).unsqueeze(0)
        H = cache.batch([prompt]) if mode == "cond" else null_H
        g = torch.Generator(device=dev).manual_seed(args.seed)
        x0 = sampler(model, H, None, nj_t, T=args.T, motion_dim=FEAT_DIM,
                            steps=args.steps, guidance=(args.guidance if mode == "cond" else 1.0),
                            H_null=null_H, null_pad_mask=None, device=dev,
                            dtype=torch.float32, generator=g)
        feat = (x0[0] * std + mean)
        joints = decode_joints(feat.unsqueeze(0))[0].cpu().numpy()
        return feat.cpu().numpy().astype(np.float32), joints

    manifest = []

    # ---- NON-uniform skeletons: long legs / long arms / gibbon / stilts ----
    if args.shape_morph:
        ref = load_centered_nj(args.ref_npz or sorted(glob.glob(os.path.join(DATA_ROOT, "*", "*.npz")))[0])
        morphs = list(MORPHS.keys())
        print(f"[morph] ref arms={group_len(ref[None], parents, 'arms')*100:.0f}cm "
              f"legs={group_len(ref[None], parents, 'legs')*100:.0f}cm; morphs {morphs}", flush=True)
        for prompt in prompts:
            slug = "".join(c if c.isalnum() else "_" for c in prompt)[:28]
            jts = {}
            for m in morphs:
                nj = rescale_limbs(ref, parents, MORPHS[m])
                nj = (nj - nj.mean(axis=0, keepdims=True)).astype(np.float32)
                feat, joints = sample_one(prompt, nj, "cond")
                np.save(os.path.join(args.out, f"{slug}__{m}.npy"), feat)
                jts[m] = joints
                ga, gl = group_len(joints, parents, "arms"), group_len(joints, parents, "legs")
                ta, tl = group_len(nj[None], parents, "arms"), group_len(nj[None], parents, "legs")
                print(f"  [{m:10s}] '{prompt[:22]}'  arms gen={ga*100:4.0f}cm (target {ta*100:3.0f})  "
                      f"legs gen={gl*100:4.0f}cm (target {tl*100:3.0f})", flush=True)
                manifest.append({"prompt": prompt, "morph": m,
                                 "arms_gen_cm": round(ga*100, 1), "arms_tgt_cm": round(ta*100, 1),
                                 "legs_gen_cm": round(gl*100, 1), "legs_tgt_cm": round(tl*100, 1)})
                render_pair(rest_pose_motion(nj, joints.shape[0]), joints, parents,
                            os.path.join(args.out, f"{slug}__{m}.mp4"),
                            caption=f"{m}  |  L=input skeleton  R=generated", skip_joints=skip, camera="fixed")
            if "long_legs" in jts and "long_arms" in jts:
                render_pair(jts["long_legs"], jts["long_arms"], parents,
                            os.path.join(args.out, f"{slug}__legs_vs_arms.mp4"),
                            caption=f"{prompt[:26]} | L=long-legs R=long-arms (both gen)", skip_joints=skip)
        json.dump({"ckpt": args.ckpt, **sample_meta,
                   "morphs": {k: MORPHS[k] for k in morphs}, "samples": manifest},
                  open(os.path.join(args.out, "shape_morph_manifest.json"), "w"), indent=2)
        print(f"[sample] shape_morph -> {args.out}")
        return

    # ---- EXTREME skeletons: scale a reference rest pose by each factor ----
    if args.shape_scales:
        ref = load_centered_nj(args.ref_npz or sorted(glob.glob(os.path.join(DATA_ROOT, "*", "*.npz")))[0])
        ref_stat = float(np.ptp(ref[:, 1]))
        scales = sorted(args.shape_scales)
        print(f"[extreme] ref stature {ref_stat:.2f} m; scales {scales} "
              f"-> targets {[round(ref_stat*s,2) for s in scales]} m", flush=True)
        for prompt in prompts:
            slug = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
            jts = {}
            for sc in scales:
                nj = (ref * sc).astype(np.float32)
                feat, joints = sample_one(prompt, nj, "cond")
                np.save(os.path.join(args.out, f"{slug}__x{sc:g}.npy"), feat)
                jts[sc] = joints
                gen_bl = bone_lengths(joints, parents)
                tgt_bl = bone_lengths(nj[None], parents)
                err = np.abs(gen_bl - tgt_bl)[tgt_bl > 0].mean()
                print(f"  [x{sc:<4g}] '{prompt[:28]}' target={ref_stat*sc:.2f}m  gen stature={np.ptp(joints[...,1]):.2f}m  "
                      f"bone-len MAE={err*100:.1f} cm", flush=True)
                manifest.append({"prompt": prompt, "scale": sc, "target_m": round(ref_stat*sc, 3),
                                 "gen_stature_m": round(float(np.ptp(joints[..., 1])), 3),
                                 "bone_mae_cm": round(float(err*100), 2)})
                render_pair(rest_pose_motion(nj, joints.shape[0]), joints, parents,
                            os.path.join(args.out, f"{slug}__x{sc:g}.mp4"),
                            caption=f"x{sc:g}  |  L=input skeleton  R=generated", skip_joints=skip, camera="fixed")
            lo, hi = scales[0], scales[-1]
            render_pair(jts[lo], jts[hi], parents, os.path.join(args.out, f"{slug}__extremes.mp4"),
                        caption=f"{prompt[:32]} | L=x{lo:g} R=x{hi:g} (both gen)", skip_joints=skip)
        json.dump({"ckpt": args.ckpt, **sample_meta, "ref_stature_m": ref_stat,
                   "scales": scales, "samples": manifest},
                  open(os.path.join(args.out, "shape_scales_manifest.json"), "w"), indent=2)
        print(f"[sample] shape_scales -> {args.out}")
        return

    # ---- shape-awareness test: same prompt, tall (left) vs short (right) skeleton ----
    if args.shape_swap:
        if args.tall_npz and args.short_npz:
            tall, short = args.tall_npz, args.short_npz
        else:
            tall, short = pick_tall_short()
        njs = {"tall": load_centered_nj(tall), "short": load_centered_nj(short)}
        for prompt in prompts:
            slug = "".join(c if c.isalnum() else "_" for c in prompt)[:36]
            jts = {}
            for name, nj in njs.items():
                feat, joints = sample_one(prompt, nj, "cond")
                np.save(os.path.join(args.out, f"{slug}__{name}.npy"), feat)
                jts[name] = joints
                gen_bl = bone_lengths(joints, parents)
                tgt_bl = bone_lengths(nj[None], parents)          # conditioned rest-pose bone lengths
                err = np.abs(gen_bl - tgt_bl)[tgt_bl > 0].mean()
                print(f"  [{name:5s}] '{prompt[:34]}' bone-len MAE vs skeleton = {err*100:.1f} cm "
                      f"(gen stature {np.ptp(joints[...,1]):.2f} m)", flush=True)
                manifest.append({"prompt": prompt, "shape": name, "bone_mae_cm": float(err * 100)})
                render_pair(rest_pose_motion(nj, joints.shape[0]), joints, parents,
                            os.path.join(args.out, f"{slug}__{name}.mp4"),
                            caption=f"{name}  |  L=input skeleton  R=generated", skip_joints=skip, camera="fixed")
            render_pair(jts["tall"], jts["short"], parents, os.path.join(args.out, f"{slug}__tall_vs_short.mp4"),
                        caption=f"{prompt[:42]} | L=tall R=short (both gen)", skip_joints=skip)
        json.dump({"ckpt": args.ckpt, **sample_meta, "tall": tall, "short": short,
                   "samples": manifest},
                  open(os.path.join(args.out, "shape_swap_manifest.json"), "w"), indent=2)
        print(f"[sample] shape_swap -> {args.out}")
        return

    # ---- text ablation: cond (left) vs null (right) ----
    if args.skeleton_npz:
        nj = load_centered_nj(args.skeleton_npz)
    else:
        nj = load_centered_nj(sorted(glob.glob(os.path.join(DATA_ROOT, "*", "*.npz")))[0])
    np.save(os.path.join(args.out, "skeleton_neutral_joints.npy"), nj)

    modes = ["cond", "null"] if args.ablation == "both" else [args.ablation]
    for prompt in prompts:
        slug = "".join(c if c.isalnum() else "_" for c in prompt)[:36]
        jts = {}
        for mode in modes:
            feat, joints = sample_one(prompt, nj, mode)
            np.save(os.path.join(args.out, f"{slug}__{mode}.npy"), feat)
            jts[mode] = joints
            manifest.append({"prompt": prompt, "mode": mode})
            print(f"  [{mode}] '{prompt[:34]}'  (feat range {feat.min():.2f}..{feat.max():.2f})", flush=True)
        if args.ablation == "both":
            render_pair(jts["cond"], jts["null"], parents, os.path.join(args.out, f"{slug}.mp4"),
                        caption=f"{prompt[:46]}  |  L=cond  R=null", skip_joints=skip)
        else:
            render_pair(None, jts[modes[0]], parents, os.path.join(args.out, f"{slug}__{modes[0]}.mp4"),
                        caption=f"{prompt[:50]} [{modes[0]}]", skip_joints=skip)
    json.dump({"n_joints": N_JOINTS, "feat_dim": FEAT_DIM, "ckpt": args.ckpt,
               "step": ck.get("step"), **sample_meta, "samples": manifest},
              open(os.path.join(args.out, "samples_manifest.json"), "w"), indent=2)
    print(f"[sample] wrote {len(manifest)} samples to {args.out}")


if __name__ == "__main__":
    main()
