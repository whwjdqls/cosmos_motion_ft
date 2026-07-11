"""Shape-awareness eval on NymeriaPlus proportional data (model trained on BONES-SEED).

Does the bones-seed-trained model honor NYMERIA actors' skeletons? Nymeria captions are NOT in the
bones-seed llm2vec cache, so each nymeria skeleton is paired with a random in-cache caption (shape
adherence is caption-independent — verified on bones-seed). We condition on the nymeria
`neutral_joints`, generate, decode (with the bones-seed stats the model was trained on), and measure
bone-length MAE vs the conditioned nymeria skeleton, with the nymeria GT decode as the floor.

Run (kimodo env, 1 GPU):
  PYTHONPATH=/home/jungbin_cho/kimodo_open:. python bs_shape_eval_nymeria.py --ckpt <run>/latest.pt \
      --split test --n 100 --out <run>/shape_eval_nymeria.json
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
from bs_dataset import MEAN_PATH, STD_PATH
from bs_model import MotionExpertInContext
from bs_text_cache import LLM2VecCache, DEFAULT_CACHE
from bs_viz import load_skeleton
from bs_sample import bone_lengths, group_len
from decode_uniego_torch import decode_joints
from uniego_layout import FEAT_DIM, canonicalize_frame0

NYMERIA = "/mnt/shared/jungbin_cho/nymeriaplus_proportional"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--mean", default=MEAN_PATH)
    ap.add_argument("--std", default=STD_PATH)
    ap.add_argument("--cache_path", default=DEFAULT_CACHE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    args = ap.parse_args()
    dev = "cuda"

    mean = torch.from_numpy(np.load(args.mean)).float().to(dev)
    std = torch.from_numpy(np.load(args.std)).float().to(dev)
    cache = LLM2VecCache(args.cache_path, device=dev)
    parents, _ = load_skeleton()
    ck = torch.load(args.ckpt, map_location="cpu"); a = ck.get("args", {})
    model = MotionExpertInContext(d=a.get("d", 512), n_layers=a.get("layers", 8), heads=a.get("heads", 8),
                                  ffn=a.get("ffn", 2048), text_dim=cache.dim, motion_dim=FEAT_DIM).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    pred = a.get("pred", "x0")
    schedule = a.get("schedule", "legacy")
    if schedule == "native":
        if pred != "x0":
            raise ValueError("native schedule checkpoints must predict x0")
        sampler = functools.partial(
            bs_native_flow.sample_x0,
            native_shift=float(a.get("native_shift", bs_native_flow.DEFAULT_SHIFT)),
            native_num_train_timesteps=int(
                a.get("native_num_train_timesteps", bs_native_flow.DEFAULT_NUM_TRAIN_TIMESTEPS)
            ),
        )
    else:
        sampler = flow.sample_v if pred == "v" else flow.sample_x0
    print(
        f"[eval] {args.ckpt} (step {ck.get('step')}, pred={pred}, schedule={schedule}) "
        f"on nymeria split={args.split}",
        flush=True,
    )
    null_H = cache.null(1)

    sp = json.load(open(os.path.join(NYMERIA, "train_test_split.json")))
    if args.split in ("test", "train"):
        files = [os.path.join(NYMERIA, "uniego_rep", r + ".npz") for r in sp[args.split]]
        files = [f for f in files if os.path.isfile(f)]
    else:
        files = sorted(glob.glob(os.path.join(NYMERIA, "uniego_rep", "*", "*.npz")))
    rng = random.Random(args.seed); rng.shuffle(files)
    caps = cache.captions  # index 0 is "" (null)
    print(f"[eval] {len(files)} nymeria clips in split", flush=True)

    @torch.no_grad()
    def generate(caption, njc):
        nj_t = torch.from_numpy(njc).float().to(dev).unsqueeze(0)
        H = cache.batch([caption])
        g = torch.Generator(device=dev).manual_seed(args.seed)
        x0 = sampler(model, H, None, nj_t, T=args.T, motion_dim=FEAT_DIM, steps=args.steps,
                     guidance=args.guidance, H_null=null_H, null_pad_mask=None, device=dev,
                     dtype=torch.float32, generator=g)
        return decode_joints((x0[0] * std + mean).unsqueeze(0))[0].cpu().numpy()

    gen_mae, gt_mae, arm_err, leg_err, statures = [], [], [], [], []
    used = 0
    for f in files:
        if used >= args.n:
            break
        try:
            z = np.load(f)
        except Exception:
            continue
        feats = z["features"].astype(np.float32); nj = z["neutral_joints"].astype(np.float32)
        if feats.ndim != 2 or feats.shape[1] != FEAT_DIM or nj.shape != (30, 3) or not np.isfinite(nj).all():
            continue
        T = min(args.T, feats.shape[0])
        if T < 10:
            continue
        s0 = rng.randint(0, feats.shape[0] - T)
        win = feats[s0:s0 + T]
        if not np.isfinite(win).all():
            continue
        njc = (nj - nj.mean(0, keepdims=True)).astype(np.float32)
        tgt = bone_lengths(njc[None], parents); m = tgt > 0
        cap = caps[rng.randrange(1, len(caps))]                  # random in-cache (bones-seed) caption

        gj = generate(cap, njc)
        gtj = decode_joints(torch.from_numpy(canonicalize_frame0(win)).unsqueeze(0))[0].numpy()
        gen_mae.append(float(np.abs(bone_lengths(gj, parents) - tgt)[m].mean()))
        gt_mae.append(float(np.abs(bone_lengths(gtj, parents) - tgt)[m].mean()))
        arm_err.append(abs(group_len(gj, parents, "arms") - group_len(njc[None], parents, "arms")))
        leg_err.append(abs(group_len(gj, parents, "legs") - group_len(njc[None], parents, "legs")))
        statures.append(float(np.ptp(gj[..., 1])))
        used += 1
        if used % 20 == 0:
            print(f"  {used}/{args.n}", flush=True)

    g = np.array(gen_mae) * 100; t = np.array(gt_mae) * 100
    res = {"dataset": "nymeriaplus_proportional", "split": args.split, "n": used,
           "schedule": schedule,
           "native_shift": a.get("native_shift") if schedule == "native" else None,
           "native_num_train_timesteps": a.get("native_num_train_timesteps") if schedule == "native" else None,
           "gen_bone_mae_cm_mean": round(float(g.mean()), 3), "gen_bone_mae_cm_std": round(float(g.std()), 3),
           "gen_bone_mae_cm_p90": round(float(np.percentile(g, 90)), 3),
           "gt_bone_mae_cm_mean": round(float(t.mean()), 3),
           "gen_arm_mae_cm": round(float(np.array(arm_err).mean() * 100), 2),
           "gen_leg_mae_cm": round(float(np.array(leg_err).mean() * 100), 2),
           "gen_stature_m_mean": round(float(np.mean(statures)), 3)}
    print(f"[nymeria {args.split}] n={used}  gen bone-MAE = {res['gen_bone_mae_cm_mean']:.2f} ± "
          f"{res['gen_bone_mae_cm_std']:.2f} cm  (p90 {res['gen_bone_mae_cm_p90']:.2f}; GT floor "
          f"{res['gt_bone_mae_cm_mean']:.2f})  arms {res['gen_arm_mae_cm']:.2f}  legs {res['gen_leg_mae_cm']:.2f}", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[eval] wrote {args.out}", flush=True)
    print("[eval] DONE", flush=True)


if __name__ == "__main__":
    main()
