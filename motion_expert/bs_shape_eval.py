"""Test-set shape-adherence eval: do generated motions keep the conditioned skeleton's bone lengths?

For each test entry (per BONES-SEED source: single / multi / natural), generate motion conditioned on
the real caption + that actor's `neutral_joints`, decode to joints, and measure **bone-length MAE** vs
the conditioned skeleton. Reports per-source aggregates (mean±std cm) + the GT motion's own bone-MAE as
the achievable floor, plus per-limb (arms/legs) errors.

Run (kimodo env, 1 GPU):
  PYTHONPATH=/home/jungbin_cho/kimodo_open:. python bs_shape_eval.py --ckpt <run>/latest.pt \
      --sources single multi --n 100 --out <run>/shape_eval_test.json
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import random

import numpy as np
import torch

import flow
import bs_native_flow
from bs_dataset import (DATA_ROOT, NATURAL_CSV, TEMPORAL_JSONL, MULTI_JSONL,
                        SPLIT_DIR, MEAN_PATH, STD_PATH)
from bs_model import MotionExpertInContext
from bs_normalization import resolve_checkpoint_normalization
from bs_text_cache import LLM2VecCache, DEFAULT_CACHE
from bs_viz import load_skeleton
from bs_sample import bone_lengths, group_len
from decode_uniego_torch import decode_joints
from uniego_layout import FEAT_DIM, canonicalize_frame0


# An entry = (uniego_npz_path, filename, seg_start_sec, seg_end_sec_or_None, caption)
def _split_relmap(split_path):
    m = {}
    for ln in open(split_path):
        rel = ln.strip()
        if rel:
            m[os.path.basename(rel)] = rel
    return m


def build_entries(source, split_path):
    """Per-source (single/multi/natural) entries restricted to a split — built directly from the
    BONES-SEED metadata so it tolerates empty sources (the kimodo dataset would raise instead)."""
    relmap = _split_relmap(split_path)
    out = []
    if source == "single":
        for ln in open(TEMPORAL_JSONL):
            o = json.loads(ln); f = o["filename"]
            if f not in relmap:
                continue
            p = os.path.join(DATA_ROOT, relmap[f] + ".npz")
            for ev in o.get("events", []):
                t = ev.get("description")
                if isinstance(t, str) and t.strip():
                    out.append((p, f, float(ev["start_time"]), float(ev["end_time"]), t.strip()))
    elif source == "multi":
        for ln in open(MULTI_JSONL):
            o = json.loads(ln); f = o["filename"]
            if f not in relmap:
                continue
            t = o.get("merged_description")
            if isinstance(t, str) and t.strip():
                p = os.path.join(DATA_ROOT, relmap[f] + ".npz")
                out.append((p, f, float(o["start_time"]), float(o["end_time"]), t.strip()))
    elif source == "natural":
        cols = ("content_natural_desc_1", "content_natural_desc_2",
                "content_natural_desc_3", "content_natural_desc_4")
        for row in csv.DictReader(open(NATURAL_CSV, newline="")):
            f = row["filename"]
            if f not in relmap:
                continue
            p = os.path.join(DATA_ROOT, relmap[f] + ".npz")
            seen = set()
            for c in cols:
                v = (row.get(c) or "").strip()
                if v and v not in seen:
                    seen.add(v)
                    out.append((p, f, 0.0, None, v))   # whole clip
    else:
        raise ValueError(source)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval_split", default=os.path.join(SPLIT_DIR, "test_content_split_paths.txt"))
    ap.add_argument("--sources", nargs="*", default=["single", "multi"])
    ap.add_argument("--n", type=int, default=100, help="samples per source")
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--mean", default=None, help="override checkpoint normalization")
    ap.add_argument("--std", default=None, help="override checkpoint normalization")
    ap.add_argument("--allow_stats_override", action="store_true")
    ap.add_argument("--cache_path", default=DEFAULT_CACHE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_index", default=None)
    args = ap.parse_args()
    dev = "cuda"

    cache = LLM2VecCache(args.cache_path, device=dev)
    parents, _ = load_skeleton()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    mean_np, std_np, normalization = resolve_checkpoint_normalization(
        ck,
        mean_override=args.mean,
        std_override=args.std,
        fallback_mean=MEAN_PATH,
        fallback_std=STD_PATH,
        allow_override=args.allow_stats_override,
    )
    mean = torch.from_numpy(mean_np).to(dev)
    std = torch.from_numpy(std_np).to(dev)
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
        f"[eval] {args.ckpt} (step {ck.get('step')}, pred={pred}, schedule={schedule}, "
        f"normalization={normalization['tag']}, "
        f"mean_sha256={normalization['mean_sha256'][:12]}, "
        f"std_sha256={normalization['std_sha256'][:12]})",
        flush=True,
    )
    null_H = cache.null(1)

    print(f"[eval] split={os.path.basename(args.eval_split)} sources={args.sources}", flush=True)

    @torch.no_grad()
    def generate(caption, njc):
        nj_t = torch.from_numpy(njc).float().to(dev).unsqueeze(0)
        H = cache.batch([caption])
        g = torch.Generator(device=dev).manual_seed(args.seed)
        x0 = sampler(model, H, None, nj_t, T=args.T, motion_dim=FEAT_DIM, steps=args.steps,
                     guidance=args.guidance, H_null=null_H, null_pad_mask=None, device=dev,
                     dtype=torch.float32, generator=g)
        return decode_joints((x0[0] * std + mean).unsqueeze(0))[0].cpu().numpy()

    rng = random.Random(args.seed)
    summary = {}
    for src in args.sources:
        entries = build_entries(src, args.eval_split)
        idxs = list(range(len(entries))); rng.shuffle(idxs)
        print(f"[eval] {src}: {len(entries)} entries in split", flush=True)
        gen_mae, gt_mae, arm_err, leg_err, statures = [], [], [], [], []
        used = 0
        for i in idxs:
            if used >= args.n:
                break
            p, f, sstart, send, text = entries[i]
            if text not in cache:
                continue
            try:
                z = np.load(p)
            except Exception:
                continue
            feats = z["features"].astype(np.float32); nj = z["neutral_joints"].astype(np.float32)
            sf = int(round(sstart * 20))
            ef = len(feats) if send is None else int(round(send * 20))
            sf = max(0, min(sf, len(feats))); ef = max(sf, min(ef, len(feats))); ef = min(ef, sf + args.T)
            win = feats[sf:ef]
            if win.shape[0] < 10 or not np.isfinite(win).all() or not np.isfinite(nj).all():
                continue
            njc = (nj - nj.mean(0, keepdims=True)).astype(np.float32)
            tgt = bone_lengths(njc[None], parents); m = tgt > 0

            gj = generate(text, njc)                                                # generated joints
            gtj = decode_joints(torch.from_numpy(canonicalize_frame0(win)).unsqueeze(0))[0].numpy()  # GT
            gen_mae.append(float(np.abs(bone_lengths(gj, parents) - tgt)[m].mean()))
            gt_mae.append(float(np.abs(bone_lengths(gtj, parents) - tgt)[m].mean()))
            arm_err.append(abs(group_len(gj, parents, "arms") - group_len(njc[None], parents, "arms")))
            leg_err.append(abs(group_len(gj, parents, "legs") - group_len(njc[None], parents, "legs")))
            statures.append(float(np.ptp(gj[..., 1])))
            used += 1
            if used % 20 == 0:
                print(f"  [{src}] {used}/{args.n}", flush=True)

        if used == 0:
            summary[src] = {"n": 0, "note": "no usable entries in this split"}
            print(f"[{src:8s}] n=0 — no entries in this split", flush=True)
            continue
        g = np.array(gen_mae) * 100; t = np.array(gt_mae) * 100
        summary[src] = {
            "n": used,
            "gen_bone_mae_cm_mean": round(float(g.mean()), 3), "gen_bone_mae_cm_std": round(float(g.std()), 3),
            "gen_bone_mae_cm_p90": round(float(np.percentile(g, 90)), 3),
            "gt_bone_mae_cm_mean": round(float(t.mean()), 3),
            "gen_arm_mae_cm": round(float(np.array(arm_err).mean() * 100), 2),
            "gen_leg_mae_cm": round(float(np.array(leg_err).mean() * 100), 2),
            "gen_stature_m_mean": round(float(np.mean(statures)), 3),
        }
        s = summary[src]
        print(f"[{src:8s}] n={used:3d}  gen bone-MAE = {s['gen_bone_mae_cm_mean']:.2f} ± "
              f"{s['gen_bone_mae_cm_std']:.2f} cm  (p90 {s['gen_bone_mae_cm_p90']:.2f}; GT floor "
              f"{s['gt_bone_mae_cm_mean']:.2f})  arms {s['gen_arm_mae_cm']:.2f}  legs {s['gen_leg_mae_cm']:.2f}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump({
            "ckpt": args.ckpt,
            "split": args.eval_split,
            "schedule": schedule,
            "normalization": normalization,
            "native_shift": a.get("native_shift") if schedule == "native" else None,
            "native_num_train_timesteps": (
                a.get("native_num_train_timesteps") if schedule == "native" else None
            ),
            "per_source": summary,
        }, open(args.out, "w"), indent=2)
        print(f"[eval] wrote {args.out}", flush=True)
    print("[eval] DONE", flush=True)


if __name__ == "__main__":
    main()
