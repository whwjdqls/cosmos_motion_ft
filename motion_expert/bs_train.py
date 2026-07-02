"""Train MotionExpertInContext on BONES-SEED uniego (flow matching, in-context llm2vec).

Only the model trains. Text comes from the cached llm2vec embeddings (no encoder in the loop);
the shape token comes from per-actor neutral_joints. Rectified-flow x0-prediction; loss =
feature-MSE(x0) + decoded centroid-relative joint L2 + decoded joint-velocity smoothness
(set w_joint=w_smooth=0 for a feature-only run — the decode is then skipped entirely).

In-training viz is **GT|gen side-by-side**, rendered with kimodo's renderer (`bs_viz`), matching
`kimodo/scripts/train.py:viz_step`.

Run (kimodo env, 1 GPU via srun/sbatch):
  PYTHONPATH=/home/jungbin_cho/kimodo_open python bs_train.py --smoke
  ... python bs_train.py --steps 200000 --batch_size 128 --run_name bs_incontext_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import flow
from bs_dataset import (BonesSeedUniegoDataset, collate, DATA_ROOT, NATURAL_CSV,
                        SPLIT_DIR, MEAN_PATH, STD_PATH)
from bs_model import MotionExpertInContext
from bs_text_cache import LLM2VecCache, DEFAULT_CACHE
from bs_viz import load_skeleton, render_pair
from decode_uniego_torch import decode_joints
from uniego_layout import FEAT_DIM, canonicalize_frame0

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = "/mnt/shared/jungbin_cho/cosmos_motion_ft_runs"


def masked_mse(a, b, valid):  # a,b [...,D] or [...,J,3]; valid [B,T] True=keep
    while valid.dim() < a.dim():
        valid = valid.unsqueeze(-1)
    se = ((a - b) ** 2) * valid
    return se.sum() / valid.expand_as(a).sum().clamp(min=1)


def load_viz_items(split_path, cache, n, dev, viz_frames):
    """Held-out (caption, neutral_joints, GT joints, length) for GT|gen viz.

    Reads the natural-description CSV directly (filename -> caption) and walks the split, so it
    does NOT need the 3-pool dataset (which requires every source non-empty — false for small
    splits). For each clip whose caption is in the cache: load its uniego features (unnormalized,
    canonicalized, capped to viz_frames), decode to GT world joints, and keep the actor's centered
    neutral_joints for conditioning the generation.
    """
    desc = {}
    with open(NATURAL_CSV, newline="") as f:
        for row in csv.DictReader(f):
            d = (row.get("content_natural_desc_1") or "").strip()
            if d:
                desc[row["filename"]] = d
    items = []
    with open(split_path) as f:
        for line in f:
            rel = line.strip()
            if not rel or rel.endswith("_M"):
                continue
            cap = desc.get(os.path.basename(rel))
            p = os.path.join(DATA_ROOT, rel + ".npz")
            if not cap or cap not in cache or not os.path.isfile(p):
                continue
            z = np.load(p)
            feats = z["features"].astype(np.float32)
            if not np.isfinite(feats).all():
                continue
            nj = z["neutral_joints"].astype(np.float32)
            nj = nj - nj.mean(axis=0, keepdims=True)
            length = int(min(feats.shape[0], viz_frames))
            gt = canonicalize_frame0(feats[:length])                  # unnormalized uniego features
            gt_joints = decode_joints(torch.from_numpy(gt).unsqueeze(0))[0].numpy()  # [L,30,3]
            items.append({"caption": cap, "nj": torch.from_numpy(nj).to(dev),
                          "gt_joints": gt_joints, "length": length})
            if len(items) >= n:
                break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--cfg_dropout", type=float, default=0.10, help="text-drop prob for CFG (shape never dropped)")
    ap.add_argument("--pred", choices=["x0", "v"], default="x0",
                    help="prediction target: x0 (clean motion) or v (velocity = eps - x0)")
    ap.add_argument("--w_feat", type=float, default=1.0)
    ap.add_argument("--w_joint", type=float, default=1.0)
    ap.add_argument("--w_smooth", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--viz_every", type=int, default=5000)
    ap.add_argument("--viz_n", type=int, default=4)
    ap.add_argument("--viz_frames", type=int, default=120)
    ap.add_argument("--viz_steps", type=int, default=50)
    ap.add_argument("--viz_guidance", type=float, default=2.0)
    ap.add_argument("--train_split", default=os.path.join(SPLIT_DIR, "train_split_paths.txt"))
    ap.add_argument("--viz_split", default=os.path.join(SPLIT_DIR, "test_content_split_paths_small.txt"))
    ap.add_argument("--mean", default=MEAN_PATH)
    ap.add_argument("--std", default=STD_PATH)
    ap.add_argument("--cache_path", default=DEFAULT_CACHE)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    mean = torch.from_numpy(np.load(args.mean)).float().to(dev)
    std = torch.from_numpy(np.load(args.std)).float().to(dev)
    cache = LLM2VecCache(args.cache_path, device=dev)
    print(f"[train] llm2vec cache: {len(cache)} captions, dim={cache.dim}", flush=True)

    model = MotionExpertInContext(d=args.d, n_layers=args.layers, heads=args.heads,
                                  ffn=args.ffn, text_dim=cache.dim, motion_dim=FEAT_DIM).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] MotionExpertInContext trainable = {n_train/1e6:.2f}M", flush=True)

    if args.smoke:
        out, tb, idx_cache = None, None, os.path.join(HERE, "_smoke_bs_index.json")
    else:
        name = args.run_name or f"bs_incontext_{time.strftime('%Y%m%d_%H%M%S')}"
        out = os.path.join(RUN_ROOT, name); os.makedirs(out, exist_ok=True)
        json.dump({**vars(args), "trainable_M": n_train}, open(os.path.join(out, "config.json"), "w"), indent=2)
        idx_cache = os.path.join(out, "bs_train_index.json")
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(out)
        except Exception:
            tb = None
        print(f"[train] run dir: {out}", flush=True)

    ds = BonesSeedUniegoDataset(args.train_split, mean_path=args.mean, std_path=args.std,
                                cache_index=idx_cache, train=True, seed=0)
    print(f"[train] dataset virtual_len={len(ds)}", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True, pin_memory=True,
                    persistent_workers=args.num_workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    def step_loss(batch):
        x0 = batch["motion"].to(dev)                         # [B,T,283] normalized
        nj = batch["neutral_joints"].to(dev)
        pad = batch["motion_pad_mask"].to(dev)
        valid = ~pad                                         # True = keep
        # CFG text-drop: replace dropped captions with "" (cache returns the null embedding).
        texts = batch["text"]
        keep = torch.rand(len(texts)) > args.cfg_dropout
        texts = [t if k else "" for t, k in zip(texts, keep.tolist())]
        text_emb = cache.batch(texts)                        # [B,1,4096]

        sigma = flow.sample_sigma_logitnormal(x0.shape[0], dev)
        x_sigma, eps = flow.add_noise(x0, sigma)
        out = model(x_sigma, sigma, text_emb, None, nj, motion_pad_mask=pad)

        if args.pred == "v":
            # velocity target v = eps - x0; feat loss is velocity-MSE. Reconstruct clean motion
            # x0_hat = x_sigma - sigma*v_hat for the decoded (pose/smooth) losses.
            v_target = eps - x0
            l_feat = masked_mse(out, v_target, valid)
            x0_hat = x_sigma - sigma.view(-1, 1, 1) * out
        else:
            x0_hat = out
            l_feat = masked_mse(x0_hat, x0, valid)
        if args.w_joint == 0.0 and args.w_smooth == 0.0:
            # feature-only: skip the decode entirely (no decode fwd/bwd, and avoids the
            # 0*inf=NaN trap if a degenerate decode produced inf).
            l_joint = x0.new_zeros(())
            l_smooth = x0.new_zeros(())
            loss = args.w_feat * l_feat
        else:
            j_hat = decode_joints(x0_hat * std + mean)           # [B,T,30,3]
            with torch.no_grad():
                j_gt = decode_joints(x0 * std + mean)
            rel_hat = j_hat - j_hat.mean(dim=2, keepdim=True)
            rel_gt = j_gt - j_gt.mean(dim=2, keepdim=True)
            l_joint = masked_mse(rel_hat, rel_gt, valid)         # centroid-relative pose
            vmask = valid[:, 1:] & valid[:, :-1]
            l_smooth = masked_mse(j_hat[:, 1:] - j_hat[:, :-1], j_gt[:, 1:] - j_gt[:, :-1], vmask)
            loss = args.w_feat * l_feat + args.w_joint * l_joint + args.w_smooth * l_smooth
        return loss, l_feat, l_joint, l_smooth

    # ---- held-out viz items (caption + GT joints + actor skeleton) ----
    parents, skip = load_skeleton()
    viz_items = []
    if args.viz_n > 0 and not args.smoke:
        viz_items = load_viz_items(args.viz_split, cache, args.viz_n, dev, args.viz_frames)
        print(f"[train] {len(viz_items)} viz captions (GT|gen side-by-side)", flush=True)

    sampler = flow.sample_v if args.pred == "v" else flow.sample_x0

    @torch.no_grad()
    def do_viz(step):
        model.eval()
        vdir = os.path.join(out, f"viz_step{step:06d}"); os.makedirs(vdir, exist_ok=True)
        null_H = cache.null(1)                               # [1,1,4096]
        for i, it in enumerate(viz_items):
            H = cache.batch([it["caption"]])                 # [1,1,4096]
            g = torch.Generator(device=dev).manual_seed(0)
            x0 = sampler(model, H, None, it["nj"].unsqueeze(0), T=it["length"],
                         motion_dim=FEAT_DIM, steps=args.viz_steps, guidance=args.viz_guidance,
                         H_null=null_H, null_pad_mask=None, device=dev,
                         dtype=torch.float32, generator=g)
            gen_joints = decode_joints((x0[0] * std + mean).unsqueeze(0))[0].cpu().numpy()
            render_pair(it["gt_joints"], gen_joints, parents,
                        os.path.join(vdir, f"{i}_{it['caption'][:30].replace(' ', '_')}.mp4"),
                        caption=it["caption"][:60], skip_joints=skip, fps=20)
        model.train()
        print(f"[viz] step {step} -> {vdir} ({len(viz_items)} clips: GT|gen)", flush=True)

    model.train()
    if args.smoke:
        ok = True
        for s, batch in enumerate(dl):
            for g in opt.param_groups:
                g["lr"] = lr_at(s)                      # mirror the real loop (warmup)
            loss, lf, lj, ls = step_loss(batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            fin = bool(torch.isfinite(loss)) and bool(torch.isfinite(gn))
            if fin:
                opt.step()
            print(f"[smoke] {s} loss={loss.item():.4f} feat={lf.item():.4f} joint={lj.item():.4f} "
                  f"smooth={ls.item():.4f} grad_norm={float(gn):.2f} finite={fin} "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            ok = ok and fin
            if s >= 4:
                break
        print("[smoke] PASS" if ok else "[smoke] FAIL")
        return

    step, t0 = 0, time.time()
    while step < args.steps:
        for batch in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, lf, lj, ls = step_loss(batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # Skip the update on a non-finite grad (a degenerate batch can NaN the decoded-joint
            # loss through the 200-frame cumulative SE(3) compose); don't let it poison the run.
            skipped = not bool(torch.isfinite(gn))
            if not skipped:
                opt.step()
            if step % args.log_every == 0:
                print(f"step {step:6d} loss={loss.item():.4f} feat={lf.item():.4f} joint={lj.item():.4f} "
                      f"smooth={ls.item():.4f} grad_norm={float(gn):.2f}{' SKIP' if skipped else ''} "
                      f"lr={lr_at(step):.2e} {(time.time()-t0)/(step+1):.3f}s/it", flush=True)
                if tb:
                    for k, v in [("loss", loss), ("feat", lf), ("joint", lj), ("smooth", ls)]:
                        tb.add_scalar(k, v.item(), step)
                    tb.add_scalar("grad_norm", float(gn) if torch.isfinite(gn) else 0.0, step)
            if step > 0 and step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "step": step, "args": vars(args)},
                           os.path.join(out, f"ckpt_step{step:06d}.pt"))
                torch.save({"model": model.state_dict(), "step": step, "args": vars(args)},
                           os.path.join(out, "latest.pt"))
                print(f"[ckpt] step {step}", flush=True)
            if viz_items and step > 0 and step % args.viz_every == 0:
                try:
                    do_viz(step)
                except Exception as e:
                    print(f"[viz] step {step} failed: {e}", flush=True)
            step += 1
            if step >= args.steps:
                break
    torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, os.path.join(out, "latest.pt"))
    print("[train] done")


if __name__ == "__main__":
    main()
