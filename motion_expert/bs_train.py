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
import functools
import json
import math
import os
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

import flow
import bs_native_flow
from bs_losses import contact_aware_losses, masked_mse
from bs_dataset import (BonesSeedUniegoDataset, collate, DATA_ROOT, NATURAL_CSV,
                        SPLIT_DIR, MEAN_PATH, STD_PATH)
from bs_model import MotionExpertInContext
from bs_text_cache import LLM2VecCache, DEFAULT_CACHE
from bs_viz import load_skeleton, render_pair
from decode_uniego_torch import decode_joints
from uniego_layout import FEAT_DIM, canonicalize_frame0

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = "/mnt/shared/jungbin_cho/cosmos_motion_ft_runs"


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


def main(argv=None, parser_defaults=None):
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
    ap.add_argument("--schedule", choices=["legacy", "native"], default="legacy",
                    help="legacy uses unshifted logit-normal training sigma and a linear 1->0 "
                         "sampler. native uses Cosmos shifted logit-normal sigma and its shifted "
                         "0.999->0 inference ladder. Native mode currently requires --pred x0.")
    ap.add_argument("--native_shift", type=float, default=bs_native_flow.DEFAULT_SHIFT,
                    help="Cosmos rational flow shift for --schedule native (Phase-2 default: 3).")
    ap.add_argument("--native_num_train_timesteps", type=int,
                    default=bs_native_flow.DEFAULT_NUM_TRAIN_TIMESTEPS,
                    help="native discrete timestep range used by the inference ladder.")
    ap.add_argument("--w_feat", type=float, default=1.0)
    ap.add_argument("--w_joint", type=float, default=1.0)
    ap.add_argument("--w_smooth", type=float, default=5.0)
    ap.add_argument("--w_contact", type=float, default=0.0)
    ap.add_argument("--w_foot_vel", type=float, default=0.0)
    ap.add_argument("--w_foot_height", type=float, default=0.0)
    ap.add_argument("--contact_logit_scale", type=float, default=10.0)
    ap.add_argument("--motion_fps", type=float, default=20.0)
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
    ap.add_argument(
        "--inline_eval_every",
        type=int,
        default=0,
        help="run the full in-process C45 overview evaluation every N steps; 0 disables it",
    )
    ap.add_argument("--inline_eval_batch_size", type=int, default=16)
    ap.add_argument("--inline_eval_steps", type=int, default=35)
    ap.add_argument("--inline_eval_guidance", type=float, default=2.0)
    ap.add_argument("--inline_eval_max_cases", type=int, default=0)
    ap.add_argument(
        "--inline_eval_solver",
        choices=["euler", "heun", "unipc"],
        default="unipc",
    )
    ap.add_argument(
        "--inline_eval_shape_counterfactual",
        choices=["none", "farthest"],
        default="farthest",
        help="same-text/same-noise shape intervention included in each inline evaluation",
    )
    ap.add_argument("--inline_eval_tmr_ckpt", default=None)
    ap.add_argument("--inline_eval_tmr_stats", default=None)
    ap.add_argument("--inline_eval_text_cache", default=None)
    ap.add_argument("--train_split", default=os.path.join(SPLIT_DIR, "train_split_paths.txt"))
    ap.add_argument("--viz_split", default=os.path.join(SPLIT_DIR, "test_content_split_paths_small.txt"))
    ap.add_argument("--mean", default=MEAN_PATH)
    ap.add_argument("--std", default=STD_PATH)
    ap.add_argument("--cache_path", default=DEFAULT_CACHE)
    ap.add_argument("--index_cache", default=None,
                    help="optional existing BONES segment-index JSON. Reusing the baseline index "
                         "avoids rebuilding a several-hundred-MB cache for schedule ablations.")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--smoke", action="store_true")
    if parser_defaults:
        ap.set_defaults(**parser_defaults)
    args = ap.parse_args(argv)
    if args.schedule == "native" and args.pred != "x0":
        ap.error("--schedule native is the clean-x0 Phase-2 POC and requires --pred x0")
    if args.native_shift <= 0:
        ap.error("--native_shift must be positive")
    if args.native_num_train_timesteps <= 1:
        ap.error("--native_num_train_timesteps must be greater than one")
    loss_weights = (
        args.w_feat,
        args.w_joint,
        args.w_smooth,
        args.w_contact,
        args.w_foot_vel,
        args.w_foot_height,
    )
    if any(weight < 0.0 for weight in loss_weights):
        ap.error("loss weights must be non-negative")
    if args.contact_logit_scale <= 0.0 or args.motion_fps <= 0.0:
        ap.error("--contact_logit_scale and --motion_fps must be positive")
    if args.inline_eval_every < 0 or args.inline_eval_max_cases < 0:
        ap.error("inline evaluation intervals and case limits must be non-negative")
    if args.inline_eval_batch_size <= 0 or args.inline_eval_steps <= 0:
        ap.error("inline evaluation batch size and steps must be positive")
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
    print(
        f"[train] schedule={args.schedule} pred={args.pred} "
        f"native_shift={args.native_shift:g} "
        f"native_num_train_timesteps={args.native_num_train_timesteps} "
        f"loss_weights=feat:{args.w_feat:g},joint:{args.w_joint:g},smooth:{args.w_smooth:g},"
        f"contact:{args.w_contact:g},foot_vel:{args.w_foot_vel:g},"
        f"foot_height:{args.w_foot_height:g}",
        flush=True,
    )

    if args.smoke:
        out, tb = None, None
        idx_cache = args.index_cache or os.path.join(HERE, "_smoke_bs_index.json")
    else:
        default_prefix = "bs_native_x0" if args.schedule == "native" else "bs_incontext"
        name = args.run_name or f"{default_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
        out = os.path.join(RUN_ROOT, name); os.makedirs(out, exist_ok=True)
        json.dump({**vars(args), "trainable_M": n_train}, open(os.path.join(out, "config.json"), "w"), indent=2)
        idx_cache = args.index_cache or os.path.join(out, "bs_train_index.json")
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

        if args.schedule == "native":
            sigma = bs_native_flow.sample_train_sigma(
                x0.shape[0], dev, shift=args.native_shift, dtype=x0.dtype
            )
        else:
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
        needs_decode = any(
            weight > 0.0
            for weight in (
                args.w_joint,
                args.w_smooth,
                args.w_contact,
                args.w_foot_vel,
                args.w_foot_height,
            )
        )
        if not needs_decode:
            # Feature-only: skip decode forward/backward and avoid a possible
            # 0*inf=NaN from a degenerate cumulative transform.
            l_joint = x0.new_zeros(())
            l_smooth = x0.new_zeros(())
            l_contact = x0.new_zeros(())
            l_foot_vel = x0.new_zeros(())
            l_foot_height = x0.new_zeros(())
        else:
            j_hat = decode_joints(x0_hat * std + mean)           # [B,T,30,3]
            with torch.no_grad():
                j_gt = decode_joints(x0 * std + mean)
            rel_hat = j_hat - j_hat.mean(dim=2, keepdim=True)
            rel_gt = j_gt - j_gt.mean(dim=2, keepdim=True)
            l_joint = masked_mse(rel_hat, rel_gt, valid)         # centroid-relative pose
            vmask = valid[:, 1:] & valid[:, :-1]
            l_smooth = masked_mse(j_hat[:, 1:] - j_hat[:, :-1], j_gt[:, 1:] - j_gt[:, :-1], vmask)
            if any(
                weight > 0.0
                for weight in (args.w_contact, args.w_foot_vel, args.w_foot_height)
            ):
                l_contact, l_foot_vel, l_foot_height = contact_aware_losses(
                    x0_hat,
                    x0,
                    j_hat,
                    valid,
                    mean,
                    std,
                    fps=args.motion_fps,
                    contact_logit_scale=args.contact_logit_scale,
                )
            else:
                l_contact = x0.new_zeros(())
                l_foot_vel = x0.new_zeros(())
                l_foot_height = x0.new_zeros(())
        loss = (
            args.w_feat * l_feat
            + args.w_joint * l_joint
            + args.w_smooth * l_smooth
            + args.w_contact * l_contact
            + args.w_foot_vel * l_foot_vel
            + args.w_foot_height * l_foot_height
        )
        return (
            loss,
            l_feat,
            l_joint,
            l_smooth,
            l_contact,
            l_foot_vel,
            l_foot_height,
            sigma.detach(),
        )

    # ---- held-out viz items (caption + GT joints + actor skeleton) ----
    parents, skip = load_skeleton()
    viz_items = []
    if args.viz_n > 0 and not args.smoke:
        viz_items = load_viz_items(args.viz_split, cache, args.viz_n, dev, args.viz_frames)
        print(f"[train] {len(viz_items)} viz captions (GT|gen side-by-side)", flush=True)

    if args.schedule == "native":
        sampler = functools.partial(
            bs_native_flow.sample_x0_unipc,
            native_shift=args.native_shift,
            native_num_train_timesteps=args.native_num_train_timesteps,
        )
    else:
        sampler = flow.sample_v if args.pred == "v" else flow.sample_x0

    inline_evaluator = None
    last_inline_eval_step = None
    if args.inline_eval_every > 0 and not args.smoke:
        from bs_tmr_eval import InlineShapeTMREvaluator

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all()
        inline_kwargs = {
            "output_dir": os.path.join(out, "inline_eval"),
            "generator_mean": mean,
            "generator_std": std,
            "batch_size": args.inline_eval_batch_size,
            "steps": args.inline_eval_steps,
            "guidance": args.inline_eval_guidance,
            "max_cases": args.inline_eval_max_cases,
            "native_solver": args.inline_eval_solver,
            "shape_counterfactual": args.inline_eval_shape_counterfactual,
            "device": dev,
        }
        if args.inline_eval_tmr_ckpt:
            inline_kwargs["tmr_ckpt"] = args.inline_eval_tmr_ckpt
        if args.inline_eval_tmr_stats:
            inline_kwargs["tmr_stats"] = args.inline_eval_tmr_stats
        if args.inline_eval_text_cache:
            inline_kwargs["text_cache"] = args.inline_eval_text_cache
        try:
            inline_evaluator = InlineShapeTMREvaluator(**inline_kwargs)
        finally:
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state_all(cuda_rng_states)
        print(
            f"[train] inline C45 eval every {args.inline_eval_every} steps "
            f"({args.inline_eval_solver}-{args.inline_eval_steps}, "
            f"cases={len(inline_evaluator.cases)}, "
            f"shape_cf={args.inline_eval_shape_counterfactual})",
            flush=True,
        )

    def do_inline_eval(eval_step: int, checkpoint_path: str) -> bool:
        nonlocal last_inline_eval_step
        if inline_evaluator is None:
            return False
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all()
        try:
            result = inline_evaluator.evaluate(
                model,
                vars(args),
                eval_step,
                checkpoint_path,
            )
            last_inline_eval_step = eval_step
            if tb:
                tb.add_scalar("eval/protocol_R03", result["tmr"]["TMR/t2m_R/R03"], eval_step)
                tb.add_scalar("eval/plain_R03", result["plain_t2m_gen"]["R03"], eval_step)
                tb.add_scalar("eval/fid_gen_gt", result["tmr"]["TMR/FID/gen_gt"], eval_step)
                tb.add_scalar(
                    "eval/contact_skate_cm_s",
                    result["physical_20fps"]["foot_skate_from_pred_contacts"] * 100.0,
                    eval_step,
                )
                tb.add_scalar(
                    "eval/contact_consistency",
                    result["physical_20fps"]["foot_contact_consistency"],
                    eval_step,
                )
                tb.add_scalar(
                    "eval/shape_bone_mae_cm",
                    result["shape"]["bone_length_mae_cm_mean"],
                    eval_step,
                )
                population = result["shape"]["population_tracking"]
                tb.add_scalar(
                    "eval/shape_centered_correlation",
                    population["actor_centered_correlation"],
                    eval_step,
                )
                tb.add_scalar(
                    "eval/shape_centered_response_slope",
                    population["actor_centered_response_slope"],
                    eval_step,
                )
                farthest = result["shape"]["counterfactuals"].get("farthest_natural")
                if farthest is not None:
                    tb.add_scalar(
                        "eval/shape_cf_delta_cosine",
                        farthest["delta_cosine"],
                        eval_step,
                    )
                    tb.add_scalar(
                        "eval/shape_cf_response_slope",
                        farthest["delta_response_slope"],
                        eval_step,
                    )
                    tb.add_scalar(
                        "eval/shape_cf_target_advantage_cm",
                        farthest["counterfactual_target_advantage_cm"],
                        eval_step,
                    )
            return True
        except Exception as exc:
            print(f"[inline-eval] step {eval_step} failed: {exc}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
            return False
        finally:
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state_all(cuda_rng_states)
            model.train()

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
            loss, lf, lj, ls, lc, lfv, lfh, sigma = step_loss(batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            fin = bool(torch.isfinite(loss)) and bool(torch.isfinite(gn))
            if fin:
                opt.step()
            print(f"[smoke] {s} loss={loss.item():.4f} feat={lf.item():.4f} joint={lj.item():.4f} "
                  f"smooth={ls.item():.4f} contact={lc.item():.4f} "
                  f"foot_vel={lfv.item():.4f} foot_height={lfh.item():.4f} "
                  f"sigma={sigma.mean().item():.3f} "
                  f"[{sigma.min().item():.3f},{sigma.max().item():.3f}] "
                  f"grad_norm={float(gn):.2f} finite={fin} "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            ok = ok and fin
            if s >= 4:
                break
        print("[smoke] PASS" if ok else "[smoke] FAIL")
        return

    def save_checkpoint(checkpoint_step: int) -> str:
        checkpoint_path = os.path.join(out, f"ckpt_step{checkpoint_step:06d}.pt")
        payload = {"model": model.state_dict(), "step": checkpoint_step, "args": vars(args)}
        torch.save(payload, checkpoint_path)
        torch.save(payload, os.path.join(out, "latest.pt"))
        print(f"[ckpt] step {checkpoint_step}", flush=True)
        return checkpoint_path

    step, t0 = 0, time.time()
    while step < args.steps:
        for batch in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, lf, lj, ls, lc, lfv, lfh, sigma = step_loss(batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # Skip the update on a non-finite grad (a degenerate batch can NaN the decoded-joint
            # loss through the 200-frame cumulative SE(3) compose); don't let it poison the run.
            skipped = not bool(torch.isfinite(gn))
            if not skipped:
                opt.step()
            if step % args.log_every == 0:
                print(f"step {step:6d} loss={loss.item():.4f} feat={lf.item():.4f} joint={lj.item():.4f} "
                      f"smooth={ls.item():.4f} contact={lc.item():.4f} "
                      f"foot_vel={lfv.item():.4f} foot_height={lfh.item():.4f} "
                      f"grad_norm={float(gn):.2f}{' SKIP' if skipped else ''} "
                      f"sigma={sigma.mean().item():.3f} "
                      f"lr={lr_at(step):.2e} {(time.time()-t0)/(step+1):.3f}s/it", flush=True)
                if tb:
                    for k, v in [
                        ("loss", loss),
                        ("feat", lf),
                        ("joint", lj),
                        ("smooth", ls),
                        ("contact", lc),
                        ("foot_vel", lfv),
                        ("foot_height", lfh),
                    ]:
                        tb.add_scalar(k, v.item(), step)
                    tb.add_scalar("grad_norm", float(gn) if torch.isfinite(gn) else 0.0, step)
                    tb.add_scalar("sigma/mean", sigma.mean().item(), step)
                    tb.add_scalar("sigma/min", sigma.min().item(), step)
                    tb.add_scalar("sigma/max", sigma.max().item(), step)
            checkpoint_path = None
            if step > 0 and step % args.save_every == 0:
                checkpoint_path = save_checkpoint(step)
            if (
                inline_evaluator is not None
                and step > 0
                and step % args.inline_eval_every == 0
            ):
                if checkpoint_path is None:
                    checkpoint_path = save_checkpoint(step)
                do_inline_eval(step, checkpoint_path)
            if viz_items and step > 0 and step % args.viz_every == 0:
                try:
                    do_viz(step)
                except Exception as e:
                    print(f"[viz] step {step} failed: {e}", flush=True)
            step += 1
            if step >= args.steps:
                break
    final_checkpoint_path = save_checkpoint(step)
    if inline_evaluator is not None and last_inline_eval_step != step:
        do_inline_eval(step, final_checkpoint_path)
    print("[train] done")


if __name__ == "__main__":
    main()
