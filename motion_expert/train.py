"""v2 Phase 2: train MotionExpert with CACHED H_R, x0-prediction, decoded losses.

Only MotionExpert + ShapeEncoder train. H_R comes from the precomputed cache (frozen reasoner
is NOT in the loop). Model predicts x0 (clean motion); loss = feature-MSE(x0) +
decoded joint-position L2 + temporal-smoothness (decoded joint-velocity match) — the decoded
terms directly penalize the crumpling/spinning that feature-MSE alone misses.

Run (cosmos env, 1 GPU):
  CUDA_VISIBLE_DEVICES=0 bash run.sh train.py --steps 100000 --run_name motionexpert_poc_v2
  bash run.sh train.py --smoke    # 3 steps, assert wiring (cached H_R, only expert grad)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio

import flow
from decode_uniego_torch import decode_joints
from hr_cache import HRCache
from motion_expert import MotionExpert
from uniego_dataset import UniegoTextMotionDataset, collate
from uniego_layout import FEAT_DIM

SKEL = "/home/jungbin_cho/cosmos_motion_ft/skeleton_soma30.npz"


def render_joints(joints, parents, out_mp4, fps=20, title=""):
    """joints (T,30,3) world Y-up,+Z-fwd → stick-figure mp4 (remap to plot, floor y=0)."""
    dx, dz, dy = -joints[..., 0], joints[..., 2], joints[..., 1]
    cx = (dx.min() + dx.max()) / 2; cz = (dz.min() + dz.max()) / 2
    half = max(dx.max() - dx.min(), dz.max() - dz.min()) / 2 * 1.1 + 1e-3
    y_top = max(float(dy.max()) * 1.05, 1.0); y_floor = min(0.0, float(dy.min()))
    edges = [(j, int(p)) for j, p in enumerate(parents) if 0 <= int(p) < len(parents) and int(p) != j]
    T = joints.shape[0]; step = max(1, T // 48); frames = []
    for t in range(0, T, step):
        fig = plt.figure(figsize=(3.5, 3.5)); ax = fig.add_subplot(111, projection="3d")
        for a, b in edges:
            ax.plot([dx[t, a], dx[t, b]], [dz[t, a], dz[t, b]], [dy[t, a], dy[t, b]], "-", color="tab:blue", lw=2)
        ax.scatter(dx[t], dz[t], dy[t], c="k", s=6)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cz - half, cz + half); ax.set_zlim(y_floor, y_top)
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
        ax.view_init(elev=20, azim=-60); ax.set_title(f"{title}\n{t}/{T}", fontsize=7)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames.append(buf.copy()); plt.close(fig)
    iio.imwrite(out_mp4, np.stack(frames), fps=fps, codec="libx264")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = "/weka/jungbin/cosmos_motion_ft_runs"
D_REASONER = 4096


def masked_mse(a, b, valid):  # a,b [...,D] or [...,J,3]; valid [B,T] True=keep
    while valid.dim() < a.dim():
        valid = valid.unsqueeze(-1)
    se = ((a - b) ** 2) * valid
    return se.sum() / valid.expand_as(a).sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--w_feat", type=float, default=1.0)
    ap.add_argument("--w_joint", type=float, default=1.0)
    ap.add_argument("--w_smooth", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--viz_every", type=int, default=5000)
    ap.add_argument("--viz_n", type=int, default=4, help="held-out val captions to sample+render each viz")
    ap.add_argument("--viz_steps", type=int, default=50)
    ap.add_argument("--viz_guidance", type=float, default=2.0)
    ap.add_argument("--pairs", default=os.path.join(HERE, "pairs_train.jsonl"))
    ap.add_argument("--cache_dir", default=os.path.join(RUN_ROOT, "hr_cache"))
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    mean = torch.from_numpy(np.load(os.path.join(HERE, "stats", "uniego283_mean.npy"))).float().to(dev)
    std = torch.from_numpy(np.load(os.path.join(HERE, "stats", "uniego283_std.npy"))).float().to(dev)
    cache = HRCache(args.cache_dir, device=dev)
    print(f"[train] H_R cache: {len(cache.captions)} captions, {cache.nshards} shards")

    model = MotionExpert(d=args.d, n_layers=args.layers, heads=args.heads, ffn=args.ffn,
                         kv_dim=D_REASONER, motion_dim=FEAT_DIM).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] MotionExpert trainable = {n_train/1e6:.2f}M")

    ds = UniegoTextMotionDataset(args.pairs, T=args.T, train=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True, pin_memory=True,
                    persistent_workers=args.num_workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    if not args.smoke:
        name = args.run_name or f"motionexpert_v2_{time.strftime('%Y%m%d_%H%M%S')}"
        out = os.path.join(RUN_ROOT, name); os.makedirs(out, exist_ok=True)
        json.dump({**vars(args), "trainable_M": n_train}, open(os.path.join(out, "config.json"), "w"), indent=2)
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(out)
        except Exception:
            tb = None
        print(f"[train] run dir: {out}")
    else:
        out, tb = None, None

    def step_loss(batch):
        x0 = batch["motion"].to(dev)                         # [B,T,283] normalized
        nj = batch["neutral_joints"].to(dev)
        valid = (~batch["motion_pad_mask"].to(dev))          # [B,T] True=keep
        H_R, h_pad = cache.batch(batch["caption"])
        sigma = flow.sample_sigma_logitnormal(x0.shape[0], dev)
        x_sigma, _ = flow.add_noise(x0, sigma)
        x0_hat = model(x_sigma, sigma, H_R, h_pad, nj, motion_pad_mask=batch["motion_pad_mask"].to(dev))

        l_feat = masked_mse(x0_hat, x0, valid)
        # decoded joints (unnormalize → decode). Use BOUNDED targets:
        #  - pose: centroid-relative joints (translation-invariant) → targets "crumpling"
        #  - smooth: per-frame joint velocity → targets "jitter/spinning/drift"
        # (absolute joint position is drift-dominated/unstable, so not used directly.)
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

    # ---- in-training viz setup: held-out val captions (in the H_R cache) + their skeletons ----
    parents = np.load(SKEL, allow_pickle=True)["parents"]
    viz_items = []
    if args.viz_n > 0 and not args.smoke:
        seen = set()
        for l in open(os.path.join(HERE, "pairs_val.jsonl")):
            r = json.loads(l)
            cap = r["caption"]
            if cap in seen:
                continue
            seen.add(cap)
            nj = np.load(r["uniego_path"])["neutral_joints"].astype(np.float32)
            nj = nj - nj.mean(0, keepdims=True)
            viz_items.append({"caption": cap, "nj": torch.from_numpy(nj).to(dev)})
            if len(viz_items) >= args.viz_n:
                break
        null_H = cache.null().unsqueeze(0); null_pad = torch.zeros(1, null_H.shape[1], dtype=torch.bool, device=dev)

    @torch.no_grad()
    def do_viz(step):
        model.eval()
        vdir = os.path.join(out, f"viz_step{step:06d}"); os.makedirs(vdir, exist_ok=True)
        for i, it in enumerate(viz_items):
            H, hp = cache.batch([it["caption"]])
            x0 = flow.sample_x0(model, H, hp, it["nj"].unsqueeze(0), T=args.T, motion_dim=FEAT_DIM,
                                steps=args.viz_steps, guidance=args.viz_guidance,
                                H_null=null_H, null_pad_mask=null_pad, device=dev)
            feat = (x0[0] * std + mean)                                  # unnormalize
            joints = decode_joints(feat.unsqueeze(0))[0].cpu().numpy()    # torch decode
            render_joints(joints, parents, os.path.join(vdir, f"{i}_{it['caption'][:30].replace(' ','_')}.mp4"),
                          title=it["caption"][:34])
        model.train()
        print(f"[viz] step {step} -> {vdir} ({len(viz_items)} clips)", flush=True)

    model.train()
    if args.smoke:
        for s, batch in enumerate(dl):
            loss, lf, lj, ls = step_loss(batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            eg = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
            print(f"[smoke] {s} loss={loss.item():.4f} feat={lf.item():.4f} joint={lj.item():.4f} "
                  f"smooth={ls.item():.4f} finite={torch.isfinite(loss).item()} expert_grad={eg:.2f} "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            opt.step()
            if s >= 2:
                break
        print("[smoke] PASS" if torch.isfinite(loss) and eg > 0 else "[smoke] FAIL")
        return

    step, t0 = 0, time.time()
    while step < args.steps:
        for batch in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, lf, lj, ls = step_loss(batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if step % args.log_every == 0:
                print(f"step {step:6d} loss={loss.item():.4f} feat={lf.item():.4f} joint={lj.item():.4f} "
                      f"smooth={ls.item():.4f} lr={lr_at(step):.2e} {(time.time()-t0)/(step+1):.3f}s/it",
                      flush=True)
                if tb:
                    for k, v in [("loss", loss), ("feat", lf), ("joint", lj), ("smooth", ls)]:
                        tb.add_scalar(k, v.item(), step)
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
