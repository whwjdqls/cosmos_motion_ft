"""Shape-aware TMR v3 trainer — SnapMoGen's RELEASED evaluator (EncoderV2 arch + losses).

Architecture (the user-prioritized change vs v1): non-VAE single-CLS-token transformers with
SPLIT heads — `cst` (contrastive/retrieval embedding) and `rec` (decoder latent) — so recon
cannot fight retrieval inside one vector. No KL, no reparam noise. Motion tower gets the
SHAPE token; decoder is shape-aware; 186-d raw-joints rep and dataset UNCHANGED; text = the
frozen pooled llm2vec caches (per user).

Losses (SnapMoGen evaluator.yaml, exp `evalv2_rec1_cst0.1_ld256`):
  1.0 * (SmoothL1 recon from rec_m AND rec_t [cross-modal, shape-aware decoder])
+ 0.1 * InfoNCE(cst_m, cst_t; temp .10, RAW-llm2vec sent filtering thre .80, batch-only)
+ 1e-5 * SmoothL1(cst_t, cst_m) latent align.
AdamW lr 2e-4 betas (.9,.99) wd 1e-5; warmup 200 -> constant -> x0.1 @120k.

Run: bash st_run.sh st_train_v3.py --out-dir .../shape_tmr/v3 --stats-path .../shape_tmr/stats_v0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from st_dataset import ShapeTMRDataset, collate_st, SPLIT_DIR
from st_model_v3 import build_shape_tmr_v3
from st_text_cache import LLM2VecCache, DEFAULT_CACHE
from st_losses_v2 import smooth_l1_recon, latent_align, info_nce_with_filtering

log = logging.getLogger("shape_tmr_v3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _lr_at(step, warmup, milestone, base_lr, gamma=0.1):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    return base_lr * (gamma if step >= milestone else 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--train-split", default=f"{SPLIT_DIR}/train_split_paths.txt")
    p.add_argument("--text-emb-cache", default=DEFAULT_CACHE)
    p.add_argument("--eval-text-cache", default=None)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=14)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-milestone", type=int, default=120_000)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--lambda-rec", type=float, default=1.0)
    p.add_argument("--lambda-align", type=float, default=1e-5)
    p.add_argument("--lambda-contrast", type=float, default=0.1)
    p.add_argument("--info-nce-temp", type=float, default=0.10)
    p.add_argument("--threshold-selfsim", type=float, default=0.80)
    p.add_argument("--aug-time-jitter-sec", type=float, default=0.3)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--output-dim", type=int, default=256)
    p.add_argument("--ff-size", type=int, default=1024)
    p.add_argument("--enc-layers", type=int, default=6)
    p.add_argument("--dec-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=5_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-cases", type=int, default=0,
                   help="training eval cases per group; 0 = full content/overview pool")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from", default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from kimodo.skeleton import SOMASkeleton30
    from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep
    cpu_motion_rep = TMRMotionRep(skeleton=SOMASkeleton30(), fps=args.fps, stats_path=args.stats_path)
    nfeats = cpu_motion_rep.motion_rep_dim

    json.dump({**vars(args), "nfeats": nfeats, "arch_version": "v3",
               "arch": "SnapMoGen EncoderV2 (non-VAE CLS + split cst/rec heads) + shape tokens",
               "text": "cached llm2vec pooled (frozen)"},
              open(out_dir / "config.json", "w"), indent=2)

    dataset = ShapeTMRDataset(
        args.train_split, motion_rep=cpu_motion_rep,
        sources=("natural", "single", "multi"),
        cache_index=str(out_dir / "st_train_index.json"),
        time_jitter_sec=args.aug_time_jitter_sec, train=True, seed=args.seed,
    )
    log.info("train dataset: virtual_len=%d pools=%s", len(dataset),
             {s: len(dataset._pools[s]) for s in dataset.SOURCES})
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True,
                        collate_fn=collate_st, persistent_workers=args.num_workers > 0)

    cache = LLM2VecCache(args.text_emb_cache, device=args.device)
    motion_encoder, text_encoder, decoder = build_shape_tmr_v3(
        nfeats=nfeats, text_dim=cache.dim, latent_dim=args.latent_dim, output_dim=args.output_dim,
        ff_size=args.ff_size, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        num_heads=args.num_heads, dropout=args.dropout, device=args.device,
    )
    n_params = sum(x.numel() for m in (motion_encoder, text_encoder, decoder) for x in m.parameters())
    log.info("trainable params = %.2fM", n_params / 1e6)
    motion_encoder.train(); text_encoder.train(); decoder.train()
    params = (list(motion_encoder.parameters()) + list(text_encoder.parameters())
              + list(decoder.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    evaluator = None
    if args.eval_every > 0:
        from st_inline_eval import TestsuiteEvaluator, BENCH_TEXT_CACHE
        eval_cache = LLM2VecCache(args.eval_text_cache or BENCH_TEXT_CACHE, device=args.device)
        evaluator = TestsuiteEvaluator(
            cpu_motion_rep, text_cache=eval_cache,
            groups=(("content", "overview"),),
            max_cases_per_group=args.eval_cases, device=args.device, seed=args.seed)

    # v3 retrieval embeddings = the cst heads
    def motion_embed_fn(feats, mask, nj):
        return motion_encoder.encode({"x": feats, "mask": mask, "neutral_joints": nj})[1]

    def text_embed_fn(llm):
        B = llm.shape[0]
        return text_encoder.encode({"x": llm, "mask": torch.ones(B, 1, dtype=torch.bool, device=args.device)})[1]

    start_step = 0
    if args.resume_from and Path(args.resume_from).is_file():
        ck = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        motion_encoder.load_state_dict(ck["motion_encoder"])
        text_encoder.load_state_dict(ck["text_encoder"])
        decoder.load_state_dict(ck["motion_decoder"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_step = int(ck.get("step", 0))
        log.info("resumed from %s at step %d", args.resume_from, start_step)

    step = start_step
    t0 = time.time()
    for _epoch in range(10_000):
        for batch in loader:
            feats = batch["features"].to(args.device, non_blocking=True)
            mask = batch["mask"].to(args.device, non_blocking=True)
            nj = batch["neutral_joints"].to(args.device, non_blocking=True)
            texts = batch["text"]

            _, cst_m, rec_m = motion_encoder.encode({"x": feats, "mask": mask, "neutral_joints": nj})
            t_emb = cache.batch(texts)                            # (B,1,4096) frozen llm2vec
            sent = t_emb[:, 0]                                    # raw sentence emb (NCE filtering)
            t_mask = torch.ones(t_emb.shape[0], 1, dtype=torch.bool, device=args.device)
            _, cst_t, rec_t = text_encoder.encode({"x": t_emb, "mask": t_mask})

            T_max = feats.shape[1]
            pred_m = decoder(rec_m, nj, T_max)                    # motion-latent recon (shape-aware)
            pred_t = decoder(rec_t, nj, T_max)                    # text-latent recon (cross-modal)
            l_rec_m = smooth_l1_recon(pred_m, feats, mask)
            l_rec_t = smooth_l1_recon(pred_t, feats, mask)
            l_align = latent_align(cst_t, cst_m)
            l_nce = info_nce_with_filtering(cst_m, cst_t, sent_emb=sent,
                                            temperature=args.info_nce_temp,
                                            threshold_selfsim=args.threshold_selfsim)

            loss = (args.lambda_rec * (l_rec_m + l_rec_t)
                    + args.lambda_align * l_align
                    + args.lambda_contrast * l_nce)

            lr = _lr_at(step, args.warmup, args.lr_milestone, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            step += 1

            if step % args.log_every == 0:
                with torch.no_grad():
                    sim = F.normalize(cst_m, dim=-1) @ F.normalize(cst_t, dim=-1).t()
                    r1 = (sim.argmax(1) == torch.arange(sim.shape[0], device=sim.device)).float().mean().item()
                rate = (step - start_step) / max(1e-9, time.time() - t0)
                log.info("step=%d lr=%.2e loss=%.4f rec_m=%.3f rec_t=%.3f align=%.4f nce=%.4f "
                         "batch_R@1=%.3f rate=%.2f steps/s",
                         step, lr, loss.item(), l_rec_m.item(), l_rec_t.item(),
                         l_align.item(), l_nce.item(), r1, rate)
                for k, v in [("loss/total", loss), ("loss/rec_m", l_rec_m), ("loss/rec_t", l_rec_t),
                             ("loss/align", l_align), ("loss/infonce", l_nce)]:
                    tb.add_scalar(k, v.item(), step)
                tb.add_scalar("train/batch_R@1", r1, step)
                tb.add_scalar("train/lr", lr, step)
                tb.add_scalar("train/steps_per_sec", rate, step)

            if evaluator is not None and step % args.eval_every == 0:
                motion_encoder.eval(); text_encoder.eval()
                em = evaluator.evaluate(motion_embed_fn=motion_embed_fn, text_embed_fn=text_embed_fn)
                motion_encoder.train(); text_encoder.train()
                for src, ms in em.items():
                    log.info("EVAL[%s] step=%d %s", src, step,
                             " ".join(f"{k}={v}" for k, v in sorted(ms.items())))
                    for k, v in ms.items():
                        tb.add_scalar(f"eval_{src}/{k}", float(v), step)

            if step % args.ckpt_every == 0:
                ckpt = {"step": step,
                        "motion_encoder": motion_encoder.state_dict(),
                        "text_encoder": text_encoder.state_dict(),
                        "motion_decoder": decoder.state_dict(),
                        "optimizer": opt.state_dict(),
                        "args": vars(args), "nfeats": nfeats, "arch_version": "v3"}
                torch.save(ckpt, out_dir / f"step_{step:08d}.pt")
                torch.save(ckpt, out_dir / "last.pt")
                log.info("ckpt step %d", step)
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break
    log.info("done after %d steps", step)


if __name__ == "__main__":
    main()
