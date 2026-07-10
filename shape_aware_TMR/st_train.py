"""Shape-aware TMR trainer (BONES-SEED proportional, all 3 text sources).

Per batch (adapted from TAP tmr_g1/training/train.py):
  1. Dataset: raw uniego window -> decode -> raw posed_joints (T,30,3) ->
     TMRMotionRep (186-d, canonicalized+normalized) + centered neutral_joints + caption.
  2. Motion encoder (SHAPE-aware: [mu, logvar, shape_tok, motion...]) -> reparam -> z_m.
  3. Text: frozen llm2vec cache -> (B,1,4096) -> text encoder (shape-free) -> z_t.
  4. Decoder(z_m, shape, T) -> recon 186-d features; masked L2.
  5. KL(m) + KL(t); InfoNCE(z_m, z_t) with memory bank + text-dup masking.
  6. AdamW; warmup+cosine.

Train-time aug: +-0.3 s temporal jitter on both window boundaries (TAP aug_time_jitter_sec).

Run (kimodo env, 1 GPU):
  bash st_run.sh st_train.py --out-dir /mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/v0 \
      --stats-path /mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/stats_v0
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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
from st_model import build_shape_tmr
from st_text_cache import LLM2VecCache, DEFAULT_CACHE
from st_losses import info_nce_membank, kl_to_standard_normal, reconstruction_loss

OFFICIAL_TMR_TEXT_ENCODER = (
    "/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/"
    "snapshots/e427752ae3446dedba49e928c93ddc9f0e413401/last_weights/text_encoder.pt"
)
OFFICIAL_TMR_MOTION_ENCODER = (
    "/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/"
    "snapshots/e427752ae3446dedba49e928c93ddc9f0e413401/last_weights/motion_encoder.pt"
)
OFFICIAL_TMR_MOTION_DECODER = (
    "/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/"
    "snapshots/e427752ae3446dedba49e928c93ddc9f0e413401/last_weights/motion_decoder.pt"
)

log = logging.getLogger("shape_tmr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _reparam(mu, logvar, training):
    if not training:
        return mu
    std = (0.5 * logvar).exp()
    return mu + torch.randn_like(std) * std


def _cosine_lr(step, warmup, total, base_lr, min_lr=1e-6):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def _load_matching_state_dict(module, checkpoint_path: str, label: str):
    """Load only tensors whose keys and shapes match this module."""
    from kimodo.model.loading import load_checkpoint_state_dict

    src = load_checkpoint_state_dict(checkpoint_path)
    dst = module.state_dict()
    matched = {k: v for k, v in src.items() if k in dst and tuple(v.shape) == tuple(dst[k].shape)}
    skipped = [k for k, v in src.items() if k not in dst or tuple(v.shape) != tuple(dst[k].shape)]
    missing = [k for k in dst.keys() if k not in matched]
    module.load_state_dict(matched, strict=False)
    log.info("loaded %s from %s: matched=%d skipped=%d module_missing=%d",
             label, checkpoint_path, len(matched), len(skipped), len(missing))
    if skipped:
        log.info("%s skipped keys (first 8): %s", label, skipped[:8])
    if missing:
        log.info("%s untouched module keys (first 8): %s", label, missing[:8])


class _EMA:
    """Polyak average over a list of modules; can temporarily swap shadow weights in."""

    def __init__(self, modules, decay):
        self.decay = decay
        self.modules = modules
        self.shadow = [{k: v.detach().clone() for k, v in m.state_dict().items()
                        if v.dtype.is_floating_point} for m in modules]
        self._backup = None

    @torch.no_grad()
    def update(self):
        for m, sh in zip(self.modules, self.shadow):
            for k, v in m.state_dict().items():
                if k in sh:
                    sh[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self):
        self._backup = [{k: v.detach().clone() for k, v in m.state_dict().items()
                         if v.dtype.is_floating_point} for m in self.modules]
        for m, sh in zip(self.modules, self.shadow):
            sd = m.state_dict()
            for k, v in sh.items():
                sd[k].copy_(v)

    @torch.no_grad()
    def swap_out(self):
        for m, bk in zip(self.modules, self._backup):
            sd = m.state_dict()
            for k, v in bk.items():
                sd[k].copy_(v)
        self._backup = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--train-split", default=f"{SPLIT_DIR}/train_split_paths.txt")
    p.add_argument("--text-emb-cache", default=DEFAULT_CACHE)
    p.add_argument("--eval-text-cache", default=None,
                   help="benchmark testsuite llm2vec cache (default: st_inline_eval.BENCH_TEXT_CACHE)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lambda-kl", type=float, default=1e-4)
    p.add_argument("--lambda-kl-motion", type=float, default=None,
                   help="motion KL weight; default inherits --lambda-kl")
    p.add_argument("--lambda-kl-text", type=float, default=None,
                   help="text KL weight; default inherits --lambda-kl")
    p.add_argument("--lambda-recon", type=float, default=0.1)
    p.add_argument("--lambda-contrastive", type=float, default=1.0)
    p.add_argument("--cross-modal-recon", action="store_true", default=False,
                   help="also reconstruct the motion from the TEXT latent (TMR grounding)")
    p.add_argument("--lambda-align", type=float, default=0.0,
                   help="SmoothL1(z_t, z_m) latent alignment (TMR's L_E)")
    p.add_argument("--pretrained-text-encoder", default="",
                   help="optional ACTORStyleEncoder state_dict for the text branch "
                        f"(official TMR default path: {OFFICIAL_TMR_TEXT_ENCODER})")
    p.add_argument("--pretrained-motion-encoder", default="",
                   help="optional ACTORStyleEncoder state_dict for the motion branch; "
                        "loads only matching tensors so shape_enc stays random "
                        f"(official TMR default path: {OFFICIAL_TMR_MOTION_ENCODER})")
    p.add_argument("--pretrained-motion-decoder", default="",
                   help="optional ACTORStyleDecoder state_dict for the motion decoder; "
                        "loads only matching tensors so decoder shape_enc stays random "
                        f"(official TMR default path: {OFFICIAL_TMR_MOTION_DECODER})")
    p.add_argument("--freeze-text-encoder", action="store_true", default=False,
                   help="freeze the text encoder; only motion encoder/decoder are optimized")
    p.add_argument("--text-use-mean", action="store_true", default=False,
                   help="use text mu instead of VAE sampling for z_t; recommended when text is frozen")
    p.add_argument("--motion-use-mean", action="store_true", default=False,
                   help="use motion mu instead of VAE sampling for z_m during training; eval already uses mu")
    p.add_argument("--aug-feat-noise-std", type=float, default=0.0,
                   help="train-time additive Gaussian noise on the normalized 186-d features")
    p.add_argument("--natural-desc4-only", action="store_true", default=False,
                   help="natural pool uses ONLY content_natural_desc_4 (the exact style of the "
                        "benchmark overview prompts; verified 400/400 string match)")
    p.add_argument("--natural-weight", type=int, default=1,
                   help="round-robin share multiplier for the natural source (2 = natural in "
                        "2 of every 4 samples instead of 1 of 3)")
    p.add_argument("--frozen-dup-filter", action="store_true", default=False,
                   help="mask InfoNCE false negatives via FROZEN raw llm2vec cosine (ungameable) "
                        "instead of the learnable projected space (which can collapse; see run c5)")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="Polyak EMA of encoder/decoder weights; inline eval + an extra *_ema ckpt "
                        "use the EMA shadow (0 = off)")
    p.add_argument("--info-nce-temp", type=float, default=0.1)
    p.add_argument("--queue-size", type=int, default=8192)
    p.add_argument("--text-dup-threshold", type=float, default=0.9)
    p.add_argument("--aug-time-jitter-sec", type=float, default=0.3,
                   help="train-time +-jitter on window start/end boundaries (sec)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--data-fps", type=float, default=20.0,
                   help="frame rate of the local motion files; can differ from --fps when "
                        "resampling raw joints into the representation frame rate")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--ff-size", type=int, default=1024)
    p.add_argument("--enc-layers", type=int, default=6)
    p.add_argument("--dec-layers", type=int, default=4)
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
    p.add_argument("--init-from", default="",
                   help="initialize encoder/decoder WEIGHTS from a ckpt but start at step 0 "
                        "(fresh optimizer/schedule) — for two-stage fine-tuning")
    args = p.parse_args()
    if args.lambda_kl_motion is None:
        args.lambda_kl_motion = args.lambda_kl
    if args.lambda_kl_text is None:
        args.lambda_kl_text = args.lambda_kl

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- CPU motion_rep (with stats) for the dataset workers + inline eval
    from kimodo.skeleton import SOMASkeleton30
    from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep
    cpu_motion_rep = TMRMotionRep(skeleton=SOMASkeleton30(), fps=args.fps, stats_path=args.stats_path)
    nfeats = cpu_motion_rep.motion_rep_dim
    log.info("TMRMotionRep dim = %d (SOMASkeleton30)", nfeats)

    json.dump({**vars(args), "nfeats": nfeats, "shape_aware": True,
               "arch": "TMR dual-encoder VAE + shape token (enc) + shape memory token (dec)"},
              open(out_dir / "config.json", "w"), indent=2)

    # ---- dataset / loader (all 3 sources, +-0.3 s boundary jitter)
    desc_cols = (("content_natural_desc_4",) if args.natural_desc4_only
                 else ("content_natural_desc_1", "content_natural_desc_2",
                       "content_natural_desc_3", "content_natural_desc_4"))
    # round-robin interleave: duplicating "natural" in SOURCES raises its per-batch share
    # (pools are keyed by name, so the index cache is unaffected by the duplication).
    sources = ("natural",) * max(1, args.natural_weight) + ("single", "multi")
    dataset = ShapeTMRDataset(
        args.train_split, motion_rep=cpu_motion_rep,
        sources=sources,
        cache_index=str(out_dir / "st_train_index.json"),
        fps=int(args.data_fps), rep_fps=args.fps,
        time_jitter_sec=args.aug_time_jitter_sec, train=True, seed=args.seed,
        natural_desc_cols=desc_cols,
    )
    log.info("train dataset: virtual_len=%d pools=%s", len(dataset),
             {s: len(dataset._pools[s]) for s in dataset.SOURCES})
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True,
                        collate_fn=collate_st, persistent_workers=args.num_workers > 0)

    # ---- model + frozen text cache
    cache = LLM2VecCache(args.text_emb_cache, device=args.device)
    motion_encoder, text_encoder, decoder = build_shape_tmr(
        nfeats=nfeats, llm_dim=cache.dim, latent_dim=args.latent_dim, ff_size=args.ff_size,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers, num_heads=args.num_heads,
        dropout=args.dropout, device=args.device,
    )
    if args.pretrained_text_encoder:
        from kimodo.model.loading import load_checkpoint_state_dict
        sd = load_checkpoint_state_dict(args.pretrained_text_encoder)
        text_encoder.load_state_dict(sd)
        log.info("loaded pretrained text encoder from %s", args.pretrained_text_encoder)
    if args.pretrained_motion_encoder:
        _load_matching_state_dict(motion_encoder, args.pretrained_motion_encoder, "pretrained motion encoder")
    if args.pretrained_motion_decoder:
        _load_matching_state_dict(decoder, args.pretrained_motion_decoder, "pretrained motion decoder")
    if args.freeze_text_encoder:
        for p_text in text_encoder.parameters():
            p_text.requires_grad_(False)
        text_encoder.eval()
        log.info("froze text encoder (%d parameters)",
                 sum(p.numel() for p in text_encoder.parameters()))
    n_params = sum(x.numel() for m in (motion_encoder, text_encoder, decoder) for x in m.parameters())
    trainable_params = sum(x.numel() for m in (motion_encoder, text_encoder, decoder)
                           for x in m.parameters() if x.requires_grad)
    log.info("params = %.2fM total / %.2fM trainable", n_params / 1e6, trainable_params / 1e6)
    motion_encoder.train(); decoder.train()
    if not args.freeze_text_encoder:
        text_encoder.train()
    params = [p for m in (motion_encoder, text_encoder, decoder) for p in m.parameters()
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ---- inline evaluator on the OFFICIAL benchmark testsuite.
    # Training selection uses the full content/overview pool only; standalone st_eval.py
    # remains the full six-group report.
    evaluator = None
    if args.eval_every > 0:
        from st_inline_eval import TestsuiteEvaluator, BENCH_TEXT_CACHE
        eval_cache = LLM2VecCache(args.eval_text_cache or BENCH_TEXT_CACHE, device=args.device)
        evaluator = TestsuiteEvaluator(
            cpu_motion_rep, eval_cache,
            groups=(("content", "overview"),),
            max_cases_per_group=args.eval_cases, src_fps=args.data_fps, rep_fps=args.fps,
            device=args.device, seed=args.seed)

    # ---- resume
    start_step = 0
    if args.init_from and Path(args.init_from).is_file():
        ck = torch.load(args.init_from, map_location=args.device, weights_only=False)
        motion_encoder.load_state_dict(ck["motion_encoder"])
        text_encoder.load_state_dict(ck["text_encoder"])
        decoder.load_state_dict(ck["motion_decoder"])
        log.info("initialized WEIGHTS from %s (step %s); starting at step 0", args.init_from, ck.get("step"))
    if args.resume_from and Path(args.resume_from).is_file():
        ck = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        motion_encoder.load_state_dict(ck["motion_encoder"])
        text_encoder.load_state_dict(ck["text_encoder"])
        decoder.load_state_dict(ck["motion_decoder"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_step = int(ck.get("step", 0))
        log.info("resumed from %s at step %d", args.resume_from, start_step)
    if args.pretrained_text_encoder and (args.init_from or args.resume_from):
        from kimodo.model.loading import load_checkpoint_state_dict
        sd = load_checkpoint_state_dict(args.pretrained_text_encoder)
        text_encoder.load_state_dict(sd)
        log.info("reloaded pretrained text encoder after checkpoint init/resume")
    if args.pretrained_motion_encoder and (args.init_from or args.resume_from):
        _load_matching_state_dict(motion_encoder, args.pretrained_motion_encoder,
                                  "reloaded pretrained motion encoder after checkpoint init/resume")
    if args.pretrained_motion_decoder and (args.init_from or args.resume_from):
        _load_matching_state_dict(decoder, args.pretrained_motion_decoder,
                                  "reloaded pretrained motion decoder after checkpoint init/resume")

    ema = _EMA([motion_encoder, text_encoder, decoder], args.ema_decay) if args.ema_decay > 0 else None

    queue_zm = queue_zt = queue_sent = None
    step = start_step
    t0 = time.time()
    for _epoch in range(10_000):                       # virtual; capped by max_steps
        for batch in loader:
            feats = batch["features"].to(args.device, non_blocking=True)
            mask = batch["mask"].to(args.device, non_blocking=True)
            nj = batch["neutral_joints"].to(args.device, non_blocking=True)
            texts = batch["text"]
            if args.aug_feat_noise_std > 0:
                feats = feats + args.aug_feat_noise_std * torch.randn_like(feats)

            m_out = motion_encoder({"x": feats, "mask": mask, "neutral_joints": nj})
            mu_m, logvar_m = m_out.unbind(1)
            z_m = mu_m if args.motion_use_mean else _reparam(mu_m, logvar_m, training=True)

            llm = cache.batch(texts)                    # (B,1,4096)
            t_mask = torch.ones(llm.shape[0], 1, dtype=torch.bool, device=args.device)
            t_out = text_encoder({"x": llm, "mask": t_mask})
            mu_t, logvar_t = t_out.unbind(1)
            z_t = mu_t if args.text_use_mean else _reparam(mu_t, logvar_t, training=True)

            T_max = feats.shape[1]
            if args.lambda_recon > 0:
                pred_m = decoder(z_m, nj, T_max)        # shape-aware reconstruction
                l_recon = reconstruction_loss(pred_m, feats, mask)
            else:
                l_recon = torch.zeros((), device=args.device)
            if args.cross_modal_recon and args.lambda_recon > 0:
                pred_t = decoder(z_t, nj, T_max)        # TEXT latent must decode the motion too
                l_recon = 0.5 * (l_recon + reconstruction_loss(pred_t, feats, mask))
            if args.lambda_align > 0:
                l_align = torch.nn.functional.smooth_l1_loss(z_t, z_m)
            else:
                l_align = torch.zeros((), device=args.device)
            l_kl_m = kl_to_standard_normal(mu_m, logvar_m)
            l_kl_t = kl_to_standard_normal(mu_t, logvar_t)

            zm_n = F.normalize(z_m, dim=-1)
            zt_n = F.normalize(z_t, dim=-1)
            if args.frozen_dup_filter:
                from st_losses_v2 import info_nce_membank_frozen
                sent = llm[:, 0]                        # frozen raw llm2vec (B,4096)
                l_nce = info_nce_membank_frozen(zm_n, zt_n, sent, queue_zm, queue_zt, queue_sent,
                                                temperature=args.info_nce_temp,
                                                threshold_selfsim=args.text_dup_threshold)
            else:
                l_nce = info_nce_membank(zm_n, zt_n, queue_zm, queue_zt,
                                         temperature=args.info_nce_temp,
                                         text_dup_threshold=args.text_dup_threshold)

            loss = (args.lambda_recon * l_recon
                    + args.lambda_kl_motion * l_kl_m
                    + args.lambda_kl_text * l_kl_t
                    + args.lambda_align * l_align
                    + args.lambda_contrastive * l_nce)

            if args.queue_size > 0:
                with torch.no_grad():
                    qm, qt = zm_n.detach(), zt_n.detach()
                    qs = llm[:, 0].detach()
                    if queue_zm is None:
                        queue_zm, queue_zt, queue_sent = qm, qt, qs
                    else:
                        queue_zm = torch.cat([queue_zm, qm], 0)[-args.queue_size:]
                        queue_zt = torch.cat([queue_zt, qt], 0)[-args.queue_size:]
                        queue_sent = torch.cat([queue_sent, qs], 0)[-args.queue_size:]

            lr = _cosine_lr(step, args.warmup, args.max_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update()
            step += 1

            if step % args.log_every == 0:
                with torch.no_grad():
                    sim = zm_n @ zt_n.t()
                    r1 = (sim.argmax(1) == torch.arange(sim.shape[0], device=sim.device)).float().mean().item()
                rate = (step - start_step) / max(1e-9, time.time() - t0)
                log.info("step=%d lr=%.2e loss=%.4f recon=%.4f kl_m=%.4f kl_t=%.4f nce=%.4f "
                         "batch_R@1=%.3f rate=%.2f steps/s",
                         step, lr, loss.item(), l_recon.item(), l_kl_m.item(), l_kl_t.item(),
                         l_nce.item(), r1, rate)
                for k, v in [("loss/total", loss), ("loss/recon", l_recon), ("loss/kl_motion", l_kl_m),
                             ("loss/kl_text", l_kl_t), ("loss/infonce", l_nce)]:
                    tb.add_scalar(k, v.item(), step)
                tb.add_scalar("train/batch_R@1", r1, step)
                tb.add_scalar("train/lr", lr, step)
                tb.add_scalar("train/steps_per_sec", rate, step)

            if evaluator is not None and step % args.eval_every == 0:
                if ema is not None:
                    ema.swap_in()                        # evaluate the EMA shadow
                em = evaluator.evaluate(motion_encoder, text_encoder)
                if args.freeze_text_encoder:
                    text_encoder.eval()
                if ema is not None:
                    ema.swap_out()
                for src, ms in em.items():
                    log.info("EVAL[%s]%s step=%d %s", src, " (EMA)" if ema else "", step,
                             " ".join(f"{k}={v}" for k, v in sorted(ms.items())))
                    for k, v in ms.items():
                        tb.add_scalar(f"eval_{src}/{k}", float(v), step)

            if step % args.ckpt_every == 0:
                if ema is not None:
                    ema.swap_in()                        # persist the EMA weights (what we eval)
                ckpt = {"step": step,
                        "motion_encoder": motion_encoder.state_dict(),
                        "text_encoder": text_encoder.state_dict(),
                        "motion_decoder": decoder.state_dict(),
                        "args": vars(args), "nfeats": nfeats,
                        "ema": args.ema_decay if ema is not None else 0.0}
                if ema is not None:
                    ema.swap_out()
                    ckpt["optimizer"] = None             # EMA ckpts are eval snapshots, not resume points
                else:
                    ckpt["optimizer"] = opt.state_dict()
                torch.save(ckpt, out_dir / f"step_{step:08d}.pt")
                torch.save(ckpt, out_dir / "last.pt")
                log.info("ckpt step %d%s", step, " (EMA weights)" if ema else "")
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break
    log.info("done after %d steps", step)


if __name__ == "__main__":
    main()
