"""Shape-swap sanity for the shape-aware TMR (run after training).

On a fixed pool of test_content samples, permute the shape inputs within the batch and
measure:
  1. ENCODER shape-invariance (expect: changes LITTLE). cos(z(motion,true_shape),
     z(motion,permuted_shape)) should be high — the retrieval latent should capture motion
     semantics, not body size (that's the fair-evaluator property). Large drop = shape
     leaking into z.
  2. DECODER shape-sensitivity (expect: changes MUCH). masked recon error with permuted
     shape should clearly exceed the true-shape recon error — proves the decoder actually
     uses the shape path. Both ~zero effect = dead shape encoders.

  bash st_run.sh st_shape_ablation.py --ckpt <run>/last.pt --stats-path <stats>
"""
from __future__ import annotations

import argparse
import logging

import torch
import torch.nn.functional as F

from st_eval import ShapeTMREmbedder
from st_inline_eval import TestsuiteEvaluator, BENCH_TEXT_CACHE
from st_losses import reconstruction_loss
from st_text_cache import LLM2VecCache

log = logging.getLogger("st_ablation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--cases", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    emb = ShapeTMREmbedder(args.ckpt, args.stats_path, device=args.device)
    eval_cache = LLM2VecCache(BENCH_TEXT_CACHE, device=args.device)
    ev = TestsuiteEvaluator(emb.rep, eval_cache, groups=(("content", "overview"),),
                            max_cases_per_group=args.cases, device=args.device, seed=args.seed)
    cases = ev.pools["content/overview"]
    dev = args.device

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(cases), generator=g)

    cos_list, rec_true, rec_perm = [], [], []
    chunk = 64
    with torch.no_grad():
        for s in range(0, len(cases), chunk):
            cs = cases[s:s + chunk]
            idxs = perm[s:s + chunk] % len(cases)
            Tm = max(c["features"].shape[0] for c in cs)
            B = len(cs)
            feats = torch.zeros(B, Tm, cs[0]["features"].shape[-1])
            mask = torch.zeros(B, Tm, dtype=torch.bool)
            for i, c in enumerate(cs):
                t = c["features"].shape[0]
                feats[i, :t] = c["features"]; mask[i, :t] = True
            nj_true = torch.stack([c["neutral_joints"] for c in cs])
            nj_perm = torch.stack([cases[int(j)]["neutral_joints"] for j in idxs])
            feats, mask = feats.to(dev), mask.to(dev)
            nj_true, nj_perm = nj_true.to(dev), nj_perm.to(dev)

            zt_retr = emb.embed_motion_features(feats, mask, nj_true)
            zp_retr = emb.embed_motion_features(feats, mask, nj_perm)
            cos_list.append(F.cosine_similarity(zt_retr, zp_retr, dim=-1).cpu())

            # decoder: recon from the true-shape latent, rendered with true vs permuted shape
            if emb.arch_version == "v3":
                z_dec = emb.motion_encoder.encode({"x": feats, "mask": mask, "neutral_joints": nj_true})[2]
            else:
                z_dec = emb.motion_encoder({"x": feats, "mask": mask, "neutral_joints": nj_true})[:, 0]
            pt = emb.decoder(z_dec, nj_true, Tm)
            pp = emb.decoder(z_dec, nj_perm, Tm)
            rec_true.append(reconstruction_loss(pt, feats, mask).item())
            rec_perm.append(reconstruction_loss(pp, feats, mask).item())

    cos = torch.cat(cos_list)
    log.info("ENCODER  cos(z_true, z_permuted-shape): mean %.4f  p10 %.4f  min %.4f  (n=%d)",
             cos.mean(), cos.quantile(0.10), cos.min(), len(cos))
    rt = sum(rec_true) / len(rec_true)
    rp = sum(rec_perm) / len(rec_perm)
    log.info("DECODER  recon true-shape %.4f | permuted-shape %.4f  (+%.1f%%)",
             rt, rp, 100 * (rp - rt) / max(rt, 1e-9))
    print(f"\nSUMMARY: encoder shape-invariance cos={cos.mean():.3f} (want high); "
          f"decoder shape-sensitivity +{100*(rp-rt)/max(rt,1e-9):.1f}% recon err (want clearly >0)")


if __name__ == "__main__":
    main()
