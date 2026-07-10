"""Ensemble retrieval eval: average unit embeddings from K shape-aware TMR (v1-arch) ckpts.

Motion emb = normalize(mean_k normalize(z_k^m)); same for text. Diversity across the
desc4-family runs (sampling shares, EMA, noise seeds) makes the mean a stronger retriever
without any new training. Reports protocol + plain metrics per testsuite group.

  bash st_run.sh st_ensemble_eval.py --ckpts A.pt B.pt C.pt --stats-path <stats> --out ens.json
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

from kimodo.metrics.tmr import compute_tmr_retrieval_metrics
from kimodo.skeleton import SOMASkeleton30
from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep

from st_eval import ShapeTMREmbedder
from st_inline_eval import TestsuiteEvaluator, BENCH_TEXT_CACHE
from st_text_cache import LLM2VecCache

log = logging.getLogger("st_ensemble")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--cases", type=int, default=100000, help="per group; default = full pools")
    p.add_argument("--groups", default="content", choices=["content", "all"])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device

    rep = TMRMotionRep(skeleton=SOMASkeleton30(), fps=20, stats_path=args.stats_path)
    cache = LLM2VecCache(BENCH_TEXT_CACHE, device=dev)

    embedders = []
    for cp in args.ckpts:
        emb = ShapeTMREmbedder(cp, args.stats_path, text_cache_path=BENCH_TEXT_CACHE, device=dev)
        embedders.append(emb)

    groups = tuple((s, g) for s in (("content",) if args.groups == "content" else ("content", "repetition"))
                   for g in ("overview", "timeline_single", "timeline_multi"))
    ev = TestsuiteEvaluator(rep, text_cache=cache, groups=groups,
                            max_cases_per_group=args.cases, device=dev)

    results = {}
    with torch.no_grad():
        for name, cases in ev.pools.items():
            zm_all, zt_all = [], []
            for s in range(0, len(cases), 64):
                cs = cases[s:s + 64]
                Tm = max(c["features"].shape[0] for c in cs); B = len(cs)
                feats = torch.zeros(B, Tm, cs[0]["features"].shape[-1])
                mask = torch.zeros(B, Tm, dtype=torch.bool)
                for i, c in enumerate(cs):
                    t = c["features"].shape[0]; feats[i, :t] = c["features"]; mask[i, :t] = True
                nj = torch.stack([c["neutral_joints"] for c in cs])
                llm = torch.cat([c["llm"] for c in cs], 0)
                feats, mask, nj, llm = (x.to(dev) for x in (feats, mask, nj, llm))
                t_mask = torch.ones(B, 1, dtype=torch.bool, device=dev)
                zm_k, zt_k = [], []
                for emb in embedders:
                    zm_k.append(emb.embed_motion_features(feats, mask, nj))
                    zt_k.append(emb.embed_llm(llm))
                zm_all.append(F.normalize(torch.stack(zm_k).mean(0), dim=-1).cpu())
                zt_all.append(F.normalize(torch.stack(zt_k).mean(0), dim=-1).cpu())
            zm = torch.cat(zm_all).numpy(); zt = torch.cat(zt_all).numpy()
            m = compute_tmr_retrieval_metrics(zm, zt)
            sim = zt @ zm.T
            ranks = (sim > sim.diagonal()[:, None]).sum(axis=1)
            res = {k.split("/")[-1]: v for k, v in m.items() if "t2m_R" in k or "FID" in k}
            res.update({"plain_R01": float((ranks < 1).mean() * 100),
                        "plain_R03": float((ranks < 3).mean() * 100),
                        "plain_MedR": float(np.median(ranks) + 1), "n": len(cases)})
            results[name] = res
            log.info("[%s] %s", name, " ".join(f"{k}={v}" for k, v in sorted(res.items())))

    if args.out:
        json.dump({"ckpts": args.ckpts, "results": results}, open(args.out, "w"), indent=2)
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
