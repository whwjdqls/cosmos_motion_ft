"""Final retrieval eval + the embedding API for scoring shape-aware generation models.

`ShapeTMREmbedder` loads a checkpoint (+stats +text cache) and exposes:
  - embed_motion(posed_joints [B,T,30,3] RAW  |or|  uniego [B,T,283] unnormalized,
                 neutral_joints [B,30,3], lengths) -> [B,256] unit vectors
  - embed_text(captions) -> [B,256] unit vectors
This is what a generation eval calls: unnormalize the generator's 283-d output with the
proportional Mean/Std, hand it here with the conditioned skeleton.

`main()` reports full retrieval on the OFFICIAL benchmark testsuite — all six groups:
{content, repetition} x {overview, timeline_single, timeline_multi} (held-out; multi included).

  bash st_run.sh st_eval.py --ckpt <run>/last.pt --stats-path <stats> --out <run>/eval.json
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

from decode_uniego import decode_joints
from st_dataset import DATA_ROOT, resample_joints_time
from st_inline_eval import BENCH_TEXT_CACHE, TESTSUITE, TestsuiteEvaluator
from st_model_agnostic import build_agnostic_tmr
from st_model import build_shape_tmr
from st_model_v2 import build_shape_tmr_v2
from st_model_v3 import build_shape_tmr_v3
from st_text_cache import LLM2VecCache, DEFAULT_CACHE

log = logging.getLogger("st_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


class ShapeTMREmbedder:
    def __init__(self, ckpt_path: str, stats_path: str,
                 text_cache_path: str = DEFAULT_CACHE, device: str = "cuda"):
        self.device = device
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        self.fps = float(a.get("fps", 20))
        self.data_fps = float(a.get("data_fps", 20))
        self.rep = TMRMotionRep(skeleton=SOMASkeleton30(), fps=int(self.fps), stats_path=stats_path)
        self.cache = LLM2VecCache(text_cache_path, device=device)
        self.arch_version = str(ck.get("arch_version", a.get("arch_version", "v1")))
        nfeats = ck.get("nfeats", self.rep.motion_rep_dim)
        if self.arch_version == "agnostic":
            self.motion_encoder, self.text_encoder, self.decoder = build_agnostic_tmr(
                nfeats=nfeats, llm_dim=self.cache.dim,
                latent_dim=a.get("latent_dim", 256), ff_size=a.get("ff_size", 1024),
                enc_layers=a.get("enc_layers", 6), dec_layers=a.get("dec_layers", 4),
                num_heads=a.get("num_heads", 4), dropout=a.get("dropout", 0.1), device=device,
            )
        elif self.arch_version == "v3":
            self.motion_encoder, self.text_encoder, self.decoder = build_shape_tmr_v3(
                nfeats=nfeats, text_dim=self.cache.dim,
                latent_dim=a.get("latent_dim", 256), output_dim=a.get("output_dim", 256),
                ff_size=a.get("ff_size", 1024), enc_layers=a.get("enc_layers", 6),
                dec_layers=a.get("dec_layers", 6), num_heads=a.get("num_heads", 4),
                dropout=a.get("dropout", 0.1), device=device,
            )
        elif self.arch_version == "v2":
            self.motion_encoder, self.text_encoder, self.decoder = build_shape_tmr_v2(
                nfeats=nfeats, text_dim=self.cache.dim,
                latent_dim=a.get("latent_dim", 256), ff_size=a.get("ff_size", 1024),
                enc_layers=a.get("enc_layers", 6), dec_layers=a.get("dec_layers", 6),
                num_heads=a.get("num_heads", 4), dropout=a.get("dropout", 0.1), device=device,
            )
        else:
            self.motion_encoder, self.text_encoder, self.decoder = build_shape_tmr(
                nfeats=nfeats, llm_dim=self.cache.dim,
                latent_dim=a.get("latent_dim", 256), ff_size=a.get("ff_size", 1024),
                enc_layers=a.get("enc_layers", 6), dec_layers=a.get("dec_layers", 4),
                num_heads=a.get("num_heads", 4), dropout=a.get("dropout", 0.1), device=device,
            )
        self.motion_encoder.load_state_dict(ck["motion_encoder"])
        self.text_encoder.load_state_dict(ck["text_encoder"])
        self.decoder.load_state_dict(ck["motion_decoder"])
        for m in (self.motion_encoder, self.text_encoder, self.decoder):
            m.eval()
        self.step = ck.get("step")
        log.info("loaded %s (arch %s, step %s)", ckpt_path, self.arch_version, self.step)

    @torch.no_grad()
    def embed_motion_features(self, feats: torch.Tensor, mask: torch.Tensor,
                              neutral_joints: torch.Tensor) -> torch.Tensor:
        x = {"x": feats.to(self.device), "mask": mask.to(self.device),
             "neutral_joints": neutral_joints.to(self.device)}
        if self.arch_version == "v3":
            return F.normalize(self.motion_encoder.encode(x)[1], dim=-1)
        return F.normalize(self.motion_encoder(x)[:, 0], dim=-1)

    @torch.no_grad()
    def embed_llm(self, llm: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(llm.shape[0], llm.shape[1], dtype=torch.bool, device=self.device)
        x = {"x": llm.to(self.device), "mask": mask}
        if self.arch_version == "v3":
            return F.normalize(self.text_encoder.encode(x)[1], dim=-1)
        return F.normalize(self.text_encoder(x)[:, 0], dim=-1)

    @torch.no_grad()
    def embed_motion(self, motion: torch.Tensor, neutral_joints: torch.Tensor,
                     lengths=None, source_fps: float | None = None) -> torch.Tensor:
        """motion: [B,T,30,3] raw joints OR [B,T,283] unnormalized uniego -> [B,256] units."""
        if motion.dim() == 3 and motion.shape[-1] == 283:
            motion = decode_joints(motion)                       # -> [B,T,30,3]
        src_fps = self.data_fps if source_fps is None else float(source_fps)
        if abs(src_fps - self.fps) >= 1e-6:
            motion = resample_joints_time(motion, src_fps, self.fps)
            lengths = None
        B, T = motion.shape[:2]
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long)
        # per-sample so canonicalization matches training (batch=1 in the rep)
        feats = torch.zeros(B, T, self.rep.motion_rep_dim)
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i in range(B):
            n = int(lengths[i])
            f = self.rep(posed_joints=motion[i:i + 1, :n],
                         to_normalize=True, to_canonicalize=True,
                         lengths=torch.tensor([n]))[0]
            feats[i, :n] = f
            mask[i, :n] = True
        nj = neutral_joints - neutral_joints.mean(dim=1, keepdim=True)
        return self.embed_motion_features(feats, mask, nj)

    @torch.no_grad()
    def embed_text(self, captions) -> torch.Tensor:
        llm = self.cache.batch(list(captions))
        return self.embed_llm(llm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--eval-text-cache", default=BENCH_TEXT_CACHE)
    p.add_argument("--testsuite", default=TESTSUITE,
                   help="Benchmark testsuite root. Default uses st_inline_eval.TESTSUITE.")
    p.add_argument("--uniego-root", default=DATA_ROOT,
                   help="Motion npz tree to resolve seed_motion.json paths. "
                        "Use this to switch between uniform and proportional motions.")
    p.add_argument("--cases", type=int, default=100000, help="per group; default = all")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    emb = ShapeTMREmbedder(args.ckpt, args.stats_path,
                           text_cache_path=args.eval_text_cache, device=args.device)
    eval_cache = LLM2VecCache(args.eval_text_cache, device=args.device)

    groups = tuple((s, g) for s in ("content", "repetition")
                   for g in ("overview", "timeline_single", "timeline_multi"))
    ev = TestsuiteEvaluator(emb.rep, eval_cache, groups=groups,
                            testsuite=args.testsuite,
                            uniego_root=args.uniego_root,
                            max_cases_per_group=args.cases,
                            src_fps=emb.data_fps, rep_fps=emb.fps, device=args.device)
    results = ev.evaluate(motion_embed_fn=emb.embed_motion_features, text_embed_fn=emb.embed_llm)

    for grp, ms in results.items():
        log.info("[%s] %s", grp, " ".join(f"{k}={v}" for k, v in sorted(ms.items())))
    if args.out:
        json.dump({"ckpt": args.ckpt, "step": emb.step, "results": results},
                  open(args.out, "w"), indent=2)
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
