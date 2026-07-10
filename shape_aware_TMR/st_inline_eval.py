"""Retrieval evaluator on the official Kimodo benchmark testsuite (TAP eval_retrieval pattern).

Pools come from `Kimodo-Motion-Gen-Benchmark-20fps/testsuite/<split>/text2motion/<group>/`
(<split> in {content, repetition}; <group> in {overview, timeline_single, timeline_multi})
— held-out benchmark cases covering ALL THREE prompt types (incl. multi, which the raw test
splits lack). Per case: text from `meta.json` (embedding from the benchmark llm2vec cache),
motion resolved via `seed_motion.json.bvh_path` -> the PROPORTIONAL uniego tree, cropped with
the 30fps->20fps-scaled indices, decoded to raw joints, featurized by TMRMotionRep
(canonicalize+normalize — must match training), plus the actor's centered neutral_joints.

Keeps TAP's posterior-collapse guard (constant embeddings fake 100% R@k -> report COLLAPSED).
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from kimodo.metrics.tmr import compute_tmr_retrieval_metrics
from kimodo.skeleton import SOMASkeleton30

from decode_uniego import decode_joints
from st_dataset import DATA_ROOT, resample_joints_time

log = logging.getLogger(__name__)

TESTSUITE = "/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite"
BENCH_TEXT_CACHE = "/home/jungbin_cho/kimodo_caches/benchmark_llm2vec.pt"


class TestsuiteEvaluator:
    """text_cache: LLM2VecCache (v1) or None (v2 — raw texts stored; pass `text_embed_fn`
    to evaluate(), which maps list[str] -> (B, latent) latents)."""

    def __init__(self, motion_rep, text_cache=None, groups=(("content", "overview"), ("content", "timeline_single")),
                 testsuite: str = TESTSUITE, uniego_root: str = DATA_ROOT,
                 max_cases_per_group: Optional[int] = 500, min_frames: int = 10,
                 src_fps: float = 20.0, rep_fps: float = 20.0,
                 device: str = "cuda", seed: int = 0):
        self.device = device
        self.pools = {}
        self.src_fps = float(src_fps)
        self.rep_fps = float(rep_fps)
        self.soma30 = SOMASkeleton30()
        self.soma30_from77 = self.soma30.get_skel_slice(self.soma30.somaskel77)
        rng = np.random.default_rng(seed)
        case_limit = None if max_cases_per_group is None or max_cases_per_group <= 0 else int(max_cases_per_group)
        for split, grp in groups:
            case_dirs = sorted(glob.glob(f"{testsuite}/{split}/text2motion/{grp}/*"))
            rng.shuffle(case_dirs)
            cases, miss = [], 0
            for cd in case_dirs:
                if case_limit is not None and len(cases) >= case_limit:
                    break
                try:
                    text = json.load(open(os.path.join(cd, "meta.json")))["text"]
                    sm = json.load(open(os.path.join(cd, "seed_motion.json")))
                except (OSError, json.JSONDecodeError, KeyError):
                    miss += 1
                    continue
                if text_cache is not None and text not in text_cache:
                    miss += 1
                    continue
                rel = sm["bvh_path"]
                rel = rel[4:] if rel.startswith("BVH/") else rel
                rel = rel[:-4] if rel.endswith(".bvh") else rel
                p = os.path.join(uniego_root, rel + ".npz")
                if not os.path.isfile(p):
                    miss += 1
                    continue
                a = int(round(int(sm["crop_start_frame_index"]) * self.src_fps / 30))
                b = int(round(int(sm["crop_end_frame_index"]) * self.src_fps / 30))
                try:
                    with np.load(p, mmap_mode="r") as d:
                        if "features" in d:
                            n_total = int(d["features"].shape[0])
                        elif "posed_joints" in d:
                            n_total = int(d["posed_joints"].shape[0])
                        else:
                            miss += 1
                            continue
                        a = max(0, min(a, n_total)); b = max(a, min(b, n_total))
                        if b - a < min_frames:
                            miss += 1
                            continue
                        if "features" in d:
                            win = np.asarray(d["features"][a:b]).astype(np.float32)
                            nj = np.asarray(d["neutral_joints"]).astype(np.float32)
                            input_kind = "uniego"
                        else:
                            win = np.asarray(d["posed_joints"][a:b]).astype(np.float32)
                            nj = self.soma30.neutral_joints.cpu().numpy().astype(np.float32)
                            input_kind = "posed_joints"
                except (OSError, KeyError, ValueError):
                    miss += 1
                    continue
                if not (np.isfinite(win).all() and np.isfinite(nj).all()):
                    miss += 1
                    continue
                T = win.shape[0]
                with torch.no_grad():
                    if input_kind == "uniego":
                        joints = decode_joints(torch.from_numpy(win).unsqueeze(0))
                    else:
                        if win.shape[1] == 77:
                            win = win[:, self.soma30_from77]
                        elif win.shape[1] != 30:
                            miss += 1
                            continue
                        joints = torch.from_numpy(win).unsqueeze(0)
                    joints = resample_joints_time(joints, self.src_fps, self.rep_fps)
                    T_rep = int(joints.shape[1])
                    feats = motion_rep(posed_joints=joints, to_normalize=True,
                                       to_canonicalize=True, lengths=torch.tensor([T_rep]))[0]
                if not torch.isfinite(feats).all():
                    miss += 1
                    continue
                nj = nj - nj.mean(axis=0, keepdims=True)
                case = {
                    "features": feats.float(),                        # (T,186) normalized
                    "neutral_joints": torch.from_numpy(nj),           # (30,3) centered
                    "text": text,
                }
                if text_cache is not None:
                    case["llm"] = text_cache.batch([text]).cpu()      # (1,1,4096)
                cases.append(case)
            self.pools[f"{split}/{grp}"] = cases
            log.info("testsuite pool[%s/%s]: %d cases (skipped %d)", split, grp, len(cases), miss)

    @torch.no_grad()
    def evaluate(self, motion_encoder=None, text_encoder=None, text_embed_fn=None,
                 motion_embed_fn=None, chunk: int = 64) -> dict:
        """v1: pass (motion_encoder, text_encoder) — VAE mu tokens, cache llm vecs in pool.
        v2/v3: pass callables — motion_embed_fn(feats, mask, nj) -> (B,D) and/or
        text_embed_fn(llm (B,1,4096) or list[str]) -> (B,D) retrieval embeddings."""
        was_training = motion_encoder.training if motion_encoder is not None else False
        if motion_encoder is not None:
            motion_encoder.eval()
        if text_encoder is not None:
            text_encoder.eval()
        out = {}
        for name, cases in self.pools.items():
            if not cases:
                continue
            zm, zt = [], []
            for s in range(0, len(cases), chunk):
                cs = cases[s:s + chunk]
                Tm = max(c["features"].shape[0] for c in cs)
                B = len(cs)
                feats = torch.zeros(B, Tm, cs[0]["features"].shape[-1])
                mask = torch.zeros(B, Tm, dtype=torch.bool)
                for i, c in enumerate(cs):
                    t = c["features"].shape[0]
                    feats[i, :t] = c["features"]; mask[i, :t] = True
                nj = torch.stack([c["neutral_joints"] for c in cs])
                feats, mask, nj = (x.to(self.device) for x in (feats, mask, nj))
                if motion_embed_fn is not None:
                    m_vec = motion_embed_fn(feats, mask, nj)
                else:
                    m_vec = motion_encoder({"x": feats, "mask": mask, "neutral_joints": nj})[:, 0]  # mu
                zm.append(F.normalize(m_vec, dim=-1).cpu())
                if text_embed_fn is not None:
                    llm = (torch.cat([c["llm"] for c in cs], dim=0).to(self.device)
                           if "llm" in cs[0] else [c["text"] for c in cs])
                    t_vec = text_embed_fn(llm)
                else:
                    llm = torch.cat([c["llm"] for c in cs], dim=0).to(self.device)
                    t_out = text_encoder({"x": llm, "mask": torch.ones(B, 1, dtype=torch.bool, device=self.device)})
                    t_vec = t_out[:, 0]
                zt.append(F.normalize(t_vec, dim=-1).cpu())
            zm = torch.cat(zm).numpy(); zt = torch.cat(zt).numpy()
            finite = np.isfinite(zm).all(axis=1) & np.isfinite(zt).all(axis=1)
            skipped_nonfinite = int((~finite).sum())
            if skipped_nonfinite:
                log.warning(
                    "testsuite pool[%s]: dropping %d non-finite embedding rows before metrics",
                    name, skipped_nonfinite,
                )
                zm = zm[finite]
                zt = zt[finite]
            if len(zm) == 0:
                out[name] = {"COLLAPSED": 1.0, "skipped_nonfinite": skipped_nonfinite}
                continue
            # posterior-collapse guard: constant embeddings fake 100% R@k
            if float(np.linalg.norm(zm.std(axis=0))) < 1e-4 or float(np.linalg.norm(zt.std(axis=0))) < 1e-4:
                out[name] = {"COLLAPSED": 1.0, "skipped_nonfinite": skipped_nonfinite}
                continue
            # plain (NO text-dedup) retrieval — honest at init, where the protocol metric's
            # 0.99 text-self-sim dedup merges everything (untrained text embs all cluster).
            sim = zt @ zm.T                                              # (N,N) t2m
            ranks = (sim > sim.diagonal()[:, None]).sum(axis=1)          # 0 = best
            plain = {"plain_R01": float((ranks < 1).mean() * 100),
                     "plain_R03": float((ranks < 3).mean() * 100),
                     "plain_MedR": float(np.median(ranks) + 1)}
            tt = zt @ zt.T
            n = tt.shape[0]
            plain["dup_frac"] = float(((tt > 0.99).sum() - n) / max(1, n * (n - 1)))  # off-diag >0.99
            # the protocol metric MERGES captions at cosine > 0.96 — measure at ITS threshold and
            # flag inflation (clustered-but-not-collapsed embeddings fake high protocol R@k while
            # plain metrics stay near-random; observed with high dropout early in training).
            merge_frac = float(((tt > 0.96).sum() - n) / max(1, n * (n - 1)))
            plain["merge_frac096"] = merge_frac
            if merge_frac > 0.01:
                plain["PROTOCOL_INFLATED"] = 1.0
            plain["skipped_nonfinite"] = skipped_nonfinite
            m = compute_tmr_retrieval_metrics(zm, zt)                    # published protocol (dedup)
            out[name] = {**{k.split("/")[-1]: v for k, v in m.items() if "t2m_R" in k or "FID" in k},
                         **plain}
        if was_training and motion_encoder is not None:
            motion_encoder.train()
            if text_encoder is not None:
                text_encoder.train()
        return out
