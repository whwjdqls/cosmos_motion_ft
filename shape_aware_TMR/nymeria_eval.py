"""Nymeria GT retrieval and a length-safe inference interface for Nymeria C45."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kimodo.metrics.tmr import compute_tmr_retrieval_metrics
from st_dataset import collate_st, resample_joints_time
from st_eval import ShapeTMREmbedder
from retrieval32 import paired_retrieval32
from decode_uniego import decode_joints
from nymeria_data import (
    NymeriaShapeTMRDataset, atomic_json, require_compute, sha256_file, stats_fingerprint,
)


def retrieval_metrics(motion, text):
    """Keep the C45 protocol; add average-tie ranks alongside its legacy strict ranks."""
    if not np.isfinite(motion).all() or not np.isfinite(text).all():
        raise ValueError("non-finite embeddings; refusing to change the candidate pool")
    motion_std = float(np.linalg.norm(motion.std(axis=0)))
    text_std = float(np.linalg.norm(text.std(axis=0)))
    n = len(motion)
    if n < 2:
        raise ValueError("retrieval requires at least two fixed cases")
    if min(motion_std, text_std) < 1e-4:
        return {"COLLAPSED": 1.0, "n_cases": n, "motion_std_norm": motion_std,
                "text_std_norm": text_std}, []
    official = compute_tmr_retrieval_metrics(motion, text)
    metrics = {key.split("/")[-1]: float(value) for key, value in official.items()
               if "t2m_R" in key}
    ranks, average_ranks = [], []
    duplicate_099 = duplicate_096 = 0
    for start in range(0, n, 256):
        end = min(start + 256, n)
        scores = text[start:end] @ motion.T
        diagonal = scores[np.arange(end - start), np.arange(start, end)][:, None]
        greater = (scores > diagonal).sum(axis=1)
        ties = (scores == diagonal).sum(axis=1)
        ranks.extend((greater + 1).tolist())
        average_ranks.extend((greater + 1 + (ties - 1) * 0.5).tolist())
        self_sim = text[start:end] @ text.T
        self_sim[np.arange(end - start), np.arange(start, end)] = -np.inf
        duplicate_099 += int((self_sim > 0.99).sum())
        duplicate_096 += int((self_sim > 0.96).sum())
    ranks = np.asarray(ranks)
    average_ranks = np.asarray(average_ranks)
    for k in (1, 3, 5, 10):
        metrics[f"plain_R{k:02d}"] = float((ranks <= k).mean() * 100)
        metrics[f"plain_avg_tie_R{k:02d}"] = float((average_ranks <= k).mean() * 100)
    metrics.update(
        plain_MedR=float(np.median(ranks)),
        plain_avg_tie_MedR=float(np.median(average_ranks)),
        dup_frac=duplicate_099 / (n * (n - 1)),
        merge_frac096=duplicate_096 / (n * (n - 1)),
        motion_std_norm=motion_std, text_std_norm=text_std, n_cases=n,
        skipped_nonfinite=0, GT_self_FID=0.0,
    )
    if metrics["merge_frac096"] > 0.01:
        metrics["PROTOCOL_INFLATED"] = 1.0
    per_case = [{"plain_rank": int(a), "average_tie_rank": float(b)}
                for a, b in zip(ranks, average_ranks)]
    return metrics, per_case


class NymeriaEvaluator:
    """GT-only validation/test retrieval; no test-based checkpoint selection."""

    def __init__(self, dataset, text_cache, *, device="cuda", split_name="validation"):
        self.dataset = dataset
        self.text_cache = text_cache
        self.device = device
        self.split_name = split_name
        self.last_cases = []
        missing = sorted({row["text"] for row in dataset.rows if row["text"] not in text_cache})
        if missing:
            raise ValueError(f"Nymeria {split_name} cache missing {len(missing)} captions: {missing[:3]}")

    @torch.no_grad()
    def evaluate(self, motion_encoder, text_encoder, chunk=64):
        old_motion, old_text = motion_encoder.training, text_encoder.training
        motion_encoder.eval()
        text_encoder.eval()
        motions, texts, lengths = [], [], []
        try:
            loader = DataLoader(self.dataset, batch_size=chunk, shuffle=False, num_workers=0,
                                collate_fn=collate_st)
            for batch in loader:
                motion = motion_encoder({
                    "x": batch["features"].to(self.device),
                    "mask": batch["mask"].to(self.device),
                    "neutral_joints": batch["neutral_joints"].to(self.device),
                })[:, 0]
                llm = self.text_cache.batch(batch["text"])
                text = text_encoder({
                    "x": llm,
                    "mask": torch.ones(llm.shape[:2], dtype=torch.bool, device=self.device),
                })[:, 0]
                motions.append(F.normalize(motion, dim=-1).cpu().numpy())
                texts.append(F.normalize(text, dim=-1).cpu().numpy())
                lengths.extend(batch["length"].tolist())
            all_motion, all_text = np.concatenate(motions), np.concatenate(texts)
            metrics, ranks = retrieval_metrics(all_motion, all_text)
            metrics.update(paired_retrieval32(all_motion, all_text))
            metrics["unique_captions"] = len({row["text"] for row in self.dataset.rows})
            self.last_cases = [
                {"case_id": row["case_id"], "uuid": row["uuid"], "text": row["text"],
                 "source_frames": row["num_frames"], "representation_frames": length,
                 **(ranks[i] if ranks else {})}
                for i, (row, length) in enumerate(zip(self.dataset.rows, lengths))
            ]
            return {f"nymeria/{self.split_name}": metrics}
        finally:
            motion_encoder.train(old_motion)
            text_encoder.train(old_text)

    def write_report(self, path, *, step, metrics, checkpoint=None, contract=None):
        atomic_json(path, {
            "dataset": "nymeria", "split": self.split_name, "step": step,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "manifest": str(self.dataset.manifest),
            "manifest_sha256": sha256_file(self.dataset.manifest),
            "GT": metrics, "per_case": self.last_cases, "contract": contract,
            "selection_allowed": self.split_name == "validation",
            "protocol": "C45 full-pool retrieval plus plain strict/average-tie ranks",
            "retrieval32_protocol": {
                "group_size": 32, "shuffle_seeds": [0, 1, 2],
                "positive": "paired diagonal only", "drop_last": True,
                "ranking": "squared Euclidean, stable order after random grouping",
                "aggregation": "mean and sample standard deviation across grouping seeds",
                "direction": "text query to motion candidates",
            },
            "physical_metrics": {
                "foot_skating": "not computed by this retrieval-only evaluator",
                "penetration": "not computed by this retrieval-only evaluator",
            },
            "FID_note": "GT-to-itself reference is zero; no generated-motion FID measured here",
        })


class NymeriaTMREmbedder(ShapeTMREmbedder):
    """Same C45 weights/API, but crop valid frames BEFORE temporal resampling."""

    def __init__(self, ckpt_path, stats_path, text_cache_path, device="cuda"):
        super().__init__(ckpt_path, stats_path, text_cache_path, device)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.contract = ck.get("nymeria_contract")
        if not self.contract or self.contract.get("dataset") != "nymeria":
            raise ValueError("expected a Nymeria evaluator checkpoint, not BONES C45")
        if stats_fingerprint(stats_path) != self.contract["stats_sha256"]:
            raise ValueError("evaluator mean/std do not match the Nymeria checkpoint")

    @torch.no_grad()
    def embed_motion(self, motion, neutral_joints, lengths=None, source_fps=None):
        source_fps = self.data_fps if source_fps is None else float(source_fps)
        if source_fps <= 0:
            raise ValueError("source FPS must be positive")
        count, padded_t = motion.shape[:2]
        lengths = [padded_t] * count if lengths is None else [int(x) for x in lengths]
        if len(lengths) != count or any(n < 1 or n > padded_t for n in lengths):
            raise ValueError("invalid true sequence lengths")
        values = []
        for i, length in enumerate(lengths):
            raw = motion[i:i + 1, :length].detach().float().cpu()
            if raw.ndim == 3 and raw.shape[-1] == 283:
                raw = decode_joints(raw)
            if raw.ndim != 4 or raw.shape[2:] != (30, 3):
                raise ValueError("expected raw UniEgo283 or raw SOMA30 joints")
            raw = resample_joints_time(raw, source_fps, self.fps)
            value = self.rep(posed_joints=raw, to_normalize=True, to_canonicalize=True,
                             lengths=torch.tensor([raw.shape[1]]))[0]
            if not torch.isfinite(value).all():
                raise ValueError("non-finite evaluator features")
            values.append(value)
        max_t = max(len(x) for x in values)
        features = torch.zeros(count, max_t, self.rep.motion_rep_dim)
        mask = torch.zeros(count, max_t, dtype=torch.bool)
        for i, value in enumerate(values):
            features[i, :len(value)] = value
            mask[i, :len(value)] = True
        neutral = neutral_joints.detach().float().cpu()
        neutral = neutral - neutral.mean(dim=1, keepdim=True)
        return self.embed_motion_features(features, mask, neutral)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--text-cache", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    require_compute()
    embedder = NymeriaTMREmbedder(args.ckpt, args.stats_path, args.text_cache, args.device)
    manifest = Path(args.data_dir) / f"{args.split}.jsonl"
    if sha256_file(manifest) != embedder.contract["manifests"][args.split]:
        raise ValueError("evaluation manifest differs from the checkpoint's held-out split")
    dataset = NymeriaShapeTMRDataset(manifest, embedder.rep, train=False,
                                    src_fps=embedder.data_fps, rep_fps=embedder.fps)
    evaluator = NymeriaEvaluator(dataset, embedder.cache, device=args.device, split_name=args.split)
    metrics = evaluator.evaluate(embedder.motion_encoder, embedder.text_encoder)
    evaluator.write_report(args.out, step=embedder.step, metrics=metrics[f"nymeria/{args.split}"],
                           checkpoint=args.ckpt, contract=embedder.contract)
    print(json.dumps({"step": embedder.step, "GT": metrics, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
