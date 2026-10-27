"""Nymeria-only Phase-2 T2M evaluation with the validation-selected evaluator.

Uses the evaluator's exact immutable held-out windows, separate generator
statistics, three paired 32-candidate retrieval groupings, and raw physics.
Training and the existing BONES evaluation entry point remain untouched.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--task-id", required=True, type=int)
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    task = next(t for t in request["tasks"] if t["id"] == args.task_id)
    root = args.request.resolve().parent
    repo = Path(__file__).resolve().parents[1]
    backend_dir = repo / ("motion_expert_joint_attention" if task["backend"] == "nano"
                          else "motion_expert_t2m_edge")
    for path in reversed([backend_dir, repo / "shape_aware_TMR",
                          repo / "motion_expert_joint_attention", repo / "motion_expert", repo]):
        sys.path.insert(0, str(path))
    os.environ["SHAPE_TMR_BUNDLE"] = request["bundle_root"]

    import config
    from nymeria_data import NymeriaShapeTMRDataset, require_compute, sha256_file
    from nymeria_eval import NymeriaTMREmbedder
    from retrieval32 import paired_retrieval32
    from st_dataset import collate_st
    import phase2_quantitative as quant
    import eval_phase2_shape_tmr as ref
    from decode_uniego_torch import decode_joints
    from shape_tmr_eval_common import EvalCase, stable_seed

    require_compute()
    if not torch.cuda.is_available():
        raise RuntimeError("An allocated CUDA GPU is required")
    torch.set_num_threads(1)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    np.random.seed(0)
    if quant.identity(task["checkpoint"]) != task["checkpoint_identity"]:
        raise RuntimeError("Selected generator checkpoint changed")
    for key, path in (("mean_sha256", config.MOTION_STATS_MEAN),
                      ("std_sha256", config.MOTION_STATS_STD)):
        if sha256_file(path) != task[key]:
            raise RuntimeError(f"Generator normalization differs from previous evaluation: {path}")

    out = root / "results" / task["label"]
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = Path(request["data_dir"]) / "test.jsonl"
    manifest_hash = sha256_file(manifest)
    if manifest_hash != request["expected_manifest_sha256"]:
        raise RuntimeError("Held-out test manifest changed")
    evaluator_identity = quant.identity(request["evaluator_checkpoint"])
    evaluator_hash = sha256_file(request["evaluator_checkpoint"])
    signature = {
        "schema": request["schema"], "task": task, "protocol": request["protocol"],
        "manifest_sha256": manifest_hash, "evaluator": evaluator_identity,
        "evaluator_sha256": evaluator_hash,
        "evaluator_stats": request["evaluator_stats"],
        "text_cache": quant.identity(request["text_cache"]),
    }
    signature_path = out / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise RuntimeError("Incompatible previous result directory")
    quant.atomic_json(signature_path, signature)
    embedder = NymeriaTMREmbedder(
        request["evaluator_checkpoint"], request["evaluator_stats"],
        request["text_cache"], str(device))
    if embedder.contract["manifests"]["test"] != manifest_hash:
        raise RuntimeError("Evaluator was trained with a different held-out split")
    dataset = NymeriaShapeTMRDataset(manifest, embedder.rep, train=False,
                                    src_fps=20.0, rep_fps=embedder.fps)
    if len(dataset) != request["expected_cases"]:
        raise RuntimeError("Unexpected case count")
    cases = [
        EvalCase(case_id=row["case_id"], cohort="nymeria_t2m", text=row["text"],
                 num_frames=row["num_frames"], seed=stable_seed("nymeria32:" + row["case_id"]),
                 motion_path=str(Path(request["data_dir"]) / row["features_file"]),
                 gt_start=0, gt_end=row["num_frames"], source_kind="nymeria",
                 floor_offset=row["floor_offset"], uuid=row["uuid"])
        for row in dataset.rows
    ]
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("Duplicate test case IDs")
    quant.atomic_json(out / "cases.json", [case.to_dict() for case in cases])
    skeleton = ref.SOMASkeleton30().to(device)

    def measured(joints, contacts, lengths, neutral, embedding=None):
        if embedding is None:
            embedding = ref._tmr_embed_joints(embedder, joints, lengths, neutral, 20.0)
        values = {
            "embedding": embedding,
            **ref._physical_batch(skeleton, joints, contacts, lengths, 20.0),
            **quant.floor_metrics(joints, lengths, [int(x) for x in skeleton.foot_joint_idx]),
        }
        bones, targets = ref._bone_vectors(
            joints, lengths, neutral, [int(x) for x in skeleton.joint_parents])
        values["bone_length_mae_cm"] = (bones - targets).abs().mean(dim=1) * 100
        return quant.numpy_values(values)

    def progress(stage, **values):
        record = {"stage": stage, "task": task["label"], "job": os.environ["SLURM_JOB_ID"],
                  "time_unix": time.time(), **values}
        quant.atomic_json(out / "progress.json", record)
        print("[progress] " + json.dumps(record), flush=True)

    reference_dir = root / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_signature = {
        "manifest_sha256": manifest_hash, "evaluator_sha256": evaluator_hash,
        "evaluator_stats": embedder.contract["stats_sha256"],
        "text_cache": signature["text_cache"], "protocol": request["protocol"],
    }
    gt_paths = []
    with (reference_dir / "build.lock").open("a") as reference_lock:
        fcntl.flock(reference_lock, fcntl.LOCK_EX)
        ref_sig = reference_dir / "signature.json"
        if ref_sig.exists() and json.loads(ref_sig.read_text()) != reference_signature:
            raise RuntimeError("GT cache belongs to another evaluator/protocol")
        quant.atomic_json(ref_sig, reference_signature)
        for start in range(0, len(cases), 64):
            batch = cases[start:start + 64]
            path = reference_dir / f"batch_{start:07d}.npz"
            gt_paths.append(path)
            if path.exists():
                continue
            packed = collate_st([dataset[i] for i in range(start, start + len(batch))])
            embedding = embedder.embed_motion_features(
                packed["features"], packed["mask"], packed["neutral_joints"])
            raw = [dataset.raw_window(i)[0] for i in range(start, start + len(batch))]
            lengths = torch.tensor([len(x) for x in raw], device=device)
            features = torch.empty(len(batch), int(lengths.max()), 283, device=device)
            for i, value in enumerate(raw):
                value = torch.from_numpy(value).to(device)
                features[i, :len(value)] = value
                features[i, len(value):] = value[-1]
            neutral = packed["neutral_joints"].to(device)
            values = measured(decode_joints(features), features[..., 279:283] > 0.5,
                              lengths, neutral, embedding=embedding)
            quant.atomic_npz(path, index=np.arange(start, start + len(batch)),
                             case_id=np.array([x.case_id for x in batch]), **values)
            if start % 1024 == 0 or start + len(batch) == len(cases):
                progress("GT", completed=start + len(batch), total=len(cases))
        ground_truth = quant.join_parts(gt_paths, cases)
        text_path = reference_dir / "text_embeddings.npz"
        if not text_path.exists():
            text = np.concatenate([
                embedder.embed_text([case.text for case in cases[start:start + 128]]).cpu().numpy()
                for start in range(0, len(cases), 128)
            ])
            quant.atomic_npz(text_path, embedding=text,
                             case_id=np.array([case.case_id for case in cases]))
        with np.load(text_path, allow_pickle=False) as saved:
            if saved["case_id"].tolist() != [case.case_id for case in cases]:
                raise RuntimeError("Text cache case order changed")
            text = np.array(saved["embedding"])
        gt_retrieval = paired_retrieval32(ground_truth["embedding"], text)
        original = json.loads(Path(request["gt_reference_report"]).read_text())
        for metric in ("R32_R01_mean", "R32_R02_mean", "R32_R03_mean"):
            if abs(gt_retrieval[metric] - original["GT"][metric]) > 0.05:
                raise RuntimeError(f"GT metric differs from selected-evaluator test: {metric}")
        gt_report = {"source": "GT", "num_cases": len(cases), "retrieval": gt_retrieval,
                     "FID": 0.0, "physics_20fps": quant.physical_summary(ground_truth),
                     "signature": reference_signature}
        quant.atomic_json(reference_dir / "summary.json", gt_report)
        progress("GT_ready", GT_R3=gt_retrieval["R32_R03_mean"])

    # Text embeddings have been materialized; retain only motion-side inference state.
    embedder.cache = None
    gc.collect()
    torch.cuda.empty_cache()
    eval_args = argparse.Namespace(
        backend=task["backend"], checkpoint=task["checkpoint"],
        expected_step=task["step"], cohort="nymeria_t2m")
    batch_size = task["batch_size"]
    generated_paths = [out / "batches" / f"batch_{start:07d}.npz"
                       for start in range(0, len(cases), batch_size)]
    if any(not path.exists() for path in generated_paths):
        progress("loading_generator")
        model, mean, std, metadata = quant.strict_model(eval_args, device)
        quant.atomic_json(out / "checkpoint_metadata.json", metadata)
        started = time.monotonic()
        newly_completed = 0
        for start, path in zip(range(0, len(cases), batch_size), generated_paths):
            if path.exists():
                continue
            batch = cases[start:start + batch_size]
            neutrals = []
            for row in dataset.rows[start:start + len(batch)]:
                neutral = np.load(Path(request["data_dir"]) / row["neutral_file"],
                                  allow_pickle=False).astype(np.float32)
                neutrals.append(neutral - neutral.mean(axis=0, keepdims=True))
            neutral = torch.from_numpy(np.stack(neutrals)).to(device)
            normalized, lengths = quant.sample_batch(
                eval_args, ref, model, batch, neutral, metadata)
            if not torch.isfinite(normalized).all():
                raise FloatingPointError(f"Nonfinite generated motion at index {start}")
            _, joints, contacts = ref._decode_generated(normalized, lengths, mean, std)
            values = measured(joints, contacts, lengths, neutral)
            if any(not np.isfinite(value).all() for value in values.values()):
                raise FloatingPointError(f"Nonfinite measured output at index {start}")
            quant.atomic_npz(path, index=np.arange(start, start + len(batch)),
                             case_id=np.array([case.case_id for case in batch]), **values)
            newly_completed += len(batch)
            elapsed = time.monotonic() - started
            done = start + len(batch)
            progress("generating", completed=done, total=len(cases),
                     cases_per_second=newly_completed / elapsed,
                     eta_hours=(len(cases) - done) * elapsed / newly_completed / 3600,
                     peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
        del model
        torch.cuda.empty_cache()
    generated = quant.join_parts(generated_paths, cases)
    from kimodo.metrics.tmr import calculate_activation_statistics, calculate_frechet_distance
    gen_mu, gen_cov = calculate_activation_statistics(generated["embedding"])
    gt_mu, gt_cov = calculate_activation_statistics(ground_truth["embedding"])
    generated_retrieval = paired_retrieval32(generated["embedding"], text)

    ids = np.array([case.case_id for case in cases])
    for seed in (0, 1, 2):
        order = np.random.default_rng(seed).permutation(len(cases))
        used = len(cases) // 32 * 32
        groups, dropped = order[:used].reshape(-1, 32), order[used:]
        data = {"group_indices": groups, "group_case_ids": ids[groups],
                "dropped_indices": dropped, "dropped_case_ids": ids[dropped]}
        for name, motion in (("generated", generated["embedding"]),
                             ("GT", ground_truth["embedding"])):
            t, m = text[groups].astype(np.float64), motion[groups].astype(np.float64)
            distances = np.maximum((t*t).sum(-1)[:, :, None] + (m*m).sum(-1)[:, None, :]
                                   - 2*np.einsum("gqd,gkd->gqk", t, m), 0)
            candidates = np.argsort(distances, axis=-1, kind="stable")
            data[name + "_paired_rank"] = (
                np.argmax(candidates == np.arange(32)[None, :, None], axis=-1) + 1)
        quant.atomic_npz(out / f"groups_seed{seed}.npz", **data)
    result = {
        "status": "complete", "task": task, "signature": signature,
        "num_cases": len(cases), "num_sequences": len({case.uuid for case in cases}),
        "coverage_verified": True, "skipped_nonfinite": 0,
        "GT": gt_report,
        "generated": {
            "retrieval": generated_retrieval,
            "FID": float(calculate_frechet_distance(gen_mu, gen_cov, gt_mu, gt_cov)),
            "physics_20fps": quant.physical_summary(generated),
        },
        "interpretation": [
            "R32 is paired-only retrieval, mean/sample SD across three grouping seeds.",
            "FID is in the NEW Nymeria evaluator space, not comparable to BONES FID.",
            "GT uses camera/head-aligned data; each generator keeps its trained representation.",
            "No generated floor correction or centroid removal; physical metrics are separate.",
        ],
    }
    quant.atomic_json(out / "summary.json", result)
    progress("complete", generated_R3=generated_retrieval["R32_R03_mean"],
             GT_R3=gt_retrieval["R32_R03_mean"], FID=result["generated"]["FID"])
    print("[summary] " + json.dumps(result), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
