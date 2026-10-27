"""GT-only C45 retrieval in fixed 32-pair groups, without generator sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nymeria_data import NymeriaShapeTMRDataset, atomic_json, require_compute, sha256_file
from nymeria_eval import NymeriaTMREmbedder
from st_dataset import collate_st


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def bones_embeddings(args):
    root = args.bones_eval_dir
    signature = json.loads((root / "signature.json").read_text())
    manifest = args.bones_manifest_dir / "bones_repetition_overview.jsonl"
    if sha256_file(manifest) != signature["manifest_sha256"]:
        raise ValueError("BONES manifest differs from saved evaluation")
    if sha256_file(args.bones_manifest_dir / "protocol.json") != signature["protocol_sha256"]:
        raise ValueError("BONES protocol differs from saved evaluation")
    if identity(signature["tmr_checkpoint"]["path"]) != signature["tmr_checkpoint"]:
        raise ValueError("BONES evaluator checkpoint identity changed")
    with manifest.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    case_ids = np.asarray([row["case_id"] for row in rows])
    reference_id = hashlib.sha256(json.dumps({
        key: signature[key] for key in ("manifest_sha256", "tmr_checkpoint", "protocol_sha256")
    }, sort_keys=True).encode()).hexdigest()[:20]
    reference = args.bones_manifest_dir / "reference_cache" / "bones_repetition_overview" / reference_id
    parts = {"index": [], "case_id": [], "embedding": []}
    for start in range(0, len(rows), 32):
        with np.load(reference / f"batch_{start:07d}.npz", allow_pickle=False) as batch:
            for key in parts:
                parts[key].append(np.asarray(batch[key]))
    values = {key: np.concatenate(value) for key, value in parts.items()}
    order = np.argsort(values["index"])
    if not np.array_equal(values["index"][order], np.arange(len(rows))):
        raise ValueError("BONES cached GT does not cover the complete cohort")
    if not np.array_equal(values["case_id"][order], case_ids):
        raise ValueError("BONES motion embedding case IDs are misaligned")
    with np.load(root / "text_embeddings.npz", allow_pickle=False) as batch:
        if not np.array_equal(batch["case_id"], case_ids):
            raise ValueError("BONES text embedding case IDs are misaligned")
        text = np.asarray(batch["embedding"])
    metadata = {
        "evaluator": signature["tmr_checkpoint"],
        "manifest": str(manifest),
        "manifest_sha256": signature["manifest_sha256"],
        "existing_GT_cache": str(reference),
        "existing_text_embeddings": str(root / "text_embeddings.npz"),
        "statistics": "Unchanged embeddings from the pinned BONES C45 evaluator",
    }
    return values["embedding"][order], text, case_ids, metadata


@torch.inference_mode()
def nymeria_embeddings(args):
    run = args.nymeria_run
    cfg = json.loads((run / "config.json").read_text())
    checkpoint = run / "best.pt"
    embedder = NymeriaTMREmbedder(
        str(checkpoint), cfg["stats_path"], cfg["eval_text_cache"], device="cuda"
    )
    manifest = Path(cfg["nymeria_data_dir"]) / "test.jsonl"
    if sha256_file(manifest) != embedder.contract["manifests"]["test"]:
        raise ValueError("Nymeria test manifest differs from the selected evaluator")
    dataset = NymeriaShapeTMRDataset(
        manifest, embedder.rep, train=False,
        src_fps=embedder.data_fps, rep_fps=embedder.fps,
    )
    missing = sorted({row["text"] for row in dataset.rows if row["text"] not in embedder.cache})
    if missing:
        raise ValueError(f"Nymeria text cache is incomplete: {missing[:3]}")
    embedder.motion_encoder.eval()
    embedder.text_encoder.eval()
    loader = DataLoader(
        dataset, batch_size=args.embedding_batch, shuffle=False,
        num_workers=0, collate_fn=collate_st,
    )
    motions, texts = [], []
    count = 0
    started = time.monotonic()
    for batch in loader:
        motion = embedder.motion_encoder({
            "x": batch["features"].to("cuda"),
            "mask": batch["mask"].to("cuda"),
            "neutral_joints": batch["neutral_joints"].to("cuda"),
        })[:, 0]
        llm = embedder.cache.batch(batch["text"])
        text = embedder.text_encoder({
            "x": llm,
            "mask": torch.ones(llm.shape[:2], dtype=torch.bool, device="cuda"),
        })[:, 0]
        motions.append(F.normalize(motion, dim=-1).cpu().numpy())
        texts.append(F.normalize(text, dim=-1).cpu().numpy())
        count += len(motion)
        if count % 1024 == 0 or count == len(dataset):
            print(f"[Nymeria GT encoding] {count}/{len(dataset)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    metadata = {
        "evaluator": identity(checkpoint),
        "selected_step": int(embedder.step),
        "selection": "Existing validation-selected checkpoint; no test-based reselection",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "statistics": cfg["stats_path"],
        "evaluator_contract": embedder.contract,
    }
    return (
        np.concatenate(motions), np.concatenate(texts),
        np.asarray([row["case_id"] for row in dataset.rows]), metadata,
    )


def score_groups(name, motion, text, case_ids, metadata, args):
    n = len(case_ids)
    if len(set(case_ids.tolist())) != n:
        raise ValueError(f"{name}: duplicate case IDs")
    if motion.ndim != 2 or motion.shape != text.shape or len(motion) != n:
        raise ValueError(f"{name}: embedding dimensions/counts disagree")
    if not np.isfinite(motion).all() or not np.isfinite(text).all():
        raise ValueError(f"{name}: nonfinite embeddings; no cases may be silently removed")
    directory = args.out / name
    directory.mkdir(parents=True, exist_ok=True)
    save_npz(directory / "gt_embeddings.npz", motion=motion, text=text, case_id=case_ids)
    motion, text = motion.astype(np.float64), text.astype(np.float64)
    norms = {}
    for label, values in (("motion", motion), ("text", text)):
        norm = np.linalg.norm(values, axis=1)
        if not np.allclose(norm, 1.0, atol=1e-4):
            raise ValueError(f"{name}: cached {label} embeddings are not C45 unit-normalized")
        norms[label] = {"min": float(norm.min()), "max": float(norm.max())}
    used = n // args.group_size * args.group_size
    if used == 0:
        raise ValueError("Not enough cases for a full retrieval group")
    trials = []
    seen = set()
    for seed in args.seeds:
        order = np.random.default_rng(seed).permutation(n)
        groups = order[:used].reshape(-1, args.group_size)
        dropped = order[used:]
        t = text[groups]
        m = motion[groups]
        # Squared Euclidean distance has exactly the same ordering as Euclidean.
        distance = (
            np.sum(t * t, axis=-1)[:, :, None]
            + np.sum(m * m, axis=-1)[:, None, :]
            - 2 * np.einsum("gqd,gkd->gqk", t, m)
        )
        distance = np.maximum(distance, 0)
        sorted_candidates = np.argsort(distance, axis=-1, kind="stable")
        target = np.arange(args.group_size)[None, :, None]
        ranks = np.argmax(sorted_candidates == target, axis=-1) + 1
        diagonal = np.diagonal(distance, axis1=1, axis2=2)[..., None]
        greater = (distance < diagonal).sum(axis=-1)
        ties = (distance == diagonal).sum(axis=-1)
        average_tie_rank = greater + 1 + (ties - 1) * 0.5
        trial = {
            "seed": seed,
            "n_groups": len(groups),
            "n_evaluated": used,
            "n_excluded_incomplete_group": len(dropped),
            "R@1": float((ranks <= 1).mean() * 100),
            "R@2": float((ranks <= 2).mean() * 100),
            "R@3": float((ranks <= 3).mean() * 100),
            "average_tie_R@1": float((average_tie_rank <= 1).mean() * 100),
            "average_tie_R@2": float((average_tie_rank <= 2).mean() * 100),
            "average_tie_R@3": float((average_tie_rank <= 3).mean() * 100),
            "queries_with_exact_distance_ties": int((ties > 1).sum()),
        }
        save_npz(
            directory / f"groups_seed{seed}.npz",
            group_indices=groups, group_case_ids=case_ids[groups],
            dropped_indices=dropped, dropped_case_ids=case_ids[dropped],
            paired_rank=ranks, average_tie_rank=average_tie_rank,
        )
        seen.update(groups.ravel().tolist())
        trials.append(trial)
        print(f"[GT {name}] " + json.dumps(trial, sort_keys=True), flush=True)
    metrics = {
        key: {
            "mean_percent": float(np.mean([trial[key] for trial in trials])),
            "std_percent": float(np.std([trial[key] for trial in trials], ddof=1)),
        }
        for key in ("R@1", "R@2", "R@3")
    }
    report = {
        "dataset": name, "source": "GT", "n_cases": n,
        "n_evaluated_per_shuffle": used, "unique_cases_evaluated_across_shuffles": len(seen),
        "metrics": metrics, "trials": trials, "embedding_norms": norms,
        "GT_self_FID": 0.0, "provenance": metadata,
    }
    atomic_json(directory / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bones-eval-dir", type=Path, required=True)
    parser.add_argument("--bones-manifest-dir", type=Path, required=True)
    parser.add_argument("--nymeria-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--embedding-batch", type=int, default=64)
    args = parser.parse_args()
    if args.group_size != 32 or len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        parser.error("This diagnostic requires 32 candidates and exactly three distinct shuffle seeds")
    require_compute()
    if not torch.cuda.is_available():
        raise RuntimeError("An owned GPU allocation is required")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        raise FileExistsError("Refusing to overwrite a completed retrieval diagnostic")
    protocol = {
        "source": "GT only; no generator loaded or sampled",
        "group_size": args.group_size, "shuffle_seeds": args.seeds,
        "direction": "text query -> motion candidates",
        "positive": "paired diagonal only; no similar-caption positive merging",
        "distance": "squared Euclidean on existing unit-normalized C45 embeddings",
        "tie_policy": "stable candidate-order sort after seeded random pair shuffling; average-tie ranks also saved",
        "remainder_policy": "drop final incomplete group independently per shuffle; exclusions saved",
        "aggregation": "mean and sample standard deviation across three shuffles",
        "uncertainty_scope": "candidate grouping only; not generator variance or a confidence interval",
        "FID": "GT self-reference zero; no generated FID or physical metrics recomputed",
        "dataset_comparability": "Same retrieval pool size, but dataset-specific trained evaluator weights",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    atomic_json(args.out / "protocol.json", protocol)
    reports = []
    for name, loader in (("bones", bones_embeddings), ("nymeria", nymeria_embeddings)):
        reports.append(score_groups(name, *loader(args), args))
    result = {"status": "complete", "protocol": protocol, "GT": reports}
    atomic_json(args.out / "summary.json", result)
    lines = [
        "# GT retrieval: 32 candidates, 3 shuffle seeds",
        "",
        "Paired positives only; text-to-motion; mean +/- sample SD in percentage points.",
        "This is C45 retrieval with a HumanML3D-style candidate pool, not the HumanML3D evaluator.",
        "",
        "| GT dataset | Total cases | Cases/shuffle | R@1 % | R@2 % | R@3 % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        cells = [
            f"{report['metrics'][key]['mean_percent']:.2f} +/- {report['metrics'][key]['std_percent']:.2f}"
            for key in ("R@1", "R@2", "R@3")
        ]
        lines.append(
            f"| {report['dataset']} | {report['n_cases']} | {report['n_evaluated_per_shuffle']} | "
            + " | ".join(cells) + " |"
        )
    (args.out / "LATEST.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[GT retrieval complete] {args.out}", flush=True)


if __name__ == "__main__":
    main()
