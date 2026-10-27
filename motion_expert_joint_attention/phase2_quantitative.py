"""Resumable full-window C45 evaluation for Cosmos Phase-2 Edge and Nano.

Generator statistics are never used for C45 normalization. Physics is measured
on raw decoded outputs, without generated grounding or centroid alignment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_features(case):
    from motion_array_cache import load_motion_window
    from uniego_layout import canonicalize_frame0, ground_features

    features, neutral = load_motion_window(case.motion_path, case.gt_start, case.gt_end)
    features = np.asarray(features, dtype=np.float32)
    neutral = np.asarray(neutral, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 283 or len(features) < 2:
        raise ValueError(f"{case.case_id}: invalid features {features.shape}")
    if case.source_kind == "nymeria" and case.floor_offset is not None:
        features = ground_features(features, float(case.floor_offset))
    features = canonicalize_frame0(features)
    neutral = neutral - neutral.mean(axis=0, keepdims=True)
    if neutral.shape != (30, 3) or not np.isfinite(features).all() or not np.isfinite(neutral).all():
        raise ValueError(f"{case.case_id}: nonfinite motion or invalid skeleton")
    return features, neutral


def prepare(args):
    import config
    import prepare_shape_tmr_eval as preparation
    from dataset import SHARED_MEAN_PATH, SHARED_STD_PATH
    from nymeria_joint_dataset import NymeriaJointDataset
    from shape_tmr_eval_common import add_bundle_python_paths, sha256_file, write_jsonl

    add_bundle_python_paths(args.bundle_root)
    from kimodo.sanitize import sanitize_text

    out = Path(args.manifest_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError(f"Refusing to overwrite immutable prepared manifests: {out}")
    preparation.load_case_features = load_features
    bones, bones_neutral, bones_audit = preparation.build_bones_cases(
        Path(args.testsuite), Path(args.bones_root), "repetition", "overview"
    )
    bones = [replace(case, text=sanitize_text(case.text)) for case in bones]
    if not bones:
        raise RuntimeError("BONES repetition/overview has no eligible cases")
    write_jsonl(out / "bones_repetition_overview.jsonl", bones)
    np.save(out / "bones_repetition_overview_neutral.npy", bones_neutral)
    print("[prepare] BONES " + json.dumps(bones_audit), flush=True)

    dataset = NymeriaJointDataset(
        split="test", num_frames=200, aligned_num_frames=97,
        task_weights={"text2motion": 0.75, "textimg2motion": 0.25},
        bones_text2motion_frac=0.0, cfg_dropout=0.0, prefer_latents=False,
        force_on_the_fly=True, reasoner_image_for_textimg=True,
        reasoner_image_size=256, train=False,
    )
    mean = np.load(SHARED_MEAN_PATH).astype(np.float32)
    std = np.load(SHARED_STD_PATH).astype(np.float32)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("Invalid common GT eligibility statistics")
    cohorts, neutrals, audits = preparation.build_nymeria_cases(dataset, mean, std)
    for name, cases in cohorts.items():
        if not cases:
            raise RuntimeError(f"{name}: no eligible cases")
        write_jsonl(out / f"{name}.jsonl", cases)
        np.save(out / f"{name}_neutral.npy", neutrals[name])
        audits[name]["windows_per_sequence"] = dict(sorted(Counter(c.uuid for c in cases).items()))
        print("[prepare] " + name + " " + json.dumps(audits[name]), flush=True)
    cohorts["bones_repetition_overview"] = bones
    caption_list = sorted({""} | {case.text for cases in cohorts.values() for case in cases})
    atomic_json(out / "evaluator_captions.json", caption_list)
    protocol = {
        "version": 1,
        "cohorts": {"bones_repetition_overview": bones_audit, **audits},
        "nymeria_split_contents": json.loads(Path(config.NYMERIA_SPLIT_FILE).read_text()),
        "sampling": {"solver": "unipc", "steps": 35, "guidance": 2.0, "shift": 3.0,
                     "fps": 20, "per_case_noise": "CPU torch.Generator seeded by manifest"},
        "nymeria_policy": "All test annotation windows in native dataset indices, including multiple captions per physical window; no random __getitem__ retries or five-person exclusion. Existing floor validity filter and common legacy-stat |z|max <= 20 eligibility guard apply identically to every model.",
        "reference_representation": "Original proportional UniEgo, floor-calibrated for Nymeria; original proportional BONES. Common GT and centered neutral shape for every checkpoint.",
        "generator_statistics": "Each checkpoint's own verified statistics; never substitute evaluator or another generator's statistics.",
        "physics": "Raw decoded SOMA-30 meters, Y-up, floor y=0, 20fps. No generated grounding, translation alignment or centroid removal. Valid frames only.",
        "penetration": "Per-frame maximum negative joint height; mean/max depth in cm and frame fraction deeper than 1cm. Both all-joint and foot-only versions; also lowest-foot height and airborne fraction.",
        "retrieval": "Full-cohort C45 text-to-motion R@3, official similar-caption positives and average ties; not minibatch retrieval.",
        "files": {name: sha256_file(out / f"{name}.jsonl") for name in cohorts},
        "sha256": {"floor_calibration": sha256_file(config.FLOOR_CALIBRATION_JSON),
                   "eligibility_mean": sha256_file(SHARED_MEAN_PATH),
                   "eligibility_std": sha256_file(SHARED_STD_PATH)},
        "bundle": str(Path(args.bundle_root).resolve()),
    }
    atomic_json(out / "protocol.json", protocol)
    print(f"[prepare] READY {out}", flush=True)


def floor_metrics(joints, lengths, feet):
    result = {
        "floor_depth_mean_cm": [], "floor_depth_max_cm": [],
        "floor_frames_below_minus1cm": [], "foot_depth_mean_cm": [],
        "foot_depth_max_cm": [], "foot_frames_below_minus1cm": [],
        "lowest_foot_height_mean_cm": [], "all_feet_above5cm_frame_fraction": [],
        "valid_frames": [],
    }
    for index, length in enumerate(lengths.tolist()):
        y = joints[index, :length, :, 1]
        floor_y = y.amin(dim=-1)
        foot_y = y[:, feet].amin(dim=-1)
        for prefix, values in (("floor", floor_y), ("foot", foot_y)):
            depth = (-values).clamp_min(0)
            result[prefix + "_depth_mean_cm"].append(depth.mean() * 100)
            result[prefix + "_depth_max_cm"].append(depth.amax() * 100)
            result[prefix + "_frames_below_minus1cm"].append((values < -0.01).float().mean())
        result["lowest_foot_height_mean_cm"].append(foot_y.mean() * 100)
        result["all_feet_above5cm_frame_fraction"].append((foot_y > 0.05).float().mean())
        result["valid_frames"].append(torch.tensor(length, device=joints.device))
    return {key: torch.stack(values) for key, values in result.items()}


def numpy_values(values):
    return {key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            for key, value in values.items()}


def measure(ref, embedder, skeleton, joints, contacts, lengths, neutral):
    values = {
        "embedding": ref._tmr_embed_joints(embedder, joints, lengths, neutral, 20.0),
        **ref._physical_batch(skeleton, joints, contacts, lengths, 20.0),
        **floor_metrics(joints, lengths, [int(x) for x in skeleton.foot_joint_idx]),
    }
    bones, targets = ref._bone_vectors(
        joints, lengths, neutral, [int(x) for x in skeleton.joint_parents]
    )
    values["bone_length_mae_cm"] = (bones - targets).abs().mean(dim=1) * 100
    return numpy_values(values)


def strict_model(args, device):
    import config

    if args.backend == "edge":
        from checkpoint import configure_attention_from_checkpoint, load_checkpoint
        from model import EdgePhase2MotionExpert
        configure_attention_from_checkpoint(args.checkpoint)
        model = EdgePhase2MotionExpert(device=device, dtype=torch.bfloat16)
        checkpoint = load_checkpoint(args.checkpoint, model=model)
    else:
        import sample
        model, cosmos, checkpoint = sample.load_joint_model(
            args.checkpoint, device=device, motion_native_solver_cli="unipc"
        )
        state = checkpoint.get("model", checkpoint)
        own = dict(model.named_all_parameters())
        own.update(dict(model.named_buffers()))
        own.update({model._NET_PREFIX + key: value for key, value in cosmos.net.named_buffers()})
        invalid = [key for key, value in state.items()
                   if key not in own or own[key].shape != value.shape]
        if not state or invalid:
            raise RuntimeError(f"Nano checkpoint overlay incomplete: {invalid[:30]}")
        if model.objective != "x0" or model.motion_schedule != "native":
            raise RuntimeError("Expected a native-schedule x0 Nano checkpoint")
        if abs(float(model.motion_shift) - 3.0) > 1e-8:
            raise RuntimeError(f"Unexpected checkpoint shift: {model.motion_shift}")
        for key in ("init_gen", "init_motion"):
            value = checkpoint.get("args", {}).get(key)
            if value and not Path(sample.resolve_legacy_path(value)).is_file():
                raise RuntimeError(f"Missing recorded initialization: {key}={value}")
    expected_step = int(getattr(args, "expected_step", 200000))
    if int(checkpoint.get("step", -1)) != expected_step:
        raise RuntimeError(f"Expected step {expected_step}, got {checkpoint.get('step')}")
    mode = "textimg2motion" if args.cohort == "nymeria_ti2m" else "text2motion"
    trained = checkpoint.get("args", {}) or {}
    weights = trained.get("task_weights")
    if isinstance(weights, dict) and float(weights.get(mode, 0)) <= 0:
        raise RuntimeError(f"Checkpoint was not trained for {mode}: {weights}")
    model.eval()
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN).astype(np.float32)).to(device)
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD).astype(np.float32)).to(device)
    metadata = {"step": checkpoint["step"], "args": trained,
                "contract": checkpoint.get("contract"), "mean": identity(config.MOTION_STATS_MEAN),
                "std": identity(config.MOTION_STATS_STD), "reasoner_image_size": getattr(
                    model, "reasoner_image_size", trained.get("reasoner_image_size", 256))}
    del checkpoint
    return model, mean, std, metadata


def sample_batch(args, ref, model, cases, neutral, metadata):
    from shape_tmr_eval_common import seeded_initial_noise

    mode = "textimg2motion" if args.cohort == "nymeria_ti2m" else "text2motion"
    lengths = torch.tensor([case.num_frames for case in cases], device=neutral.device)
    max_t = int(lengths.max().item())
    noise = seeded_initial_noise(cases, max_t, neutral.device)
    if args.backend == "nano":
        context = ref._conditioning_context(model, cases, mode, 20.0)
        return ref._sample_batch(model, cases, neutral, context, mode, noise,
                                 steps=35, guidance=2.0)
    import flow
    from caption_policy import normalize_bones_caption

    captions = [
        normalize_bones_caption(case.text).caption if case.source_kind == "bones" else case.text
        for case in cases
    ]
    images = ref._load_images(cases, 20.0) if mode == "textimg2motion" else [None] * len(cases)
    modes = [mode] * len(cases)
    conditional = model.prepare_conditions(
        captions, modes=modes, reasoner_images=images,
        image_size=int(metadata["reasoner_image_size"]),
    )
    null = model.prepare_conditions(
        [""] * len(cases), modes=modes, reasoner_images=images,
        image_size=int(metadata["reasoner_image_size"]),
    )
    pad = torch.arange(max_t, device=neutral.device)[None] >= lengths[:, None]

    def closure(condition):
        return lambda noisy, sigma: model(
            reasoner_inputs=condition, x_sigma=noisy, sigma=sigma,
            neutral_joints=neutral, motion_pad_mask=pad,
        )

    generated = flow.sample_x0_unipc(
        closure(conditional), predict_null=closure(null), T=max_t, motion_dim=283,
        steps=35, guidance=2.0, batch=len(cases), device=neutral.device,
        initial_noise=noise, shift=3.0,
    )
    generated.masked_fill_(pad.unsqueeze(-1), 0)
    return generated, lengths


def join_parts(paths, cases):
    pieces = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as value:
            for key in value.files:
                pieces.setdefault(key, []).append(np.asarray(value[key]))
    arrays = {key: np.concatenate(values, axis=0) for key, values in pieces.items()}
    if not arrays:
        raise RuntimeError("No completed result batches")
    order = np.argsort(arrays["index"])
    arrays = {key: value[order] for key, value in arrays.items()}
    if not np.array_equal(arrays["index"], np.arange(len(cases))):
        raise RuntimeError("Results do not cover every manifest case exactly once")
    if arrays["case_id"].tolist() != [case.case_id for case in cases]:
        raise RuntimeError("Result case IDs do not match immutable manifest")
    for key, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite result field {key}")
    return arrays


def physical_summary(arrays):
    result = {key + "_mean": float(np.mean(value)) for key, value in arrays.items()
              if key not in {"embedding", "case_id", "index"}}
    weights = arrays["valid_frames"].astype(np.float64)
    for key in ("floor_depth_mean_cm", "foot_depth_mean_cm", "floor_frames_below_minus1cm",
                "foot_frames_below_minus1cm", "lowest_foot_height_mean_cm",
                "all_feet_above5cm_frame_fraction"):
        result[key + "_frame_weighted"] = float(np.average(arrays[key], weights=weights))
    result["floor_depth_max_cm_worst_case"] = float(arrays["floor_depth_max_cm"].max())
    result["foot_depth_max_cm_worst_case"] = float(arrays["foot_depth_max_cm"].max())
    return result


def retrieval_chunked(motion, text, chunk=256):
    # Preserve the bundle's rescaled-similarity threshold and average ties.
    ranks = np.empty(len(text), dtype=np.float64)
    for start in range(0, len(text), chunk):
        end = min(start + chunk, len(text))
        scores = np.einsum("bi,ci->bc", text[start:end], motion) / 2 + 0.5
        similarities = np.einsum("bi,ci->bc", text[start:end], text) / 2 + 0.5
        positive = similarities > (2 * 0.99 - 1)
        if not positive.any(axis=1).all():
            raise RuntimeError("C45 retrieval query has no positive")
        best = np.where(positive, scores, -np.inf).max(axis=1, keepdims=True)
        ranks[start:end] = (scores > best).sum(axis=1) + ((scores == best).sum(axis=1) - 1) / 2
    result = {f"R{k:02d}": round(float(np.mean(ranks < k) * 100), 2) for k in (1, 2, 3, 5, 10)}
    result["MedR"] = round(float(np.median(ranks) + 1), 2)
    return result


def finalize(args, ref, embedder, cases, generated, ground_truth, signature):
    from kimodo.metrics.tmr import (
        calculate_activation_statistics, calculate_frechet_distance, contrastive_metrics,
    )

    mu_gen, cov_gen = calculate_activation_statistics(generated["embedding"])
    mu_gt, cov_gt = calculate_activation_statistics(ground_truth["embedding"])
    metrics = {
        "FID": float(calculate_frechet_distance(mu_gen, cov_gen, mu_gt, cov_gt)),
        "generated_physics_20fps": physical_summary(generated),
        "ground_truth_physics_20fps": physical_summary(ground_truth),
    }
    cache_header = torch.load(args.text_cache, map_location="meta", weights_only=False, mmap=True)
    missing = sorted({case.text for case in cases} - set(cache_header["captions"]))
    del cache_header
    if missing and not args.allow_missing_text:
        raise RuntimeError(f"Evaluator text cache lacks {len(missing)} captions")
    if not missing:
        embeddings = []
        for start in range(0, len(cases), 128):
            embeddings.append(embedder.embed_text(
                [case.text for case in cases[start:start + 128]]).cpu().numpy())
        text_embedding = np.concatenate(embeddings)
        if not np.isfinite(text_embedding).all():
            raise RuntimeError("Nonfinite text embeddings")
        # Verify bounded-memory ranks against the pinned official implementation
        # on actual cohort embeddings before publishing full-cohort retrieval.
        count = min(128, len(cases))
        small_text, small_motion = text_embedding[:count], generated["embedding"][:count]
        score = np.einsum("bi,ci->bc", small_text, small_motion) / 2 + 0.5
        text_score = np.einsum("bi,ci->bc", small_text, small_text) / 2 + 0.5
        official = contrastive_metrics(score, text_score, threshold=0.99)
        chunked = retrieval_chunked(small_motion, small_text, chunk=31)
        if any(official[key] != value for key, value in chunked.items()):
            raise RuntimeError(f"Retrieval differs from bundle: {official} vs {chunked}")
        metrics["t2m_retrieval"] = retrieval_chunked(generated["embedding"], text_embedding)
        metrics["ground_truth_t2m_retrieval"] = retrieval_chunked(ground_truth["embedding"], text_embedding)
        metrics["R@3"] = metrics["t2m_retrieval"]["R03"]
        atomic_npz(Path(args.output_dir) / "text_embeddings.npz",
                   embedding=text_embedding, case_id=np.array([case.case_id for case in cases]))
    result = {
        "status": "complete" if not missing else "partial_missing_evaluator_text",
        "signature": signature, "cohort": args.cohort, "checkpoint": args.checkpoint,
        "backend": args.backend, "num_cases": len(cases),
        "num_sequences": len({case.uuid for case in cases if case.uuid}),
        "num_missing_evaluator_captions": len(missing),
        "retrieval_pool": "full cohort, similar-caption positives, average ties",
        "physics_units": "meter/second velocities; fractions 0..1; penetration and bone error centimeters",
        "caption_conditioning": "BONES main_actor_camera_wearer_v1 for Edge; original BONES for Nano; humanized Nymeria for both. Evaluator BONES text uses official sanitize_text.",
        "metrics": metrics, "coverage_verified": True,
    }
    atomic_json(Path(args.output_dir) / "summary.json", result)
    print("[summary] " + json.dumps(result, sort_keys=True), flush=True)


@torch.inference_mode()
def evaluate(args):
    import config
    import eval_phase2_shape_tmr as ref
    from decode_uniego_torch import decode_joints
    from shape_tmr_eval_common import read_jsonl, sha256_file

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cases = read_jsonl(Path(args.manifest_dir) / f"{args.cohort}.jsonl")
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("Empty or duplicate manifest")
    neutrals = np.load(Path(args.manifest_dir) / f"{args.cohort}_neutral.npy", mmap_mode="r")
    if neutrals.shape != (len(cases), 30, 3):
        raise RuntimeError("Neutral skeleton count/shape mismatch")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": 1, "checkpoint": identity(args.checkpoint), "backend": args.backend,
        "manifest_sha256": sha256_file(Path(args.manifest_dir) / f"{args.cohort}.jsonl"),
        "protocol_sha256": sha256_file(Path(args.manifest_dir) / "protocol.json"),
        "tmr_checkpoint": identity(Path(args.bundle_root) / "artifacts/evaluator/c45_step_00005000.pt"),
        "steps": 35, "guidance": 2.0, "shift": 3.0,
        "batch_size": args.batch_size, "cohort": args.cohort,
        "mean_sha256": sha256_file(config.MOTION_STATS_MEAN),
        "std_sha256": sha256_file(config.MOTION_STATS_STD),
    }
    if int(getattr(args, "expected_step", 200000)) != 200000:
        signature["expected_step"] = int(args.expected_step)
    signature_path = out / "signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise RuntimeError("Refusing stale/incompatible result resume")
    else:
        atomic_json(signature_path, signature)
    embedder = ref.ShapeTMREmbedder(
        str(Path(args.bundle_root) / "artifacts/evaluator/c45_step_00005000.pt"),
        str(Path(args.bundle_root) / "artifacts/evaluator/stats/motion"),
        text_cache_path=args.text_cache, device=str(device),
    )
    if embedder.arch_version not in {"v1", "v2", "v3"}:
        raise RuntimeError(f"Expected a shape-aware C45 architecture, got {embedder.arch_version}")
    skeleton = ref.SOMASkeleton30().to(device)
    reference_id = hashlib.sha256(json.dumps({
        key: signature[key] for key in ("manifest_sha256", "tmr_checkpoint", "protocol_sha256")
    }, sort_keys=True).encode()).hexdigest()[:20]
    reference_dir = Path(args.manifest_dir) / "reference_cache" / args.cohort / reference_id
    reference_dir.mkdir(parents=True, exist_ok=True)
    gt_parts = []
    with (reference_dir / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for start in range(0, len(cases), 32):
            batch = cases[start:start + 32]
            path = reference_dir / f"batch_{start:07d}.npz"
            gt_parts.append(path)
            if path.exists():
                continue
            data = [load_features(case)[0] for case in batch]
            lengths = torch.tensor([len(value) for value in data], device=device)
            features = torch.empty(len(data), int(lengths.max()), 283, device=device)
            for index, value in enumerate(data):
                value = torch.from_numpy(value).to(device)
                features[index, :len(value)] = value
                features[index, len(value):] = value[-1]
            neutral = torch.from_numpy(np.array(neutrals[start:start + len(batch)])).to(device)
            joints = decode_joints(features)
            values = measure(ref, embedder, skeleton, joints, features[..., 279:283] > 0.5, lengths, neutral)
            atomic_npz(path, index=np.arange(start, start + len(batch)),
                       case_id=np.array([case.case_id for case in batch]), **values)
            if start % 1024 == 0 or start + len(batch) == len(cases):
                print(f"[GT {args.cohort}] {start + len(batch)}/{len(cases)}", flush=True)
    ground_truth = join_parts(gt_parts, cases)

    generated_paths = [
        out / "batches" / f"batch_{start:07d}.npz" for start in range(0, len(cases), args.batch_size)
    ]
    if any(not path.exists() for path in generated_paths):
        if args.finalize_only:
            raise RuntimeError("Finalization requested before all generation batches exist")
        model, mean, std, metadata = strict_model(args, device)
        atomic_json(out / "checkpoint_metadata.json", metadata)
        started = time.monotonic()
        newly_completed = 0
        for start, path in zip(range(0, len(cases), args.batch_size), generated_paths):
            if path.exists():
                continue
            batch = cases[start:start + args.batch_size]
            neutral = torch.from_numpy(np.array(neutrals[start:start + len(batch)])).to(device)
            normalized, lengths = sample_batch(args, ref, model, batch, neutral, metadata)
            if not torch.isfinite(normalized).all():
                raise FloatingPointError(f"Nonfinite generation at {batch[0].case_id}; no cases substituted")
            _, joints, contacts = ref._decode_generated(normalized, lengths, mean, std)
            values = measure(ref, embedder, skeleton, joints, contacts, lengths, neutral)
            for key, value in values.items():
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite {key} at batch {start}")
            atomic_npz(path, index=np.arange(start, start + len(batch)),
                       case_id=np.array([case.case_id for case in batch]), **values)
            newly_completed += len(batch)
            elapsed = time.monotonic() - started
            done = start + len(batch)
            eta = (len(cases) - done) * elapsed / newly_completed
            print(f"[GEN {args.backend} {args.cohort}] {done}/{len(cases)} "
                  f"{newly_completed/elapsed:.3f} cases/s eta={eta/3600:.2f}h "
                  f"peak_gpu={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
        del model
        torch.cuda.empty_cache()
    generated = join_parts(generated_paths, cases)
    finalize(args, ref, embedder, cases, generated, ground_truth, signature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--backend", choices=("edge", "nano"), default="nano")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--testsuite")
    parser.add_argument("--bones-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--expected-step", type=int, default=200000,
                        help="Require this saved training step (default: 200000)")
    parser.add_argument("--cohort", choices=("bones_repetition_overview", "nymeria_t2m", "nymeria_ti2m"))
    parser.add_argument("--output-dir")
    parser.add_argument("--text-cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-missing-text", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.expected_step < 0:
        parser.error("--expected-step must be nonnegative")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Run inside an allocated GPU compute step, never on a login node")
    repo = Path(__file__).resolve().parents[1]
    backend_dir = repo / ("motion_expert_joint_attention" if args.prepare or args.backend == "nano"
                          else "motion_expert_t2m_edge")
    sys.path.insert(0, str(backend_dir))
    os.environ["SHAPE_TMR_BUNDLE"] = args.bundle_root
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if args.prepare:
        if not args.testsuite or not args.bones_root:
            parser.error("--prepare requires --testsuite and --bones-root")
        prepare(args)
    else:
        if not all((args.checkpoint, args.cohort, args.output_dir, args.text_cache)) or args.batch_size < 1:
            parser.error("Require checkpoint, cohort, output-dir, text-cache and positive batch-size")
        evaluate(args)


if __name__ == "__main__":
    main()
