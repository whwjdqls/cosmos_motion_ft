"""In-memory BONES generator evaluation with the shape-aware C45 TMR.

The benchmark's file pipeline is deliberately collapsed into one GPU process:

  text + proportional skeleton -> BONES generator -> normalized UniEgo [B,T,283]
  -> unnormalize -> SOMA-30 joints at 20 fps -> C45 TMR features at 30 fps

Generated motions and embeddings are never written.  The only output is one aggregate JSON
containing the Kimodo benchmark TMR metrics, plain retrieval diagnostics, foot-skating/contact
metrics, bone-length adherence, and a paired same-text/same-noise shape intervention.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import bs_native_flow
from bs_dataset import DATA_ROOT, MEAN_PATH, STD_PATH
from bs_model import MotionExpertInContext
from bs_shape_metrics import (
    counterfactual_shape_response,
    farthest_shape_indices,
    population_shape_tracking,
)
from bs_text_cache import LLM2VecCache
from decode_uniego_torch import decode_joints
from uniego_layout import FEAT_DIM, FOOT_SLICE, canonicalize_frame0


ROOT = Path(__file__).resolve().parents[1]
SHAPE_TMR_DIR = ROOT / "shape_aware_TMR"
if str(SHAPE_TMR_DIR) not in sys.path:
    sys.path.insert(0, str(SHAPE_TMR_DIR))

from st_dataset import resample_joints_time  # noqa: E402
from st_eval import ShapeTMREmbedder  # noqa: E402
from st_inline_eval import BENCH_TEXT_CACHE, TESTSUITE  # noqa: E402

from kimodo.metrics import (  # noqa: E402
    FootContactConsistency,
    FootSkateFromContacts,
    FootSkateFromHeight,
    FootSkateRatio,
    aggregate_metrics,
    compute_metrics,
    compute_tmr_retrieval_metrics,
)
from kimodo.skeleton import SOMASkeleton30  # noqa: E402


DEFAULT_TMR_CKPT = (
    "/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/"
    "c45_official30fps_balanced_10k/step_00005000.pt"
)
DEFAULT_TMR_STATS = (
    "/home/jungbin_cho/.cache/huggingface/hub/"
    "models--nvidia--TMR-SOMA-RP-v1/snapshots/"
    "e427752ae3446dedba49e928c93ddc9f0e413401/stats/motion"
)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    text: str
    motion_path: str
    crop_start: int
    crop_end: int
    num_frames: int
    seed: int
    neutral_joints: np.ndarray


def _parse_generator(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("generator must be LABEL=/absolute/checkpoint.pt")
    label, path = spec.split("=", 1)
    label, path = label.strip(), os.path.abspath(path.strip())
    if not label or not path:
        raise argparse.ArgumentTypeError("generator must be LABEL=/absolute/checkpoint.pt")
    return label, path


def _motion_path(uniego_root: str, seed_motion: dict) -> str:
    rel = seed_motion["bvh_path"]
    rel = rel[4:] if rel.startswith("BVH/") else rel
    rel = rel[:-4] if rel.endswith(".bvh") else rel
    return os.path.join(uniego_root, rel + ".npz")


def build_cases(
    testsuite: str,
    split: str,
    group: str,
    uniego_root: str,
    generator_cache: LLM2VecCache,
    eval_cache: LLM2VecCache,
    fps: float,
    min_frames: int,
    max_cases: int,
) -> tuple[list[EvalCase], dict]:
    case_dirs = sorted(glob.glob(os.path.join(testsuite, split, "text2motion", group, "*")))
    cases: list[EvalCase] = []
    skipped: Counter[str] = Counter()

    for case_dir in case_dirs:
        if max_cases > 0 and len(cases) >= max_cases:
            break
        try:
            meta = json.load(open(os.path.join(case_dir, "meta.json")))
            seed_motion = json.load(open(os.path.join(case_dir, "seed_motion.json")))
            text = str(meta["text"])
            duration = float(meta["duration"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            skipped["bad_metadata"] += 1
            continue
        if text not in generator_cache:
            skipped["missing_generator_text"] += 1
            continue
        if text not in eval_cache:
            skipped["missing_evaluator_text"] += 1
            continue

        path = _motion_path(uniego_root, seed_motion)
        if not os.path.isfile(path):
            skipped["missing_motion"] += 1
            continue
        try:
            with np.load(path, mmap_mode="r") as data:
                if "features" not in data or "neutral_joints" not in data:
                    skipped["wrong_motion_schema"] += 1
                    continue
                n_total = int(data["features"].shape[0])
                start = int(round(int(seed_motion["crop_start_frame_index"]) * fps / 30.0))
                end = int(round(int(seed_motion["crop_end_frame_index"]) * fps / 30.0))
                start = max(0, min(start, n_total))
                end = max(start, min(end, n_total))
                gt = np.asarray(data["features"][start:end])
                neutral = np.asarray(data["neutral_joints"]).astype(np.float32)
        except (OSError, KeyError, TypeError, ValueError, EOFError):
            skipped["bad_motion"] += 1
            continue
        if end - start < min_frames:
            skipped["too_short_gt"] += 1
            continue
        if not np.isfinite(gt).all():
            skipped["nonfinite_gt"] += 1
            continue
        if neutral.shape != (30, 3) or not np.isfinite(neutral).all():
            skipped["bad_neutral_joints"] += 1
            continue

        num_frames = int(duration * fps)
        if num_frames < min_frames:
            skipped["too_short_request"] += 1
            continue
        neutral = neutral - neutral.mean(axis=0, keepdims=True)
        cases.append(
            EvalCase(
                case_id=os.path.basename(case_dir),
                text=text,
                motion_path=path,
                crop_start=start,
                crop_end=end,
                num_frames=num_frames,
                seed=int(meta.get("seed", 0)),
                neutral_joints=neutral,
            )
        )

    # Length sorting reduces attention padding while preserving paired text/motion ordering.
    cases.sort(key=lambda c: (c.num_frames, c.case_id))
    audit = {
        "discovered": len(case_dirs),
        "used": len(cases),
        "skipped": dict(sorted(skipped.items())),
        "min_generated_frames": min((c.num_frames for c in cases), default=0),
        "max_generated_frames": max((c.num_frames for c in cases), default=0),
    }
    return cases, audit


def _chunks(items: list[EvalCase], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _load_gt_batch(cases: list[EvalCase], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = []
    for case in cases:
        with np.load(case.motion_path, mmap_mode="r") as data:
            feat = np.asarray(data["features"][case.crop_start:case.crop_end]).astype(np.float32)
        arrays.append(canonicalize_frame0(feat))

    lengths = torch.tensor([len(x) for x in arrays], dtype=torch.long, device=device)
    max_t = int(lengths.max().item())
    features = torch.empty(len(arrays), max_t, FEAT_DIM, dtype=torch.float32, device=device)
    for i, feat in enumerate(arrays):
        x = torch.from_numpy(feat).to(device)
        features[i, :len(feat)] = x
        features[i, len(feat):] = x[-1]
    joints = decode_joints(features)
    contacts = features[..., FOOT_SLICE] > 0.5
    return joints, contacts, lengths


def _initial_noise(cases: list[EvalCase], max_t: int, device: str) -> torch.Tensor:
    x = torch.zeros(len(cases), max_t, FEAT_DIM, dtype=torch.float32)
    for i, case in enumerate(cases):
        generator = torch.Generator(device="cpu").manual_seed(case.seed)
        x[i, :case.num_frames] = torch.randn(
            case.num_frames,
            FEAT_DIM,
            generator=generator,
            dtype=torch.float32,
        )
    return x.to(device)


@torch.inference_mode()
def sample_batch(
    model: MotionExpertInContext,
    checkpoint_args: dict,
    text: torch.Tensor,
    null_text: torch.Tensor,
    neutral_joints: torch.Tensor,
    cases: list[EvalCase],
    steps: int,
    guidance: float,
    native_solver: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([c.num_frames for c in cases], dtype=torch.long, device=device)
    max_t = int(lengths.max().item())
    pad = torch.arange(max_t, device=device)[None] >= lengths[:, None]
    valid = (~pad).unsqueeze(-1)
    x = _initial_noise(cases, max_t, device)
    pred = str(checkpoint_args.get("pred", "x0"))
    schedule = str(checkpoint_args.get("schedule", "legacy"))

    if schedule == "native":
        if pred != "x0":
            raise ValueError("native BONES checkpoints must predict x0")
        native_shift = float(checkpoint_args.get("native_shift", bs_native_flow.DEFAULT_SHIFT))
        native_n = int(
            checkpoint_args.get(
                "native_num_train_timesteps",
                bs_native_flow.DEFAULT_NUM_TRAIN_TIMESTEPS,
            )
        )
        if native_solver == "unipc":
            scheduler = bs_native_flow.create_unipc_scheduler(
                steps,
                shift=native_shift,
                num_train_timesteps=native_n,
                device=device,
            )
            timesteps = scheduler.timesteps
        else:
            sigmas, timesteps = bs_native_flow.inference_schedule(
                steps,
                shift=native_shift,
                num_train_timesteps=native_n,
                device=device,
            )

        def predict_x0(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            model_sigma = (timestep.float() / float(native_n)).expand(len(cases))
            x0_cond = model(state, model_sigma, text, None, neutral_joints, pad).float()
            x0_null = model(state, model_sigma, null_text, None, neutral_joints, pad).float()
            return x0_null + guidance * (x0_cond - x0_null)

        if native_solver == "unipc":
            for i, timestep in enumerate(timesteps):
                x0 = predict_x0(x, timestep).masked_fill(~valid, 0.0)
                sigma = scheduler.sigmas[i].to(device=x.device, dtype=torch.float32)
                velocity = (x.float() - x0) / sigma.clamp(min=1e-6)
                x = scheduler.step(
                    model_output=velocity,
                    timestep=timestep,
                    sample=x.float(),
                    return_dict=False,
                    generator=None,
                )[0]
                x = x.masked_fill(~valid, 0.0)
            return x, lengths

        for i in range(steps):
            sigma = sigmas[i].float().clamp(min=1e-6)
            next_sigma = sigmas[i + 1].float()
            delta = next_sigma - sigma
            x0 = predict_x0(x, timesteps[i])
            velocity = (x.float() - x0) / sigma
            euler_state = (x.float() + delta * velocity).masked_fill(~valid, 0.0)

            if native_solver == "heun" and i < steps - 1:
                x0_next = predict_x0(euler_state, timesteps[i + 1])
                velocity_next = (euler_state - x0_next) / next_sigma.clamp(min=1e-6)
                x = x.float() + 0.5 * delta * (velocity + velocity_next)
            else:
                x = euler_state
            x = x.masked_fill(~valid, 0.0)
        return x, lengths

    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        sigma = sigmas[i].clamp(min=1e-3)
        model_sigma = sigma.expand(len(cases))
        cond = model(x, model_sigma, text, None, neutral_joints, pad).float()
        null = model(x, model_sigma, null_text, None, neutral_joints, pad).float()
        prediction = null + guidance * (cond - null)
        if pred == "v":
            x = x - (sigmas[i] - sigmas[i + 1]) * prediction
        elif pred == "x0":
            eps = (x.float() - (1.0 - sigma) * prediction) / sigma
            next_sigma = sigmas[i + 1]
            x = (1.0 - next_sigma) * prediction + next_sigma * eps
        else:
            raise ValueError(f"unsupported prediction target: {pred}")
        x = x.masked_fill(~valid, 0.0)
    return x, lengths


def _decode_generated(
    normalized: torch.Tensor,
    lengths: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = normalized * std + mean
    for i, length in enumerate(lengths.tolist()):
        features[i, length:] = features[i, length - 1]
    joints = decode_joints(features)
    contacts = features[..., FOOT_SLICE] > 0.5
    return features, joints, contacts


@torch.inference_mode()
def _tmr_embed_joints(
    embedder: ShapeTMREmbedder,
    joints: torch.Tensor,
    lengths: torch.Tensor,
    neutral_joints: torch.Tensor,
    source_fps: float,
) -> torch.Tensor:
    features = []
    for i, length in enumerate(lengths.tolist()):
        posed = joints[i, :length].detach().cpu()
        posed = resample_joints_time(posed, source_fps, embedder.fps)
        n = int(posed.shape[0])
        feat = embedder.rep(
            posed_joints=posed.unsqueeze(0),
            to_normalize=True,
            to_canonicalize=True,
            lengths=torch.tensor([n]),
        )[0]
        features.append(feat.float())

    max_t = max(len(x) for x in features)
    dim = int(features[0].shape[-1])
    padded = torch.zeros(len(features), max_t, dim, dtype=torch.float32)
    mask = torch.zeros(len(features), max_t, dtype=torch.bool)
    for i, feat in enumerate(features):
        padded[i, :len(feat)] = feat
        mask[i, :len(feat)] = True
    return embedder.embed_motion_features(padded, mask, neutral_joints.detach().cpu())


def _foot_metrics(skeleton: SOMASkeleton30, fps: float):
    kwargs = {"skeleton": skeleton, "fps": fps}
    return [
        FootSkateFromHeight(**kwargs),
        FootSkateFromContacts(**kwargs),
        FootContactConsistency(**kwargs),
        FootSkateRatio(**kwargs),
    ]


def _aggregate_physical(metrics_list) -> dict[str, float]:
    values = aggregate_metrics(metrics_list)
    return {key: float(value.mean().item()) for key, value in values.items()}


def _compute_physical(metrics_list, joints, contacts, lengths):
    compute_metrics(
        metrics_list,
        {"posed_joints": joints, "foot_contacts": contacts, "lengths": lengths},
    )


def _bone_mae_cm(
    joints: torch.Tensor,
    lengths: torch.Tensor,
    neutral_joints: torch.Tensor,
    parents: list[int],
) -> list[float]:
    motion_len, target_len = _bone_length_vectors(joints, lengths, neutral_joints, parents)
    return ((motion_len - target_len).abs().mean(dim=1) * 100.0).cpu().tolist()


def _bone_length_vectors(
    joints: torch.Tensor,
    lengths: torch.Tensor,
    neutral_joints: torch.Tensor,
    parents: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return generated and conditioned bone lengths as [batch, non-root bones] tensors."""
    edges = [(j, p) for j, p in enumerate(parents) if 0 <= p < len(parents) and p != j]
    child = torch.tensor([x[0] for x in edges], device=joints.device)
    parent = torch.tensor([x[1] for x in edges], device=joints.device)
    motion_lengths = []
    for i, length in enumerate(lengths.tolist()):
        motion_lengths.append(torch.linalg.vector_norm(
            joints[i, :length, child] - joints[i, :length, parent], dim=-1
        ).mean(dim=0))
    target_lengths = torch.linalg.vector_norm(
        neutral_joints[:, child] - neutral_joints[:, parent], dim=-1
    )
    return torch.stack(motion_lengths), target_lengths


def _natural_shape_counterfactuals(
    cases: list[EvalCase],
    parents: list[int],
) -> tuple[dict[str, np.ndarray], dict]:
    """Pair every case with the most different held-out natural skeleton."""
    neutral = np.stack([case.neutral_joints for case in cases]).astype(np.float32)
    edges = [(j, p) for j, p in enumerate(parents) if 0 <= p < len(parents) and p != j]
    child = np.asarray([x[0] for x in edges])
    parent = np.asarray([x[1] for x in edges])
    target_bones = np.linalg.norm(neutral[:, child] - neutral[:, parent], axis=-1)
    pair_indices = farthest_shape_indices(target_bones)
    paired_bones = target_bones[pair_indices]
    return {"farthest_natural": neutral[pair_indices]}, {
        "strategy": "most different natural held-out skeleton by Euclidean bone-length distance",
        "same_text_and_noise": True,
        "num_pairs": len(cases),
        "unique_counterfactual_case_indices": int(np.unique(pair_indices).size),
        "requested_bone_delta_cm_mean": float(
            np.abs(paired_bones - target_bones).mean() * 100.0
        ),
    }


def _plain_retrieval(motion: np.ndarray, text: np.ndarray) -> dict[str, float]:
    similarity = text @ motion.T
    ranks = (similarity > similarity.diagonal()[:, None]).sum(axis=1)
    return {
        "R01": float((ranks < 1).mean() * 100.0),
        "R02": float((ranks < 2).mean() * 100.0),
        "R03": float((ranks < 3).mean() * 100.0),
        "R05": float((ranks < 5).mean() * 100.0),
        "R10": float((ranks < 10).mean() * 100.0),
        "MedR": float(np.median(ranks) + 1),
        "paired_cosine": float(similarity.diagonal().mean()),
    }


def _load_generator(checkpoint_path: str, text_dim: int, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    model = MotionExpertInContext(
        d=args.get("d", 512),
        n_layers=args.get("layers", 8),
        heads=args.get("heads", 8),
        ffn=args.get("ffn", 2048),
        text_dim=text_dim,
        motion_dim=FEAT_DIM,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, args


@torch.inference_mode()
def precompute_references(
    cases: list[EvalCase],
    batch_size: int,
    embedder: ShapeTMREmbedder,
    skeleton: SOMASkeleton30,
    fps: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    gt_embeddings, text_embeddings = [], []
    metrics = _foot_metrics(skeleton, fps)
    bone_errors = []
    generated_bones, target_bones = [], []
    parents = [int(x) for x in skeleton.joint_parents]
    started = time.time()
    for batch_index, batch_cases in enumerate(_chunks(cases, batch_size), start=1):
        neutral = torch.from_numpy(np.stack([c.neutral_joints for c in batch_cases])).to(device)
        joints, contacts, lengths = _load_gt_batch(batch_cases, device)
        gt_embeddings.append(
            _tmr_embed_joints(embedder, joints, lengths, neutral, fps).cpu().numpy()
        )
        text_embeddings.append(embedder.embed_text([c.text for c in batch_cases]).cpu().numpy())
        _compute_physical(metrics, joints, contacts, lengths)
        bone_errors.extend(_bone_mae_cm(joints, lengths, neutral, parents))
        motion_lengths, neutral_lengths = _bone_length_vectors(
            joints, lengths, neutral, parents
        )
        generated_bones.append(motion_lengths.cpu().numpy())
        target_bones.append(neutral_lengths.cpu().numpy())
        print(
            f"[reference] batch {batch_index} cases={sum(len(x) for x in gt_embeddings)}/{len(cases)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    physical = _aggregate_physical(metrics)
    shape = {
        "bone_length_mae_cm_mean": float(np.mean(bone_errors)),
        "bone_length_mae_cm_std": float(np.std(bone_errors)),
        "population_tracking": population_shape_tracking(
            np.concatenate(generated_bones),
            np.concatenate(target_bones),
        ),
    }
    return np.concatenate(gt_embeddings), np.concatenate(text_embeddings), physical, shape


@torch.inference_mode()
def evaluate_model(
    label: str,
    model: MotionExpertInContext,
    checkpoint_args: dict,
    checkpoint_step: int,
    checkpoint_path: str,
    cases: list[EvalCase],
    batch_size: int,
    steps: int,
    guidance: float,
    generator_cache: LLM2VecCache,
    embedder: ShapeTMREmbedder,
    skeleton: SOMASkeleton30,
    mean: torch.Tensor,
    std: torch.Tensor,
    gt_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    fps: float,
    native_solver: str,
    device: str,
    shape_counterfactuals: dict[str, np.ndarray] | None = None,
) -> dict:
    schedule = str(checkpoint_args.get("schedule", "legacy"))
    pred = str(checkpoint_args.get("pred", "x0"))
    solver = native_solver if schedule == "native" else "legacy_euler"
    denoiser_evaluations = (2 * steps - 1) if solver == "heun" else steps
    print(
        f"[{label}] evaluating step={checkpoint_step} schedule={schedule} pred={pred} "
        f"solver={solver} steps={steps} denoiser_evals={denoiser_evaluations} "
        f"guidance={guidance}",
        flush=True,
    )

    metrics = _foot_metrics(skeleton, fps)
    parents = [int(x) for x in skeleton.joint_parents]
    embeddings = []
    bone_errors = []
    generated_bones, target_bones = [], []
    generated_count = 0
    started = time.time()
    for batch_index, batch_cases in enumerate(_chunks(cases, batch_size), start=1):
        neutral = torch.from_numpy(np.stack([c.neutral_joints for c in batch_cases])).to(device)
        text = generator_cache.batch([c.text for c in batch_cases])
        null_text = generator_cache.null(len(batch_cases))
        normalized, lengths = sample_batch(
            model,
            checkpoint_args,
            text,
            null_text,
            neutral,
            batch_cases,
            steps,
            guidance,
            native_solver,
            device,
        )
        _, joints, contacts = _decode_generated(normalized, lengths, mean, std)
        embeddings.append(
            _tmr_embed_joints(embedder, joints, lengths, neutral, fps).cpu().numpy()
        )
        _compute_physical(metrics, joints, contacts, lengths)
        bone_errors.extend(_bone_mae_cm(joints, lengths, neutral, parents))
        motion_lengths, neutral_lengths = _bone_length_vectors(
            joints, lengths, neutral, parents
        )
        generated_bones.append(motion_lengths.cpu().numpy())
        target_bones.append(neutral_lengths.cpu().numpy())
        generated_count += len(batch_cases)
        print(
            f"[{label}] batch {batch_index} generated={generated_count}/{len(cases)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    generated = np.concatenate(embeddings)
    finite = np.isfinite(generated).all(axis=1)
    if not finite.all():
        raise RuntimeError(f"{label}: {int((~finite).sum())} non-finite generated TMR embeddings")
    protocol = compute_tmr_retrieval_metrics(generated, text_embeddings, gt_embeddings)
    plain_generated = _plain_retrieval(generated, text_embeddings)
    physical = _aggregate_physical(metrics)
    generated_bones_np = np.concatenate(generated_bones)
    target_bones_np = np.concatenate(target_bones)
    shape = {
        "bone_length_mae_cm_mean": float(np.mean(bone_errors)),
        "bone_length_mae_cm_std": float(np.std(bone_errors)),
        "population_tracking": population_shape_tracking(
            generated_bones_np,
            target_bones_np,
        ),
        "counterfactuals": {},
    }

    for counterfactual_name, counterfactual_neutral in (shape_counterfactuals or {}).items():
        counterfactual_neutral = np.asarray(counterfactual_neutral, dtype=np.float32)
        expected_shape = (len(cases), 30, 3)
        if counterfactual_neutral.shape != expected_shape:
            raise ValueError(
                f"{counterfactual_name}: expected counterfactual skeletons {expected_shape}, "
                f"got {counterfactual_neutral.shape}"
            )
        cf_embeddings, cf_generated_bones, cf_target_bones = [], [], []
        generated_count = 0
        for batch_index, start in enumerate(range(0, len(cases), batch_size), start=1):
            batch_cases = cases[start:start + batch_size]
            neutral = torch.from_numpy(
                counterfactual_neutral[start:start + len(batch_cases)]
            ).to(device)
            text = generator_cache.batch([c.text for c in batch_cases])
            null_text = generator_cache.null(len(batch_cases))
            normalized, lengths = sample_batch(
                model,
                checkpoint_args,
                text,
                null_text,
                neutral,
                batch_cases,
                steps,
                guidance,
                native_solver,
                device,
            )
            _, joints, _ = _decode_generated(normalized, lengths, mean, std)
            cf_embeddings.append(
                _tmr_embed_joints(embedder, joints, lengths, neutral, fps).cpu().numpy()
            )
            motion_lengths, neutral_lengths = _bone_length_vectors(
                joints, lengths, neutral, parents
            )
            cf_generated_bones.append(motion_lengths.cpu().numpy())
            cf_target_bones.append(neutral_lengths.cpu().numpy())
            generated_count += len(batch_cases)
            print(
                f"[{label}/shape:{counterfactual_name}] batch {batch_index} "
                f"generated={generated_count}/{len(cases)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

        cf_generated = np.concatenate(cf_embeddings)
        if not np.isfinite(cf_generated).all():
            bad = int((~np.isfinite(cf_generated).all(axis=1)).sum())
            raise RuntimeError(
                f"{label}/{counterfactual_name}: {bad} non-finite generated TMR embeddings"
            )
        cf_protocol = compute_tmr_retrieval_metrics(
            cf_generated,
            text_embeddings,
            gt_embeddings,
        )
        cf_plain = _plain_retrieval(cf_generated, text_embeddings)
        response = counterfactual_shape_response(
            generated_bones_np,
            np.concatenate(cf_generated_bones),
            target_bones_np,
            np.concatenate(cf_target_bones),
        )
        shape["counterfactuals"][counterfactual_name] = {
            **response,
            "tmr": {key: float(value) for key, value in cf_protocol.items()},
            "plain_t2m": cf_plain,
            "protocol_R03_delta": float(
                cf_protocol["TMR/t2m_R/R03"] - protocol["TMR/t2m_R/R03"]
            ),
            "plain_R03_delta": float(cf_plain["R03"] - plain_generated["R03"]),
        }

    sampling_passes = 1 + len(shape["counterfactuals"])
    result = {
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "training_schedule": schedule,
        "prediction": pred,
        "sampler_solver": solver,
        "sampling_steps": steps,
        "denoiser_evaluations": denoiser_evaluations,
        "model_forward_calls_with_cfg": 2 * denoiser_evaluations,
        "sampling_passes": sampling_passes,
        "total_denoiser_evaluations_per_case": sampling_passes * denoiser_evaluations,
        "total_model_forward_calls_with_cfg_per_case": sampling_passes * 2 * denoiser_evaluations,
        "guidance": guidance,
        "num_motions": len(cases),
        "tmr": {key: float(value) for key, value in protocol.items()},
        "plain_t2m_gen": plain_generated,
        "plain_t2m_gt": _plain_retrieval(gt_embeddings, text_embeddings),
        "physical_20fps": physical,
        "shape": shape,
        "elapsed_sec": time.time() - started,
    }
    return result


@torch.inference_mode()
def evaluate_generator(
    label: str,
    checkpoint_path: str,
    cases: list[EvalCase],
    batch_size: int,
    steps: int,
    guidance: float,
    generator_cache: LLM2VecCache,
    embedder: ShapeTMREmbedder,
    skeleton: SOMASkeleton30,
    mean: torch.Tensor,
    std: torch.Tensor,
    gt_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    fps: float,
    native_solver: str,
    device: str,
    shape_counterfactuals: dict[str, np.ndarray] | None = None,
) -> dict:
    model, checkpoint, checkpoint_args = _load_generator(
        checkpoint_path,
        generator_cache.dim,
        device,
    )
    try:
        return evaluate_model(
            label,
            model,
            checkpoint_args,
            int(checkpoint.get("step", 0)),
            checkpoint_path,
            cases,
            batch_size,
            steps,
            guidance,
            generator_cache,
            embedder,
            skeleton,
            mean,
            std,
            gt_embeddings,
            text_embeddings,
            fps,
            native_solver,
            device,
            shape_counterfactuals,
        )
    finally:
        del model, checkpoint
        torch.cuda.empty_cache()


class InlineShapeTMREvaluator:
    """Reusable full-overview evaluator for live training checkpoints."""

    def __init__(
        self,
        output_dir: str,
        generator_mean: torch.Tensor,
        generator_std: torch.Tensor,
        *,
        tmr_ckpt: str = DEFAULT_TMR_CKPT,
        tmr_stats: str = DEFAULT_TMR_STATS,
        text_cache: str = BENCH_TEXT_CACHE,
        testsuite: str = TESTSUITE,
        split: str = "content",
        group: str = "overview",
        uniego_root: str = DATA_ROOT,
        fps: float = 20.0,
        batch_size: int = 16,
        steps: int = 35,
        guidance: float = 2.0,
        min_frames: int = 10,
        max_cases: int = 0,
        native_solver: str = "unipc",
        shape_counterfactual: str = "farthest",
        device: str = "cuda",
    ):
        if batch_size <= 0 or steps <= 0:
            raise ValueError("inline evaluation batch size and steps must be positive")
        if native_solver not in {"euler", "heun", "unipc"}:
            raise ValueError(f"unsupported native solver: {native_solver}")
        if shape_counterfactual not in {"none", "farthest"}:
            raise ValueError(f"unsupported shape counterfactual: {shape_counterfactual}")
        if not os.path.isfile(tmr_ckpt):
            raise FileNotFoundError(f"missing inline TMR checkpoint: {tmr_ckpt}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.fps = float(fps)
        self.native_solver = native_solver
        self.shape_counterfactual = shape_counterfactual
        self.device = device
        self.tmr_ckpt = tmr_ckpt
        self.tmr_stats = tmr_stats
        self.generator_cache = LLM2VecCache(text_cache, device=device)

        self.cases, self.audit = build_cases(
            testsuite,
            split,
            group,
            uniego_root,
            self.generator_cache,
            self.generator_cache,
            self.fps,
            min_frames,
            max_cases,
        )
        if not self.cases:
            raise RuntimeError(f"no usable inline {split}/{group} cases: {self.audit}")
        print(f"[inline-eval] cases {json.dumps(self.audit, sort_keys=True)}", flush=True)

        self.embedder = ShapeTMREmbedder(
            tmr_ckpt,
            tmr_stats,
            text_cache_path=text_cache,
            device=device,
        )
        self.skeleton = SOMASkeleton30().to(device)
        parents = [int(x) for x in self.skeleton.joint_parents]
        if self.shape_counterfactual == "farthest":
            (
                self.shape_counterfactuals,
                self.shape_counterfactual_audit,
            ) = _natural_shape_counterfactuals(self.cases, parents)
        else:
            self.shape_counterfactuals = {}
            self.shape_counterfactual_audit = {"strategy": "disabled", "num_pairs": 0}
        self.mean = generator_mean.detach()
        self.std = generator_std.detach()
        (
            self.gt_embeddings,
            self.text_embeddings,
            self.gt_physical,
            self.gt_shape,
        ) = precompute_references(
            self.cases,
            self.batch_size,
            self.embedder,
            self.skeleton,
            self.fps,
            self.device,
        )
        self.protocol = {
            "testsuite": testsuite,
            "split": split,
            "group": group,
            "case_audit": self.audit,
            "generator_representation": "normalized 283-D proportional UniEgo at 20 fps",
            "tmr_conversion": (
                "unnormalize -> decode SOMA-30 joints -> resample 20 to 30 fps "
                "-> C45 TMRMotionRep"
            ),
            "shape_conditioning": (
                "same centered proportional neutral_joints supplied to generator and C45"
            ),
            "shape_counterfactual": self.shape_counterfactual_audit,
            "generated_artifacts_saved": False,
            "foot_metric_fps": self.fps,
            "contact_threshold": 0.5,
            "sampling_steps": self.steps,
            "requested_native_solver": self.native_solver,
            "guidance": self.guidance,
            "per_case_seeded_noise": True,
            "execution": "in-process training callback",
        }
        self.history_path = self.output_dir / "history.json"
        self.history = []
        if self.history_path.is_file():
            try:
                loaded = json.loads(self.history_path.read_text())
                if isinstance(loaded, list):
                    self.history = loaded
            except (OSError, ValueError, json.JSONDecodeError):
                pass

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)

    def evaluate(
        self,
        model: MotionExpertInContext,
        checkpoint_args: dict,
        checkpoint_step: int,
        checkpoint_path: str,
    ) -> dict:
        was_training = model.training
        model.eval()
        try:
            result = evaluate_model(
                f"inline_step_{checkpoint_step:06d}",
                model,
                checkpoint_args,
                int(checkpoint_step),
                checkpoint_path,
                self.cases,
                self.batch_size,
                self.steps,
                self.guidance,
                self.generator_cache,
                self.embedder,
                self.skeleton,
                self.mean,
                self.std,
                self.gt_embeddings,
                self.text_embeddings,
                self.fps,
                self.native_solver,
                self.device,
                self.shape_counterfactuals,
            )
        finally:
            model.train(was_training)

        payload = {
            "protocol": self.protocol,
            "evaluator": {
                "checkpoint": self.tmr_ckpt,
                "checkpoint_step": self.embedder.step,
                "stats": self.tmr_stats,
                "fps": self.embedder.fps,
            },
            "ground_truth": {
                "num_motions": len(self.cases),
                "plain_t2m": _plain_retrieval(self.gt_embeddings, self.text_embeddings),
                "physical_20fps": self.gt_physical,
                "shape": self.gt_shape,
            },
            "generators": {f"step_{checkpoint_step:06d}": result},
        }
        result_path = self.output_dir / f"step_{checkpoint_step:06d}.json"
        self._write_json(result_path, payload)

        summary = {
            "step": int(checkpoint_step),
            "result": str(result_path),
            "protocol_R03": result["tmr"]["TMR/t2m_R/R03"],
            "plain_R03": result["plain_t2m_gen"]["R03"],
            "fid_gen_gt": result["tmr"]["TMR/FID/gen_gt"],
            "contact_skate_cm_s": (
                result["physical_20fps"]["foot_skate_from_pred_contacts"] * 100.0
            ),
            "height_skate_cm_s": (
                result["physical_20fps"]["foot_skate_from_height"] * 100.0
            ),
            "contact_consistency": result["physical_20fps"]["foot_contact_consistency"],
            "skate_ratio": result["physical_20fps"]["foot_skate_ratio"],
            "bone_mae_cm": result["shape"]["bone_length_mae_cm_mean"],
            "shape_centered_correlation": result["shape"]["population_tracking"][
                "actor_centered_correlation"
            ],
            "shape_centered_response_slope": result["shape"]["population_tracking"][
                "actor_centered_response_slope"
            ],
            "shape_centered_variance_ratio": result["shape"]["population_tracking"][
                "actor_centered_variance_ratio"
            ],
        }
        farthest = result["shape"]["counterfactuals"].get("farthest_natural")
        if farthest is not None:
            summary.update({
                "shape_cf_delta_cosine": farthest["delta_cosine"],
                "shape_cf_response_slope": farthest["delta_response_slope"],
                "shape_cf_magnitude_ratio": farthest["delta_magnitude_ratio"],
                "shape_cf_target_mae_cm": farthest["counterfactual_target_bone_mae_cm"],
                "shape_cf_target_advantage_cm": farthest[
                    "counterfactual_target_advantage_cm"
                ],
                "shape_cf_plain_R03": farthest["plain_t2m"]["R03"],
            })
        self.history = [row for row in self.history if row.get("step") != checkpoint_step]
        self.history.append(summary)
        self.history.sort(key=lambda row: row["step"])
        self._write_json(self.history_path, self.history)
        print(f"[inline-eval] {json.dumps(summary, sort_keys=True)}", flush=True)
        torch.cuda.empty_cache()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator",
        action="append",
        type=_parse_generator,
        required=True,
        help="repeatable LABEL=/absolute/checkpoint.pt",
    )
    parser.add_argument("--tmr-ckpt", required=True)
    parser.add_argument("--tmr-stats", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--testsuite", default=TESTSUITE)
    parser.add_argument("--split", default="content")
    parser.add_argument("--group", default="overview")
    parser.add_argument("--uniego-root", default=DATA_ROOT)
    parser.add_argument("--generator-mean", default=MEAN_PATH)
    parser.add_argument("--generator-std", default=STD_PATH)
    parser.add_argument(
        "--generator-text-cache",
        default=BENCH_TEXT_CACHE,
        help="benchmark LLM2Vec cache; same frozen representation as training, with all eval texts",
    )
    parser.add_argument("--eval-text-cache", default=BENCH_TEXT_CACHE)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--native-solver",
        choices=["euler", "heun", "unipc"],
        default="euler",
        help="unipc calls NVIDIA Cosmos-3 FlowUniPCMultistepScheduler directly",
    )
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=0, help="0 evaluates every usable case")
    parser.add_argument(
        "--shape-counterfactual",
        choices=["none", "farthest"],
        default="farthest",
        help="paired same-text/same-noise intervention using the farthest natural test skeleton",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("bs_tmr_eval.py requires a Slurm GPU allocation")
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("--steps and --batch-size must be positive")
    for label, checkpoint_path in args.generator:
        if not os.path.isfile(checkpoint_path):
            parser.error(f"missing generator checkpoint for {label}: {checkpoint_path}")
    if not os.path.isfile(args.tmr_ckpt):
        parser.error(f"missing TMR checkpoint: {args.tmr_ckpt}")

    generator_cache = LLM2VecCache(args.generator_text_cache, device=args.device)
    evaluator_cache = LLM2VecCache(args.eval_text_cache, device=args.device)
    cases, audit = build_cases(
        args.testsuite,
        args.split,
        args.group,
        args.uniego_root,
        generator_cache,
        evaluator_cache,
        args.fps,
        args.min_frames,
        args.max_cases,
    )
    if not cases:
        raise SystemExit(f"no usable {args.split}/{args.group} cases: {audit}")
    print(f"[cases] {json.dumps(audit, sort_keys=True)}", flush=True)

    embedder = ShapeTMREmbedder(
        args.tmr_ckpt,
        args.tmr_stats,
        text_cache_path=args.eval_text_cache,
        device=args.device,
    )
    skeleton = SOMASkeleton30().to(args.device)
    parents = [int(x) for x in skeleton.joint_parents]
    if args.shape_counterfactual == "farthest":
        shape_counterfactuals, shape_counterfactual_audit = _natural_shape_counterfactuals(
            cases,
            parents,
        )
    else:
        shape_counterfactuals = {}
        shape_counterfactual_audit = {"strategy": "disabled", "num_pairs": 0}
    print(
        f"[shape] {json.dumps(shape_counterfactual_audit, sort_keys=True)}",
        flush=True,
    )
    mean = torch.from_numpy(np.load(args.generator_mean)).float().to(args.device)
    std = torch.from_numpy(np.load(args.generator_std)).float().to(args.device)

    gt_embeddings, text_embeddings, gt_physical, gt_shape = precompute_references(
        cases,
        args.batch_size,
        embedder,
        skeleton,
        args.fps,
        args.device,
    )
    results = {}
    for label, checkpoint_path in args.generator:
        if label in results:
            raise ValueError(f"duplicate generator label: {label}")
        results[label] = evaluate_generator(
            label,
            checkpoint_path,
            cases,
            args.batch_size,
            args.steps,
            args.guidance,
            generator_cache,
            embedder,
            skeleton,
            mean,
            std,
            gt_embeddings,
            text_embeddings,
            args.fps,
            args.native_solver,
            args.device,
            shape_counterfactuals,
        )
        r = results[label]
        print(
            f"[{label}] R03={r['tmr']['TMR/t2m_R/R03']:.2f} "
            f"plain_R03={r['plain_t2m_gen']['R03']:.2f} "
            f"FID_gen_gt={r['tmr']['TMR/FID/gen_gt']:.4f} "
            f"skate_cm_s={r['physical_20fps']['foot_skate_from_pred_contacts'] * 100.0:.3f} "
            f"bone_mae_cm={r['shape']['bone_length_mae_cm_mean']:.3f} "
            f"shape_corr={r['shape']['population_tracking']['actor_centered_correlation']:.3f}",
            flush=True,
        )

    payload = {
        "protocol": {
            "testsuite": args.testsuite,
            "split": args.split,
            "group": args.group,
            "case_audit": audit,
            "generator_representation": "normalized 283-D proportional UniEgo at 20 fps",
            "tmr_conversion": "unnormalize -> decode SOMA-30 joints -> resample 20 to 30 fps -> C45 TMRMotionRep",
            "shape_conditioning": "same centered proportional neutral_joints supplied to generator and C45",
            "shape_counterfactual": shape_counterfactual_audit,
            "generated_artifacts_saved": False,
            "foot_metric_fps": args.fps,
            "contact_threshold": 0.5,
            "sampling_steps": args.steps,
            "requested_native_solver": args.native_solver,
            "guidance": args.guidance,
            "per_case_seeded_noise": True,
        },
        "evaluator": {
            "checkpoint": args.tmr_ckpt,
            "checkpoint_step": embedder.step,
            "stats": args.tmr_stats,
            "fps": embedder.fps,
        },
        "ground_truth": {
            "num_motions": len(cases),
            "plain_t2m": _plain_retrieval(gt_embeddings, text_embeddings),
            "physical_20fps": gt_physical,
            "shape": gt_shape,
        },
        "generators": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[eval] wrote aggregate metrics only: {out_path}", flush=True)


if __name__ == "__main__":
    main()
