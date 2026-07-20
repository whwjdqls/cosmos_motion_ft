"""Distributed C45 evaluation for the native-schedule Phase-2 T2M/TI2M expert.

The evaluator intentionally keeps three normalization spaces separate:

* Phase-2 normalized UniEgo is used only by the generator.
* Generated and GT motion are decoded to raw SOMA-30 joints at 20 FPS.
* C45 applies its bundled official 30 FPS TMR statistics only inside its motion rep.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from dataset import SHARED_MEAN_PATH, SHARED_STD_PATH
from decode_uniego_torch import decode_joints
import flow
from bs_shape_metrics import (
    counterfactual_shape_response,
    population_shape_tracking,
)
from render_motion import render_conditioned_motion_mp4, render_motion_mp4
import sample as joint_sample
from shape_tmr_eval_common import (
    DEFAULT_BUNDLE_ROOT,
    SUITES,
    EvalCase,
    add_bundle_python_paths,
    chunked_farthest_indices,
    load_gt_batch,
    read_jsonl,
    seeded_initial_noise,
    sha256_file,
)
from uniego_layout import FEAT_DIM, FOOT_SLICE


BUNDLE_ROOT = add_bundle_python_paths(
    os.environ.get("SHAPE_TMR_BUNDLE", str(DEFAULT_BUNDLE_ROOT))
)

from kimodo.metrics import (  # noqa: E402
    FootContactConsistency,
    FootSkateFromContacts,
    FootSkateFromHeight,
    FootSkateRatio,
    compute_metrics,
    compute_tmr_retrieval_metrics,
)
from kimodo.skeleton import SOMASkeleton30  # noqa: E402
from st_dataset import resample_joints_time  # noqa: E402
from st_eval import ShapeTMREmbedder  # noqa: E402


PHYSICAL_KEYS = (
    "foot_skate_from_height",
    "foot_skate_from_pred_contacts",
    "foot_skate_max_vel",
    "foot_contact_consistency",
    "foot_skate_ratio",
)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def _file_identity(path: str | os.PathLike[str]) -> dict[str, str | int]:
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _distributed_device():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-2 C45 evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    return rank, world, torch.device("cuda", local_rank)


def _barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def _neutral_bones(neutral: np.ndarray, parents: list[int]) -> np.ndarray:
    edges = [(joint, parent) for joint, parent in enumerate(parents) if 0 <= parent < len(parents)]
    child = np.asarray([edge[0] for edge in edges])
    parent = np.asarray([edge[1] for edge in edges])
    return np.linalg.norm(neutral[:, child] - neutral[:, parent], axis=-1)


def _bone_vectors(
    joints: torch.Tensor,
    lengths: torch.Tensor,
    neutral: torch.Tensor,
    parents: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    edges = [(joint, parent) for joint, parent in enumerate(parents) if 0 <= parent < len(parents)]
    child = torch.tensor([edge[0] for edge in edges], device=joints.device)
    parent = torch.tensor([edge[1] for edge in edges], device=joints.device)
    motion = []
    for index, length in enumerate(lengths.tolist()):
        motion.append(
            torch.linalg.vector_norm(
                joints[index, :length, child] - joints[index, :length, parent], dim=-1
            ).mean(dim=0)
        )
    target = torch.linalg.vector_norm(neutral[:, child] - neutral[:, parent], dim=-1)
    return torch.stack(motion), target


def _physical_batch(
    skeleton: SOMASkeleton30,
    joints: torch.Tensor,
    contacts: torch.Tensor,
    lengths: torch.Tensor,
    fps: float,
) -> dict[str, torch.Tensor]:
    metrics = [
        FootSkateFromHeight(skeleton=skeleton, fps=fps),
        FootSkateFromContacts(skeleton=skeleton, fps=fps),
        FootContactConsistency(skeleton=skeleton, fps=fps),
        FootSkateRatio(skeleton=skeleton, fps=fps),
    ]
    return compute_metrics(
        metrics,
        {"posed_joints": joints, "foot_contacts": contacts, "lengths": lengths},
    )


@torch.inference_mode()
def _tmr_embed_joints(
    embedder: ShapeTMREmbedder,
    joints: torch.Tensor,
    lengths: torch.Tensor,
    neutral: torch.Tensor,
    source_fps: float,
) -> torch.Tensor:
    features = []
    for index, length in enumerate(lengths.tolist()):
        posed = joints[index, :length].detach().cpu()
        posed = resample_joints_time(posed, source_fps, embedder.fps)
        count = int(posed.shape[0])
        feature = embedder.rep(
            posed_joints=posed.unsqueeze(0),
            to_normalize=True,
            to_canonicalize=True,
            lengths=torch.tensor([count]),
        )[0]
        features.append(feature.float())
    max_t = max(len(feature) for feature in features)
    dim = int(features[0].shape[-1])
    padded = torch.zeros(len(features), max_t, dim, dtype=torch.float32)
    mask = torch.zeros(len(features), max_t, dtype=torch.bool)
    for index, feature in enumerate(features):
        padded[index, : len(feature)] = feature
        mask[index, : len(feature)] = True
    return embedder.embed_motion_features(padded, mask, neutral.detach().cpu())


def _plain_retrieval(motion: np.ndarray, text: np.ndarray, chunk: int = 512) -> dict[str, float]:
    ranks = np.empty(len(text), dtype=np.int64)
    paired = np.empty(len(text), dtype=np.float64)
    for start in range(0, len(text), chunk):
        end = min(start + chunk, len(text))
        similarity = text[start:end] @ motion.T
        local = np.arange(end - start)
        diagonal = similarity[local, np.arange(start, end)]
        ranks[start:end] = (similarity > diagonal[:, None]).sum(axis=1)
        paired[start:end] = diagonal
    return {
        "R01": float((ranks < 1).mean() * 100.0),
        "R02": float((ranks < 2).mean() * 100.0),
        "R03": float((ranks < 3).mean() * 100.0),
        "R05": float((ranks < 5).mean() * 100.0),
        "R10": float((ranks < 10).mean() * 100.0),
        "MedR": float(np.median(ranks) + 1),
        "paired_cosine": float(paired.mean()),
    }


def _decode_generated(
    normalized: torch.Tensor,
    lengths: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = normalized.float() * std + mean
    for index, length in enumerate(lengths.tolist()):
        features[index, length:] = features[index, length - 1]
    joints = decode_joints(features)
    contacts = features[..., FOOT_SLICE] > 0.5
    return features, joints, contacts


def _load_images(cases: list[EvalCase], fps: float) -> list[torch.Tensor]:
    from nymeria_camera_dataset import decode_window_pyav

    images = []
    for case in cases:
        if case.image_path is None or case.image_start is None:
            raise ValueError(f"{case.case_id}: TI2M case is missing image metadata")
        frames = decode_window_pyav(case.image_path, case.image_start, 1, fps)
        if len(frames) != 1:
            raise RuntimeError(f"{case.case_id}: decoded {len(frames)} condition frames")
        images.append(
            torch.from_numpy(np.ascontiguousarray(frames[0])).permute(2, 0, 1).contiguous()
        )
    return images


def _conditioning_context(model, cases: list[EvalCase], mode: str, fps: float):
    cond_ids = []
    null_ids = []
    cond_reasoner = []
    null_reasoner = []
    images = None
    if mode == "textimg2motion":
        images = _load_images(cases, fps)
        for case, image in zip(cases, images):
            conditional = model.cosmos.encode_reasoner_image_text(
                case.text, image, image_size=model.reasoner_image_size
            )
            unconditional = model.cosmos.encode_reasoner_image_text(
                "", image, image_size=model.reasoner_image_size
            )
            cond_reasoner.append(conditional)
            null_reasoner.append(unconditional)
            cond_ids.append(conditional["input_ids"].view(1, -1))
            null_ids.append(unconditional["input_ids"].view(1, -1))
    else:
        for case in cases:
            cond_ids.append(model.cosmos.tokenize(case.text))
            null_ids.append(model.cosmos.tokenize(""))
        cond_reasoner = [None] * len(cases)
        null_reasoner = [None] * len(cases)
    return {
        "cond_ids": cond_ids,
        "null_ids": null_ids,
        "cond_reasoner": cond_reasoner,
        "null_reasoner": null_reasoner,
        "images": images,
    }


@torch.inference_mode()
def _sample_batch(
    model,
    cases: list[EvalCase],
    neutral: torch.Tensor,
    context: dict,
    mode: str,
    initial_noise: torch.Tensor,
    *,
    steps: int,
    guidance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model.objective != "x0" or model.motion_schedule != "native":
        raise ValueError(
            f"expected native x0 motion checkpoint, got {model.objective}/{model.motion_schedule}"
        )
    batch = len(cases)
    lengths = torch.tensor([case.num_frames for case in cases], device=neutral.device)
    max_t = int(lengths.max().item())
    pad = torch.arange(max_t, device=neutral.device)[None] >= lengths[:, None]
    noisy = ~pad
    modes = [mode] * batch
    empty_video = [None] * batch
    conditional = model.predict_closure(
        input_ids_list=context["cond_ids"],
        neutral_joints=neutral,
        motion_pad_mask=pad,
        noisy_frame_mask=noisy,
        modes=modes,
        video_latents=empty_video,
        reasoner_inputs=context["cond_reasoner"],
    )
    unconditional = None
    if guidance != 1.0:
        unconditional = model.predict_closure(
            input_ids_list=context["null_ids"],
            neutral_joints=neutral,
            motion_pad_mask=pad,
            noisy_frame_mask=noisy,
            modes=modes,
            video_latents=empty_video,
            reasoner_inputs=context["null_reasoner"],
        )
    sampled = flow.sample_x0_native_unipc(
        conditional,
        T=max_t,
        motion_dim=FEAT_DIM,
        steps=steps,
        guidance=guidance,
        predict_null=unconditional,
        batch=batch,
        device=neutral.device,
        dtype=torch.float32,
        initial_noise=initial_noise,
        native_shift=model.motion_shift,
        native_num_train_timesteps=model.motion_num_train_timesteps,
    )
    sampled.masked_fill_(pad.unsqueeze(-1), 0.0)
    return sampled, lengths


def _append(store: dict[str, list[np.ndarray]], key: str, value) -> None:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    store[key].append(np.asarray(value))


def _variant_metrics(
    store: dict[str, list[np.ndarray]],
    prefix: str,
    normalized: torch.Tensor,
    lengths: torch.Tensor,
    neutral: torch.Tensor,
    embedder: ShapeTMREmbedder,
    skeleton: SOMASkeleton30,
    parents: list[int],
    mean: torch.Tensor,
    std: torch.Tensor,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _features, joints, contacts = _decode_generated(normalized, lengths, mean, std)
    embedding = _tmr_embed_joints(embedder, joints, lengths, neutral, fps)
    physical = _physical_batch(skeleton, joints, contacts, lengths, fps)
    bones, targets = _bone_vectors(joints, lengths, neutral, parents)
    _append(store, f"{prefix}_embedding", embedding)
    _append(store, f"{prefix}_bones", bones)
    _append(store, f"{prefix}_target_bones", targets)
    for key in PHYSICAL_KEYS:
        _append(store, f"{prefix}_{key}", physical[key])
    return joints, bones, targets


def _save_visualizations(
    out_dir: Path,
    cohort: str,
    prefix: str,
    cases: list[EvalCase],
    normalized: torch.Tensor,
    lengths: torch.Tensor,
    gt_joints: torch.Tensor,
    gt_lengths: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    images: list[torch.Tensor] | None,
    limit: int,
    fps: int,
) -> None:
    if limit <= 0:
        return
    features, generated, _contacts = _decode_generated(normalized, lengths, mean, std)
    root = out_dir / "viz" / cohort / prefix
    root.mkdir(parents=True, exist_ok=True)
    for index in range(min(limit, len(cases))):
        stem = cases[index].case_id[:160]
        length = int(lengths[index].item())
        gt_length = int(gt_lengths[index].item())
        np.save(root / f"{stem}_generated_uniego.npy", features[index, :length].cpu().numpy())
        np.save(root / f"{stem}_gt_joints.npy", gt_joints[index, :gt_length].cpu().numpy())
        caption = cases[index].text[:120]
        if images is None:
            render_motion_mp4(
                generated[index, :length].cpu().numpy(),
                str(root / f"{stem}.mp4"),
                caption=caption,
                fps=fps,
                gt_joints=gt_joints[index, :gt_length].cpu().numpy(),
            )
        else:
            render_conditioned_motion_mp4(
                condition_image=images[index],
                gen_joints=generated[index, :length].cpu().numpy(),
                gt_joints=gt_joints[index, :gt_length].cpu().numpy(),
                out_path=str(root / f"{stem}.mp4"),
                condition_out_path=str(root / f"{stem}_condition.png"),
                caption=caption,
                fps=fps,
            )


def _evaluate_local_shard(
    model,
    embedder: ShapeTMREmbedder,
    skeleton: SOMASkeleton30,
    cases: list[EvalCase],
    global_indices: list[int],
    all_neutral: np.ndarray,
    counterfactual_indices: np.ndarray | None,
    mode: str,
    variants: list[tuple[str, float]],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    steps: int,
    fps: float,
    rank: int,
    out_dir: Path,
    viz_limit: int,
) -> dict[str, np.ndarray]:
    parents = [int(value) for value in skeleton.joint_parents]
    store: dict[str, list[np.ndarray]] = defaultdict(list)
    started = time.time()
    visualized = False
    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start : start + batch_size]
        batch_global = global_indices[start : start + batch_size]
        neutral = torch.from_numpy(all_neutral[batch_global]).to(device=device, dtype=torch.float32)
        context = _conditioning_context(model, batch_cases, mode, fps)
        max_t = max(case.num_frames for case in batch_cases)
        initial = seeded_initial_noise(batch_cases, max_t, device)

        _gt_features, gt_joints, gt_contacts, gt_lengths = load_gt_batch(batch_cases, device)
        gt_embedding = _tmr_embed_joints(embedder, gt_joints, gt_lengths, neutral, fps)
        text_embedding = embedder.embed_text([case.text for case in batch_cases])
        gt_physical = _physical_batch(skeleton, gt_joints, gt_contacts, gt_lengths, fps)
        gt_bones, target_bones = _bone_vectors(gt_joints, gt_lengths, neutral, parents)
        _append(store, "global_index", np.asarray(batch_global, dtype=np.int64))
        _append(store, "case_id", np.asarray([case.case_id for case in batch_cases], dtype=str))
        _append(store, "gt_embedding", gt_embedding)
        _append(store, "text_embedding", text_embedding)
        _append(store, "gt_bones", gt_bones)
        _append(store, "target_bones", target_bones)
        for key in PHYSICAL_KEYS:
            _append(store, f"gt_{key}", gt_physical[key])

        for prefix, guidance in variants:
            normalized, lengths = _sample_batch(
                model,
                batch_cases,
                neutral,
                context,
                mode,
                initial.clone(),
                steps=steps,
                guidance=guidance,
            )
            _variant_metrics(
                store,
                prefix,
                normalized,
                lengths,
                neutral,
                embedder,
                skeleton,
                parents,
                mean,
                std,
                fps,
            )
            if rank == 0 and not visualized:
                _save_visualizations(
                    out_dir,
                    batch_cases[0].cohort,
                    prefix,
                    batch_cases,
                    normalized,
                    lengths,
                    gt_joints,
                    gt_lengths,
                    mean,
                    std,
                    context["images"],
                    viz_limit,
                    int(fps),
                )

            if counterfactual_indices is not None:
                cf_global = counterfactual_indices[np.asarray(batch_global)]
                cf_neutral = torch.from_numpy(all_neutral[cf_global]).to(
                    device=device, dtype=torch.float32
                )
                cf_normalized, cf_lengths = _sample_batch(
                    model,
                    batch_cases,
                    cf_neutral,
                    context,
                    mode,
                    initial.clone(),
                    steps=steps,
                    guidance=guidance,
                )
                _cf_features, cf_joints, _cf_contacts = _decode_generated(
                    cf_normalized, cf_lengths, mean, std
                )
                cf_embedding = _tmr_embed_joints(
                    embedder, cf_joints, cf_lengths, cf_neutral, fps
                )
                cf_bones, cf_targets = _bone_vectors(
                    cf_joints, cf_lengths, cf_neutral, parents
                )
                _append(store, f"{prefix}_cf_embedding", cf_embedding)
                _append(store, f"{prefix}_cf_bones", cf_bones)
                _append(store, f"{prefix}_cf_target_bones", cf_targets)
        visualized = True
        completed = min(start + len(batch_cases), len(cases))
        print(
            f"[rank {rank}][{cases[0].cohort}] {completed}/{len(cases)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    return {key: np.concatenate(values, axis=0) for key, values in store.items()}


def _mean_physical(arrays: dict[str, np.ndarray], prefix: str) -> dict[str, float]:
    return {key: float(np.mean(arrays[f"{prefix}_{key}"])) for key in PHYSICAL_KEYS}


def _finalize_variant(
    arrays: dict[str, np.ndarray],
    prefix: str,
    *,
    steps: int,
    guidance: float,
    counterfactual: bool,
) -> dict:
    generated = arrays[f"{prefix}_embedding"]
    gt = arrays["gt_embedding"]
    text = arrays["text_embedding"]
    protocol = compute_tmr_retrieval_metrics(generated, text, gt)
    generated_bones = arrays[f"{prefix}_bones"]
    target_bones = arrays[f"{prefix}_target_bones"]
    bone_error = np.abs(generated_bones - target_bones).mean(axis=1) * 100.0
    shape = {
        "bone_length_mae_cm_mean": float(bone_error.mean()),
        "bone_length_mae_cm_std": float(bone_error.std()),
        "population_tracking": population_shape_tracking(generated_bones, target_bones),
        "counterfactuals": {},
    }
    if counterfactual:
        cf_embedding = arrays[f"{prefix}_cf_embedding"]
        cf_bones = arrays[f"{prefix}_cf_bones"]
        cf_targets = arrays[f"{prefix}_cf_target_bones"]
        cf_protocol = compute_tmr_retrieval_metrics(cf_embedding, text, gt)
        cf_plain = _plain_retrieval(cf_embedding, text)
        response = counterfactual_shape_response(
            generated_bones,
            cf_bones,
            target_bones,
            cf_targets,
        )
        shape["counterfactuals"]["farthest_natural"] = {
            **response,
            "tmr": {key: float(value) for key, value in cf_protocol.items()},
            "plain_t2m": cf_plain,
            "protocol_R03_delta": float(
                cf_protocol["TMR/t2m_R/R03"] - protocol["TMR/t2m_R/R03"]
            ),
        }
    sampling_passes = 2 if counterfactual else 1
    forward_calls = steps if guidance == 1.0 else 2 * steps
    return {
        "training_schedule": "native",
        "prediction": "x0",
        "sampler_solver": "unipc",
        "sampling_steps": steps,
        "denoiser_evaluations": steps,
        "model_forward_calls_with_cfg": forward_calls,
        "sampling_passes": sampling_passes,
        "total_denoiser_evaluations_per_case": sampling_passes * steps,
        "total_model_forward_calls_per_case": sampling_passes * forward_calls,
        "guidance": guidance,
        "num_motions": len(generated),
        "tmr": {key: float(value) for key, value in protocol.items()},
        "plain_t2m_gen": _plain_retrieval(generated, text),
        "plain_t2m_gt": _plain_retrieval(gt, text),
        "physical_20fps": _mean_physical(arrays, prefix),
        "shape": shape,
    }


def _finalize_cohort(
    cohort: str,
    result_path: Path,
    shard_dir: Path,
    world: int,
    total_cases: int,
    variants: list[tuple[str, float]],
    *,
    mode: str,
    steps: int,
    counterfactual: bool,
    protocol: dict,
    checkpoint: str,
    tmr_ckpt: str,
    tmr_stats: str,
    text_cache: str,
    native_shift: float,
    native_num_train_timesteps: int,
) -> dict:
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    for rank in range(world):
        shard_path = shard_dir / f"{cohort}_rank{rank:04d}_of_{world:04d}.npz"
        with np.load(shard_path, allow_pickle=False) as shard:
            for key in shard.files:
                pieces[key].append(np.asarray(shard[key]))
    arrays = {key: np.concatenate(values, axis=0) for key, values in pieces.items()}
    order = np.argsort(arrays["global_index"])
    arrays = {key: value[order] for key, value in arrays.items()}
    expected = np.arange(total_cases, dtype=np.int64)
    if not np.array_equal(arrays["global_index"], expected):
        raise RuntimeError(f"{cohort}: distributed shards do not cover every global case exactly")
    if len(set(arrays["case_id"].tolist())) != total_cases:
        raise RuntimeError(f"{cohort}: case IDs are not unique")
    if not all(np.isfinite(value).all() for key, value in arrays.items() if value.dtype.kind in "fc"):
        raise RuntimeError(f"{cohort}: non-finite shard values")

    gt_bone_error = np.abs(arrays["gt_bones"] - arrays["target_bones"]).mean(axis=1) * 100.0
    gt_shape = {
        "bone_length_mae_cm_mean": float(gt_bone_error.mean()),
        "bone_length_mae_cm_std": float(gt_bone_error.std()),
        "population_tracking": population_shape_tracking(
            arrays["gt_bones"], arrays["target_bones"]
        ),
    }
    payload = {
        "protocol": {
            "cohort": cohort,
            "mode": mode,
            "case_audit": protocol,
            "generator_representation": "Phase-2 normalized 283-D proportional UniEgo at 20 FPS",
            "tmr_conversion": (
                "generator-stat unnormalize -> SOMA-30 joints -> resample 20 to 30 FPS -> "
                "C45 TMRMotionRep with evaluator-only stats"
            ),
            "shape_conditioning": (
                "same centered proportional neutral_joints supplied to Phase-2 and C45"
            ),
            "counterfactual": (
                "same text and exact initial noise with farthest natural skeleton"
                if counterfactual else "disabled (TI2M image fixes actor appearance)"
            ),
            "ti2m_cfg_semantics": (
                "conditional and null branches share the same image; only text is dropped"
                if mode == "textimg2motion" else None
            ),
            "sampling_steps": steps,
            "native_shift": native_shift,
            "native_num_train_timesteps": native_num_train_timesteps,
            "per_case_seeded_noise": True,
        },
        "evaluator": {
            "checkpoint": tmr_ckpt,
            "stats": tmr_stats,
            "text_cache": text_cache,
            "fps": 30,
        },
        "generator": {"checkpoint": checkpoint},
        "ground_truth": {
            "num_motions": total_cases,
            "plain_t2m": _plain_retrieval(arrays["gt_embedding"], arrays["text_embedding"]),
            "physical_20fps": _mean_physical(arrays, "gt"),
            "shape": gt_shape,
        },
        "generators": {},
    }
    for prefix, guidance in variants:
        payload["generators"][prefix] = _finalize_variant(
            arrays,
            prefix,
            steps=steps,
            guidance=guidance,
            counterfactual=counterfactual,
        )
    _atomic_json(result_path, payload)
    return payload


def _summary_row(payload: dict, prefix: str) -> dict:
    result = payload["generators"][prefix]
    row = {
        "n": result["num_motions"],
        "protocol_R01": result["tmr"]["TMR/t2m_R/R01"],
        "protocol_R02": result["tmr"]["TMR/t2m_R/R02"],
        "protocol_R03": result["tmr"]["TMR/t2m_R/R03"],
        "protocol_R05": result["tmr"]["TMR/t2m_R/R05"],
        "protocol_R10": result["tmr"]["TMR/t2m_R/R10"],
        "plain_R03": result["plain_t2m_gen"]["R03"],
        "fid_gen_gt": result["tmr"]["TMR/FID/gen_gt"],
        "contact_skate_cm_s": result["physical_20fps"][
            "foot_skate_from_pred_contacts"
        ] * 100.0,
        "height_skate_cm_s": result["physical_20fps"]["foot_skate_from_height"] * 100.0,
        "max_contact_velocity_cm_s": result["physical_20fps"]["foot_skate_max_vel"] * 100.0,
        "contact_consistency": result["physical_20fps"]["foot_contact_consistency"],
        "skate_ratio": result["physical_20fps"]["foot_skate_ratio"],
        "bone_mae_cm": result["shape"]["bone_length_mae_cm_mean"],
        "shape_centered_correlation": result["shape"]["population_tracking"][
            "actor_centered_correlation"
        ],
        "shape_response_slope": result["shape"]["population_tracking"][
            "actor_centered_response_slope"
        ],
        "shape_variance_ratio": result["shape"]["population_tracking"][
            "actor_centered_variance_ratio"
        ],
    }
    farthest = result["shape"]["counterfactuals"].get("farthest_natural")
    if farthest:
        row.update(
            {
                "shape_cf_delta_cosine": farthest["delta_cosine"],
                "shape_cf_response_slope": farthest["delta_response_slope"],
                "shape_cf_target_advantage_cm": farthest[
                    "counterfactual_target_advantage_cm"
                ],
            }
        )
    return row


def _bones_weighted_summary(summaries: dict[str, dict]) -> dict | None:
    cohorts = [f"bones_{split}_{group}" for split, group in SUITES]
    if any(cohort not in summaries for cohort in cohorts):
        return None
    rows = [summaries[cohort]["t2m_cfg2"] for cohort in cohorts]
    total = sum(int(row["n"]) for row in rows)
    weighted = {}
    for key in sorted(set.intersection(*(set(row) for row in rows))):
        if key == "n":
            continue
        values = [row[key] for row in rows]
        if all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            weighted[key] = sum(
                float(row[key]) * int(row["n"]) for row in rows
            ) / total
    return {
        "total_cases": total,
        "included_cohorts": cohorts,
        "case_weighted_mean_of_suite_metrics": weighted,
        "note": (
            "Case-weighted means of six independently computed suite metrics; retrieval and "
            "FID are not recomputed over one merged distractor pool."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument(
        "--tmr-ckpt",
        default=str(BUNDLE_ROOT / "artifacts" / "evaluator" / "c45_step_00005000.pt"),
    )
    parser.add_argument(
        "--tmr-stats",
        default=str(BUNDLE_ROOT / "artifacts" / "evaluator" / "stats" / "motion"),
    )
    parser.add_argument("--cohorts", default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--viz-limit", type=int, default=1)
    parser.add_argument("--no-counterfactual", action="store_true")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.steps <= 0:
        parser.error("--batch-size and --steps must be positive")
    rank, world, device = _distributed_device()
    manifest_dir = Path(args.manifest_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    protocol_all = json.loads((manifest_dir / "protocol.json").read_text())
    available = [path.stem for path in sorted(manifest_dir.glob("*.jsonl"))]
    requested = available if args.cohorts == "all" else args.cohorts.split(",")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown cohorts {unknown}; available={available}")

    mean = torch.from_numpy(np.load(SHARED_MEAN_PATH)).float().to(device).view(1, 1, -1)
    std = torch.from_numpy(np.load(SHARED_STD_PATH)).float().to(device).view(1, 1, -1)
    model, _cosmos, checkpoint_meta = joint_sample.load_joint_model(
        args.checkpoint,
        device=device,
        motion_native_solver_cli="unipc",
    )
    if int(checkpoint_meta.get("step", -1)) != 200000:
        print(f"[rank {rank}] WARNING: checkpoint step={checkpoint_meta.get('step')}", flush=True)
    embedder = ShapeTMREmbedder(
        args.tmr_ckpt,
        args.tmr_stats,
        text_cache_path=args.text_cache,
        device=str(device),
    )
    skeleton = SOMASkeleton30().to(device)
    parents = [int(value) for value in skeleton.joint_parents]
    protocol_sha = sha256_file(manifest_dir / "protocol.json")
    checkpoint_identity = _file_identity(args.checkpoint)
    text_cache_identity = _file_identity(args.text_cache)
    tmr_checkpoint_identity = _file_identity(args.tmr_ckpt)
    summaries = {}

    for cohort in requested:
        all_cases = read_jsonl(manifest_dir / f"{cohort}.jsonl")
        all_neutral = np.load(manifest_dir / f"{cohort}_neutral.npy").astype(np.float32)
        if len(all_neutral) != len(all_cases):
            raise RuntimeError(f"{cohort}: neutral/case count mismatch")
        if args.max_cases > 0:
            all_cases = all_cases[: args.max_cases]
            all_neutral = all_neutral[: args.max_cases]
        total_cases = len(all_cases)
        if total_cases == 0:
            raise RuntimeError(f"{cohort}: no cases")
        is_ti2m = cohort == "nymeria_ti2m"
        mode = "textimg2motion" if is_ti2m else "text2motion"
        variants = (
            [("ti2m_cfg2", 2.0), ("ti2m_no_cfg", 1.0)]
            if is_ti2m else [("t2m_cfg2", 2.0)]
        )
        use_counterfactual = not is_ti2m and not args.no_counterfactual
        counterfactual_indices = None
        if use_counterfactual:
            counterfactual_indices = chunked_farthest_indices(
                _neutral_bones(all_neutral, parents)
            )
        global_indices = list(range(rank, total_cases, world))
        local_cases = [all_cases[index] for index in global_indices]
        shard_dir = out_dir / "shards"
        shard_path = shard_dir / f"{cohort}_rank{rank:04d}_of_{world:04d}.npz"
        shard_meta = shard_path.with_suffix(".json")
        expected_meta = {
            "cohort": cohort,
            "rank": rank,
            "world": world,
            "total_cases": total_cases,
            "local_cases": len(local_cases),
            "checkpoint": checkpoint_identity,
            "text_cache": text_cache_identity,
            "tmr_checkpoint": tmr_checkpoint_identity,
            "steps": args.steps,
            "variants": [
                {"name": name, "guidance": guidance} for name, guidance in variants
            ],
            "counterfactual": use_counterfactual,
            "protocol_sha256": protocol_sha,
            "manifest_sha256": sha256_file(manifest_dir / f"{cohort}.jsonl"),
            "neutral_sha256": sha256_file(manifest_dir / f"{cohort}_neutral.npy"),
        }
        reuse = False
        if shard_path.is_file() and shard_meta.is_file():
            reuse = json.loads(shard_meta.read_text()) == expected_meta
        if reuse:
            print(f"[rank {rank}][{cohort}] reuse {shard_path}", flush=True)
        else:
            arrays = _evaluate_local_shard(
                model,
                embedder,
                skeleton,
                local_cases,
                global_indices,
                all_neutral,
                counterfactual_indices,
                mode,
                variants,
                mean,
                std,
                device,
                batch_size=args.batch_size,
                steps=args.steps,
                fps=args.fps,
                rank=rank,
                out_dir=out_dir,
                viz_limit=args.viz_limit,
            )
            _atomic_npz(shard_path, arrays)
            _atomic_json(shard_meta, expected_meta)
        _barrier(world)

        if rank == 0:
            cohort_protocol = (
                protocol_all["nymeria"].get(cohort)
                if cohort.startswith("nymeria_")
                else protocol_all["bones"].get(cohort)
            )
            result_path = out_dir / "results" / f"{cohort}.json"
            payload = _finalize_cohort(
                cohort,
                result_path,
                shard_dir,
                world,
                total_cases,
                variants,
                mode=mode,
                steps=args.steps,
                counterfactual=use_counterfactual,
                protocol=cohort_protocol,
                checkpoint=str(Path(args.checkpoint).resolve()),
                tmr_ckpt=str(Path(args.tmr_ckpt).resolve()),
                tmr_stats=str(Path(args.tmr_stats).resolve()),
                text_cache=str(Path(args.text_cache).resolve()),
                native_shift=float(model.motion_shift),
                native_num_train_timesteps=int(model.motion_num_train_timesteps),
            )
            summaries[cohort] = {
                prefix: _summary_row(payload, prefix) for prefix, _guidance in variants
            }
            print(f"[result][{cohort}] {json.dumps(summaries[cohort], sort_keys=True)}", flush=True)
        _barrier(world)

    if rank == 0:
        summary = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_step": int(checkpoint_meta.get("step", -1)),
            "tmr_checkpoint": str(Path(args.tmr_ckpt).resolve()),
            "tmr_stats_evaluator_only": str(Path(args.tmr_stats).resolve()),
            "generator_mean": SHARED_MEAN_PATH,
            "generator_std": SHARED_STD_PATH,
            "sampler": {
                "schedule": "native",
                "solver": "unipc",
                "steps": args.steps,
                "shift": float(model.motion_shift),
                "num_train_timesteps": int(model.motion_num_train_timesteps),
            },
            "cohorts": summaries,
        }
        bones_summary = _bones_weighted_summary(summaries)
        if bones_summary is not None:
            summary["bones_six_suite_summary"] = bones_summary
        _atomic_json(out_dir / "summary.json", summary)
        print(f"[eval] wrote {out_dir / 'summary.json'}", flush=True)

    _barrier(world)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
