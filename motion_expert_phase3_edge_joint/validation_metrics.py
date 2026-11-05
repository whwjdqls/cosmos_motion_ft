"""Matched Body23/TMR and lossless RGB metrics for periodic validation.

These reuse the reviewed reconstruction/retrieval implementations. I3D source
and all evaluator assets are supplied and checksum-pinned by validation config.
"""
from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from .settings import atomic_json


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def frechet(a, b):
    # Same stable float64 formulation used by the reviewed EgoControl metrics;
    # low-rank covariance is expected for the 71-window cohort.
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2 or not all(np.isfinite(x).all() for x in (a, b)):
        raise ValueError("Invalid FVD feature arrays")
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    def root(cov):
        eig, vec = np.linalg.eigh((cov + cov.T) / 2)
        return (vec * np.sqrt(np.maximum(eig, 0))) @ vec.T
    trace = np.linalg.svd(root(ca) @ root(cb), compute_uv=False).sum()
    value = float(np.square(a.mean(0) - b.mean(0)).sum() + np.trace(ca) + np.trace(cb) - 2 * trace)
    if not np.isfinite(value) or value < -1e-6:
        raise FloatingPointError("Invalid FVD")
    return max(value, 0.)


def physical(raw, target):
    from decode_uniego_torch import decode_transforms
    from nymeria_baselines.body_shape import BODY_JOINTS, CONTACT_JOINTS
    from nymeria_baselines.evaluate_uem_phase3 import measure
    from kimodo.metrics.foot_skate import FootSkateFromHeight
    a, b = [np.take(decode_transforms(torch.from_numpy(x)[None])[0].numpy(), BODY_JOINTS, axis=1)[..., :3, 3]
            for x in (raw, target)]
    skate = FootSkateFromHeight(skeleton=SimpleNamespace(foot_joint_idx=list(CONTACT_JOINTS)), fps=20)
    values = measure(a, b, skate)
    return {name: float(values[key]) * scale for name, key, scale in (
        ("mpjpe_cm", "mpjpe_m", 100), ("pa_mpjpe_cm", "pa_mpjpe_m", 100),
        ("root_ade_cm", "root_err_m", 100), ("root_fde_cm", "root_fde_m", 100),
        ("relative_root_ade_cm", "root_relative_ade_m", 100),
        ("skate_cm_s", "foot_skate_cm_s", 1), ("penetration_cm", "foot_depth_mean_cm", 1),
        ("airborne_percent", "all_feet_above5cm_frame_fraction", 100),
        ("acceleration_error_m_s2", "accel_err_m_s2", 1))}


@torch.inference_mode()
def motion_scores(config, cases, folder, contract):
    from .validation import load_case, local_path
    from runtime_paths import resolve_legacy_path
    root = Path(__file__).resolve().parents[1]
    if str(root / "shape_aware_TMR") not in sys.path:
        sys.path.insert(0, str(root / "shape_aware_TMR"))
    from nymeria_eval import NymeriaTMREmbedder
    from nymeria_baselines.body23_tmr_adapter import KEEP, complete_body
    from nymeria_baselines.evaluate_reconstruction_tmr import fid, retrieval_details
    from decode_uniego_torch import decode_transforms
    paths = [local_path(resolve_legacy_path(contract[k])) for k in ("motion_stats_mean", "motion_stats_std")]
    from .settings import sha256
    if [sha256(p) for p in paths] != [contract["motion_mean"], contract["motion_std"]]:
        raise ValueError("Motion metric generator statistics differ")
    mean, std = [np.load(p).astype(np.float32) for p in paths]
    adapter = json.loads(Path(config["body_adapter"]).read_text())
    embedder = NymeriaTMREmbedder(config["tmr_checkpoint"], config["tmr_stats"], config["text_cache"], "cuda")
    metrics, positions, neutrals = [], {"pred": [], "gt": []}, {"pred": [], "gt": []}
    for row in cases:
        motion, neutral, _ = load_case(row, contract)
        with np.load(folder / "video2motion" / (row["id"] + ".npz")) as archive:
            generated = archive["generated"].copy()
            if int(archive["seed"]) != row["seed"] or generated.shape != motion.shape or not np.isfinite(generated).all():
                raise ValueError("Invalid V2M prediction")
        raw, gt = generated * std + mean, motion * std + mean
        metrics.append(physical(raw, gt))
        for name, value in (("pred", raw), ("gt", gt)):
            transforms = decode_transforms(torch.from_numpy(value)[None])[0].numpy()
            body = np.take(transforms, KEEP, axis=1)
            full, n = complete_body(body[..., :3, 3], body[..., :3, :3], neutral[KEEP], adapter)
            positions[name].append(full)
            neutrals[name].append(n)
    embeddings = {}
    for name in positions:
        values = []
        for start in range(0, len(cases), 32):
            joints = torch.from_numpy(np.stack(positions[name][start:start + 32]).astype(np.float32))
            neutral = torch.from_numpy(np.stack(neutrals[name][start:start + 32]).astype(np.float32))
            values.append(embedder.embed_motion(joints, neutral, lengths=[97] * len(joints), source_fps=20).cpu().numpy())
        embeddings[name] = np.concatenate(values)
    text = embedder.embed_text([row["caption"] for row in cases]).cpu().numpy()
    retrieval, _ = retrieval_details(embeddings["pred"], text, [row["id"] for row in cases])
    gt_retrieval, _ = retrieval_details(embeddings["gt"], text, [row["id"] for row in cases])
    result = {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}
    result.update(fid=fid(embeddings["pred"], embeddings["gt"]), **retrieval)
    result["matching_distance"] = float(np.linalg.norm(embeddings["pred"] - text, axis=-1).mean())
    distances = np.linalg.norm(embeddings["pred"][:, None] - embeddings["pred"][None], axis=-1)
    result["diversity"] = float(distances[np.triu_indices(len(cases), 1)].mean())
    atomic_json(folder / "v2m_per_case.json", metrics)
    atomic_json(folder / "v2m_gt_reference.json", gt_retrieval)
    return result


class VideoMetrics:
    def __init__(self, config):
        from native_phase_training.evaluate_inverse_forward import LPIPSAlex
        from dreamsim import dreamsim
        source = module(config["i3d_source"], "periodic_validation_i3d")
        preprocessing = module(config["i3d_preprocess_source"], "periodic_validation_i3d_preprocess")
        self.preprocess_i3d = preprocessing.preprocess_single
        import lpips
        expected_assets = (Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth",
                           Path(lpips.__file__).parent / "weights/v0.1/alex.pth")
        if any(Path(config[key]).resolve() != expected.resolve() for key, expected in
               zip(("lpips_alexnet", "lpips_linear"), expected_assets, strict=True)):
            raise ValueError("LPIPS would load assets outside the pinned metric configuration")
        self.lpips = LPIPSAlex("cuda", batch_size=32)
        self.dream, self.preprocess = dreamsim(pretrained=True, device="cuda", cache_dir=config["dreamsim_cache"],
            dreamsim_type="ensemble", normalize_embeds=True)
        self.dream.eval().requires_grad_(False)
        self.i3d = source.InceptionI3d(400, in_channels=3).cuda().eval().requires_grad_(False)
        self.i3d.load_state_dict(torch.load(config["i3d_weights"], map_location="cpu", weights_only=True), strict=True)

    @torch.inference_mode()
    def paired(self, gt, predicted):
        from native_phase_training.evaluate_inverse_forward import _frame_metrics
        from PIL import Image
        if gt.shape != predicted.shape or gt.shape != (97, 256, 256, 3) or gt.dtype != predicted.dtype or gt.dtype != np.uint8:
            raise ValueError("Invalid decoded validation RGB")
        gt, predicted = gt[1:], predicted[1:]
        values = _frame_metrics(gt, predicted, self.lpips)
        dream = []
        for start in range(0, len(gt), 32):
            a = torch.cat([self.preprocess(Image.fromarray(x)) for x in gt[start:start + 32]]).cuda()
            b = torch.cat([self.preprocess(Image.fromarray(x)) for x in predicted[start:start + 32]]).cuda()
            dream.append(self.dream(a, b).float().cpu().numpy().reshape(-1))
        values["dreamsim"] = np.concatenate(dream)
        features = []
        for rgb in (gt, predicted):
            video = self.preprocess_i3d(torch.from_numpy(rgb), 224) * 2
            chunks = torch.stack(list(torch.split(video, 16, dim=1))).cuda()
            feature = self.i3d(chunks).float().cpu().numpy()
            if feature.shape != (6, 400) or not np.isfinite(feature).all():
                raise FloatingPointError("Invalid I3D features")
            features.append(feature)
        if any(value.shape != (96,) or not np.isfinite(value).all() for value in values.values()):
            raise FloatingPointError("Invalid paired RGB metrics")
        return values, *features


@torch.inference_mode()
def load_scoring_vae(config):
    """Call on every training rank: the native loader synchronizes parameters."""
    from motion_expert_phase3_edge.visualize import load_vae
    return load_vae(Path(config["vae"]), "256", 97, "cuda")


@torch.inference_mode()
def video_scores(config, cases, folder, contract, policy, *, vae=None):
    from .validation import load_case
    if vae is None:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError("Distributed scoring requires the VAE initialized collectively on all ranks")
        vae = load_scoring_vae(config)
    if vae.get_output_range() != "[-1,1]":
        raise ValueError("Unexpected VAE output range")
    metrics = VideoMetrics(config)
    batch_size, offset = policy["decode_batch_size"], 0
    rows, gt_features, pred_features = [], [], []
    while offset < len(cases):
        selected = cases[offset:offset + batch_size]
        latents, ground_truth = [], []
        for row in selected:
            _, _, video = load_case(row, contract)
            with np.load(row["bundle"]) as bundle:
                ground_truth.append(bundle["rgb"].copy())
            with np.load(folder / "motimg2video" / (row["id"] + ".npz")) as archive:
                generated = archive["generated"].copy()
                if (generated.shape != video.shape or int(archive["seed"]) != row["seed"]
                        or not np.array_equal(generated[:, 0], video[:, 0]) or not np.isfinite(generated).all()):
                    raise ValueError("Invalid M2V prediction or changed conditioning prefix")
                latents.append(generated)
        try:
            pixels = vae.decode(torch.from_numpy(np.stack(latents)).cuda().float())
            if pixels.shape != (len(selected), 3, 97, 256, 256) or not torch.isfinite(pixels).all():
                raise FloatingPointError("Invalid VAE decode")
            decoded = ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 4, 1).cpu().numpy()
            del pixels
        except torch.cuda.OutOfMemoryError:
            pixels = None
            gc.collect()
            torch.cuda.empty_cache()
            if len(selected) == 1:
                raise
            batch_size = max(1, len(selected) // 2)
            continue
        for row, gt, pred in zip(selected, ground_truth, decoded, strict=True):
            values, a, b = metrics.paired(gt, pred)
            rows.append({"id": row["id"], **{k: float(v.mean()) for k, v in values.items()}})
            gt_features.append(a)
            pred_features.append(b)
        offset += len(selected)
        print(f"[phase3/validation] M2V metrics {offset}/{len(cases)}", flush=True)
    a, b = np.concatenate(gt_features), np.concatenate(pred_features)
    result = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "id"}
    result.update(fvd=frechet(a, b), gt_self_fvd=frechet(a, a), fvd_segments=len(a))
    if result["gt_self_fvd"] > 1e-4 or len(a) != len(cases) * 6:
        raise ValueError("FVD reference identity/coverage failed")
    atomic_json(folder / "m2v_per_case.json", rows)
    return result


def score(config, records, folder, contract, policy, *, vae=None):
    # Do not keep TMR and all video evaluators resident simultaneously.
    motion = motion_scores(config, records, folder, contract)
    gc.collect()
    torch.cuda.empty_cache()
    video = video_scores(config, records, folder, contract, policy, vae=vae)
    result = dict(v2m=motion, m2v=video)
    atomic_json(folder / "summary.json", result)
    return result
