"""Explicit phase-3 recipe; no changes to specialist training defaults."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path


PHASE1_RUN = "edge_phase1_T97_20fps_4gpu_bs8_camera_wearer_lora_exclude5_yonsei_20260905"
PHASE2_RUN = (
    "edge_7layer_self_cross_final_hidden_knorm_captiononly_nymeria_bones50_t2m_ti2m_"
    "nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921"
)

BRIDGE_PARAMETERIZATIONS = ("shared", "separate_norm", "separate_norm_out", "mmdit")


def add_bridge_options(parser):
    from .runtime import add_execution_options
    add_execution_options(parser)
    parser.add_argument("--bridge-parameterization", choices=BRIDGE_PARAMETERIZATIONS,
                        help="Bridge parameter sharing; training defaults to shared, inference uses checkpoint")
    return parser


def configure_bridge(parameterization):
    if parameterization not in BRIDGE_PARAMETERIZATIONS:
        raise ValueError("Unknown bridge parameterization")
    # Legacy recipes omit this key. Preserve their semantic defaults.
    RECIPE.pop("bridge_parameterization", None)
    if parameterization != "shared":
        RECIPE["bridge_parameterization"] = parameterization


def configure_checkpoint_bridge(payload, *, requested=None):
    saved = payload.get("scheduler", {}).get("recipe", {}).get("bridge_parameterization", "shared")
    if requested is not None and requested != saved:
        raise ValueError("Requested bridge parameterization conflicts with checkpoint")
    configure_bridge(saved)


def task_probabilities(value=None):
    default = {"video2motion": 1/3, "motimg2video": 1/3, "joint": 1/3}
    if value is None:
        return default
    supplied = json.loads(value)
    if not isinstance(supplied, dict) or set(supplied) != set(default):
        raise ValueError("Joint task weights must specify video2motion, motimg2video and joint")
    weights = {key: float(supplied[key]) for key in default}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Joint task probabilities must be finite and nonnegative")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Joint task probabilities must sum to one")
    return {key: weight for key, weight in weights.items() if weight > 0}


def configure_conditioning(attention, clean_motion_timestep):
    if attention not in {"directional", "bidirectional"}:
        raise ValueError("Unknown bridge conditioning attention")
    if clean_motion_timestep not in {"omit", "zero"}:
        raise ValueError("Unknown clean motion timestep policy")
    RECIPE["conditioning_attention"] = attention
    RECIPE["clean_motion_timestep"] = clean_motion_timestep


def configure_checkpoint_conditioning(payload, *, attention=None, clean_motion_timestep=None):
    saved = payload.get("scheduler", {}).get("recipe", {})
    expected_attention = saved.get("conditioning_attention", "directional")
    expected_time = saved.get("clean_motion_timestep", "omit")
    if attention is not None and attention != expected_attention:
        raise ValueError("Requested conditioning attention conflicts with checkpoint")
    if clean_motion_timestep is not None and clean_motion_timestep != expected_time:
        raise ValueError("Requested clean motion timestep conflicts with checkpoint")
    configure_conditioning(expected_attention, expected_time)
    configure_checkpoint_bridge(payload)
    if "tasks" in saved:
        RECIPE["tasks"] = saved["tasks"].copy()


RECIPE = {
    "schema": 1, "frames": 97, "fps": 20.0, "latent_shape": [48, 25, 16, 16],
    "image_hw": [256, 256], "motion_dim": 283, "hidden": 2048,
    "bridge_heads": 16, "bridge_head_dim": 128,
    "bridge_layers": [3, 7, 11, 15, 19, 23, 27],
    "tasks": task_probabilities(os.environ.get("PHASE3_JOINT_TASK_WEIGHTS")),
    "joint_conditioning": "first_image_and_text",
    "joint_noise": "independent_specialist_schedules",
    "video_weight": 10.0, "video_normalize_by_active": False,
    "video_sample_level_loss_averaging": False, "video_time_weight": "uniform",
    "motion_weights": {"w_feat": 1.0, "w_joint": 1.0, "w_smooth": 5.0,
                       "w_contact": 0.05, "w_foot_vel": 1.0, "w_foot_height": 10.0},
    "joint_loss_mode": "centered", "contact_logit_scale": 2.0,
    "motion_loss_reduction": "per_task", "motion_decoded_nonfinite": "skip",
    "motion_noise": "logitnormal", "video_noise": "waver", "training_shift": 3.0,
    "motion_inference_shift": 3.0, "video_inference_shift": 10.0,
    "sampler": "unipc", "sampling_steps": 35, "guidance": 1.0,
    "cfg_dropout": 0.1, "bones_fraction": 0.0, "actor_exclusions": [],
    "global_batch": 32, "steps": 200000, "lr": 2e-4, "warmup": 1000,
    "lr_floor": 0.0, "betas": [0.9, 0.99], "weight_decay": 0.0,
    "clip_grad": 1.0, "seed": 0, "save_every": 10000,
}


def sample_task(rng):
    draw = rng.random()
    total = sum(RECIPE["tasks"].values())
    cumulative = 0.0
    for name, weight in RECIPE["tasks"].items():
        cumulative += weight / total
        if draw < cumulative:
            return name
    return next(reversed(RECIPE["tasks"]))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_compute_node():
    from native_phase_training.compute_access import require_compute_access
    require_compute_access()


def add_paths(parser: argparse.ArgumentParser):
    add_bridge_options(parser)
    parser.add_argument("--conditioning-attention", choices=["directional", "bidirectional"],
                        default=os.environ.get("PHASE3_CONDITIONING_ATTENTION", "directional"))
    parser.add_argument("--clean-motion-timestep", choices=["omit", "zero"],
                        default=os.environ.get("PHASE3_CLEAN_MOTION_TIMESTEP", "omit"))
    weka = Path(os.environ.get("WEKA_ROOT", "/lustre/whwjdqls99/cosmos/weka"))
    runs = Path(os.environ.get("RUN_ROOT", weka / "cosmos_motion_ft_runs"))
    scratch = (Path("/scratch2/whwjdqls99/cosmos") if os.environ.get("COSMOS_SITE") == "yonsei"
               else runs)
    phase2_runs = scratch / "cosmos_motion_ft_runs" if os.environ.get("COSMOS_SITE") == "yonsei" else runs
    nymeria = weka / "nymeriaplus_kimodo_proportional"
    parser.add_argument("--phase1", type=Path, default=os.environ.get("EDGE_PHASE3_PHASE1",
        runs / "cosmos3_camera/camera_world_edge" / PHASE1_RUN / "checkpoints/iter_000200000"))
    parser.add_argument("--phase2", type=Path, default=os.environ.get("EDGE_PHASE3_PHASE2",
        phase2_runs / "motion_expert_t2m_edge" / PHASE2_RUN / "checkpoints/step_000200000.pt"))
    parser.add_argument("--latent-root", type=Path, default=nymeria / "joint_latents_T97_256tier_256")
    parser.add_argument("--cache", type=Path, default=scratch / "caches/phase3_edge_camhead_T97_v1")
    parser.add_argument("--manifest", type=Path, default=nymeria / "video/manifest_video.jsonl")
    parser.add_argument("--split-file", type=Path, default=nymeria / "train_test_split.json")
    parser.add_argument("--floor-calibration", type=Path, default=nymeria / "metadata/floor_calibration.json")
    parser.add_argument("--out", type=Path, default=phase2_runs / "motion_expert_phase3_edge" /
                        "edge_p3_p1ema200k_crosscaption200k_bridge7_joint3_equal_T97_gbs32_wv10_yonsei_20261002")
    return parser
