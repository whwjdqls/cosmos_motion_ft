"""Pinned contract for Cosmos-3 Edge Phase-2 T2M + TI2M pretraining."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))
WEKA_ROOT = Path(os.environ.get("WEKA_ROOT", "/mnt/projects/ll/jungbinc/weka"))
RUN_ROOT = Path(os.environ.get("RUN_ROOT", WEKA_ROOT / "cosmos_motion_ft_runs"))
EDGE_FRAMEWORK_ROOT = Path(
    os.environ.get("COSMOS_FRAMEWORK_EDGE_ROOT", "/mnt/projects/ll/jungbinc/cosmos-framework-edge")
)
EDGE_MODEL_ROOT = Path(os.environ.get("EDGE_MODEL_ROOT", WEKA_ROOT / "Cosmos3-Edge"))
EDGE_DCP_ROOT = Path(os.environ.get("BASE_CHECKPOINT_PATH", WEKA_ROOT / "cosmos3_edge_dcp"))

MODEL_FAMILY = "cosmos3_edge_nemotron_2b_dense_vl"
MOTION_REPRESENTATION = "camera_head_recanonicalization_v1"
HIDDEN_SIZE = 2048
NUM_LAYERS = 28
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
MOTION_DIM = 283
NUM_JOINTS = 30
MOTION_INTERMEDIATE_SIZE = 3072
MOTION_LAYER_STRIDE = 4
MOTION_LAYER_INDICES = tuple(
    i for i in range(NUM_LAYERS) if (i + 1) % MOTION_LAYER_STRIDE == 0
)
# Preserve the original Cosmos-3 Nano Phase-2 motion expert.  Only the
# backbone-facing residual/attention geometry changes for Edge; the fresh
# motion FFN remains the same three-linear SwiGLU block.
MOTION_MLP_TYPE = "motion_swiglu"
TEXT_PACKING = "edge_native_generation_prefix"
MOTION_MROPE = "cosmos3d"
TIMESTEP_CONVENTION = "normalized_sigma"
FPS = 20.0
DEFAULT_T = 200
TI2M_FRAMES = 97
REASONER_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 128
DEFAULT_GRAD_ACCUM = 1
TASK_WEIGHTS = {"text2motion": 0.75, "textimg2motion": 0.25}
BONES_TEXT2M_FRAC = 0.0
BONES_REPRESENTATION = "legacy_uniego283_motion_only"
BONES_CAMERA_HEAD_EQUIVALENT = False
CAPTION_SUBJECT_POLICY = "standalone_C_to_sentence_aware_camera_wearer"
CONTRACT_SCHEMA_VERSION = 3

MOTION_STATS_MEAN = Path(
    os.environ.get(
        "MOTION_STATS_MEAN",
        RUN_ROOT
        / "nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_mean.npy",
    )
)
MOTION_STATS_STD = Path(
    os.environ.get(
        "MOTION_STATS_STD",
        RUN_ROOT
        / "nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_std.npy",
    )
)
MOTION_STATS_SUMMARY = Path(
    os.environ.get(
        "MOTION_STATS_SUMMARY",
        RUN_ROOT / "nymeria_camera_head_recanonicalization_v1/stats/summary.json",
    )
)
NYMERIA_UNIEGO_ROOT = Path(
    os.environ.get(
        "NYMERIA_UNIEGO_ROOT",
        WEKA_ROOT / "nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1",
    )
)
EDGE_VISION_WEIGHTS = EDGE_MODEL_ROOT / "vision_encoder/model.safetensors"
BONES_PAIRS_TRAIN = RUN_ROOT / "joint_attention/bones_pairs_train.jsonl"
BONES_PAIRS_VAL = RUN_ROOT / "joint_attention/bones_pairs_val.jsonl"

EXPECTED_STATS_SHA256 = {
    "mean": "4043893ec7ba2004f90dbc614a081e2f383b2798cdc4a39dce4d4ea6d47101c5",
    "std": "684e9b354f60f4cd1f8e251a1f8fcf3541d76887b2bcb9570773ce460dd69439",
}
EXPECTED_STATS_SUMMARY_SHA256 = (
    "d594cb1a1509d8dcec76d9ed085b910bc6e2aff131b6522b2ee13caadda1b587"
)
EXPECTED_STATS_POPULATION = {
    "split": "train",
    "raw_usable_captioned_windows": 128_102,
    "dropped_windows": 7_173,
    "kept_windows_after_floor_filter": 120_929,
    "runtime_guard_rejected": 0,
    "stats_windows": 120_929,
    "frames": 11_888_119,
}
EXPECTED_EDGE_DCP_METADATA_SHA256 = (
    "f0d19c6bdbe43663e3a1a6fcb3437a5718d92565fd5db2fdc1ec499bc9d5e1ec"
)
EXPECTED_EDGE_MODEL_INDEX_SHA256 = (
    "ee48f9da9fbab206b6d2902eb109a842dde7b1347f5716a177f0be00968acf33"
)
EXPECTED_EDGE_VISION_WEIGHTS_SHA256 = (
    "2180ad739ecc96b5c1e9386892d3c5c08bfa42b9cdab9aabc53b028671db89b3"
)
EXPECTED_EDGE_FRAMEWORK_COMMIT = "d4599e2e43fbd06168e9884205b9b66c3902d8f6"


def _load_shared_config() -> Any:
    """Load the existing data-only constants under a private module name.

    `nymeria_joint_dataset.py` historically imports a top-level module named
    `config`.  The Edge run puts this directory first on PYTHONPATH, so re-export
    the old data constants below while keeping all Edge geometry under distinct
    names.  No Nano model code is imported.
    """

    path = REPO_ROOT / "motion_expert_joint_attention/config.py"
    spec = importlib.util.spec_from_file_location("_shared_motion_data_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared motion data config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SHARED = _load_shared_config()
for _name in dir(_SHARED):
    if _name.isupper() and _name not in globals():
        globals()[_name] = getattr(_SHARED, _name)

# Force the new representation-specific paths even when this module is imported
# by the historical dataset implementation.
SHARED_MEAN = str(MOTION_STATS_MEAN)
SHARED_STD = str(MOTION_STATS_STD)
globals()["MOTION_STATS_MEAN"] = str(MOTION_STATS_MEAN)
globals()["MOTION_STATS_STD"] = str(MOTION_STATS_STD)
globals()["NYMERIA_UNIEGO_ROOT"] = str(NYMERIA_UNIEGO_ROOT)
RUNS_ROOT = str(RUN_ROOT)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def framework_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(EDGE_FRAMEWORK_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


@lru_cache(maxsize=2)
def validate_artifacts(*, include_bones: bool = False) -> dict[str, str]:
    required = {
        "edge_model_index": EDGE_MODEL_ROOT / "model.safetensors.index.json",
        "edge_processor": EDGE_MODEL_ROOT / "preprocessor_config.json",
        "edge_tokenizer": EDGE_MODEL_ROOT / "tokenizer.json",
        "edge_vision_weights": EDGE_VISION_WEIGHTS,
        "edge_dcp_metadata": EDGE_DCP_ROOT / "model/.metadata",
        "motion_mean": Path(globals()["MOTION_STATS_MEAN"]),
        "motion_std": Path(globals()["MOTION_STATS_STD"]),
        "motion_stats_summary": MOTION_STATS_SUMMARY,
        "uniego_root": Path(globals()["NYMERIA_UNIEGO_ROOT"]),
    }
    if include_bones:
        required.update(
            bones_pairs_train=BONES_PAIRS_TRAIN,
            bones_pairs_val=BONES_PAIRS_VAL,
        )
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Edge Phase-2 artifacts: " + ", ".join(missing))

    hashes = {
        "edge_model_index": sha256_file(required["edge_model_index"]),
        "edge_dcp_metadata": sha256_file(required["edge_dcp_metadata"]),
        "edge_vision_weights": sha256_file(required["edge_vision_weights"]),
        "motion_mean": sha256_file(required["motion_mean"]),
        "motion_std": sha256_file(required["motion_std"]),
        "motion_stats_summary": sha256_file(required["motion_stats_summary"]),
        "edge_framework_commit": framework_commit(),
    }
    expected = {
        "edge_model_index": EXPECTED_EDGE_MODEL_INDEX_SHA256,
        "edge_dcp_metadata": EXPECTED_EDGE_DCP_METADATA_SHA256,
        "edge_vision_weights": EXPECTED_EDGE_VISION_WEIGHTS_SHA256,
        "motion_mean": EXPECTED_STATS_SHA256["mean"],
        "motion_std": EXPECTED_STATS_SHA256["std"],
        "motion_stats_summary": EXPECTED_STATS_SUMMARY_SHA256,
        "edge_framework_commit": EXPECTED_EDGE_FRAMEWORK_COMMIT,
    }
    drift = {key: (hashes[key], value) for key, value in expected.items() if hashes[key] != value}
    if drift:
        detail = ", ".join(f"{key}: actual={a} expected={e}" for key, (a, e) in drift.items())
        raise RuntimeError(f"Edge Phase-2 artifact drift: {detail}")
    summary = json.loads(MOTION_STATS_SUMMARY.read_text())
    population_drift = {
        key: (summary.get(key), expected_value)
        for key, expected_value in EXPECTED_STATS_POPULATION.items()
        if summary.get(key) != expected_value
    }
    if population_drift:
        detail = ", ".join(
            f"{key}: actual={actual!r} expected={expected!r}"
            for key, (actual, expected) in population_drift.items()
        )
        raise RuntimeError(f"Nymeria motion-stat population drift: {detail}")
    return hashes


def architecture_contract() -> dict[str, Any]:
    hashes = validate_artifacts()
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "phase": 2,
        "tasks": ["text2motion", "textimg2motion"],
        "default_task_weights": dict(TASK_WEIGHTS),
        "default_bones_text2motion_frac": BONES_TEXT2M_FRAC,
        "caption_subject_policy": CAPTION_SUBJECT_POLICY,
        "ti2m_valid_frames": TI2M_FRAMES,
        "reasoner_image_size": REASONER_IMAGE_SIZE,
        "reasoner_image_conditioning": "frozen_edge_siglip2_visual_tokens",
        "generator_tokens": False,
        "attention_roles": ["causal_reasoner", "full_motion"],
        "model_family": MODEL_FAMILY,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "head_dim": HEAD_DIM,
        "motion_dim": MOTION_DIM,
        "motion_intermediate_size": MOTION_INTERMEDIATE_SIZE,
        "motion_layer_indices": list(MOTION_LAYER_INDICES),
        "motion_mlp_type": MOTION_MLP_TYPE,
        "text_packing": TEXT_PACKING,
        "motion_mrope": MOTION_MROPE,
        "timestep_convention": TIMESTEP_CONVENTION,
        "motion_representation": MOTION_REPRESENTATION,
        "bones_representation": BONES_REPRESENTATION,
        "bones_camera_head_equivalent": BONES_CAMERA_HEAD_EQUIVALENT,
        "bones_pairs_train": str(BONES_PAIRS_TRAIN),
        "bones_pairs_val": str(BONES_PAIRS_VAL),
        "nymeria_uniego_root": str(globals()["NYMERIA_UNIEGO_ROOT"]),
        "motion_stats_mean": str(globals()["MOTION_STATS_MEAN"]),
        "motion_stats_std": str(globals()["MOTION_STATS_STD"]),
        "motion_stats_summary": str(MOTION_STATS_SUMMARY),
        "motion_stats_population": dict(EXPECTED_STATS_POPULATION),
        "base_dcp_root": str(EDGE_DCP_ROOT),
        "edge_model_root": str(EDGE_MODEL_ROOT),
        **hashes,
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "architecture_contract",
    "framework_commit",
    "sha256_file",
    "validate_artifacts",
]
