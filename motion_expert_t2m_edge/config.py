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
MOTION_REPRESENTATION = os.environ.get(
    "EDGE_MOTION_REPRESENTATION", "camera_head_recanonicalization_v1"
)
if MOTION_REPRESENTATION not in {
    "camera_head_recanonicalization_v1",
    "legacy_uniego283",
}:
    raise ValueError(f"unsupported EDGE_MOTION_REPRESENTATION={MOTION_REPRESENTATION!r}")
LEGACY_MOTION_REPRESENTATION = MOTION_REPRESENTATION == "legacy_uniego283"
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
MOTION_SELF_ATTENTION_PER_GAP = int(
    os.environ.get("EDGE_MOTION_SELF_ATTENTION_PER_GAP", "0")
)
if MOTION_SELF_ATTENTION_PER_GAP not in (0, 1):
    raise ValueError("EDGE_MOTION_SELF_ATTENTION_PER_GAP must be 0 or 1")
MOTION_SELF_ATTENTION_AFTER_JOINT_LAYERS = (
    MOTION_LAYER_INDICES[:-1] if MOTION_SELF_ATTENTION_PER_GAP else ()
)
_motion_self_attention_before_joint = os.environ.get(
    "EDGE_MOTION_SELF_ATTENTION_BEFORE_JOINT", "0"
)
if _motion_self_attention_before_joint not in {"0", "1"}:
    raise ValueError("EDGE_MOTION_SELF_ATTENTION_BEFORE_JOINT must be 0 or 1")
MOTION_SELF_ATTENTION_BEFORE_JOINT = _motion_self_attention_before_joint == "1"
# Preserve the original Cosmos-3 Nano Phase-2 motion expert.  Only the
# backbone-facing residual/attention geometry changes for Edge; the fresh
# motion FFN remains the same three-linear SwiGLU block.
MOTION_MLP_TYPE = "motion_swiglu"
# Legacy Edge checkpoints rounded all I/O weights except time to BF16.
# Nano trained its complete motion heads and final norm in FP32.
MOTION_IO_DTYPE = os.environ.get("EDGE_MOTION_IO_DTYPE", "bfloat16")
if MOTION_IO_DTYPE not in {"bfloat16", "float32"}:
    raise ValueError(f"unsupported EDGE_MOTION_IO_DTYPE={MOTION_IO_DTYPE!r}")
# Independent opt-ins: preserve every existing architecture by default.
MOTION_NORM_DTYPE = os.environ.get("EDGE_MOTION_NORM_DTYPE", "bfloat16")
if MOTION_NORM_DTYPE not in {"bfloat16", "float32"}:
    raise ValueError(f"unsupported EDGE_MOTION_NORM_DTYPE={MOTION_NORM_DTYPE!r}")
_reasoner_k_norm_trainable = os.environ.get("EDGE_REASONER_K_NORM_TRAINABLE", "0")
if _reasoner_k_norm_trainable not in {"0", "1"}:
    raise ValueError("EDGE_REASONER_K_NORM_TRAINABLE must be 0 or 1")
REASONER_K_NORM_TRAINABLE = _reasoner_k_norm_trainable == "1"
_reasoner_k_norm_bypass = os.environ.get("EDGE_REASONER_K_NORM_BYPASS", "0")
if _reasoner_k_norm_bypass not in {"0", "1"}:
    raise ValueError("EDGE_REASONER_K_NORM_BYPASS must be 0 or 1")
REASONER_K_NORM_BYPASS = _reasoner_k_norm_bypass == "1"
if REASONER_K_NORM_BYPASS and REASONER_K_NORM_TRAINABLE:
    raise ValueError("as-is reasoner K/V and a trainable interface K norm are mutually exclusive")
REASONER_KV_PROJECTION = os.environ.get("EDGE_REASONER_KV_PROJECTION", "none")
if REASONER_KV_PROJECTION not in {"none", "linear_identity_v1"}:
    raise ValueError(f"unsupported EDGE_REASONER_KV_PROJECTION={REASONER_KV_PROJECTION!r}")
if REASONER_KV_PROJECTION != "none" and REASONER_K_NORM_BYPASS:
    raise ValueError("projected reasoner K/V requires the post-projection interface K norm")
_reasoner_v_norm_trainable = os.environ.get("EDGE_REASONER_V_NORM_TRAINABLE", "0")
if _reasoner_v_norm_trainable not in {"0", "1"}:
    raise ValueError("EDGE_REASONER_V_NORM_TRAINABLE must be 0 or 1")
REASONER_V_NORM_TRAINABLE = _reasoner_v_norm_trainable == "1"
if REASONER_V_NORM_TRAINABLE and (
    REASONER_KV_PROJECTION != "linear_identity_v1" or not REASONER_K_NORM_TRAINABLE
):
    raise ValueError("learned projected-V normalization requires Projection A with its learned K norm")
MOTION_ATTENTION_MODE = os.environ.get("EDGE_MOTION_ATTENTION_MODE", "joint")
CROSS_ATTENTION_K_NORM = os.environ.get("EDGE_CROSS_ATTENTION_K_NORM", "none")
CROSS_ATTENTION_V_NORM = os.environ.get("EDGE_CROSS_ATTENTION_V_NORM", "none")
_cross_attention_reasoner_rope = os.environ.get("EDGE_CROSS_ATTENTION_REASONER_ROPE", "1")
if _cross_attention_reasoner_rope not in {"0", "1"}:
    raise ValueError("EDGE_CROSS_ATTENTION_REASONER_ROPE must be 0 or 1")
CROSS_ATTENTION_REASONER_ROPE = _cross_attention_reasoner_rope == "1"


def validate_attention_configuration() -> None:
    if MOTION_SELF_ATTENTION_BEFORE_JOINT and (
        MOTION_ATTENTION_MODE != "joint" or MOTION_SELF_ATTENTION_PER_GAP
    ):
        raise ValueError("pre-joint motion self-attention requires joint mode without gap blocks")
    if CROSS_ATTENTION_K_NORM not in {"none", "rmsnorm"}:
        raise ValueError(f"unsupported EDGE_CROSS_ATTENTION_K_NORM={CROSS_ATTENTION_K_NORM!r}")
    if CROSS_ATTENTION_K_NORM != "none" and MOTION_ATTENTION_MODE != "self_cross_final_hidden":
        raise ValueError("cross-key RMSNorm is supported only for final-hidden self/cross attention")
    if CROSS_ATTENTION_V_NORM not in {"none", "rmsnorm"}:
        raise ValueError(f"unsupported EDGE_CROSS_ATTENTION_V_NORM={CROSS_ATTENTION_V_NORM!r}")
    if MOTION_ATTENTION_MODE != "self_cross_final_hidden" and (
        CROSS_ATTENTION_V_NORM != "none" or not CROSS_ATTENTION_REASONER_ROPE
    ):
        raise ValueError("cross-value RMSNorm and unrotated cross keys require final-hidden self/cross attention")
    if MOTION_ATTENTION_MODE not in {"joint", "self_cross_final_hidden", "self_cross_layer_kv"}:
        raise ValueError(f"unsupported EDGE_MOTION_ATTENTION_MODE={MOTION_ATTENTION_MODE!r}")
    if MOTION_ATTENTION_MODE != "joint":
        if REASONER_K_NORM_TRAINABLE or REASONER_KV_PROJECTION != "none":
            raise ValueError("self/cross ablations have no learned reasoner-interface norm or adapter")
        if MOTION_SELF_ATTENTION_PER_GAP:
            raise ValueError("self/cross blocks cannot be combined with extra joint-attention gap blocks")


def attention_mode_contract(mode: str | None = None) -> dict[str, Any]:
    mode = MOTION_ATTENTION_MODE if mode is None else mode
    joint = mode == "joint"
    return {
        "motion_attention_mode": mode,
        "motion_block_order": (
            ["self_attention", "joint_attention", "ffn"]
            if joint and MOTION_SELF_ATTENTION_BEFORE_JOINT
            else ["joint_attention", "ffn"] if joint
            else ["self_attention", "cross_attention", "ffn"]
        ),
        "cross_attention_memory": (
            None if joint else "final_post_norm_hidden" if mode == "self_cross_final_hidden"
            else "selected_layer_native_kv"
        ),
        "cross_attention_key_normalization": (
            None if joint else CROSS_ATTENTION_K_NORM if mode == "self_cross_final_hidden" else "frozen_reasoner_native"
        ),
        "cross_attention_value_normalization": (
            None if joint else CROSS_ATTENTION_V_NORM if mode == "self_cross_final_hidden" else "none"
        ),
        "cross_attention_reasoner_rope": (
            None if joint else CROSS_ATTENTION_REASONER_ROPE if mode == "self_cross_final_hidden" else True
        ),
        "cross_attention_position_encoding": (
            None if joint else "motion_native_mrope_reasoner_unrotated"
            if mode == "self_cross_final_hidden" and not CROSS_ATTENTION_REASONER_ROPE
            else "native_mrope_once"
        ),
    }


validate_attention_configuration()
TEXT_PACKING = os.environ.get("EDGE_TEXT_PACKING", "edge_native_generation_prefix")
if TEXT_PACKING not in {"edge_native_generation_prefix", "nano_caption_only"}:
    raise ValueError(f"unsupported EDGE_TEXT_PACKING={TEXT_PACKING!r}")
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
BONES_CAMERA_HEAD_EQUIVALENT = LEGACY_MOTION_REPRESENTATION
CAPTION_SUBJECT_POLICY = "standalone_C_to_sentence_aware_camera_wearer"
BONES_CAPTION_SUBJECT_POLICY = os.environ.get("EDGE_BONES_CAPTION_POLICY", "original")
if BONES_CAPTION_SUBJECT_POLICY not in {"original", "main_actor_camera_wearer_v1"}:
    raise ValueError(f"unsupported EDGE_BONES_CAPTION_POLICY={BONES_CAPTION_SUBJECT_POLICY!r}")
CONTRACT_SCHEMA_VERSION = 3

MOTION_STATS_MEAN = Path(
    os.environ.get(
        "MOTION_STATS_MEAN",
        REPO_ROOT / "motion_expert/stats/uniego283_mean.npy"
        if LEGACY_MOTION_REPRESENTATION
        else RUN_ROOT
        / "nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_mean.npy",
    )
)
MOTION_STATS_STD = Path(
    os.environ.get(
        "MOTION_STATS_STD",
        REPO_ROOT / "motion_expert/stats/uniego283_std.npy"
        if LEGACY_MOTION_REPRESENTATION
        else RUN_ROOT
        / "nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_std.npy",
    )
)
_motion_stats_summary = os.environ.get("MOTION_STATS_SUMMARY")
MOTION_STATS_SUMMARY = (
    Path(_motion_stats_summary)
    if _motion_stats_summary
    else None
    if LEGACY_MOTION_REPRESENTATION
    else RUN_ROOT / "nymeria_camera_head_recanonicalization_v1/stats/summary.json"
)
NYMERIA_UNIEGO_ROOT = Path(
    os.environ.get(
        "NYMERIA_UNIEGO_ROOT",
        WEKA_ROOT / "nymeriaplus_kimodo_proportional/uniego_rep"
        if LEGACY_MOTION_REPRESENTATION
        else WEKA_ROOT / "nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1",
    )
)
EDGE_VISION_WEIGHTS = EDGE_MODEL_ROOT / "vision_encoder/model.safetensors"
BONES_PAIRS_TRAIN = RUN_ROOT / "joint_attention/bones_pairs_train.jsonl"
BONES_PAIRS_VAL = RUN_ROOT / "joint_attention/bones_pairs_val.jsonl"

EXPECTED_STATS_SHA256 = (
    {
        "mean": "bd1d6bdc9a3b026fe1e5b28899441655ee36672c69c3e6e6389e9baff4b400d3",
        "std": "ee069e3aa9f3cd1a1e70135cc00bc751030f8045fae6bbfb7b4f5b32fa65f28c",
    }
    if LEGACY_MOTION_REPRESENTATION
    else {
        "mean": "4043893ec7ba2004f90dbc614a081e2f383b2798cdc4a39dce4d4ea6d47101c5",
        "std": "684e9b354f60f4cd1f8e251a1f8fcf3541d76887b2bcb9570773ce460dd69439",
    }
)
EXPECTED_STATS_SUMMARY_SHA256 = (
    None
    if LEGACY_MOTION_REPRESENTATION
    else "d594cb1a1509d8dcec76d9ed085b910bc6e2aff131b6522b2ee13caadda1b587"
)
EXPECTED_STATS_POPULATION = {} if LEGACY_MOTION_REPRESENTATION else {
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

# Yonsei's local conversion has different DCP serialization metadata. On
# 2026-09-08 all 254 frozen reasoner tensors matched the original pinned Edge
# safetensors at DCP precision. The stats summary differs only in site paths:
# replacing Yonsei paths with Grasp paths reproduces the Grasp summary hash.
# Model/vision weights, framework revision, normalization arrays, and the
# complete expected statistics population retain the shared pins above.
EXPECTED_YONSEI_METADATA_SHA256 = {
    "edge_dcp_metadata": "3b9312f86564fe79b43d2fbda9bba5757f8ff7c82a45eed37b574ced07b983e9",
    "motion_stats_summary": "98c5aaf5cda21e1f5379706f027d44010c71f1a6f4e719f1fe28201f110ac449",
}


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
        "uniego_root": Path(globals()["NYMERIA_UNIEGO_ROOT"]),
    }
    if MOTION_STATS_SUMMARY is not None:
        required["motion_stats_summary"] = MOTION_STATS_SUMMARY
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
        "edge_framework_commit": framework_commit(),
    }
    if "motion_stats_summary" in required:
        hashes["motion_stats_summary"] = sha256_file(required["motion_stats_summary"])
    else:
        hashes["motion_stats_summary"] = None
    expected = {
        "edge_model_index": EXPECTED_EDGE_MODEL_INDEX_SHA256,
        "edge_dcp_metadata": EXPECTED_EDGE_DCP_METADATA_SHA256,
        "edge_vision_weights": EXPECTED_EDGE_VISION_WEIGHTS_SHA256,
        "motion_mean": EXPECTED_STATS_SHA256["mean"],
        "motion_std": EXPECTED_STATS_SHA256["std"],
        "edge_framework_commit": EXPECTED_EDGE_FRAMEWORK_COMMIT,
    }
    if EXPECTED_STATS_SUMMARY_SHA256 is not None:
        expected["motion_stats_summary"] = EXPECTED_STATS_SUMMARY_SHA256
    artifact_site = os.environ.get("COSMOS_SITE")
    if artifact_site == "custom":
        # Restored archives retain the originating site's exact metadata bytes.
        # Select known pins explicitly; never rewrite provenance to match a host.
        artifact_site = os.environ.get("EDGE_ARTIFACT_SOURCE_SITE", "grasp")
        if artifact_site not in {"grasp", "yonsei"}:
            raise ValueError("EDGE_ARTIFACT_SOURCE_SITE must be grasp or yonsei")
    if artifact_site == "yonsei":
        expected.update(
            (key, value)
            for key, value in EXPECTED_YONSEI_METADATA_SHA256.items()
            if key in expected
        )
    drift = {key: (hashes[key], value) for key, value in expected.items() if hashes[key] != value}
    if drift:
        detail = ", ".join(f"{key}: actual={a} expected={e}" for key, (a, e) in drift.items())
        raise RuntimeError(f"Edge Phase-2 artifact drift: {detail}")
    if MOTION_STATS_SUMMARY is not None:
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
        "bones_caption_subject_policy": BONES_CAPTION_SUBJECT_POLICY,
        "ti2m_valid_frames": TI2M_FRAMES,
        "reasoner_image_size": REASONER_IMAGE_SIZE,
        "reasoner_image_conditioning": "frozen_edge_siglip2_visual_tokens",
        "generator_tokens": False,
        "attention_roles": (
            ["causal_reasoner", "motion_self", "full_motion"]
            if MOTION_ATTENTION_MODE == "joint" and MOTION_SELF_ATTENTION_BEFORE_JOINT
            else ["causal_reasoner", "full_motion"] if MOTION_ATTENTION_MODE == "joint"
            else ["causal_reasoner", "motion_self", "motion_cross"]
        ),
        **attention_mode_contract(),
        "model_family": MODEL_FAMILY,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "head_dim": HEAD_DIM,
        "motion_dim": MOTION_DIM,
        "motion_intermediate_size": MOTION_INTERMEDIATE_SIZE,
        "motion_layer_indices": list(MOTION_LAYER_INDICES),
        "motion_self_attention_per_gap": MOTION_SELF_ATTENTION_PER_GAP,
        "motion_self_attention_before_joint": MOTION_SELF_ATTENTION_BEFORE_JOINT,
        "motion_self_attention_after_joint_layers": list(
            MOTION_SELF_ATTENTION_AFTER_JOINT_LAYERS
        ),
        "motion_mlp_type": MOTION_MLP_TYPE,
        "motion_io_dtype": MOTION_IO_DTYPE,
        "motion_norm_dtype": MOTION_NORM_DTYPE,
        "reasoner_k_norm_trainable": REASONER_K_NORM_TRAINABLE,
        "reasoner_k_norm_bypass": REASONER_K_NORM_BYPASS,
        "reasoner_kv_projection": REASONER_KV_PROJECTION,
        "reasoner_v_norm_trainable": REASONER_V_NORM_TRAINABLE,
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
        "motion_stats_summary": (
            str(MOTION_STATS_SUMMARY) if MOTION_STATS_SUMMARY is not None else None
        ),
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
