"""Hydra experiment registration for native-compatible cached-latent Phase 1."""

from __future__ import annotations

import copy
import glob
import os
import sys

from hydra.core.config_store import ConfigStore
from torch.utils.data import DistributedSampler

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

ROOT = "/home/jungbin_cho/cosmos_motion_ft"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
NYMERIA_WORLD = os.path.join(ROOT, "nymeria_world")
if NYMERIA_WORLD not in sys.path:
    sys.path.insert(0, NYMERIA_WORLD)

from native_phase_training.latent_nymeria_dataset import (  # noqa: E402
    CyclingDataLoader,
    LatentAwareIterativeJointDataLoader,
    get_nymeria_camera_latent_sft_dataset,
)
from native_phase_training.checkpoint_eval_callback import NativeCheckpointEvalSubmitter  # noqa: E402
from native_phase_training.latent_omni_model import LatentOmniMoTModel  # noqa: E402
from nymeria_camera_rgb_dataset import MODE_WEIGHTS  # noqa: E402

_NUM_FRAMES = int(os.environ.get("NYMERIA_NUM_FRAMES", "97"))
_MODEL_RESOLUTION = os.environ.get("NYMERIA_RESOLUTION", "256").strip()
if not _MODEL_RESOLUTION:
    raise ValueError("NYMERIA_RESOLUTION must not be empty")
_MODE = os.environ.get("NYMERIA_MODE", "mixture")
_ALL_TASKS = frozenset({"forward_dynamics", "inverse_dynamics", "policy", "image2video"})
_DROPPED_MODES = frozenset(
    mode.strip() for mode in os.environ.get("NYMERIA_DROP_MODES", "").split(",") if mode.strip()
)
_UNKNOWN_DROPPED_MODES = _DROPPED_MODES - _ALL_TASKS
if _UNKNOWN_DROPPED_MODES:
    raise ValueError(f"unknown NYMERIA_DROP_MODES: {sorted(_UNKNOWN_DROPPED_MODES)}")
if not MODE_WEIGHTS:
    raise ValueError("NYMERIA_DROP_MODES removed every training task")
if _MODE != "mixture" and _MODE not in MODE_WEIGHTS:
    raise ValueError(f"NYMERIA_MODE={_MODE!r} is not active; active tasks={sorted(MODE_WEIGHTS)}")
_LATENT_ROOT = os.environ.get(
    "NYMERIA_LATENT_ROOT",
    f"/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T{_NUM_FRAMES}",
)
_QUALITY_FILTER_PATH = os.environ.get("NYMERIA_QUALITY_FILTER", "").strip()
_REPLACE_STANDALONE_C = os.environ.get("NYMERIA_REPLACE_STANDALONE_C", "0").lower() in {
    "1",
    "true",
    "yes",
}
_FULL_FT = bool(os.environ.get("NYMERIA_FULL_FT", ""))
_LORA_LR = float(os.environ.get("NATIVEP1_LORA_LR", "5.0e-05"))
_FULL_FT_LR = float(os.environ.get("NATIVEP1_FULL_FT_LR", "1.0e-04"))
_ACTION_LR_MULT = float(os.environ.get("NATIVEP1_ACTION_LR_MULT", "4.0"))
_ACTION_LOSS_WEIGHT = float(os.environ.get("NATIVEP1_ACTION_LOSS_WEIGHT", "10.0"))
if _ACTION_LOSS_WEIGHT < 0.0:
    raise ValueError("NATIVEP1_ACTION_LOSS_WEIGHT must be non-negative")
_NORMALIZE_LOSS_BY_ACTIVE = os.environ.get("NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE", "0").lower() in {
    "1",
    "true",
    "yes",
}
_ADAPTATION_MODE = os.environ.get("NATIVEP1_ADAPTATION_MODE", "global_lora").strip().lower()
_ADAPTATION_MODES = frozenset({"global_lora", "action_only", "camera_kv_lora"})
if _ADAPTATION_MODE not in _ADAPTATION_MODES:
    raise ValueError(
        f"NATIVEP1_ADAPTATION_MODE must be one of {sorted(_ADAPTATION_MODES)}, got {_ADAPTATION_MODE!r}"
    )
if _ADAPTATION_MODE in {"action_only", "camera_kv_lora"} and "image2video" in MODE_WEIGHTS:
    raise ValueError(
        f"{_ADAPTATION_MODE} has no trainable path for image2video. "
        "Set NYMERIA_DROP_MODES=image2video; I2V remains an evaluation-only frozen-prior check."
    )


def _parse_csv_ints(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default)
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list, got {raw!r}") from error
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _parse_optional_csv_floats(name: str) -> list[float] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated float list, got {raw!r}") from error


_PREFIX_LENGTHS = _parse_csv_ints("NATIVEP1_PREFIX_LENGTHS", "1")
_PREFIX_SAMPLING_WEIGHTS = _parse_optional_csv_floats("NATIVEP1_PREFIX_SAMPLING_WEIGHTS")
_EXPECTED_LATENT_HW_ENV = os.environ.get("NATIVEP1_EXPECTED_LATENT_HW", "").strip()
_EXPECTED_LATENT_HW = int(_EXPECTED_LATENT_HW_ENV) if _EXPECTED_LATENT_HW_ENV else None
_EXPECTED_IMAGE_HW_ENV = os.environ.get("NATIVEP1_EXPECTED_IMAGE_HW", "").strip()
_EXPECTED_IMAGE_HW = int(_EXPECTED_IMAGE_HW_ENV) if _EXPECTED_IMAGE_HW_ENV else None
if _EXPECTED_LATENT_HW is not None and _EXPECTED_LATENT_HW <= 0:
    raise ValueError("NATIVEP1_EXPECTED_LATENT_HW must be positive")
if _EXPECTED_IMAGE_HW is not None and _EXPECTED_IMAGE_HW <= 0:
    raise ValueError("NATIVEP1_EXPECTED_IMAGE_HW must be positive")
_REQUIRE_LATENT_CACHE_CONTRACT = os.environ.get(
    "NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT", "0"
).lower() in {"1", "true", "yes"}
_SHIFT_OVERRIDE_ENV = os.environ.get("NATIVEP1_SHIFT_OVERRIDE", "").strip()
_SHIFT_OVERRIDE = float(_SHIFT_OVERRIDE_ENV) if _SHIFT_OVERRIDE_ENV else None
if _SHIFT_OVERRIDE is not None and _SHIFT_OVERRIDE <= 0.0:
    raise ValueError("NATIVEP1_SHIFT_OVERRIDE must be positive")
_CLIPS_PER_GPU = int(os.environ.get("NATIVEP1_CLIPS_PER_GPU", "4"))
if _CLIPS_PER_GPU < 0:
    raise ValueError("NATIVEP1_CLIPS_PER_GPU must be >= 0 (use 0 for native token-budget packing)")
_USE_TOKEN_BUDGET_PACKING = _CLIPS_PER_GPU == 0
_MAX_SAMPLES_ENV = os.environ.get("NYMERIA_MAX_SAMPLES", "")
_MAX_SAMPLES = int(_MAX_SAMPLES_ENV) if _MAX_SAMPLES_ENV else None
_AUTO_EVAL = os.environ.get("NATIVEP1_AUTO_EVAL", "0").lower() in {"1", "true", "yes"}
_AUTO_EVAL_EVERY = int(os.environ.get("NATIVEP1_AUTO_EVAL_EVERY", "0"))
if _AUTO_EVAL_EVERY < 0:
    raise ValueError("NATIVEP1_AUTO_EVAL_EVERY must be non-negative")
_EVAL_INPUT_DIR = os.environ.get(
    "NATIVEP1_EVAL_INPUT_DIR",
    "/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_viz5_256_T97_v2",
)
_AUTO_EVAL_FULL71 = os.environ.get(
    "NATIVEP1_AUTO_EVAL_FULL71", "1" if _AUTO_EVAL else "0"
).lower() in {"1", "true", "yes"}
_AUTO_EVAL_FULL71_EVERY = int(
    os.environ.get("NATIVEP1_FULL71_EVAL_EVERY", str(_AUTO_EVAL_EVERY))
)
if _AUTO_EVAL_FULL71_EVERY < 0:
    raise ValueError("NATIVEP1_FULL71_EVAL_EVERY must be non-negative")
_FULL71_EVAL_INPUT_DIR = os.environ.get(
    "NATIVEP1_FULL71_EVAL_INPUT_DIR",
    "/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2",
)


def _latest_existing_dir(pattern: str) -> str | None:
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_text_tokenizer_path() -> str:
    explicit = os.environ.get("COSMOS_TEXT_TOKENIZER_PATH")
    if explicit:
        if not os.path.isfile(os.path.join(explicit, "tokenizer.json")):
            raise FileNotFoundError(f"COSMOS_TEXT_TOKENIZER_PATH has no tokenizer.json: {explicit}")
        return explicit

    cache = os.path.expanduser("~/.cache/huggingface/hub")
    cosmos_tok = _latest_existing_dir(os.path.join(cache, "models--nvidia--Cosmos3-Nano/snapshots/*/text_tokenizer"))
    if cosmos_tok and os.path.isfile(os.path.join(cosmos_tok, "tokenizer.json")):
        return cosmos_tok

    qwen_snap = _latest_existing_dir(os.path.join(cache, "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*"))
    if qwen_snap and os.path.isfile(os.path.join(qwen_snap, "tokenizer.json")):
        return qwen_snap

    if os.environ.get("ALLOW_HF_TOKENIZER_DOWNLOAD", "0") == "1":
        return "Qwen/Qwen3-VL-8B-Instruct"

    raise FileNotFoundError(
        "No local Cosmos/Qwen tokenizer snapshot found. Set COSMOS_TEXT_TOKENIZER_PATH "
        "or set ALLOW_HF_TOKENIZER_DOWNLOAD=1 explicitly."
    )


_MODEL_CONFIG = copy.deepcopy(NANO_MODEL_CONFIG)
_MODEL_CONFIG["resolution"] = _MODEL_RESOLUTION
_MODEL_CONFIG["tokenizer"]["vae_path"] = os.environ.get(
    "WAN_VAE_PATH",
    _MODEL_CONFIG["tokenizer"]["vae_path"],
)
_MODEL_CONFIG["vlm_config"]["tokenizer"]["pretrained_model_name"] = _resolve_text_tokenizer_path()
_MODEL_CONFIG["rectified_flow_training_config"]["action_loss_weight"] = _ACTION_LOSS_WEIGHT
_MODEL_CONFIG["rectified_flow_training_config"]["normalize_loss_by_active"] = _NORMALIZE_LOSS_BY_ACTIVE
if _SHIFT_OVERRIDE is not None:
    effective_shifts = dict(_MODEL_CONFIG["rectified_flow_training_config"]["shift"])
    if _MODEL_RESOLUTION not in effective_shifts:
        raise ValueError(
            f"cannot override shift for resolution {_MODEL_RESOLUTION!r}; "
            f"available={sorted(effective_shifts)}"
        )
    effective_shifts[_MODEL_RESOLUTION] = _SHIFT_OVERRIDE
    _MODEL_CONFIG["rectified_flow_training_config"]["shift"] = effective_shifts


def _task_stream(task: str):
    dataset = L(get_nymeria_camera_latent_sft_dataset)(
        num_frames=_NUM_FRAMES,
        fps=20.0,
        mode=task,
        latent_root=_LATENT_ROOT,
        quality_filter_path=_QUALITY_FILTER_PATH,
        replace_standalone_c=_REPLACE_STANDALONE_C,
        prefix_lengths=_PREFIX_LENGTHS,
        prefix_sampling_weights=_PREFIX_SAMPLING_WEIGHTS,
        prefix_seed=42,
        model_resolution_tier=_MODEL_RESOLUTION,
        expected_latent_hw=_EXPECTED_LATENT_HW,
        expected_image_hw=_EXPECTED_IMAGE_HW,
        require_latent_cache_contract=_REQUIRE_LATENT_CACHE_CONTRACT,
        split="train",
        max_action_dim="${model.config.max_action_dim}",
        cfg_dropout_rate=0.1,
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        max_samples=_MAX_SAMPLES,
    )
    return dict(
        ratio=int(round(MODE_WEIGHTS[task] * 100)),
        dataloader=L(CyclingDataLoader)(
            dataset=dataset,
            batch_size=1,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            in_order=False,
            sampler=L(DistributedSampler)(dataset=dataset, shuffle=True, seed=42, drop_last=True),
        ),
    )


_DATALOADERS = {t: _task_stream(t) for t in MODE_WEIGHTS} if _MODE == "mixture" else {_MODE: _task_stream(_MODE)}

world_camera_nymeria_latent_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(project="cosmos3_camera", group="camera_world", name="world_camera_nymeria_latent_nano", wandb_mode="disabled"),
        model=L(LatentOmniMoTModel)(
            config=copy.deepcopy(_MODEL_CONFIG),
            adaptation_mode=_ADAPTATION_MODE,
            _recursive_=False,
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=(
                ["moe_gen", "time_embedder", "vae2llm", "llm2vae", "action2llm", "llm2action", "action_modality_embed"]
                if _FULL_FT
                else (
                    ["action2llm", "llm2action", "action_modality_embed"]
                    if _ADAPTATION_MODE == "action_only"
                    else ["lora_A", "lora_B", "action2llm", "llm2action", "action_modality_embed"]
                )
            ),
            lr=(_FULL_FT_LR if _FULL_FT else _LORA_LR),
            lr_multipliers=(
                {}
                if _FULL_FT
                else {
                    "action2llm": _ACTION_LR_MULT,
                    "llm2action": _ACTION_LR_MULT,
                    "action_modality_embed": _ACTION_LR_MULT,
                }
            ),
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.4],
            verbosity_interval=0,
            warm_up_steps=[500],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                checkpoint_eval=L(NativeCheckpointEvalSubmitter)(
                    enabled=_AUTO_EVAL,
                    sbatch_script=os.path.join(ROOT, "native_phase_training", "sbatch_checkpoint_eval.sh"),
                    eval_input_dir=_EVAL_INPUT_DIR,
                    output_subdir="checkpoint_evals",
                    every_n_iterations=_AUTO_EVAL_EVERY,
                ),
                checkpoint_eval_full71=L(NativeCheckpointEvalSubmitter)(
                    enabled=_AUTO_EVAL_FULL71,
                    sbatch_script=os.path.join(
                        ROOT, "native_phase_training", "sbatch_checkpoint_eval_full71.sh"
                    ),
                    eval_input_dir=_FULL71_EVAL_INPUT_DIR,
                    output_subdir="eval_full71_inverse_forward",
                    every_n_iterations=_AUTO_EVAL_FULL71_EVERY,
                    required_input_files=("fd_input.jsonl", "invdyn_input.jsonl"),
                ),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(enabled=False, export_every_n=1, hf_repo_id=None, upload_to_object_store=dict(bucket="", credentials="", enabled=False)),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(LatentAwareIterativeJointDataLoader)(
            audio_sample_rate=48000,
            max_samples_per_batch=None if _USE_TOKEN_BUDGET_PACKING else _CLIPS_PER_GPU,
            max_sequence_length=(
                "${model.config.max_num_tokens_after_packing}" if _USE_TOKEN_BUDGET_PACKING else None
            ),
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            seed=42,
            dataloaders=_DATALOADERS,
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

world_camera_nymeria_latent_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [_NUM_FRAMES]
world_camera_nymeria_latent_nano["model"]["config"]["lora_enabled"] = (
    not _FULL_FT and _ADAPTATION_MODE != "action_only"
)
world_camera_nymeria_latent_nano["model"]["config"]["lora_rank"] = 16
world_camera_nymeria_latent_nano["model"]["config"]["lora_alpha"] = 32
world_camera_nymeria_latent_nano["model"]["config"]["lora_target_modules"] = (
    "k_proj_moe_gen,v_proj_moe_gen"
    if _ADAPTATION_MODE == "camera_kv_lora"
    else "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
)
world_camera_nymeria_latent_nano["model"]["config"]["lora_keep_trainable_modules"] = (
    "action2llm,llm2action,action_modality_embed"
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="world_camera_nymeria_latent_nano",
    node=world_camera_nymeria_latent_nano,
)
