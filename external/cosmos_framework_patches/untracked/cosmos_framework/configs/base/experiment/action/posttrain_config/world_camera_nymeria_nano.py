# SPDX-License-Identifier: OpenMDW-1.1
"""``world_camera_nymeria_nano`` — Cosmos3-Nano egocentric camera-action world-model SFT.

Camera-only Phase 2 on NymeriaPlus (ego-video + text + camera action), 4-task mixture:
  forward_dynamics (img+action[+text]->video), inverse_dynamics (video->camera, no text),
  policy (img->action+video[+text]), image2video (img[+text]->video).
Text dropped 10% (CFG null). Camera action = preprocessed UPRIGHT-RGB 9D pseudo-action.

Clones ``action_policy_droid_nano`` (FusedAdamW, LambdaLinear, reasoner frozen, gen+vae+action
heads trained, action heads re-init), swapping the DROID dataset for the NymeriaPlus camera dataset.

Usage (1 node, 8 GPU)::

    BASE_CHECKPOINT_PATH=/weka/jungbin/cosmos3_nano_dcp \\
    WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/world_camera_nymeria_repro.toml
"""
import copy
import os
import sys

# the NymeriaPlus dataset factory lives outside the package
sys.path.insert(0, "/home/jungbin_cho/cosmos_motion_ft/nymeria_world")

# window length (4N+1) — env-driven so the 33f and 97f runs share one config.
_NUM_FRAMES = int(os.environ.get("NYMERIA_NUM_FRAMES", "97"))
# task mode — "mixture" (4-task) or a fixed mode (for diagnostics).
_MODE = os.environ.get("NYMERIA_MODE", "mixture")
# NYMERIA_FULL_FT=1 -> full-parameter finetune of the GENERATOR pathway (no LoRA); else LoRA.
_FULL_FT = bool(os.environ.get("NYMERIA_FULL_FT", ""))

from torch.utils.data import DataLoader, DistributedSampler

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.joint_dataloader import (
    IterativeJointDataLoader,
)
from nymeria_camera_rgb_dataset import get_nymeria_camera_sft_dataset, MODE_WEIGHTS  # noqa: E402


def _task_stream(task: str):
    """One single-task stream for ``IterativeJointDataLoader``.

    ``IterativeJointDataLoader.__iter__`` picks ONE stream per step with a
    rank-independent RNG seeded by ``seed + global_id`` (joint_dataloader.py:558-560:
    ``rng = np.random.RandomState(self.seed + self.global_id); index_id = rng.choice(...)``).
    ``global_id`` is kept in lock-step across ranks by the trainer's
    ``dataloader_train.set_start_iteration(iteration * grad_accum_iter)`` call
    (joint_dataloader.py:495-496) plus the per-yield ``self.global_id += 1``, so
    every rank selects the SAME task at the SAME step -> identical loss/backward
    graph + collectives -> no multi-GPU NCCL desync. Within the chosen task, each
    rank draws a DIFFERENT, disjoint, shuffled data shard via the per-task
    ``DistributedSampler`` below (sharded across the full world). ``IterativeJointDataLoader``
    also does the token packing itself, so no outer ``PackingDataLoader`` is needed
    (and it injects ``custom_collate_fn`` at instantiate time -> we must NOT set
    ``collate_fn`` on the inner ``DataLoader``).

    Returns the ``{"dataloader", "ratio"}`` dict shape that ``JointDataLoader``
    expects (joint_dataloader.py:223-224).
    """
    _dataset = L(get_nymeria_camera_sft_dataset)(
        num_frames=_NUM_FRAMES, fps=20.0, mode=task, resolution="256", split="train",
        max_action_dim="${model.config.max_action_dim}", cfg_dropout_rate=0.1,
        tokenizer_config="${model.config.vlm_config.tokenizer}",
    )
    return dict(
        ratio=int(round(MODE_WEIGHTS[task] * 100)),
        dataloader=L(DataLoader)(
            dataset=_dataset,
            batch_size=1, num_workers=4, persistent_workers=True,
            pin_memory=True, prefetch_factor=4, in_order=False,
            # Disjoint per-rank shard of THIS task across the whole world, so the
            # synchronized task-per-step selection still feeds each rank a different
            # sample. shuffle=True so the shards are not in manifest order.
            sampler=L(DistributedSampler)(dataset=_dataset, shuffle=True, seed=42, drop_last=True),
        ),
    )


# Synchronized task-per-step: one stream per task (default). _MODE != "mixture" -> single fixed task.
if _MODE == "mixture":
    _DATALOADERS = {t: _task_stream(t) for t in MODE_WEIGHTS}
else:
    _DATALOADERS = {_MODE: _task_stream(_MODE)}

cs = ConfigStore.instance()

world_camera_nymeria_nano = LazyDict(
    dict(
        defaults=[
            # NOTE: native multi-GPU always uses FSDP2 fully_shard (dp_enabled is forced on when
            # world_size>1). With dp_shard=1 (set in TOML) fully_shard runs in PURE-REPLICATE mode:
            # NO param sharding — full 16B replicated per GPU, grad all-reduce = DDP-equivalent for LoRA.
            # (distributed_parallelism="ddp" would wrap torch-DDP on top of the FSDP mesh → crash.)
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
        job=dict(project="cosmos3", group="camera_world", name="world_camera_nymeria_nano",
                 wandb_mode="disabled"),
        model=dict(config=copy.deepcopy(NANO_MODEL_CONFIG)),  # action_gen + vision_gen, max_action_dim 64
        optimizer=dict(
            betas=[0.9, 0.99], eps=1.0e-08, fused=True,
            # NYMERIA_FULL_FT: full-parameter finetune of the GENERATOR pathway (_moe_gen) + gen/VAE/action
            # heads, reasoner frozen. Else: LoRA adapters + action heads only.
            keys_to_select=(
                ["moe_gen", "time_embedder", "vae2llm", "llm2vae",
                 "action2llm", "llm2action", "action_modality_embed"]
                if _FULL_FT else
                ["lora_A", "lora_B", "action2llm", "llm2action", "action_modality_embed"]
            ),
            lr=(1.0e-04 if _FULL_FT else 2.0e-04),  # full-ft: lower rate to protect the pretrained generator
            lr_multipliers={},
            optimizer_type="FusedAdam", weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear", cycle_lengths=[100], f_max=[0.4], f_min=[0.0],
            f_start=[0.4], verbosity_interval=0, warm_up_steps=[500],
        ),
        trainer=dict(
            distributed_parallelism="fsdp", grad_accum_iter=1, logging_iter=1, max_iter=100,
            max_val_iter=None, run_validation=False, run_validation_on_start=False,
            save_zero_checkpoint=False, seed=42, timeout_period=999999999, validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(every_n=200, log_memory_detail=True, save_s3=False,
                                    step_size=1, upload_every_n_mul=5),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False, dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True, keys_not_to_resume=[],
            # camera_pose is a PRETRAINED embodiment (domain 2) in Nano's action heads — the
            # zero-shot camera inference used them. So LOAD action2llm/llm2action/action_modality_embed
            # (don't re-init) and fine-tune them to our metric scale. Only skip net_ema.
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False, load_path="???", load_training_state=False,
            only_load_scheduler_state=False, save_iter=100, strict_resume=False, verbose=True,
            hf_export=dict(enabled=False, export_every_n=1, hf_repo_id=None,
                           upload_to_object_store=dict(bucket="", credentials="", enabled=False)),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        # Synchronized multi-task mixture: IterativeJointDataLoader selects ONE task per step with a
        # rank-independent RNG (seed+global_id, joint_dataloader.py:558-560) so EVERY rank runs the
        # SAME task each step (identical graph/collectives -> no NCCL desync), while each task's
        # per-rank DistributedSampler hands every rank a DIFFERENT disjoint shard. It also packs the
        # token budget itself, so no outer PackingDataLoader is needed.
        dataloader_train=L(IterativeJointDataLoader)(
            audio_sample_rate=48000,
            max_samples_per_batch=32, max_sequence_length=None, patch_spatial=2,
            sound_latent_fps=0, tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            seed=42,  # shared across ranks -> synchronized task-per-step selection
            dataloaders=_DATALOADERS,  # one stream per task -> synchronized task-per-step across ranks
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

# window length (4N+1) -> pin the VAE encode duration to match (env-driven: 33 or 97).
world_camera_nymeria_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [_NUM_FRAMES]

# LoRA on the generator attention (q/k/v/o_proj_moe_gen); reasoner + base gen stay frozen.
world_camera_nymeria_nano["model"]["config"]["lora_enabled"] = (not _FULL_FT)
world_camera_nymeria_nano["model"]["config"]["lora_rank"] = 16
world_camera_nymeria_nano["model"]["config"]["lora_alpha"] = 32
world_camera_nymeria_nano["model"]["config"]["lora_target_modules"] = (
    "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
)
# Full-train the PRETRAINED camera action heads alongside LoRA (load them, fine-tune to our
# metric scale). LoRA otherwise freezes them; this keeps them trainable.
world_camera_nymeria_nano["model"]["config"]["lora_keep_trainable_modules"] = (
    "action2llm,llm2action,action_modality_embed"
)

for _item in [world_camera_nymeria_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
