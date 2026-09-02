"""Train the multimodal joint-attention/bridge model on Cosmos-3 Nano (cosmos env).

This generalizes the original text->motion trainer to seven base tasks plus opt-in Phase-3
joint-target tasks (text / image / video / camera / motion in ONE packed sequence). The per-task
contract lives in ``task_plan.py``; the data seam in ``nymeria_joint_dataset.py``; the
per-modality rectified-flow helpers in ``flow.py``; the gen I/O adapter in ``gen_heads.py``;
and the packed forward (real ``gen_idx`` + per-modality encode/decode + dict output) in
``joint_motion_model.py``. THIS file drives the loop:

  per batch -> read ``mode`` + every present modality -> per sample noise each NOISED modality
  with its PER-MODALITY objective (motion: ``flow.add_noise_x0_masked`` by default -- logit-
  normal sigma, target = x0; ``--objective velocity`` selects the old velocity noiser for
  ablation. vision/camera: ALWAYS ``flow.add_noise_velocity_masked`` -- velocity target
  ``eps - x0`` with checkpointed legacy-uniform or native shifted-Waver time; clean condition frames
  pass through untouched either way) -> call ``model.forward(modes=..., x_t=noised_motion,
  video_latents=noised_latents, camera_action=noised_action, ...)`` -> the model encodes via
  gen_heads + motion_heads, runs the shared joint attention, decodes each SUPERVISED modality
  -> apply the per-task flow losses (motion: feat + decoded joint + smooth on x0_hat; vision:
  latent flow MSE; camera: flow MSE chan[:9] x10) and SUM only the supervised modalities ->
  backward -> log per-modality + per-task scalars.

The seven base tasks:
  inverse_dynamics  video                 -> camera     (no text)
  forward_dynamics  camera + text + image -> video
  policy            text + image          -> camera + video
  text2motion       text                  -> motion     (existing trained path)
  textimg2motion    text + image          -> motion
  motimg2video      motion + text + image -> video
  video2motion      video                 -> motion     (no text)

Optional Phase-3 bridge tasks (zero default weight):
  video2camera_motion  video              -> camera + motion
  camimg2video_motion  camera + image     -> video + motion

TRAIN SCOPE (DESIGN_7TASK.md section 5; all toggles, defaults reproduce text->motion):
  motion is ALWAYS fully trained (_moe_motion + MotionHeads + norm_moe_motion).
  reasoner: --reasoner_lora (else frozen).  generator: --gen_lora | --gen_full (else frozen);
  the FIRST run uses --gen_lora per the confirmed plan. model.freeze() routes requires_grad and
  --smoke asserts finite loss + grad ONLY on the active-trainable set (zero grad on frozen ones).

Run (cosmos env, 1 GPU debug):
  bash run.sh train.py --smoke
  bash run.sh train.py --gen_lora --batch_size 16 --steps 200000 --out joint7_v1
Multi-GPU (PURE DDP by default -- trainable REPLICATED, optimizer state replicated -> resumable):
  torchrun --standalone --nproc_per_node=8 train.py --ddp --gen_lora --out joint7_v1
Multi-GPU (FSDP opt-in -- shard the trainable across ranks; optimizer state NOT checkpointed):
  torchrun --standalone --nproc_per_node=8 train.py --ddp --fsdp --gen_lora --out joint7_v1
Resume a run (weights + optimizer + step; LR schedule continues at the resumed step):
  torchrun --standalone --nproc_per_node=8 train.py --ddp --gen_lora --out joint7_v1 --resume auto
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import flow
import task_plan as TP
from checkpoint_utils import load_gen_init_state, load_joint_pt
from cosmos_loader import FrozenCosmos
from decode_uniego_torch import decode_joints
from head_camera_alignment import (
    DEFAULT_CALIBRATION,
    head_camera_alignment_losses,
    head_camera_errors,
)
from joint_motion_model import JointMotionModel
from motion_losses import contact_aware_losses
from nymeria_joint_dataset import NymeriaJointDataset, collate_joint
from uniego_layout import FEAT_DIM

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = config.RUNS_ROOT
D_REASONER = config.HIDDEN


# --------------------------------------------------------------------------------------
# DDP helpers
# --------------------------------------------------------------------------------------
def _ddp_env():
    """Return (rank, world, local_rank) reading torchrun env (defaults to single proc)."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local


def _is_rank0(rank):
    return rank == 0


# --------------------------------------------------------------------------------------
# loss utils
# --------------------------------------------------------------------------------------
def masked_mse(a, b, valid):
    """a,b [...,D] or [...,J,3]; valid [B,T] True=keep. Broadcast valid over trailing dims."""
    while valid.dim() < a.dim():
        valid = valid.unsqueeze(-1)
    se = ((a - b) ** 2) * valid
    return se.sum() / valid.expand_as(a).sum().clamp(min=1)


def lr_factor(step, warmup, total, schedule):
    """Linear warmup to 1 then cosine decay to 0 (reuse of train_motion_ft.lr_factor)."""
    if schedule == "constant":
        return 1.0
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    prog = float(step - warmup) / float(max(1, total - warmup))
    prog = min(max(prog, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def clip_grads(params, max_norm):
    """Grad clip that tolerates FSDP (DTensor + plain Tensor can't mix in one foreach
    clip_grad_norm_ call). Clip the two groups separately. Mirrors train_motion_ft.clip_grads."""
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = ()
    dtensor_params, plain_params = [], []
    for p in params:
        if p.grad is None:
            continue
        (dtensor_params if isinstance(p.grad, DTensor) else plain_params).append(p)
    if dtensor_params:
        torch.nn.utils.clip_grad_norm_(dtensor_params, max_norm)
    if plain_params:
        torch.nn.utils.clip_grad_norm_(plain_params, max_norm)


def grads_finite(params):
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(g, DTensor):
                g = g.to_local()
        except Exception:
            pass
        if not torch.isfinite(g).all():
            return False
    return True


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # data / mixture
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="subset of task_plan.TASKS to train (default: configured positive-weight "
                         "base tasks). Tasks not listed get zero mixture weight; experimental "
                         "joint-target tasks default to zero.")
    ap.add_argument("--task_weights", default=None,
                    help="JSON dict of {mode: weight} overriding config.TASK_WEIGHTS (relative).")
    ap.add_argument("--bones_frac", type=float, default=0.5,
                    help="fraction of text2motion mass routed to the BONES motion-only stream.")
    ap.add_argument("--precomputed_latents", action="store_true", default=True,
                    help="load precomputed Wan-VAE video latents (else decode raw frames; heavy).")
    ap.add_argument("--no_precomputed_latents", dest="precomputed_latents", action="store_false")
    ap.add_argument("--force_on_the_fly", action="store_true",
                    help="NEVER read the precomputed latent cache; VAE-encode every video window "
                         "live in the training step. Auto-fallback (use valid cache, else encode "
                         "live) happens WITHOUT this flag whenever the cache is missing or its "
                         "T_lat mismatches the current --T; this just forces the live path.")
    ap.add_argument("--wan_vae_path",
                    default=os.environ.get("WAN_VAE_PATH",
                                           "/weka/jungbin/wan22_vae/Wan2.2_VAE.pth"),
                    help="Wan2.2-VAE checkpoint for ON-THE-FLY latent encoding (loaded once, "
                         "frozen/bf16/eval on the training GPU when a video task is enabled).")
    ap.add_argument("--vae_resolution", default="256",
                    help="resolution bucket for the on-the-fly VAE resize/pad pipeline (256 = "
                         "the precompute default; must match the precomputed cache's bucket).")
    ap.add_argument("--objective", choices=["velocity", "x0"], default="x0",
                    help="flow objective for the MOTION pathway ONLY (per-modality design): "
                         "'x0' (DEFAULT -- the proven motion_expert/bs_train.py recipe: "
                         "logit-normal sigma, net predicts clean x0, MSE vs x0, geometric "
                         "losses on the prediction directly) or 'velocity' (kept selectable "
                         "for backward compat / ablation of older velocity-motion runs). "
                         "VISION/CAMERA targets are ALWAYS velocity (uniform t, target = "
                         "eps - x0) regardless of this flag -- that matches the pretrained "
                         "Cosmos generator's native rectified-flow objective and never changes.")
    ap.add_argument("--motion_schedule", choices=["legacy", "native"], default="legacy",
                    help="noise-time schedule for x0 MOTION targets. legacy keeps the historical "
                         "unshifted logit-normal training sigma and linear 1->0 DDIM ladder. "
                         "native uses Cosmos's shifted logit-normal sigma and shifted 1000-step "
                         "inference ladder. Explicitly select native for new aligned Phase-2 runs; "
                         "legacy remains the default so old launchers/checkpoints do not drift.")
    ap.add_argument("--motion_shift", type=float, default=flow.NATIVE_MOTION_SHIFT,
                    help="rational flow shift for --motion_schedule native (default: 3).")
    ap.add_argument("--motion_num_train_timesteps", type=int,
                    default=flow.NATIVE_NUM_TRAIN_TIMESTEPS,
                    help="native motion scheduler timestep scale/quantization range (default: 1000).")
    ap.add_argument("--motion_native_solver", choices=["euler", "unipc"], default="unipc",
                    help="inference/viz solver for native-schedule x0 checkpoints. unipc converts "
                         "x0 predictions to velocity and runs NVIDIA's official Phase-1 UniPC "
                         "solver (default); euler remains available for historical comparisons.")
    ap.add_argument("--gen_schedule", choices=["legacy", "native"], default="legacy",
                    help="noise-time schedule for VIDEO/CAMERA velocity targets. legacy is the "
                         "historical uniform-time training plus linear Euler sampler. native "
                         "uses Cosmos-3 Waver training time, resolution shift, and the official "
                         "UniPC inference contract. New Phase-3 bridge runs from native Phase 1 "
                         "must select native; legacy remains the old-checkpoint default.")
    ap.add_argument("--gen_shift", type=float, default=flow.NATIVE_MOTION_SHIFT,
                    help="native Cosmos video resolution shift (256px Phase 1 uses 3).")
    ap.add_argument("--gen_num_train_timesteps", type=int,
                    default=flow.NATIVE_NUM_TRAIN_TIMESTEPS,
                    help="native generator scheduler timestep scale (default: 1000).")
    ap.add_argument("--gen_native_solver", choices=["unipc"], default="unipc",
                    help="native generator inference solver; NVIDIA Cosmos-3 uses UniPC.")
    ap.add_argument("--gen_packing", choices=["legacy", "native"], default="legacy",
                    help="generator reasoner/mRoPE packing. native uses [BOS,text,EOS,SOG], "
                         "FPS-modulated float 3D-mRoPE, and the Cosmos temporal modality margin; "
                         "legacy preserves historical joint checkpoints.")
    ap.add_argument("--gen_fps", type=float, default=20.0,
                    help="source FPS for native generator 3D-mRoPE modulation.")
    ap.add_argument("--gen_temporal_margin", type=float, default=15000.0,
                    help="native Cosmos text-to-generator temporal mRoPE margin.")
    ap.add_argument("--motion_intermediate", type=int, default=config.MOTION_INTERMEDIATE_SIZE,
                    help="motion-expert FFN width (the only size knob; smaller=lighter expert). "
                         "The motion expert is always randomly initialized, never from the generator.")
    ap.add_argument("--motion_layer_stride", type=int, default=config.MOTION_LAYER_STRIDE,
                    help="SPARSE-DEPTH: the 3-way joint attention fires every Nth backbone layer "
                         "(stride=3 -> 12 motion blocks; stride=6 -> 6). The frozen reasoner+"
                         "generator still run all layers.")
    ap.add_argument("--motion_mrope", choices=["legacy", "cosmos3d"], default="legacy",
                    help="motion rotary-position convention. legacy keeps old checkpoints/runs "
                         "unchanged; cosmos3d uses official Cosmos-style 3D-mRoPE for motion as "
                         "a T x 1 x 1 temporal grid, so motion frame k aligns with video/camera "
                         "time k in gen+motion tasks.")
    ap.add_argument("--coupling", choices=["joint", "bridge_local"], default="joint",
                    help="gen-motion coupling mode. joint is the historical 3-way joint attention; "
                         "bridge_local keeps native gen attention separate and applies a local "
                         "directional gen-motion bridge at motion layers.")
    ap.add_argument("--textimg_condition", choices=["generator", "reasoner"], default="reasoner",
                    help="textimg2motion image-conditioning path. reasoner sends frame 0 through "
                         "the Qwen-VL reasoner image path and packs no generator rows for "
                         "textimg2motion. generator is deprecated and retained only for old "
                         "checkpoint/run compatibility.")
    ap.add_argument("--reasoner_image_size", type=int, default=256,
                    help="square pixel size presented to the frozen Qwen visual tower for "
                         "reasoner-side textimg2motion. 256 is the released processor's minimum "
                         "image area and yields 64 visual tokens (640 would yield 400).")
    ap.add_argument("--batch_size", type=int, default=config.TRAIN_DEFAULTS["batch_size"])
    ap.add_argument("--steps", type=int, default=config.TRAIN_DEFAULTS["steps"])
    ap.add_argument("--T", type=int, default=config.VIDEO_NUM_FRAMES,
                    help="shared window length (4N+1 for the Wan VAE); default 33.")
    ap.add_argument("--ti2m_frames", type=int, default=None,
                    help="valid aligned Nymeria frames for reasoner-side textimg2motion. When "
                         "set below --T, TI2M is padded/loss-masked to --T while T2M retains the "
                         "full output capacity (production Phase-2: --T 200 --ti2m_frames 97).")
    # optimization
    ap.add_argument("--lr", type=float, default=config.TRAIN_DEFAULTS["lr"])
    ap.add_argument("--warmup", type=int, default=config.TRAIN_DEFAULTS["warmup"])
    ap.add_argument("--lr_schedule", choices=["cosine", "constant"],
                    default=config.TRAIN_DEFAULTS["lr_schedule"])
    ap.add_argument("--grad_clip", type=float, default=config.TRAIN_DEFAULTS["grad_clip"])
    # loss weights
    ap.add_argument("--w_feat", type=float, default=config.TRAIN_DEFAULTS["w_feat"])
    ap.add_argument("--w_joint", type=float, default=config.TRAIN_DEFAULTS["w_joint"])
    ap.add_argument("--w_smooth", type=float, default=config.TRAIN_DEFAULTS["w_smooth"])
    ap.add_argument("--w_contact", type=float, default=0.0,
                    help="balanced raw-contact BCE weight (GT occupancy supplies pos_weight).")
    ap.add_argument("--w_foot_vel", type=float, default=0.0,
                    help="GT-contact-masked horizontal foot velocity weight, in physical m/s.")
    ap.add_argument("--w_foot_height", type=float, default=0.0,
                    help="GT-contact-masked raw foot-height reconstruction weight, in metres.")
    ap.add_argument("--contact_logit_scale", type=float, default=2.0,
                    help="raw-contact logit slope around the evaluation boundary 0.5.")
    ap.add_argument("--motion_fps", type=float, default=20.0,
                    help="motion frame rate used to convert planted-foot displacement to m/s.")
    ap.add_argument("--w_vision", type=float, default=config.TRAIN_DEFAULTS["w_vision"])
    ap.add_argument("--w_camera", type=float, default=config.TRAIN_DEFAULTS["w_camera"])
    ap.add_argument("--head_camera_alignment", action="store_true",
                    help="Phase-3 bridge variant: derive a clean upright-camera condition from "
                         "clean motion for motimg2video and supervise video2motion head-relative "
                         "SE(3) against synchronized camera actions. GT camera is never a V2M input.")
    ap.add_argument("--head_camera_calibration", default=DEFAULT_CALIBRATION,
                    help="train-split rigid head-joint to upright-camera calibration JSON.")
    ap.add_argument("--w_head_camera_trans", type=float, default=0.0,
                    help="V2M relative head-camera translation-loss weight.")
    ap.add_argument("--w_head_camera_rot", type=float, default=0.0,
                    help="V2M relative head-camera rotation-loss weight.")
    ap.add_argument("--head_camera_translation_scale", type=float, default=0.02,
                    help="metres corresponding to one normalized robust translation-error unit.")
    ap.add_argument("--head_camera_rotation_scale_deg", type=float, default=5.0,
                    help="degrees corresponding to one normalized robust rotation-error unit.")
    ap.add_argument("--cfg_dropout", type=float, default=config.TRAIN_DEFAULTS["cfg_dropout"])
    # train scope toggles (DESIGN_7TASK.md section 5)
    ap.add_argument("--gen_lora", action="store_true",
                    help="inject LoRA on q/k/v/o_proj_moe_gen (generator base stays frozen).")
    ap.add_argument("--gen_lora_rank", type=int, default=16,
                    help="generator LoRA rank; native Phase 1 used rank 16.")
    ap.add_argument("--gen_lora_alpha", type=int, default=16,
                    help="generator LoRA alpha. Set 32 when loading native Phase-1 rank16/alpha32; "
                         "the historical joint-training default was 16.")
    ap.add_argument("--gen_full", action="store_true",
                    help="full generator FT: all _moe_gen + gen I/O heads (excl. with --gen_lora).")
    ap.add_argument("--freeze_gen", action="store_true",
                    help="freeze generator LoRA/full/action-head params even when --gen_lora or "
                         "--gen_full is used to instantiate/warm-start them. Intended for "
                         "bridge-only Phase-3 runs.")
    ap.add_argument("--reasoner_lora", action="store_true",
                    help="inject LoRA on the reasoner q/k/v/o_proj (else reasoner fully frozen).")
    ap.add_argument("--freeze_motion", action="store_true",
                    help="PHASE-1: freeze the motion pathway (_moe_motion + motion heads + "
                         "norm_moe_motion) -- exclude it from the optimizer/grad-clip/all-reduce so "
                         "ONLY the gen-LoRA trains (camera-only tasks). The motion expert is still "
                         "built but never stepped.")
    # curriculum warm-start (Phase 3 = both): load a PRIOR checkpoint's params by SUBSET.
    ap.add_argument("--init_gen", default=None,
                    help="Phase-3 warm-start: load ONLY the generator/gen-LoRA params (the "
                         "cosmos.net lora_/_moe_gen/gen-IO-head keys) from a Phase-1 checkpoint "
                         "(latest.pt / ckpt_step*.pt) by name, strict=False.")
    ap.add_argument("--init_gen_dcp_weights", choices=["ema", "regular"], default="ema",
                    help="when --init_gen is a native Cosmos DCP directory, load net_ema "
                         "(official inference/default) or the non-EMA net tensors.")
    ap.add_argument("--init_motion", default=None,
                    help="Phase-3 warm-start: load ONLY the motion pathway (_moe_motion + motion "
                         "heads + norm_moe_motion) from a Phase-2 checkpoint by name, strict=False.")
    # misc
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=2,
                    help="batches prefetched by each DataLoader worker; ignored with zero workers")
    ap.add_argument("--dataloader_timeout", type=float, default=0.0,
                    help="seconds to wait for a worker batch before failing; zero disables timeout")
    ap.add_argument("--fp32_master", action="store_true",
                    help="cast trainable params to fp32 master (else keep bf16)")
    ap.add_argument("--out", default=None, help="run name under RUN_ROOT")
    ap.add_argument("--save_every", type=int, default=config.TRAIN_DEFAULTS["save_every"])
    ap.add_argument("--viz_every", type=int, default=config.TRAIN_DEFAULTS["viz_every"])
    ap.add_argument("--viz_n", type=int, default=4)
    ap.add_argument("--viz_steps", type=int, default=50)
    ap.add_argument("--viz_guidance", type=float, default=2.0)
    ap.add_argument("--require_viz", action="store_true",
                    help="fail the run if held-out visualization setup, sampling, or rendering "
                         "fails. Use for production runs where checkpoint visuals are required.")
    ap.add_argument("--viz_only", action="store_true",
                    help="build the configured specialists and held-out visualization set, write "
                         "step-0 visualizations, then exit without training or checkpoint saving.")
    ap.add_argument("--viz_frame_stride", type=int, default=2,
                    help="render every Nth motion frame and divide MP4 fps by N so duration is "
                         "preserved (default 2 halves matplotlib rendering cost).")
    ap.add_argument("--log_every", type=int, default=config.TRAIN_DEFAULTS["log_every"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ddp", action="store_true",
                    help="torchrun standalone multi-GPU. DEFAULT = PURE DDP: the tiny trainable set "
                         "is REPLICATED on every rank (plain tensors) and grads are all-reduced + "
                         "averaged manually; the optimizer state is a normal replicated state_dict "
                         "(cheap to save/restore -> trivial resume). The frozen 16B backbone is "
                         "replicated in both modes. Add --fsdp to shard the trainable instead.")
    ap.add_argument("--fsdp", action="store_true",
                    help="OPT-IN: FSDP2-shard the trainable motion expert across ranks (fully_shard "
                         "each MoTJointLayer) instead of replicating it under pure DDP. Use only when "
                         "the trainable set is large enough that replicating its optimizer is costly. "
                         "In this mode the optimizer state is sharded/DTensor and is NOT saved to the "
                         "checkpoint (so --resume restores weights + step only).")
    ap.add_argument("--resume", default=None,
                    help="Resume a run: 'auto' loads <out>/latest.pt if present, or pass an explicit "
                         "checkpoint path. Restores trainable weights (by name, strict=False), the "
                         "optimizer state (pure-DDP / single-GPU), and continues from ckpt['step']+1 "
                         "(LR schedule/warmup resume at that step). No-op if no checkpoint exists. "
                         "Takes precedence over --init_gen/--init_motion (which are weight-only "
                         "warm-starts). WARNs loudly if the checkpoint's --T/--tasks/--objective "
                         "differ from this run.")
    args = ap.parse_args()

    # ---- PER-MODALITY objectives (HISTORY: x0 was once SAMPLER-ONLY) ----
    # A fail-fast SystemExit used to live here: step_loss noised velocity-only, so `--objective
    # x0` trained a velocity denoiser that sample.py then decoded as clean-x0 -> pure-noise
    # samples (loss still dropped; this exact train/sample mismatch cost a long debug -- see
    # memory cosmos_joint_objective_mismatch). The objective is now PER-MODALITY:
    #   MOTION        -> args.objective. "x0" (default): flow.add_noise_x0_masked -- logit-
    #                    normal sigma, target = x0 itself, geometric losses on x0_hat =
    #                    prediction directly (the proven motion_expert/bs_train.py recipe).
    #                    "velocity" stays selectable for ablation / older velocity-motion runs.
    #   VISION/CAMERA -> ALWAYS velocity (flow.add_noise_velocity_masked, target = eps - x0).
    #                    --gen_schedule selects legacy uniform time or native shifted-Waver time;
    #                    args.objective NEVER touches the gen pathway.
    # The seven base tasks noise at most one specialist. Experimental Phase-3 joint-target tasks
    # noise both, using separate per-sample coordinates: motion keeps its pretrained x0/logit-normal
    # marginal and generator targets keep their velocity/Waver marginal. The checkpoint stores both
    # contracts so sampling can co-integrate them on the common native UniPC ladder.
    if args.objective not in ("velocity", "x0"):
        raise SystemExit(f"--objective {args.objective!r} not implemented (choices: velocity, x0)")
    if args.motion_schedule == "native" and args.objective != "x0":
        ap.error("--motion_schedule native requires --objective x0")
    if args.motion_shift <= 0.0:
        ap.error("--motion_shift must be positive")
    if args.motion_num_train_timesteps <= 1:
        ap.error("--motion_num_train_timesteps must be greater than one")
    if args.gen_shift <= 0.0:
        ap.error("--gen_shift must be positive")
    if args.gen_num_train_timesteps <= 1:
        ap.error("--gen_num_train_timesteps must be greater than one")
    if args.gen_lora_rank <= 0 or args.gen_lora_alpha <= 0:
        ap.error("--gen_lora_rank and --gen_lora_alpha must be positive")
    if args.gen_fps <= 0.0:
        ap.error("--gen_fps must be positive")
    if args.gen_temporal_margin < 0.0:
        ap.error("--gen_temporal_margin must be non-negative")
    if args.gen_schedule == "native" and args.gen_packing != "native":
        ap.error("--gen_schedule native requires --gen_packing native for Phase-1 parity")
    if args.reasoner_image_size <= 0:
        ap.error("--reasoner_image_size must be positive")
    motion_loss_weights = (
        args.w_feat,
        args.w_joint,
        args.w_smooth,
        args.w_contact,
        args.w_foot_vel,
        args.w_foot_height,
    )
    if any(weight < 0.0 for weight in motion_loss_weights):
        ap.error("motion loss weights must be non-negative")
    if args.w_head_camera_trans < 0.0 or args.w_head_camera_rot < 0.0:
        ap.error("head-camera loss weights must be non-negative")
    if args.head_camera_translation_scale <= 0.0 or args.head_camera_rotation_scale_deg <= 0.0:
        ap.error("head-camera translation/rotation scales must be positive")
    if (args.w_head_camera_trans > 0.0 or args.w_head_camera_rot > 0.0) \
            and not args.head_camera_alignment:
        ap.error("head-camera loss weights require --head_camera_alignment")
    if args.head_camera_alignment and not os.path.isfile(args.head_camera_calibration):
        ap.error(f"head-camera calibration JSON is missing: {args.head_camera_calibration}")
    if args.contact_logit_scale <= 0.0 or args.motion_fps <= 0.0:
        ap.error("--contact_logit_scale and --motion_fps must be positive")
    if args.viz_frame_stride <= 0:
        ap.error("--viz_frame_stride must be positive")
    if args.num_workers < 0:
        ap.error("--num_workers must be non-negative")
    if args.prefetch_factor <= 0:
        ap.error("--prefetch_factor must be positive")
    if args.dataloader_timeout < 0.0:
        ap.error("--dataloader_timeout must be non-negative")
    if args.require_viz and args.viz_n <= 0:
        ap.error("--require_viz requires --viz_n > 0")
    if args.viz_only and args.viz_n <= 0:
        ap.error("--viz_only requires --viz_n > 0")
    if args.viz_n > 0 and args.viz_every <= 0:
        ap.error("--viz_every must be positive when --viz_n > 0")
    if not 0.0 <= args.bones_frac <= 1.0:
        ap.error("--bones_frac must be in [0,1]")
    motion_x0 = args.objective == "x0"

    # ---- mutually-exclusive gen scope ----
    config.validate_train_scope({"gen_lora": args.gen_lora, "gen_full": args.gen_full})

    rank, world, local = _ddp_env()
    if args.ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
    # Rank 0 can spend tens of minutes sampling and rendering while peer ranks wait. Keep that
    # control-plane wait off the 10-minute NCCL training group: a CPU/Gloo group with an explicit
    # long timeout carries only the tiny visualization setup/status flags.
    viz_sync_group = None
    if args.ddp and world > 1 and args.viz_n > 0:
        viz_sync_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    dev = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0 + rank)

    def log(msg):
        if _is_rank0(rank):
            print(msg, flush=True)

    # ---- task mixture (subset + override) ------------------------------------------------
    task_weights = dict(config.TASK_WEIGHTS)
    if args.task_weights:
        task_weights.update({k: float(v) for k, v in json.loads(args.task_weights).items()})
    if args.tasks:
        unknown = [t for t in args.tasks if t not in TP.TASKS]
        if unknown:
            raise ValueError(f"--tasks has unknown modes {unknown}; expected {TP.TASKS}")
        task_weights = {t: task_weights.get(t, 0.0) for t in args.tasks}
    negative_weights = {m: w for m, w in task_weights.items() if w < 0.0}
    if negative_weights:
        raise ValueError(f"task weights must be non-negative, got {negative_weights}")
    task_weights = {m: w for m, w in task_weights.items() if w > 0.0}
    if not task_weights:
        raise ValueError("no active tasks (empty positive-weight mixture)")
    active_tasks = list(task_weights)
    if args.head_camera_alignment:
        unsupported_alignment_tasks = set(active_tasks) - {"video2motion", "motimg2video"}
        if unsupported_alignment_tasks:
            ap.error(
                "--head_camera_alignment is isolated to Phase-3 V2M/M2V; unsupported active "
                f"tasks: {sorted(unsupported_alignment_tasks)}"
            )
        if args.coupling != "bridge_local":
            ap.error("--head_camera_alignment requires --coupling bridge_local")
    aligned_frames = args.T if args.ti2m_frames is None else int(args.ti2m_frames)
    if not 1 <= aligned_frames <= args.T:
        ap.error(f"--ti2m_frames must be in [1, --T]; got {aligned_frames} with --T {args.T}")
    if aligned_frames != args.T:
        if set(active_tasks) - {"text2motion", "textimg2motion"}:
            ap.error("--ti2m_frames < --T is only supported for Phase-2 T2M+TI2M")
        if "textimg2motion" not in active_tasks or args.textimg_condition != "reasoner":
            ap.error("--ti2m_frames < --T requires reasoner-side textimg2motion")
    log(f"[tasks] active={active_tasks}")
    log(f"[tasks] weights={ {k: round(v, 4) for k, v in task_weights.items()} }")
    weight_total = sum(task_weights.values())
    t2m_prob = task_weights.get("text2motion", 0.0) / weight_total
    log(
        f"[tasks] effective source mass: nymeria_t2m={t2m_prob * (1.0 - args.bones_frac):.4f} "
        f"bones_t2m={t2m_prob * args.bones_frac:.4f} "
        f"nymeria_ti2m={task_weights.get('textimg2motion', 0.0) / weight_total:.4f}"
    )
    log(
        f"[motion_flow] objective={args.objective} schedule={args.motion_schedule} "
        f"shift={args.motion_shift:g} timesteps={args.motion_num_train_timesteps} "
        f"native_solver={args.motion_native_solver}"
    )
    log(
        f"[gen_flow] objective=velocity schedule={args.gen_schedule} "
        f"distribution={'waver' if args.gen_schedule == 'native' else 'uniform'} "
        f"shift={args.gen_shift:g} timesteps={args.gen_num_train_timesteps} "
        f"native_solver={args.gen_native_solver} packing={args.gen_packing} "
        f"fps={args.gen_fps:g} temporal_margin={args.gen_temporal_margin:g}"
    )
    log(
        f"[motion_lengths] output_T={args.T} ti2m_aligned_T={aligned_frames} "
        f"reasoner_image={args.reasoner_image_size}x{args.reasoner_image_size}"
    )
    log(
        f"[motion_loss] feat={args.w_feat:g} joint={args.w_joint:g} "
        f"smooth={args.w_smooth:g} contact={args.w_contact:g} "
        f"foot_vel={args.w_foot_vel:g} foot_height={args.w_foot_height:g} "
        f"contact_logit_scale={args.contact_logit_scale:g} fps={args.motion_fps:g}"
    )
    log(
        f"[head_camera] enabled={args.head_camera_alignment} "
        f"calibration={args.head_camera_calibration} "
        f"w_trans={args.w_head_camera_trans:g} w_rot={args.w_head_camera_rot:g} "
        f"trans_scale_m={args.head_camera_translation_scale:g} "
        f"rot_scale_deg={args.head_camera_rotation_scale_deg:g}"
    )
    log(
        f"[motion_stats] mean={config.MOTION_STATS_MEAN} std={config.MOTION_STATS_STD}"
    )
    if args.textimg_condition == "generator" and "textimg2motion" in active_tasks:
        log("[DEPRECATED] --textimg_condition generator packs the TI2M image as a clean generator "
            "latent frame. Use --textimg_condition reasoner for new TI2M training; generator mode "
            "is retained only to reproduce/load historical runs.")

    # ---- normalization stats (283-d motion z-score) --------------------------------------
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).float().to(dev)
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).float().to(dev)

    # ---- frozen Cosmos reasoner + generator ----
    log("[build] loading FrozenCosmos (reasoner + generator)...")
    cosmos = FrozenCosmos(device=dev)

    # ---- joint model: trainable _moe_motion + heads + (toggled) gen/reasoner adapters ----
    model = JointMotionModel(
        cosmos,
        objective=args.objective,
        motion_schedule=args.motion_schedule,
        motion_shift=args.motion_shift,
        motion_num_train_timesteps=args.motion_num_train_timesteps,
        motion_native_solver=args.motion_native_solver,
        gen_schedule=args.gen_schedule,
        gen_shift=args.gen_shift,
        gen_num_train_timesteps=args.gen_num_train_timesteps,
        gen_native_solver=args.gen_native_solver,
        gen_packing=args.gen_packing,
        gen_fps=args.gen_fps,
        gen_temporal_margin=args.gen_temporal_margin,
        motion_intermediate_size=args.motion_intermediate,
        motion_layer_stride=args.motion_layer_stride,
        motion_mrope=args.motion_mrope,
        coupling=args.coupling,
        textimg_condition=args.textimg_condition,
        reasoner_image_size=args.reasoner_image_size,
        head_camera_alignment=args.head_camera_alignment,
        head_camera_calibration=args.head_camera_calibration,
        gen_lora=args.gen_lora,
        gen_lora_rank=args.gen_lora_rank,
        gen_lora_alpha=args.gen_lora_alpha,
        gen_full=args.gen_full,
        freeze_gen=args.freeze_gen,
        reasoner_lora=args.reasoner_lora,
        freeze_motion=args.freeze_motion,
    ).to(dev)
    model.freeze()  # motion always (unless --freeze_motion); gen/reasoner per the toggles

    # ---- PHASE-1 sanity: under --freeze_motion the motion pathway must carry no grad ----------
    if args.freeze_motion:
        model.assert_motion_frozen()
        log("[freeze_motion] motion pathway (_moe_motion + heads + norm_moe_motion) EXCLUDED "
            "from the trainable set (requires_grad=False); training ONLY the gen/reasoner adapters")
    if args.freeze_gen:
        log("[freeze_gen] generator LoRA/full/action-head params EXCLUDED from optimization "
            "(may still be warm-started by --init_gen)")

    # ---- curriculum warm-start: load prior-checkpoint params by SUBSET (strict=False) --------
    if args.init_gen:
        gsd = load_gen_init_state(
            args.init_gen,
            native_weights=args.init_gen_dcp_weights,
        )
        n_load, n_miss, n_shape = model.load_gen_subset(gsd)
        log(f"[init_gen] {args.init_gen}: loaded {n_load} gen keys "
            f"(skipped missing={n_miss} shape-mismatch={n_shape}) from {len(gsd)} ckpt tensors "
            f"(native_dcp_weights={args.init_gen_dcp_weights})")
    if args.init_motion:
        msd = load_joint_pt(args.init_motion)
        n_load, n_miss, n_shape = model.load_motion_subset(msd)
        log(f"[init_motion] {args.init_motion}: loaded {n_load} motion keys "
            f"(skipped missing={n_miss} shape-mismatch={n_shape}) from {len(msd)} ckpt tensors")

    if args.fp32_master:
        for p in model.trainable_parameters():
            if p.dtype != torch.float32:
                p.data = p.data.float()

    # Includes the generator/reasoner LoRA + gen_full params, which live under cosmos.net and
    # are NOT visible via model.parameters() -- model.trainable_parameters() unions both views.
    trainable = model.trainable_parameters()
    n_train = sum(p.numel() for p in trainable)
    log(f"[params] trainable = {n_train/1e6:.2f}M ({len(trainable)} tensors)  "
        f"gen_lora={args.gen_lora} gen_full={args.gen_full} reasoner_lora={args.reasoner_lora} "
        f"freeze_gen={args.freeze_gen} freeze_motion={args.freeze_motion}")

    # ---- multi-GPU parallelism: PURE DDP (default) vs FSDP (opt-in) ----------------------
    # PURE DDP (default, world>1, no --fsdp): the tiny trainable set stays plain REPLICATED
    #   tensors (NO fully_shard). Grads are synced by the manual all-reduce (+ mean) below in the
    #   train loop; the optimizer state is a normal replicated state_dict -> trivial to save/resume.
    # FSDP (--fsdp): shard each MoTJointLayer's trainable motion expert across ranks. Only worth it
    #   for a large trainable set; the optimizer state is sharded and is NOT checkpointed.
    fsdp_sharded = bool(args.fsdp and world > 1)
    if fsdp_sharded:
        from torch.distributed.fsdp import fully_shard
        for layer in model.layers:
            fully_shard(layer)
        log(f"[fsdp] sharded {len(model.layers)} MoTJointLayers across {world} GPUs")
    elif args.ddp and world > 1:
        # PURE DDP requires every replica to START from IDENTICAL trainable weights: the motion
        # expert (and any freshly-init'd LoRA) are randomly initialized, and `manual_seed(0+rank)`
        # gives each rank a DIFFERENT init -> averaging grads over divergent replicas is meaningless.
        # Broadcast rank-0's trainable params so all replicas agree before the first step. (FSDP
        # doesn't need this: it shards from a single logical copy.)
        import torch.distributed as dist
        with torch.no_grad():
            for p in trainable:
                dist.broadcast(p.data, src=0)
        log(f"[ddp] PURE DDP across {world} GPUs: trainable REPLICATED (no fully_shard), "
            f"broadcast rank-0 init to all replicas; grads all-reduced+averaged manually; "
            f"optimizer state replicated (resumable)")

    # ---- ON-THE-FLY Wan-VAE video-latent encoder (loaded ONCE, frozen/bf16/eval) ----------
    # Any task that packs image/video may need a latent the precomputed cache does not have at
    # the current --T (the cache is keyed at T=33 -> T_lat=9; at other T the lookup misses). We
    # then VAE-encode the sample's raw frames live in step_loss -- the EXACT precompute_latents
    # path (resize/pad via ActionTransformPipeline -> /127.5-1 -> Wan2.2-VAE.encode -> crop
    # padding in latent) so the live (C,T_lat,h,w) latents are byte-conventions-identical to the
    # cached ones gen_heads expects. Built only when a video/image task is active (motion-only
    # runs skip it entirely) AND live encoding may be needed (missing/mismatched cache or forced).
    T_lat_expected = (args.T - 1) // 4 + 1
    def _task_needs_generator_vision(m: str) -> bool:
        if args.textimg_condition == "reasoner" and m == "textimg2motion":
            return False
        p = TP.build_task_plan(m)
        return p.video.present or p.image.present

    video_tasks_active = any(_task_needs_generator_vision(m) for m in active_tasks)
    # T-specific precomputed-latent root (also handed to the dataset below): T=VIDEO_NUM_FRAMES
    # (33) uses the default cache; any other T uses a per-T root (e.g. joint_latents_T97) so
    # caches at different T never collide.
    _lat_root = (config.VIDEO_LATENT_ROOT if args.T == config.VIDEO_NUM_FRAMES
                 else f"{config.VIDEO_LATENT_ROOT}_T{args.T}")

    def _probe_latent_cache(root, t_lat_expected, n_probe=3):
        """True iff the ACTUAL per-T latent cache at ``root`` exists and its latents' T_lat
        matches ``t_lat_expected`` (probe a few .npz files, layout <root>/<subj>/<key>.npz).
        This replaces the old hardcoded ``T_lat==9`` check so a COMPLETE per-T cache (e.g.
        joint_latents_T97) is recognized and the Wan VAE is NOT loaded for it."""
        import glob as _glob
        if not os.path.isdir(root):
            return False
        checked = 0
        for path in _glob.iglob(os.path.join(root, "*", "*.npz")):
            try:
                with np.load(path) as d:
                    key = "latents" if "latents" in d else d.files[0]
                    t_lat = int(d[key].shape[1])
            except Exception:  # noqa: BLE001 -- unreadable probe file: try the next one
                continue
            if t_lat != t_lat_expected:
                return False
            checked += 1
            if checked >= n_probe:
                break
        return checked > 0

    cache_matches_T = video_tasks_active and _probe_latent_cache(_lat_root, T_lat_expected)
    need_live_vae = video_tasks_active and (
        args.force_on_the_fly or (not args.precomputed_latents) or (not cache_matches_T)
    )
    wan_vae = wan_pipe = None
    if need_live_vae:
        from precompute_latents import encode_window as _pc_encode_window
        from precompute_latents import load_vae as _pc_load_vae
        from precompute_latents import make_pipeline as _pc_make_pipeline
        log(f"[wan_vae] loading Wan2.2-VAE for ON-THE-FLY encode "
            f"(T={args.T} -> T_lat={T_lat_expected}, res={args.vae_resolution}, "
            f"force_on_the_fly={args.force_on_the_fly}, cache_hits_at_T={cache_matches_T})...")
        t_vae0 = time.time()
        # encode_exact_durations=[args.T] -> the VAE encodes at exactly this window length (no
        # padding inflation), mirroring precompute_latents.load_vae(num_frames=T).
        wan_vae = _pc_load_vae(args.wan_vae_path, args.vae_resolution, args.T, dev)
        wan_pipe = _pc_make_pipeline()
        log(f"[wan_vae] loaded ({wan_vae.model.count_param()/1e6:.1f}M params) "
            f"in {time.time()-t_vae0:.1f}s; on-the-fly latents ACTIVE")
    elif video_tasks_active:
        log(f"[wan_vae] not loaded: T={args.T} (T_lat={T_lat_expected}) hits the precomputed "
            f"cache at {_lat_root}; no live VAE encode needed (fast path).")

    def encode_frames_live(frames_uint8):
        """One raw window (uint8 [3,T,H,W]) -> (C,T_lat,h,w) float latents via the EXACT
        precompute_latents path (no grad, bf16 autocast). Returns None if the VAE is disabled."""
        if wan_vae is None:
            return None
        # precompute_latents.encode_window expects (T,H,W,3) uint8 numpy; frames are (3,T,H,W).
        frames_np = frames_uint8.permute(1, 2, 3, 0).contiguous().cpu().numpy()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latents_np, _image_size = _pc_encode_window(
                wan_vae, wan_pipe, args.vae_resolution, frames_np, dev
            )  # (C,T_lat,h,w) fp16 numpy
        return torch.from_numpy(np.ascontiguousarray(latents_np)).float().to(dev)

    # ---- dataset: 5-modality aligned NymeriaPlus (+ BONES text2motion) -------------------
    # `_lat_root` (the T-specific precomputed-latent root) is computed above, next to the cache
    # probe. If the per-T cache is absent, the dataset emits raw frames and the trainer
    # VAE-encodes live.
    def build_ds(
        split,
        train,
        cfg_dropout,
        max_samples=None,
        *,
        task_weights_override=None,
        bones_frac_override=None,
    ):
        return NymeriaJointDataset(
            split=split, num_frames=args.T, aligned_num_frames=aligned_frames,
            task_weights=task_weights if task_weights_override is None else task_weights_override,
            bones_text2motion_frac=(
                args.bones_frac if bones_frac_override is None else bones_frac_override
            ),
            cfg_dropout=cfg_dropout,
            prefer_latents=args.precomputed_latents, latent_root=_lat_root,
            force_on_the_fly=args.force_on_the_fly, train=train,
            reasoner_image_for_textimg=(args.textimg_condition == "reasoner"),
            reasoner_image_size=args.reasoner_image_size,
            camera_head_alignment=args.head_camera_alignment,
            max_samples=max_samples, seed=rank,
        )

    ds = build_ds("train", train=True, cfg_dropout=args.cfg_dropout)
    if args.bones_frac > 0.0 and "text2motion" in active_tasks and not ds.has_bones:
        raise RuntimeError(
            "BONES was requested by --bones_frac but failed to load; refusing to silently "
            "change the Phase-2 source mixture"
        )
    sampler = None
    if args.ddp and world > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world, rank=rank, shuffle=True)
    loader_kwargs = {
        "dataset": ds,
        "batch_size": args.batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": collate_joint,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "timeout": args.dataloader_timeout,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dl = DataLoader(**loader_kwargs)
    log(
        f"[data] dataset={len(ds)} output_T={args.T} aligned_T={aligned_frames} "
        f"has_bones={ds.has_bones} workers={args.num_workers} "
        f"prefetch={args.prefetch_factor if args.num_workers > 0 else 0} "
        f"timeout={args.dataloader_timeout:g}s"
    )

    # ---- optimizer ----
    # fused AdamW works on plain tensors (single-GPU + pure DDP); it can't mix DTensors, so
    # disable it only under FSDP sharding.
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99),
                            fused=not fsdp_sharded)

    # ---- run dir / logging ----
    out, tb = None, None
    data_policy = {
        "nymeria_text2motion_window": "native_caption_span_up_to_T",
        "nymeria_textimg2motion_window": (
            f"{aligned_frames}_frame_video_aligned_motion_with_first_frame_reasoner_image"
            if args.textimg_condition == "reasoner"
            else "T_frame_video_aligned_generator_image"
        ),
        "caption_dropout": float(args.cfg_dropout),
        "reasoner_image_dropout": 0.0,
        "reasoner_image_size": int(args.reasoner_image_size),
        "ti2m_frames": int(aligned_frames),
        "bones_overview_caption": "content_natural_desc_4",
        "motion_schedule": args.motion_schedule,
        "motion_shift": float(args.motion_shift),
        "motion_num_train_timesteps": int(args.motion_num_train_timesteps),
        "motion_native_solver": args.motion_native_solver,
        "motion_stats_mean": config.MOTION_STATS_MEAN,
        "motion_stats_std": config.MOTION_STATS_STD,
        "nymeria_uniego_root": config.NYMERIA_UNIEGO_ROOT,
        "head_camera_alignment": {
            "enabled": bool(args.head_camera_alignment),
            "calibration": args.head_camera_calibration,
            "m2v_condition": "relative_camera_action_derived_only_from_clean_motion",
            "v2m_supervision": "training_only_synchronized_camera_action_target",
            "absolute_pose_used": False,
            "w_translation": float(args.w_head_camera_trans),
            "w_rotation": float(args.w_head_camera_rot),
            "translation_scale_m": float(args.head_camera_translation_scale),
            "rotation_scale_deg": float(args.head_camera_rotation_scale_deg),
        },
        "motion_loss": {
            "w_feat": float(args.w_feat),
            "w_joint": float(args.w_joint),
            "w_smooth": float(args.w_smooth),
            "w_contact": float(args.w_contact),
            "w_foot_vel": float(args.w_foot_vel),
            "w_foot_height": float(args.w_foot_height),
            "contact_logit_scale": float(args.contact_logit_scale),
            "motion_fps": float(args.motion_fps),
            "contact_mask": "ground_truth",
            "foot_joints": [24, 25, 28, 29],
        },
        "gen_schedule": args.gen_schedule,
        "gen_train_time_distribution": (
            "waver" if args.gen_schedule == "native" else "uniform"
        ),
        "gen_shift": float(args.gen_shift),
        "gen_num_train_timesteps": int(args.gen_num_train_timesteps),
        "gen_native_solver": args.gen_native_solver,
        "gen_packing": args.gen_packing,
        "gen_fps": float(args.gen_fps),
        "gen_temporal_margin": float(args.gen_temporal_margin),
        "init_gen_dcp_weights": args.init_gen_dcp_weights,
    }
    if not args.smoke and _is_rank0(rank):
        name = args.out or f"joint7_{time.strftime('%Y%m%d_%H%M%S')}"
        out = os.path.join(RUN_ROOT, name)
        os.makedirs(out, exist_ok=True)
        json.dump({**vars(args), "trainable_M": n_train / 1e6, "world": world,
                   "active_tasks": active_tasks, "task_weights": task_weights,
                   "data_policy": data_policy},
                  open(os.path.join(out, "config.json"), "w"), indent=2, default=str)
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(out)
        except Exception:
            tb = None
        log(f"[run] {out}")
    logf = open(os.path.join(out, "train.log"), "a") if out else None

    def logline(msg):
        log(msg)
        if logf is not None:
            logf.write(msg + "\n")
            logf.flush()

    # ----------------------------------------------------------------------------------
    # RESUME: restore trainable weights + optimizer + step from a checkpoint (orthogonal to
    # --init_gen/--init_motion; --resume takes precedence). No-op if no checkpoint is found.
    # In pure DDP / single-GPU the optimizer state is a normal replicated state_dict (restored);
    # under --fsdp only weights + step are restored (the optimizer state was not checkpointed).
    # ----------------------------------------------------------------------------------
    start_step = 0
    if args.resume:
        # Resolve the resume path on ALL ranks (not just rank0). `out` is rank0-only, so for
        # `--resume auto` we recompute the run dir from RUN_ROOT/args.out deterministically -> every
        # rank restores the same weights/optimizer/step and the DDP replicas stay identical.
        if args.resume == "auto":
            _run_dir = os.path.join(RUN_ROOT, args.out) if args.out else out
            resume_path = os.path.join(_run_dir, "latest.pt") if _run_dir else None
        else:
            resume_path = args.resume
        if resume_path and os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            # config-drift guard: resuming across a --T/--tasks/--objective change is usually wrong.
            prev_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
            for key in (
                "T", "tasks", "objective", "motion_schedule", "motion_shift",
                "motion_num_train_timesteps", "motion_native_solver", "motion_mrope",
                "gen_schedule", "gen_shift", "gen_num_train_timesteps", "gen_native_solver",
                "gen_packing", "gen_fps", "gen_temporal_margin",
                "gen_lora_rank", "gen_lora_alpha", "init_gen_dcp_weights",
                "coupling", "textimg_condition", "reasoner_image_size", "ti2m_frames",
                "w_feat", "w_joint", "w_smooth", "w_contact", "w_foot_vel",
                "w_foot_height", "contact_logit_scale", "motion_fps",
                "head_camera_alignment", "head_camera_calibration",
                "w_head_camera_trans", "w_head_camera_rot",
                "head_camera_translation_scale", "head_camera_rotation_scale_deg",
            ):
                prev = prev_args.get(key)
                cur = getattr(args, key)
                if prev is not None and prev != cur:
                    logline(f"[resume][WARN] config drift: --{key} was {prev!r} in the checkpoint "
                            f"but is {cur!r} now -- resuming across a config change is usually WRONG "
                            f"(proceeding anyway).")
            # restore trainable weights by name (strict=False), same approach as the subset loaders.
            sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            n_load, n_miss, n_shape = model._load_subset(sd, lambda _n: True)
            # restore optimizer (present in pure-DDP / single-GPU checkpoints).
            opt_state = ckpt.get("optimizer") if isinstance(ckpt, dict) else None
            opt_restored = False
            if opt_state is not None:
                try:
                    opt.load_state_dict(opt_state)
                    opt_restored = True
                except Exception as e:  # noqa: BLE001
                    logline(f"[resume][WARN] optimizer restore FAILED: {type(e).__name__}: {e}")
            prev_step = int(ckpt.get("step", -1)) if isinstance(ckpt, dict) else -1
            start_step = prev_step + 1 if prev_step >= 0 else 0
            logline(f"[resume] {resume_path}: resumed from step {prev_step} -> continuing at "
                    f"step {start_step}; weights loaded={n_load} (missing={n_miss} "
                    f"shape-mismatch={n_shape}); optimizer_restored={opt_restored} "
                    f"(fsdp_sharded={fsdp_sharded})")
        else:
            logline(f"[resume] no checkpoint at {resume_path!r}; starting fresh (step 0)")

    # small helper: latent [C,T_lat,h,w] shape (handles list element) --------------------
    def vlat_shape(vlat):
        if vlat is None:
            return (0, 0, 0, 0)
        C, T_lat, h, w = vlat.shape
        return C, T_lat, h, w

    # ----------------------------------------------------------------------------------
    # one training-step loss (mode-aware, multi-modality)
    #
    # Returns (total_loss, scalars) where scalars is a dict of detached python floats:
    #   per-modality flow losses (motion_feat / motion_joint / motion_smooth / vision / camera)
    #   and per-task totals (loss/task/<mode>), for TB logging.
    # ----------------------------------------------------------------------------------
    def step_loss(batch):
        modes = batch["mode"]                                   # List[str] length B
        B = len(modes)
        captions = batch["caption"]
        input_ids_list = [
            (
                cosmos.tokenize_generation(c)
                if args.gen_packing == "native" and TP.build_task_plan(modes[s]).has_gen
                else cosmos.tokenize(c)
            )
            for s, c in enumerate(captions)
        ]
        reasoner_inputs = [None] * B
        if args.textimg_condition == "reasoner":
            for s, mode_s in enumerate(modes):
                if mode_s != "textimg2motion":
                    continue
                img = batch.get("reasoner_image", [None] * B)[s]
                if img is None:
                    raise RuntimeError(
                        "textimg_condition=reasoner needs batch['reasoner_image'] for "
                        "textimg2motion. Enable dataset reasoner-image frames or force raw frames."
                    )
                r = cosmos.encode_reasoner_image_text(
                    captions[s], img, image_size=args.reasoner_image_size
                )
                reasoner_inputs[s] = r
                input_ids_list[s] = r["input_ids"].view(1, -1)

        # ---- per-sample flow time/sigma (PER-SPECIALIST objectives) ----------------------
        # The exact tensor used to noise a specialist is also sent to that specialist's timestep
        # embedding. Motion and generator coordinates are separate because their frozen Phase-2
        # and Phase-1 pretraining distributions differ:
        #   motion-noised samples (text2motion/textimg2motion/video2motion): the MOTION
        #     objective's schedule -- logit-normal sigma for x0 (bs_train recipe), uniform t
        #     for velocity.
        #   gen-noised samples (fwd/inv/policy/motimg2video): velocity with either the legacy
        #     uniform schedule or native shifted-Waver schedule.
        # Samples whose modality is condition-only get sigma_eff=0 per-token, so that specialist's
        # sampled value is inert. Joint-target samples intentionally draw the two values independently.
        motion_noised = torch.tensor(
            [TP.build_task_plan(m).motion.supervised for m in modes],
            dtype=torch.bool,
            device=dev,
        )
        gen_noised = torch.tensor(
            [
                TP.build_task_plan(m).video.supervised
                or TP.build_task_plan(m).camera.supervised
                for m in modes
            ],
            dtype=torch.bool,
            device=dev,
        )
        motion_sigma = torch.rand(B, device=dev)
        if motion_x0:
            if args.motion_schedule == "native":
                sampled_motion_sigma = flow.sample_sigma_native_logitnormal(
                    B,
                    dev,
                    shift=args.motion_shift,
                    dtype=motion_sigma.dtype,
                )
            else:
                sampled_motion_sigma = flow.sample_sigma_logitnormal(B, dev)
            motion_sigma = torch.where(motion_noised, sampled_motion_sigma, motion_sigma)

        gen_sigma = torch.rand(B, device=dev)
        if args.gen_schedule == "native":
            sampled_gen_sigma = flow.sample_sigma_native_waver(
                B,
                dev,
                shift=args.gen_shift,
                dtype=gen_sigma.dtype,
            )
            gen_sigma = torch.where(gen_noised, sampled_gen_sigma, gen_sigma)

        # ---- motion modality (dense [B,Tm,283]; None if no sample carries motion) --------
        motion = batch["motion"]
        nj = batch["neutral_joints"]
        pad = batch["motion_pad_mask"]
        x0 = x_t = t_motion = target_motion = None
        valid = noisy_frame_mask = None
        if motion is not None:
            x0 = motion.to(dev)
            nj = nj.to(dev)
            pad = pad.to(dev)
            valid = ~pad                                       # [B,Tm] True=keep
            # per-sample: motion-target tasks noise all valid frames; motion-condition tasks
            # (motimg2video, clean_policy=="all") keep every frame clean.
            cond_motion = torch.ones_like(valid)               # True=CLEAN
            for s in range(B):
                plan = TP.build_task_plan(modes[s])
                if plan.motion.present and plan.motion.clean_policy != "all":
                    cond_motion[s] = pad[s]                     # clean only the pad frames
            # MOTION objective dispatch (per-modality design; see the flag/guard above):
            #   x0 (default) -> add_noise_x0_masked  : motion_sigma rows are logit-normal sigma,
            #                   target_motion = x0 itself.
            #   velocity     -> add_noise_velocity_masked : uniform t, target = eps - x0.
            # Both take (x0, condition_mask, t_or_sigma) positionally and return the same
            # tuple structure; passing motion_sigma keeps the noiser and forward's motion sigma
            # on the SAME per-sample value (t_motion comes back == motion_sigma).
            motion_noiser = (flow.add_noise_x0_masked if motion_x0
                             else flow.add_noise_velocity_masked)
            x_t, t_motion, target_motion, noised_motion = motion_noiser(
                x0, cond_motion, motion_sigma)
            # the model's motion encoder expects noisy_frame_mask True=NOISED (loss target).
            noisy_frame_mask = noised_motion & valid           # noised AND valid

        # ---- generator modalities: noise per-sample with the resolved condition_mask -----
        # We feed the model the NOISED latents/actions + the per-token condition_mask is rebuilt
        # internally from the same plan; the target velocity (eps - x0) is kept here for the loss.
        cam_dense = batch["camera_action"]                     # [B,Tc,9] or None
        cam_pad = batch["camera_pad_mask"]                     # [B,Tc] True=pad or None
        alignment_dense = batch.get("camera_alignment_action")
        alignment_pad = batch.get("camera_alignment_pad_mask")
        if alignment_dense is not None:
            alignment_dense = alignment_dense.to(dev)
            alignment_pad = alignment_pad.to(dev)
        vid_list = list(batch["video_latents"])                # list[Optional[C,T_lat,h,w]]
        frm_list = batch["video_frames"]                       # list[Optional[uint8 [3,T,H,W]]]

        # ---- ON-THE-FLY VAE encode: for any sample whose task needs video/image but has NO
        # precomputed latent (cache missing / T_lat mismatched at this T), encode its raw frames
        # live into a (C,T_lat,h,w) latent so the rest of the step is identical to the cached path.
        # The fast path (valid cache present) is untouched -- vid_list[s] stays the cached latent
        # and no VAE runs for it.
        if wan_vae is not None:
            for s in range(B):
                plan_s = TP.build_task_plan(modes[s])
                if args.textimg_condition == "reasoner" and modes[s] == "textimg2motion":
                    continue
                if not (plan_s.video.present or plan_s.image.present):
                    continue
                if vid_list[s] is not None:                    # cache hit -> skip VAE
                    continue
                if frm_list[s] is None:                        # no frames to encode
                    continue
                lat = encode_frames_live(frm_list[s])          # (C,T_lat,h,w) on dev
                if lat is not None:
                    vid_list[s] = lat

        noised_vid: list = [None] * B
        target_vid: list = [None] * B
        noised_cam_dense = camera_target = None
        if cam_dense is not None:
            cam_dense = cam_dense.to(dev)
            cam_pad = cam_pad.to(dev)

        # build per-sample noised gen inputs + targets
        for s in range(B):
            plan = TP.build_task_plan(modes[s])
            reasoner_textimg = args.textimg_condition == "reasoner" and modes[s] == "textimg2motion"
            ts = gen_sigma[s:s + 1]
            # video / image
            vlat = vid_list[s]
            if (plan.video.present or plan.image.present) and (not reasoner_textimg) and vlat is not None:
                vlat = vlat.to(dev)                            # [C,T_lat,h,w]
                C, T_lat, h, w = vlat.shape
                if plan.image.present and not plan.video.present:
                    # single CLEAN image frame: pass through untouched, no target.
                    noised_vid[s] = vlat[:, :1]
                else:
                    # per-frame clean mask from the plan (frame0 clean for frame0 policy, etc.)
                    cmask_frames = torch.tensor(
                        TP._video_condition_mask(plan.video.clean_policy, T_lat),
                        dtype=torch.bool, device=dev)         # [T_lat] True=CLEAN
                    # noise the latent (T_lat-first); [1,T_lat,C*h*w] view then back.
                    flat = vlat.permute(1, 0, 2, 3).reshape(1, T_lat, -1)   # [1,T_lat,C*h*w]
                    nfx, _, tgt, _ = flow.add_noise_velocity_masked(
                        flat, cmask_frames.view(1, T_lat), t=ts)
                    noised_vid[s] = nfx.view(T_lat, C, h, w).permute(1, 0, 2, 3).contiguous()
                    if plan.video.supervised:
                        target_vid[s] = tgt.view(T_lat, C, h, w)  # [T_lat,C,h,w] velocity target
            # camera is noised below in a single batched pass (per-sample condition masks).

        # camera: batch-noise (homogeneous-ish) with per-sample condition masks.
        if cam_dense is not None and any(TP.build_task_plan(m).camera.present for m in modes):
            Tc = cam_dense.shape[1]
            cond_cam = torch.ones((B, Tc), dtype=torch.bool, device=dev)   # True=CLEAN
            for s in range(B):
                plan = TP.build_task_plan(modes[s])
                if not plan.camera.present:
                    continue
                if plan.camera.clean_policy == "all":
                    cond_cam[s] = torch.ones(Tc, dtype=torch.bool, device=dev)  # condition
                else:
                    cond_cam[s] = cam_pad[s]                   # clean only pads -> noise valid
            noised_cam_dense, _, camera_target, _ = flow.add_noise_velocity_masked(
                cam_dense, cond_cam, t=gen_sigma)
            # zero out the velocity target on pad frames (kept clean) for the loss mask later.
        noised_cam_list = [None] * B
        if noised_cam_dense is not None:
            for s in range(B):
                if TP.build_task_plan(modes[s]).camera.present:
                    # drop pad frames so the model sees only the (T-1) real action frames.
                    keep = ~cam_pad[s]
                    noised_cam_list[s] = noised_cam_dense[s][keep]          # [n_cam,9]

        # ---- forward (dict): real gen_idx + per-modality encode/decode ------------------
        out = model.forward(
            input_ids_list,
            x_t=x_t,
            motion_t_or_sigma=motion_sigma,
            gen_t_or_sigma=gen_sigma,
            neutral_joints=nj if motion is not None else None,
            motion_pad_mask=pad if motion is not None else None,
            noisy_frame_mask=noisy_frame_mask,
            modes=modes,
            video_latents=noised_vid,
            camera_action=noised_cam_list,
            reasoner_inputs=reasoner_inputs,
            return_dict=True,
        )

        scalars = {}
        if motion_x0 and motion_noised.any():
            active_sigma = motion_sigma[motion_noised]
            scalars["motion_sigma_mean"] = float(active_sigma.mean())
            scalars["motion_sigma_min"] = float(active_sigma.min())
            scalars["motion_sigma_max"] = float(active_sigma.max())
        if gen_noised.any():
            active_gen_sigma = gen_sigma[gen_noised]
            scalars["gen_sigma_mean"] = float(active_gen_sigma.mean())
            scalars["gen_sigma_min"] = float(active_gen_sigma.min())
            scalars["gen_sigma_max"] = float(active_gen_sigma.max())
        joint_noised = motion_noised & gen_noised
        if joint_noised.any():
            scalars["joint_sigma_abs_delta"] = float(
                (motion_sigma[joint_noised] - gen_sigma[joint_noised]).abs().mean()
            )
        total = torch.zeros((), device=dev)
        per_task = {m: torch.zeros((), device=dev) for m in set(modes)}

        # ---- MOTION loss (feat MSE vs the objective's target + decoded joint + smooth),
        # masked to NOISED frames. target_motion is eps-x0 (velocity) or x0 itself (x0). ----
        motion_pred = out.get("motion_pred")
        if motion_pred is not None and target_motion is not None:
            # supervise only frames that were noised (motion-target tasks), and valid.
            mloss_mask = noisy_frame_mask                      # [B,Tm] True=supervised
            if mloss_mask.any():
                # Decode once, then reduce per task so experimental two-target modes can use a
                # half branch weight without changing historical one-target task weights.
                if motion_x0:
                    x0_hat = motion_pred
                else:
                    tb_ = t_motion.view(-1, *([1] * (x_t.dim() - 1)))
                    x0_hat = x_t - tb_ * motion_pred
                contact_enabled = any(
                    weight > 0.0
                    for weight in (args.w_contact, args.w_foot_vel, args.w_foot_height)
                )
                need_decoded = args.w_joint > 0 or args.w_smooth > 0 or contact_enabled
                j_hat = j_gt = rel_hat = rel_gt = None
                if need_decoded:
                    j_hat = decode_joints(x0_hat * std + mean)
                    with torch.no_grad():
                        j_gt = decode_joints(x0 * std + mean).detach()
                    if torch.isfinite(j_hat).all():
                        rel_hat = j_hat - j_hat.mean(dim=2, keepdim=True)
                        rel_gt = j_gt - j_gt.mean(dim=2, keepdim=True)

                metric_terms = {k: [] for k in (
                    "motion_feat", "motion_joint", "motion_smooth", "motion_contact",
                    "motion_foot_vel", "motion_foot_height"
                )}
                for mode in sorted(set(modes)):
                    plan = TP.build_task_plan(mode)
                    if not plan.motion.supervised:
                        continue
                    rows = torch.tensor(
                        [m == mode for m in modes], dtype=torch.bool, device=dev
                    )
                    mode_mask = mloss_mask & rows[:, None]
                    if not mode_mask.any():
                        continue
                    l_feat = masked_mse(motion_pred, target_motion, mode_mask)
                    l_joint = x0.new_zeros(())
                    l_smooth = x0.new_zeros(())
                    l_contact = x0.new_zeros(())
                    l_foot_vel = x0.new_zeros(())
                    l_foot_height = x0.new_zeros(())
                    if rel_hat is not None:
                        if args.w_joint > 0:
                            l_joint = masked_mse(rel_hat, rel_gt, mode_mask)
                        if args.w_smooth > 0:
                            vmask = mode_mask[:, 1:] & mode_mask[:, :-1]
                            l_smooth = masked_mse(
                                j_hat[:, 1:] - j_hat[:, :-1],
                                j_gt[:, 1:] - j_gt[:, :-1],
                                vmask,
                            )
                        if contact_enabled:
                            l_contact, l_foot_vel, l_foot_height = contact_aware_losses(
                                x0_hat, x0, j_hat, mode_mask, mean, std,
                                fps=args.motion_fps,
                                contact_logit_scale=args.contact_logit_scale,
                            )
                    m_total = (
                        args.w_feat * l_feat
                        + args.w_joint * l_joint
                        + args.w_smooth * l_smooth
                        + args.w_contact * l_contact
                        + args.w_foot_vel * l_foot_vel
                        + args.w_foot_height * l_foot_height
                    )
                    count = int(rows.sum().item())
                    branch_scale = plan.motion.loss_weight / TP.W_MOTION
                    contribution = m_total * (float(count) / float(B)) * branch_scale
                    total = total + contribution
                    per_task[mode] = per_task[mode] + contribution
                    for key, value in zip(metric_terms, (
                        l_feat, l_joint, l_smooth, l_contact, l_foot_vel, l_foot_height
                    )):
                        metric_terms[key].append((value, count))

                for key, terms in metric_terms.items():
                    if terms:
                        denom = float(sum(count for _, count in terms))
                        value = sum(term * count for term, count in terms) / denom
                        scalars[key] = float(value.detach())

        # ---- VISION loss (per-sample latent flow MSE on noised frames) -----------------
        video_pred = out.get("video_pred") or [None] * B
        vis_accum = []
        for s in range(B):
            vp = video_pred[s]
            tgt = target_vid[s]
            if vp is None or tgt is None:
                continue
            # vp: [1,C,T_lat,h,w] with predictions at the noised frames; tgt: [T_lat,C,h,w].
            plan = TP.build_task_plan(modes[s])
            C, T_lat, h, w = vlat_shape(vid_list[s])
            cmask = torch.tensor(
                TP._video_condition_mask(plan.video.clean_policy, T_lat),
                dtype=torch.bool, device=dev)                  # [T_lat] True=CLEAN
            noised_frames = ~cmask                             # [T_lat]
            vp5 = vp[0].permute(1, 0, 2, 3)                    # [T_lat,C,h,w]
            # flatten to [1, T_lat, C*h*w] and mask to noised frames.
            p_flat = vp5.reshape(1, T_lat, -1)
            t_flat = tgt.reshape(1, T_lat, -1)
            branch_scale = plan.video.loss_weight / TP.W_VISION
            lv = flow.vision_flow_loss(
                p_flat, t_flat, noised_frames.view(1, T_lat),
                weight=args.w_vision * branch_scale,
            )
            vis_accum.append(lv)
            batch_lv = lv / float(B)
            total = total + batch_lv
            per_task[modes[s]] = per_task[modes[s]] + batch_lv
        if vis_accum:
            scalars["vision"] = float((sum(vis_accum) / len(vis_accum)).detach())

        # ---- CAMERA loss (per-sample flow MSE on chan[:9], noised frames) --------------
        camera_pred = out.get("camera_pred") or [None] * B
        cam_accum = []
        for s in range(B):
            cp = camera_pred[s]
            if cp is None or camera_target is None:
                continue
            plan = TP.build_task_plan(modes[s])
            if not plan.camera.supervised:
                continue
            keep = ~cam_pad[s]                                 # [Tc] real action frames
            tgt = camera_target[s][keep]                       # [n_cam,9] velocity target
            # camera_pred[s] holds only the NOISED action rows; for a fully-noised target
            # (clean_policy=="none") that is every real frame.
            n = cp.shape[0]
            noised_mask = torch.ones((1, n), dtype=torch.bool, device=dev)
            branch_scale = plan.camera.loss_weight / TP.ACTION_LOSS_WEIGHT
            lc = flow.camera_flow_loss(
                cp.unsqueeze(0), tgt[:n].unsqueeze(0), noised_mask,
                weight=args.w_camera * branch_scale,
            )
            cam_accum.append(lc)
            batch_lc = lc / float(B)
            total = total + batch_lc
            per_task[modes[s]] = per_task[modes[s]] + batch_lc
        if cam_accum:
            scalars["camera"] = float((sum(cam_accum) / len(cam_accum)).detach())

        # ---- RELATIVE HEAD<->CAMERA alignment -------------------------------------------
        # V2M receives only clean video and predicts motion. The synchronized camera action is
        # a training-only target here; it is never packed into the V2M forward. M2V receives a
        # clean camera condition derived deterministically from its clean motion inside the model,
        # never the GT camera action. Absolute world positions are intentionally unused.
        if args.head_camera_alignment:
            if alignment_dense is None or alignment_pad is None:
                raise RuntimeError(
                    "head-camera alignment is enabled but the dataset emitted no camera target"
                )
            if motion_pred is None or x0 is None or valid is None:
                raise RuntimeError("head-camera alignment requires motion rows in every sample")
            if motion_x0:
                alignment_x0_hat = motion_pred
            else:
                alignment_t = t_motion.view(-1, *([1] * (x_t.dim() - 1)))
                alignment_x0_hat = x_t - alignment_t * motion_pred
            predicted_alignment = model.motion_to_camera_action(alignment_x0_hat)
            if predicted_alignment.shape[:2] != alignment_dense.shape[:2]:
                raise RuntimeError(
                    "head-camera transition mismatch: derived "
                    f"{tuple(predicted_alignment.shape)} vs target {tuple(alignment_dense.shape)}"
                )

            v2m_rows = torch.tensor(
                [mode == "video2motion" for mode in modes],
                dtype=torch.bool,
                device=dev,
            )
            transition_valid = valid[:, 1:] & valid[:, :-1] & (~alignment_pad)
            v2m_transition_mask = transition_valid & v2m_rows[:, None]
            n_v2m = int(v2m_rows.sum().item())
            if v2m_transition_mask.any():
                head_trans, head_rot = head_camera_alignment_losses(
                    predicted_alignment,
                    alignment_dense,
                    v2m_transition_mask,
                    translation_scale_m=args.head_camera_translation_scale,
                    rotation_scale_deg=args.head_camera_rotation_scale_deg,
                )
                head_total = (
                    args.w_head_camera_trans * head_trans
                    + args.w_head_camera_rot * head_rot
                )
                head_contrib = head_total * (float(n_v2m) / float(B))
                total = total + head_contrib
                per_task["video2motion"] = per_task.get(
                    "video2motion", total.new_zeros(())
                ) + head_contrib
                with torch.no_grad():
                    trans_m, rot_deg = head_camera_errors(
                        predicted_alignment,
                        alignment_dense,
                        v2m_transition_mask,
                    )
                scalars["head_camera_v2m_trans_loss"] = float(head_trans.detach())
                scalars["head_camera_v2m_rot_loss"] = float(head_rot.detach())
                scalars["head_camera_v2m_trans_m"] = float(trans_m)
                scalars["head_camera_v2m_rot_deg"] = float(rot_deg)

            # Calibration/input diagnostic for M2V. This is not a loss: the derived camera is a
            # deterministic condition, and reporting its GT discrepancy makes calibration quality
            # visible without accidentally supervising M2V with target camera tokens.
            derived_camera = out.get("derived_camera_action") or [None] * B
            m2v_trans = []
            m2v_rot = []
            with torch.no_grad():
                for s, mode in enumerate(modes):
                    if mode != "motimg2video" or derived_camera[s] is None:
                        continue
                    n = int(derived_camera[s].shape[0])
                    target_s = alignment_dense[s:s + 1, :n]
                    pred_s = derived_camera[s].unsqueeze(0).to(target_s.dtype)
                    mask_s = (~alignment_pad[s:s + 1, :n])
                    trans_m, rot_deg = head_camera_errors(pred_s, target_s, mask_s)
                    m2v_trans.append(trans_m)
                    m2v_rot.append(rot_deg)
            if m2v_trans:
                scalars["head_camera_m2v_condition_trans_m"] = float(torch.stack(m2v_trans).mean())
                scalars["head_camera_m2v_condition_rot_deg"] = float(torch.stack(m2v_rot).mean())

        if args.coupling == "bridge_local":
            gates_g = torch.stack([b.gate_g.detach().float() for b in model.bridges.values()])
            gates_m = torch.stack([b.gate_m.detach().float() for b in model.bridges.values()])
            scalars["bridge_gate_g_abs_mean"] = float(gates_g.abs().mean())
            scalars["bridge_gate_m_abs_mean"] = float(gates_m.abs().mean())
            scalars["bridge_gate_g_abs_max"] = float(gates_g.abs().max())
            scalars["bridge_gate_m_abs_max"] = float(gates_m.abs().max())

        for m, v in per_task.items():
            scalars[f"task/{m}"] = float(v.detach())
        return total, scalars

    # ----------------------------------------------------------------------------------
    # SMOKE: run a few steps over a couple tasks; prove wiring + freeze partition
    # ----------------------------------------------------------------------------------
    if args.smoke:
        # build small per-mode batches so we exercise >=1 motion task + (if latents avail) a
        # video task, asserting finite loss + grad ONLY on the active-trainable set.
        def one_sample_batch(target_mode):
            """Draw a single item whose mode == target_mode (retry up to a cap), collate to B=1."""
            for _ in range(2000):
                item = ds[np.random.randint(len(ds))]
                if item["mode"] == target_mode:
                    return collate_joint([item])
            return None

        # candidate smoke tasks: text2motion (always), + a video task if latents exist.
        smoke_tasks = [t for t in ("text2motion", "video2motion", "textimg2motion",
                                   "forward_dynamics", "inverse_dynamics", "policy",
                                   "motimg2video", "video2camera_motion",
                                   "camimg2video_motion") if t in active_tasks]

        model.train()
        ok = True
        ran = 0
        bridge_core_seen = False
        for tmode in smoke_tasks:
            batch = one_sample_batch(tmode)
            if batch is None:
                logline(f"[smoke] could not draw a '{tmode}' sample; skipping")
                continue
            # if this task needs gen modalities but the drawn sample carries none (no cached
            # latent, no raw frames to encode live, and no camera), skip gracefully.
            plan = TP.build_task_plan(tmode)
            no_video = (all(v is None for v in batch["video_latents"])
                        and all(f is None for f in batch["video_frames"]))
            needs_generator_payload = _task_needs_generator_vision(tmode) or plan.camera.present
            if needs_generator_payload and no_video and batch["camera_action"] is None:
                logline(f"[smoke] '{tmode}' needs gen modalities but none present; skipping")
                continue
            try:
                loss, sc = step_loss(batch)
            except Exception as e:  # noqa: BLE001
                logline(f"[smoke] '{tmode}' step_loss FAILED: {type(e).__name__}: {e}")
                ok = False
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()

            # grad on active-trainable set must be > 0; frozen pathways must be 0 / None.
            train_grad = sum(p.grad.abs().sum().item()
                             for n, p in model.named_all_parameters()
                             if model._is_trainable_name(n) and p.grad is not None)
            bridge_gate_grad = sum(
                p.grad.abs().sum().item()
                for n, p in model.named_all_parameters()
                if "bridges." in n and ".gate_" in n and p.grad is not None
            )
            bridge_core_grad = sum(
                p.grad.abs().sum().item()
                for n, p in model.named_all_parameters()
                if "bridges." in n and ".gate_" not in n and p.grad is not None
            )
            bridge_core_seen = bridge_core_seen or bridge_core_grad > 0.0
            try:
                model.assert_frozen_grads_zero()
                frozen_ok = True
            except AssertionError as e:
                logline(f"[smoke] '{tmode}' FROZEN-GRAD LEAK: {str(e)[:200]}")
                frozen_ok = False
            fin = bool(torch.isfinite(loss).item())
            step_finite = fin and grads_finite(trainable)
            if tmode == "video2camera_motion":
                step_finite = step_finite and "motion_feat" in sc and "camera" in sc
            if tmode == "camimg2video_motion":
                step_finite = step_finite and "motion_feat" in sc and "vision" in sc
            if args.grad_clip > 0:
                clip_grads(trainable, args.grad_clip)
            smoke_lrf = lr_factor(ran, args.warmup, args.steps, args.lr_schedule)
            for group in opt.param_groups:
                group["lr"] = args.lr * smoke_lrf
            logline(f"[smoke] {tmode:16s} loss={loss.item():.4f} finite={fin} "
                    f"train_grad={train_grad:.3f} gate_grad={bridge_gate_grad:.3f} "
                    f"core_grad={bridge_core_grad:.3f} frozen_ok={frozen_ok} "
                    f"{ {k: round(v, 3) for k, v in sc.items()} }")
            ok = ok and step_finite and (train_grad > 0) and frozen_ok
            if step_finite:
                opt.step()
            ran += 1
        if ran == 0:
            logline("[smoke] FAIL: ran 0 tasks")
            ok = False
        if args.coupling == "bridge_local" and ran >= 3 and not bridge_core_seen:
            logline("[smoke] FAIL: bridge q/k/v/o parameters never received a nonzero gradient")
            ok = False
        print("[smoke] PASS" if ok else "[smoke] FAIL", flush=True)
        if not ok:
            raise RuntimeError("training smoke failed")
        return

    # ----------------------------------------------------------------------------------
    # In-train held-out visualization. Motion pretraining balances T2M/TI2M. Bridge training
    # covers every active corner/joint mode. Samples are fixed for the entire run so checkpoints
    # are comparable.
    # ----------------------------------------------------------------------------------
    viz_items = []
    viz_modes = [
        m for m in (
            "text2motion", "textimg2motion", "video2motion", "motimg2video",
            "video2camera_motion", "camimg2video_motion",
        ) if m in active_tasks
    ]
    viz_vae = wan_vae
    viz_vae_is_temporary = False
    viz_setup_error = ""
    if args.viz_n > 0 and out is not None and viz_modes:
        try:
            n_total = max(args.viz_n, len(viz_modes))
            quotas = {mode: n_total // len(viz_modes) for mode in viz_modes}
            for mode in viz_modes[: n_total % len(viz_modes)]:
                quotas[mode] += 1

            def _viz_record(item, mode):
                gt_motion = torch.as_tensor(item["motion"]).float()
                pad_mask = item.get("motion_pad_mask")
                if pad_mask is not None:
                    gt_motion = gt_motion[~torch.as_tensor(pad_mask)]
                video_latents_v = item.get("video_latents")
                return {
                    "mode": mode,
                    "caption": item["caption"],
                    "source": item.get("source", "nymeria"),
                    "gt_motion": gt_motion,
                    "sample_T": (
                        int(gt_motion.shape[0]) if mode == "textimg2motion" else args.T
                    ),
                    "neutral_joints": torch.as_tensor(item["neutral_joints"]).float(),
                    "reasoner_image": (
                        torch.as_tensor(item["reasoner_image"]).clone()
                        if item.get("reasoner_image") is not None
                        else None
                    ),
                    "video_latents": (
                        torch.as_tensor(video_latents_v).float().clone()
                        if video_latents_v is not None
                        else None
                    ),
                    "camera_action": (
                        torch.as_tensor(item["camera_action"]).float().clone()
                        if item.get("camera_action") is not None
                        else None
                    ),
                }

            if quotas.get("text2motion", 0) > 0:
                target = quotas["text2motion"]
                vds = build_ds(
                    "test",
                    train=False,
                    cfg_dropout=0.0,
                    max_samples=4096,
                    task_weights_override={"text2motion": 1.0},
                )
                selected = []
                seen = set()
                per_source = max(1, target // 2)
                for strict in (True, False):
                    for i in range(len(vds)):
                        if len(selected) >= target:
                            break
                        item = vds[i]
                        cap = item.get("caption", "")
                        src = item.get("source", "nymeria")
                        if not cap or cap in seen or item.get("motion") is None:
                            continue
                        if strict and sum(v["source"] == src for v in selected) >= per_source:
                            continue
                        seen.add(cap)
                        selected.append(_viz_record(item, "text2motion"))
                    if len(selected) >= target:
                        break
                viz_items.extend(selected)

            if quotas.get("textimg2motion", 0) > 0:
                target = quotas["textimg2motion"]
                vds = build_ds(
                    "test",
                    train=False,
                    cfg_dropout=0.0,
                    max_samples=4096,
                    task_weights_override={"textimg2motion": 1.0},
                    bones_frac_override=0.0,
                )
                seen = set()
                for i in range(len(vds)):
                    if sum(v["mode"] == "textimg2motion" for v in viz_items) >= target:
                        break
                    item = vds[i]
                    cap = item.get("caption", "")
                    if (
                        not cap
                        or cap in seen
                        or item.get("motion") is None
                        or item.get("reasoner_image") is None
                    ):
                        continue
                    seen.add(cap)
                    viz_items.append(_viz_record(item, "textimg2motion"))

            for mode in (
                "video2motion", "motimg2video", "video2camera_motion",
                "camimg2video_motion",
            ):
                target = quotas.get(mode, 0)
                if target <= 0:
                    continue
                vds = build_ds(
                    "test",
                    train=False,
                    cfg_dropout=0.0,
                    max_samples=4096,
                    task_weights_override={mode: 1.0},
                    bones_frac_override=0.0,
                )
                selected = 0
                for i in range(len(vds)):
                    if selected >= target:
                        break
                    item = vds[i]
                    if (
                        item.get("motion") is None
                        or item.get("neutral_joints") is None
                        or item.get("video_latents") is None
                        or (
                            mode in ("video2camera_motion", "camimg2video_motion")
                            and item.get("camera_action") is None
                        )
                    ):
                        continue
                    viz_items.append(_viz_record(item, mode))
                    selected += 1

            counts = {}
            for item in viz_items:
                key = f"{item['mode']}:{item['source']}"
                counts[key] = counts.get(key, 0) + 1
            mode_counts = {
                mode: sum(item["mode"] == mode for item in viz_items)
                for mode in viz_modes
            }
            missing = {
                mode: (mode_counts.get(mode, 0), quotas[mode])
                for mode in viz_modes
                if mode_counts.get(mode, 0) < quotas[mode]
            }
            if missing:
                raise RuntimeError(f"could not build required held-out viz samples: {missing}")
            logline(f"[viz] setup: {len(viz_items)} held-out samples = {counts}")
        except Exception as e:
            viz_setup_error = f"{type(e).__name__}: {str(e)[:240]}"
            logline(f"[viz] setup skipped: {viz_setup_error}")

    # Setup is rank-0-only because only rank 0 owns `out`. Broadcast its status so a required
    # visualization failure terminates every DDP rank coherently instead of stranding peers in
    # the next gradient collective.
    viz_setup_failed = int(bool(viz_setup_error))
    if args.ddp and world > 1 and args.viz_n > 0:
        import torch.distributed as dist

        setup_flag = torch.tensor(viz_setup_failed, dtype=torch.int32)
        dist.broadcast(setup_flag, src=0, group=viz_sync_group)
        viz_setup_failed = int(setup_flag.item())
    if viz_setup_failed and args.require_viz:
        raise RuntimeError(
            "required held-out visualization setup failed"
            + (f": {viz_setup_error}" if viz_setup_error else " on rank 0")
        )

    def _get_viz_vae():
        nonlocal viz_vae, viz_vae_is_temporary
        if viz_vae is None:
            from precompute_latents import load_vae

            logline(
                "[viz] loading rank-local Wan2.2-VAE for V2M/M2V checkpoint videos..."
            )
            viz_vae = load_vae(
                args.wan_vae_path,
                args.vae_resolution,
                args.T,
                dev,
                rank_local=True,
            )
            viz_vae_is_temporary = True
            logline("[viz] Wan2.2-VAE loaded")
        return viz_vae

    def _release_temporary_viz_vae():
        nonlocal viz_vae, viz_vae_is_temporary
        if not viz_vae_is_temporary:
            return
        viz_vae = None
        viz_vae_is_temporary = False
        gc.collect()
        torch.cuda.empty_cache()
        logline("[viz] released rank-local Wan2.2-VAE")

    def _write_latent_video(latents, path, *, frame_stride=1):
        import eval_camera as EC

        vae = _get_viz_vae()
        lat_np = (
            latents.detach().float().cpu().numpy()
            if torch.is_tensor(latents)
            else np.asarray(latents, dtype=np.float32)
        )
        frames = EC.latent_to_frames(vae, lat_np, dev)
        frames = frames[::frame_stride]
        fps = max(1, round(20 / frame_stride))
        EC.frames_to_mp4(frames, path, fps)
        return frames

    def _hstack_videos(left_path, right_path, out_path, left_label, right_label, *, fps=None):
        """Best-effort labeled comparison; the two component MP4s remain authoritative."""
        import subprocess

        rate = f"fps={int(fps)}," if fps is not None else ""
        fc = (
            f"[0:v]{rate}setpts=PTS-STARTPTS,scale=-2:400,pad=iw:ih+26:0:26:black,"
            f"drawtext=text='{left_label}':"
            "x=6:y=3:fontcolor=yellow:fontsize=18[a];"
            f"[1:v]{rate}setpts=PTS-STARTPTS,scale=-2:400,pad=iw:ih+26:0:26:black,"
            f"drawtext=text='{right_label}':"
            "x=6:y=3:fontcolor=yellow:fontsize=18[b];"
            "[a][b]hstack=inputs=2:shortest=1"
        )
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", left_path, "-i", right_path,
             "-filter_complex", fc, out_path],
            check=False,
        )
        return out_path if result.returncode == 0 else None

    def _plot_camera_actions(gt_action, pred_action, out_path):
        """Plot metric relative-action trajectories from a common identity origin."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import eval_inverse_dynamics as EID

        gt = np.asarray(gt_action, dtype=np.float64)
        pred = np.asarray(pred_action, dtype=np.float64)
        n = min(len(gt), len(pred))
        gt = gt[:n]
        pred = pred[:n]
        gt_pos = EID.pred_abs(gt[:, :3], EID.rot6d_to_R(gt[:, 3:9]))
        pred_pos = EID.pred_abs(pred[:, :3], EID.rot6d_to_R(pred[:, 3:9]))
        ate = float(np.sqrt(np.square(gt_pos - pred_pos).sum(axis=1).mean()))
        fig = plt.figure(figsize=(6.4, 5.6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*gt_pos.T, color="tab:green", linewidth=1.8, label="GT camera")
        ax.plot(*pred_pos.T, color="tab:red", linewidth=1.8, label="generated camera")
        ax.scatter(*gt_pos[0], color="black", s=24)
        all_pos = np.concatenate([gt_pos, pred_pos], axis=0)
        center = all_pos.mean(axis=0)
        radius = float(np.abs(all_pos - center).max() * 1.1 + 1e-6)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_title(f"Joint video -> camera + motion | direct ATE={ate:.3f} m")
        ax.legend(fontsize=8)
        ax.view_init(elev=24, azim=-60)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return ate

    def _render_joint_motion(x0_np, it, vdir, stem):
        """Decode and render normalized generated motion with its aligned GT."""
        from render_motion import render_motion_mp4

        x0v = torch.from_numpy(np.asarray(x0_np, dtype=np.float32)).to(dev).unsqueeze(0)
        joints = decode_joints(x0v[0].float().unsqueeze(0) * std + mean)[0].cpu().numpy()
        gt_feat = it["gt_motion"].to(dev) * std + mean
        gt_joints = decode_joints(gt_feat.unsqueeze(0))[0].cpu().numpy()
        pred_path = os.path.join(vdir, stem + "_pred_motion.npy")
        gt_path = os.path.join(vdir, stem + "_gt_motion.npy")
        np.save(pred_path, joints)
        np.save(gt_path, gt_joints)
        motion_mp4 = os.path.join(vdir, stem + "_gt_pred_motion.mp4")
        render_motion_mp4(
            joints,
            motion_mp4,
            caption="joint physical generation (no text condition)",
            fps=max(1, round(20 / args.viz_frame_stride)),
            gt_joints=gt_joints,
            frame_stride=args.viz_frame_stride,
        )
        return {
            "pred_joints_npy": pred_path,
            "gt_joints_npy": gt_path,
            "motion_video": motion_mp4,
        }

    @torch.no_grad()
    def do_viz(step):
        if not viz_items or out is None:
            return
        model.eval()
        try:
            vdir = os.path.join(out, f"viz_step{step:06d}")
            os.makedirs(vdir, exist_ok=True)
            manifest = []
            for i, it in enumerate(viz_items):
                mode = it["mode"]
                nj_v = it["neutral_joints"].to(dev).unsqueeze(0)
                sample_T = int(it["sample_T"])
                src = it.get("source", "nymeria")
                tag = it["caption"][:28].replace(" ", "_").replace("/", "_") or "no_text"
                stem = f"{i}_{mode}_{src}_{tag}"

                if mode == "video2camera_motion":
                    from sample import sample_video2camera_motion

                    vlat = it["video_latents"].to(dev)
                    gt_camera = it["camera_action"].to(dev)
                    generated = sample_video2camera_motion(
                        model,
                        video_latents=vlat,
                        neutral_joints=nj_v,
                        T=sample_T,
                        camera_T=int(gt_camera.shape[0]),
                        steps=args.viz_steps,
                        guidance=args.viz_guidance,
                        device=dev,
                        seed=i,
                    )
                    motion_files = _render_joint_motion(
                        generated["motion"], it, vdir, stem
                    )
                    condition_video = os.path.join(vdir, stem + "_condition_video.mp4")
                    _write_latent_video(
                        vlat, condition_video, frame_stride=args.viz_frame_stride
                    )
                    comparison_video = os.path.join(vdir, stem + ".mp4")
                    comparison = _hstack_videos(
                        condition_video,
                        motion_files["motion_video"],
                        comparison_video,
                        "condition video",
                        "GT | jointly generated motion",
                    )
                    gt_camera_path = os.path.join(vdir, stem + "_gt_camera.npy")
                    pred_camera_path = os.path.join(vdir, stem + "_pred_camera.npy")
                    camera_plot = os.path.join(vdir, stem + "_camera_trajectory.png")
                    np.save(gt_camera_path, gt_camera.float().cpu().numpy())
                    np.save(pred_camera_path, generated["camera"])
                    camera_ate = _plot_camera_actions(
                        gt_camera.float().cpu().numpy(), generated["camera"], camera_plot
                    )
                    manifest.append({
                        "i": i,
                        "mode": mode,
                        "source": src,
                        "caption": "",
                        "condition_video": condition_video,
                        "comparison_video": comparison,
                        "gt_camera_npy": gt_camera_path,
                        "generated_camera_npy": pred_camera_path,
                        "camera_trajectory_png": camera_plot,
                        "camera_direct_ate_m": camera_ate,
                        **motion_files,
                        "T": sample_T,
                        "sampling_steps": args.viz_steps,
                        "guidance": args.viz_guidance,
                        "joint_sampler": "single_state_unipc",
                    })
                    continue

                if mode == "camimg2video_motion":
                    from sample import sample_camimg2video_motion

                    vlat = it["video_latents"].to(dev)
                    gt_camera = it["camera_action"].to(dev)
                    generated = sample_camimg2video_motion(
                        model,
                        image_latent=vlat[:, 0],
                        camera_action=gt_camera,
                        neutral_joints=nj_v,
                        T=sample_T,
                        T_lat=int(vlat.shape[1]),
                        steps=args.viz_steps,
                        guidance=args.viz_guidance,
                        device=dev,
                        seed=i,
                    )
                    motion_files = _render_joint_motion(
                        generated["motion"], it, vdir, stem
                    )
                    gt_npz = os.path.join(vdir, stem + "_gt_latents.npz")
                    gen_npz = os.path.join(vdir, stem + "_generated_latents.npz")
                    np.savez(gt_npz, latents=vlat.float().cpu().numpy().astype(np.float16))
                    np.savez(gen_npz, latents=generated["video"].astype(np.float16))
                    gt_mp4 = os.path.join(vdir, stem + "_gt_video.mp4")
                    gen_mp4 = os.path.join(vdir, stem + "_generated_video.mp4")
                    _write_latent_video(vlat, gt_mp4)
                    _write_latent_video(generated["video"], gen_mp4)
                    video_comparison = os.path.join(vdir, stem + "_gt_generated_video.mp4")
                    _hstack_videos(
                        gt_mp4, gen_mp4, video_comparison, "GT video", "jointly generated video"
                    )
                    joint_comparison_path = os.path.join(vdir, stem + ".mp4")
                    joint_comparison = _hstack_videos(
                        gen_mp4,
                        motion_files["motion_video"],
                        joint_comparison_path,
                        "jointly generated video",
                        "GT | jointly generated motion",
                        fps=max(1, round(20 / args.viz_frame_stride)),
                    )
                    manifest.append({
                        "i": i,
                        "mode": mode,
                        "source": src,
                        "caption": "",
                        "gt_latents_npz": gt_npz,
                        "generated_latents_npz": gen_npz,
                        "gt_video": gt_mp4,
                        "generated_video": gen_mp4,
                        "video_comparison": video_comparison,
                        "comparison_video": joint_comparison,
                        **motion_files,
                        "T": sample_T,
                        "sampling_steps": args.viz_steps,
                        "guidance": args.viz_guidance,
                        "joint_sampler": "single_state_unipc",
                    })
                    continue

                if mode == "motimg2video":
                    from sample import sample_motimg2video

                    vlat = it["video_latents"].to(dev)
                    gen_latents = sample_motimg2video(
                        model,
                        caption=it["caption"],
                        image_latent=vlat[:, 0],
                        motion=it["gt_motion"].to(dev).unsqueeze(0),
                        neutral_joints=nj_v,
                        T_lat=int(vlat.shape[1]),
                        steps=args.viz_steps,
                        guidance=args.viz_guidance,
                        device=dev,
                        seed=i,
                    )
                    gt_npz = os.path.join(vdir, stem + "_gt_latents.npz")
                    gen_npz = os.path.join(vdir, stem + "_generated_latents.npz")
                    np.savez(gt_npz, latents=vlat.float().cpu().numpy().astype(np.float16))
                    np.savez(gen_npz, latents=gen_latents.astype(np.float16))
                    gt_mp4 = os.path.join(vdir, stem + "_gt.mp4")
                    gen_mp4 = os.path.join(vdir, stem + "_generated.mp4")
                    _write_latent_video(vlat, gt_mp4)
                    _write_latent_video(gen_latents, gen_mp4)
                    comparison_mp4 = os.path.join(vdir, stem + ".mp4")
                    comparison = _hstack_videos(
                        gt_mp4, gen_mp4, comparison_mp4, "GT video", "generated video"
                    )
                    if comparison is None:
                        logline(f"[viz] ffmpeg M2V comparison failed; kept {gt_mp4} and {gen_mp4}")
                    manifest.append({
                        "i": i,
                        "mode": mode,
                        "source": src,
                        "caption": it["caption"],
                        "gt_latents_npz": gt_npz,
                        "generated_latents_npz": gen_npz,
                        "gt_video": gt_mp4,
                        "generated_video": gen_mp4,
                        "comparison_video": comparison,
                        "T": sample_T,
                        "gen_schedule": args.gen_schedule,
                        "gen_shift": args.gen_shift,
                        "gen_native_solver": args.gen_native_solver,
                        "sampling_steps": args.viz_steps,
                        "guidance": args.viz_guidance,
                    })
                    continue

                if mode == "textimg2motion":
                    from sample import sample_textimg2motion

                    x0_np = sample_textimg2motion(
                        model,
                        caption=it["caption"],
                        neutral_joints=nj_v,
                        T=sample_T,
                        reasoner_image=it["reasoner_image"],
                        steps=args.viz_steps,
                        guidance=args.viz_guidance,
                        device=dev,
                        seed=i,
                    )
                    x0v = torch.from_numpy(x0_np).to(dev).unsqueeze(0)
                elif mode == "video2motion":
                    from sample import sample_video2motion

                    vlat = it["video_latents"].to(dev)
                    x0_np = sample_video2motion(
                        model,
                        neutral_joints=nj_v,
                        T=sample_T,
                        video_latents=vlat,
                        steps=args.viz_steps,
                        guidance=args.viz_guidance,
                        device=dev,
                        seed=i,
                    )
                    x0v = torch.from_numpy(x0_np).to(dev).unsqueeze(0)
                else:
                    x0v = model.sample(
                        caption=it["caption"], neutral_joints=nj_v, T=sample_T,
                        steps=args.viz_steps, guidance=args.viz_guidance,
                        tokenizer=cosmos.tokenize, device=dev, seed=i,
                    )
                feat = x0v[0].float() * std + mean
                joints = decode_joints(feat.unsqueeze(0))[0].cpu().numpy()
                npy_path = os.path.join(vdir, stem + ".npy")
                np.save(npy_path, joints)

                gt_joints = gt_path = None
                gtm = it.get("gt_motion")
                if gtm is not None and gtm.numel() > 0:
                    gt_feat = gtm.to(dev) * std + mean
                    gt_joints = decode_joints(gt_feat.unsqueeze(0))[0].cpu().numpy()
                    gt_path = os.path.join(vdir, stem + "_gt.npy")
                    np.save(gt_path, gt_joints)

                condition_path = None
                comparison_path = None
                motion_mp4_path = None
                try:
                    from render_motion import render_conditioned_motion_mp4, render_motion_mp4

                    motion_mp4_path = os.path.join(
                        vdir, stem + ("_gt_pred_motion.mp4" if mode == "video2motion" else ".mp4")
                    )
                    render_fps = max(1, round(20 / args.viz_frame_stride))
                    if mode == "textimg2motion":
                        condition_path = os.path.join(vdir, stem + "_condition.png")
                        render_conditioned_motion_mp4(
                            condition_image=it["reasoner_image"],
                            gen_joints=joints,
                            gt_joints=gt_joints,
                            out_path=motion_mp4_path,
                            condition_out_path=condition_path,
                            caption=it["caption"],
                            fps=render_fps,
                            frame_stride=args.viz_frame_stride,
                        )
                    else:
                        render_motion_mp4(
                            joints, motion_mp4_path, caption=it["caption"], fps=render_fps,
                            gt_joints=gt_joints,
                            frame_stride=args.viz_frame_stride,
                        )
                    if mode == "video2motion":
                        condition_path = os.path.join(vdir, stem + "_condition_video.mp4")
                        _write_latent_video(
                            it["video_latents"],
                            condition_path,
                            frame_stride=args.viz_frame_stride,
                        )
                        requested_comparison = os.path.join(vdir, stem + ".mp4")
                        comparison_path = _hstack_videos(
                            condition_path,
                            motion_mp4_path,
                            requested_comparison,
                            "condition video",
                            "GT | predicted motion",
                        )
                        if comparison_path is None:
                            logline(
                                f"[viz] ffmpeg V2M comparison failed; kept {condition_path} "
                                f"and {motion_mp4_path}"
                            )
                except Exception as e:  # noqa: BLE001 -- rendering must never break training
                    if args.require_viz:
                        raise
                    logline(f"[viz] mp4 render skipped: {type(e).__name__}: {str(e)[:120]}")

                manifest.append({
                    "i": i,
                    "mode": it["mode"],
                    "source": src,
                    "caption": it["caption"],
                    "joints_npy": npy_path,
                    "gt_joints_npy": gt_path,
                    "motion_video": motion_mp4_path,
                    "condition_media": condition_path,
                    "comparison_video": comparison_path,
                    "T": int(joints.shape[0]),
                    "motion_schedule": args.motion_schedule,
                    "motion_shift": args.motion_shift,
                    "motion_native_solver": args.motion_native_solver,
                    "render_frame_stride": args.viz_frame_stride,
                    "render_fps": max(1, round(20 / args.viz_frame_stride)),
                })
            json.dump(manifest, open(os.path.join(vdir, "manifest.json"), "w"), indent=2)
            logline(
                f"[viz] step {step} -> {vdir} "
                f"({len(manifest)} held-out records; modes="
                f"{sorted({row['mode'] for row in manifest})})"
            )
        finally:
            model.train()

    def run_checkpoint_viz(step):
        """Run rank-0 visualization and synchronize success/failure across DDP ranks."""
        if args.viz_n <= 0:
            return
        if args.ddp and world > 1:
            import torch.distributed as dist

            dist.barrier(group=viz_sync_group)
        failed = 0
        error_text = ""
        if _is_rank0(rank):
            try:
                do_viz(step)
            except Exception as e:  # noqa: BLE001 -- propagate coherently after broadcast
                failed = 1
                error_text = f"{type(e).__name__}: {str(e)[:240]}"
                logline(f"[viz] step {step} failed: {error_text}")
            finally:
                _release_temporary_viz_vae()
        if args.ddp and world > 1:
            flag = torch.tensor(failed, dtype=torch.int32)
            dist.broadcast(flag, src=0, group=viz_sync_group)
            failed = int(flag.item())
        if failed and args.require_viz:
            raise RuntimeError(
                f"required checkpoint visualization failed at step {step}"
                + (f": {error_text}" if error_text else " on rank 0")
            )

    if args.viz_only:
        run_checkpoint_viz(0)
        logline("[viz_only] done; no optimizer step or checkpoint save was performed")
        if tb is not None:
            tb.close()
        if logf is not None:
            logf.close()
        if args.ddp:
            import torch.distributed as dist

            dist.destroy_process_group()
        return

    def save_ckpt(step):
        if out is None:
            return
        try:
            sd = model.trainable_state_dict()
        except Exception:
            sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        if not _is_rank0(rank):
            return
        # Save the optimizer state whenever it is NOT FSDP-sharded. In pure DDP (or single-GPU) it
        # is a normal replicated state_dict -> save it so --resume restores weights + optimizer +
        # step. Under --fsdp the state is sharded/DTensor -> skip it (resume restores weights+step).
        payload = {"model": sd,
                   "optimizer": None if fsdp_sharded else opt.state_dict(),
                   "step": step, "args": vars(args), "task_weights": task_weights,
                   "data_policy": data_policy}
        torch.save(payload, os.path.join(out, f"ckpt_step{step:06d}.pt"))
        torch.save(payload, os.path.join(out, "latest.pt"))
        logline(f"[ckpt] step {step}")

    # ----------------------------------------------------------------------------------
    # train loop
    # ----------------------------------------------------------------------------------
    model.train()
    torch.cuda.reset_peak_memory_stats()
    step, t0 = start_step, time.time()
    n_skipped = 0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in dl:
            lrf = lr_factor(step, args.warmup, args.steps, args.lr_schedule)
            for g in opt.param_groups:
                g["lr"] = args.lr * lrf

            loss, sc = step_loss(batch)

            # BATCH-LOSS BOMB GUARD (defense-in-depth; see nymeria_joint_dataset._load_motion's
            # feature guard). Run 2727 was destroyed at step 14140 by a single loss=34.6 batch
            # (grad_clip=1.0 did NOT save it: the spike also poisons Adam's moment estimates and
            # the damage took ~4k steps to re-learn). If THIS rank's batch loss is non-finite or
            # explosive, zero the loss INSTEAD of skipping: backward still runs (producing zero
            # grads) and the DDP all-reduce still executes on every rank in lockstep — a per-rank
            # skip would desync the collectives (the multi-task NCCL-hang class). Threshold: an
            # EMA of recent losses x25, floored at 50 (generous: healthy batches are <~2).
            _l = float(loss.detach())
            _ema = getattr(main, "_loss_ema", None) or 1.0
            if not math.isfinite(_l) or _l > max(50.0, 25.0 * _ema):
                logline(f"[bomb-guard] step {step}: batch loss {_l:.1f} > "
                        f"max(50, 25x ema {_ema:.2f}) — zeroing this rank's contribution")
                # Multiplying NaN/Inf by zero remains NaN. `where` gives a graph-connected,
                # finite zero with zero derivative for non-finite losses; ordinary finite spikes
                # can use the cheaper multiply-by-zero path.
                loss = (
                    loss * 0.0
                    if math.isfinite(_l)
                    else torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
                )
            else:
                main._loss_ema = 0.99 * _ema + 0.01 * _l

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # PURE-DDP grad sync: the trainable set is REPLICATED (plain tensors), so we must
            # manually all-reduce + AVERAGE each grad (all_reduce defaults to SUM; DDP needs the
            # mean, else the effective LR is world x too large). Under --fsdp this is SKIPPED --
            # FSDP2 reduce-scatters the sharded grads internally, so touching them here would
            # double-reduce.
            if args.ddp and world > 1 and not fsdp_sharded:
                import torch.distributed as dist
                # MULTI-TASK COLLECTIVE SAFETY: ranks pick DIFFERENT tasks per step (per-rank RNG),
                # so a trainable param may have a grad on some ranks and None on others (its task
                # wasn't sampled there). all_reduce is a COLLECTIVE -- EVERY rank must call it for the
                # SAME params in the SAME order or the ranks desync -> NCCL ALLREDUCE timeout/hang
                # (this was the recurring multi-task hang; single-task runs never hit it). So we
                # all_reduce EVERY trainable param, materializing a zero grad where it's None. A rank
                # that didn't touch the param contributes 0, and dividing the SUM by `world` yields
                # exactly the mixed-task batch-mean gradient (0-grad samples are part of the batch).
                for p in trainable:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad)              # SUM across ranks (incl. zeros)
                    p.grad /= world                      # -> mixed-task batch mean

            if args.grad_clip > 0:
                clip_grads(trainable, args.grad_clip)

            step_ok = bool(torch.isfinite(loss.detach()).item()) and grads_finite(trainable)
            if args.ddp and world > 1:
                # Every rank must either step or skip. This is essential for FSDP shards, where a
                # non-finite gradient can be local to one rank; rank-local optimizer decisions
                # would permanently diverge replicas/shards.
                import torch.distributed as dist
                ok_flag = torch.tensor(1 if step_ok else 0, device=dev, dtype=torch.int32)
                dist.all_reduce(ok_flag, op=dist.ReduceOp.MIN)
                step_ok = bool(ok_flag.item())
            if step_ok:
                opt.step()
            else:
                n_skipped += 1
                if n_skipped <= 20 or n_skipped % 50 == 0:
                    logline(f"[skip] step {step}: non-finite loss/grad (total skipped={n_skipped})")

            # DEBUG (smoke only): assert the REPLICATED trainable params are BIT-IDENTICAL across
            # ranks after the optimizer step -- proves pure-DDP grad-averaging keeps replicas in
            # sync. Enabled via COSMOS_DDP_PARAM_CHECK=1; a no-op otherwise / in single-GPU.
            if (os.environ.get("COSMOS_DDP_PARAM_CHECK") == "1"
                    and args.ddp and world > 1 and not fsdp_sharded):
                import torch.distributed as dist
                p0 = trainable[0].detach().float().clone()
                ref = p0.clone()
                dist.broadcast(ref, src=0)
                max_diff = (p0 - ref).abs().max().item()
                mism = torch.tensor([1.0 if max_diff > 0 else 0.0], device=dev)
                dist.all_reduce(mism)
                logline(f"[ddp_param_check] step {step}: rank{rank} max|p0-rank0|={max_diff:.3e} "
                        f"total_mismatched_ranks={int(mism.item())}")

            if step % args.log_every == 0:
                peak = torch.cuda.max_memory_allocated() / 1e9
                cur_lr = args.lr * lrf
                mod_str = " ".join(f"{k}={v:.3f}" for k, v in sc.items() if "/" not in k)
                n_done = step - start_step + 1
                logline(f"step {step:6d} loss={loss.item():.4f} {mod_str} lr={cur_lr:.2e} "
                        f"mem={peak:.1f}GB {(time.time()-t0)/n_done:.3f}s/it skip={n_skipped}")
                if tb:
                    tb.add_scalar("loss/total", loss.item(), step)
                    for k, v in sc.items():
                        tb.add_scalar(f"loss/{k}", v, step)
                    tb.add_scalar("lr", cur_lr, step)
                    tb.add_scalar("mem/peak_gb", peak, step)
                    tb.add_scalar("n_skipped", n_skipped, step)

            if step > 0 and step % args.save_every == 0:
                save_ckpt(step)
            if step > 0 and step % args.viz_every == 0:
                # Rank 0 samples/renders while other ranks block in the synchronized helper.
                run_checkpoint_viz(step)

            step += 1
            if step >= args.steps:
                break

    save_ckpt(step)
    # The loop saves periodic checkpoints before incrementing `step`, while the final checkpoint
    # is written at exactly `args.steps`. Visualize that final state explicitly as well.
    run_checkpoint_viz(step)
    logline("[train] done")
    if logf is not None:
        logf.close()
    if args.ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
