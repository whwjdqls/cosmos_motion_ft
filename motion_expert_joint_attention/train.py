"""Train the 7-TASK joint-attention multimodal model on Cosmos-3 Nano (cosmos env).

This generalizes the original text->motion trainer to the full 7-task joint-attention
model (text / image / video / camera / motion in ONE packed sequence). The single per-task
contract lives in ``task_plan.py``; the data seam in ``nymeria_joint_dataset.py``; the
per-modality rectified-flow helpers in ``flow.py``; the gen I/O adapter in ``gen_heads.py``;
and the packed forward (real ``gen_idx`` + per-modality encode/decode + dict output) in
``joint_motion_model.py``. THIS file drives the loop:

  per batch -> read ``mode`` + every present modality -> per sample noise each NOISED modality
  with its PER-MODALITY objective (motion: ``flow.add_noise_x0_masked`` by default -- logit-
  normal sigma, target = x0; ``--objective velocity`` selects the old velocity noiser for
  ablation. vision/camera: ALWAYS ``flow.add_noise_velocity_masked`` -- uniform t, target =
  eps - x0, the pretrained Cosmos generator's native rectified flow; clean condition frames
  pass through untouched either way) -> call ``model.forward(modes=..., x_t=noised_motion,
  video_latents=noised_latents, camera_action=noised_action, ...)`` -> the model encodes via
  gen_heads + motion_heads, runs the shared joint attention, decodes each SUPERVISED modality
  -> apply the per-task flow losses (motion: feat + decoded joint + smooth on x0_hat; vision:
  latent flow MSE; camera: flow MSE chan[:9] x10) and SUM only the supervised modalities ->
  backward -> log per-modality + per-task scalars.

The 7 tasks (``task_plan.TASKS``):
  inverse_dynamics  video                 -> camera     (no text)
  forward_dynamics  camera + text + image -> video
  policy            text + image          -> camera + video
  text2motion       text                  -> motion     (existing trained path)
  textimg2motion    text + image          -> motion
  motimg2video      motion + text + image -> video
  video2motion      video                 -> motion     (no text)

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
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import flow
import task_plan as TP
from cosmos_loader import FrozenCosmos
from decode_uniego_torch import decode_joints
from joint_motion_model import JointMotionModel
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
                    help="subset of task_plan.TASKS to train (default: all 7). Tasks not listed "
                         "get zero mixture weight.")
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
    ap.add_argument("--batch_size", type=int, default=config.TRAIN_DEFAULTS["batch_size"])
    ap.add_argument("--steps", type=int, default=config.TRAIN_DEFAULTS["steps"])
    ap.add_argument("--T", type=int, default=config.VIDEO_NUM_FRAMES,
                    help="shared window length (4N+1 for the Wan VAE); default 33.")
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
    ap.add_argument("--w_vision", type=float, default=config.TRAIN_DEFAULTS["w_vision"])
    ap.add_argument("--w_camera", type=float, default=config.TRAIN_DEFAULTS["w_camera"])
    ap.add_argument("--cfg_dropout", type=float, default=config.TRAIN_DEFAULTS["cfg_dropout"])
    # train scope toggles (DESIGN_7TASK.md section 5)
    ap.add_argument("--gen_lora", action="store_true",
                    help="inject LoRA on q/k/v/o_proj_moe_gen (generator base stays frozen).")
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
    ap.add_argument("--init_motion", default=None,
                    help="Phase-3 warm-start: load ONLY the motion pathway (_moe_motion + motion "
                         "heads + norm_moe_motion) from a Phase-2 checkpoint by name, strict=False.")
    # misc
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--fp32_master", action="store_true",
                    help="cast trainable params to fp32 master (else keep bf16)")
    ap.add_argument("--out", default=None, help="run name under RUN_ROOT")
    ap.add_argument("--save_every", type=int, default=config.TRAIN_DEFAULTS["save_every"])
    ap.add_argument("--viz_every", type=int, default=config.TRAIN_DEFAULTS["viz_every"])
    ap.add_argument("--viz_n", type=int, default=4)
    ap.add_argument("--viz_steps", type=int, default=50)
    ap.add_argument("--viz_guidance", type=float, default=2.0)
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
    #   VISION/CAMERA -> ALWAYS velocity (flow.add_noise_velocity_masked, uniform t, target =
    #                    eps - x0), matching the pretrained Cosmos generator's native rectified
    #                    flow. args.objective NEVER touches the gen pathway.
    # This split is well-defined because NO task noises motion AND vision/camera in the same
    # sample (task_plan.py: motion-target tasks keep video/image clean; gen-target tasks keep
    # motion clean/absent), so each sample carries exactly ONE noised-objective semantics and
    # ONE per-sample t_or_sigma (sampled up front in step_loss and fed to BOTH the noiser and
    # model.forward -- the invariant whose violation caused the bug above). The checkpoint
    # stores vars(args) (incl. objective), so sample.load_joint_model keeps auto-pairing the
    # MOTION sampler (sample_x0 vs sample_velocity) with the trained motion objective; the
    # gen samplers are velocity unconditionally.
    if args.objective not in ("velocity", "x0"):
        raise SystemExit(f"--objective {args.objective!r} not implemented (choices: velocity, x0)")
    motion_x0 = args.objective == "x0"

    # ---- mutually-exclusive gen scope ----
    config.validate_train_scope({"gen_lora": args.gen_lora, "gen_full": args.gen_full})

    rank, world, local = _ddp_env()
    if args.ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
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
    task_weights = {m: w for m, w in task_weights.items() if w > 0.0}
    if not task_weights:
        raise ValueError("no active tasks (empty positive-weight mixture)")
    active_tasks = list(task_weights)
    log(f"[tasks] active={active_tasks}")
    log(f"[tasks] weights={ {k: round(v, 4) for k, v in task_weights.items()} }")
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

    # ---- 7-task joint model: trainable _moe_motion + heads + (toggled) gen/reasoner adapters ----
    model = JointMotionModel(
        cosmos,
        objective=args.objective,
        motion_intermediate_size=args.motion_intermediate,
        motion_layer_stride=args.motion_layer_stride,
        motion_mrope=args.motion_mrope,
        coupling=args.coupling,
        textimg_condition=args.textimg_condition,
        gen_lora=args.gen_lora,
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
    def _load_ckpt_sd(path):
        """Return the name->tensor param dict from a train.py checkpoint (the trainable_state_dict
        under the ``model`` key; tolerant of a bare state_dict)."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sd = payload.get("model", payload) if isinstance(payload, dict) else payload
        return sd

    if args.init_gen:
        gsd = _load_ckpt_sd(args.init_gen)
        n_load, n_miss, n_shape = model.load_gen_subset(gsd)
        log(f"[init_gen] {args.init_gen}: loaded {n_load} gen keys "
            f"(skipped missing={n_miss} shape-mismatch={n_shape}) from {len(gsd)} ckpt tensors")
    if args.init_motion:
        msd = _load_ckpt_sd(args.init_motion)
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
    def build_ds(split, train, cfg_dropout, max_samples=None):
        return NymeriaJointDataset(
            split=split, num_frames=args.T, task_weights=task_weights,
            bones_text2motion_frac=args.bones_frac, cfg_dropout=cfg_dropout,
            prefer_latents=args.precomputed_latents, latent_root=_lat_root,
            force_on_the_fly=args.force_on_the_fly, train=train,
            reasoner_image_for_textimg=(args.textimg_condition == "reasoner"),
            max_samples=max_samples, seed=rank,
        )

    ds = build_ds("train", train=True, cfg_dropout=args.cfg_dropout)
    sampler = None
    if args.ddp and world > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_joint, drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    log(f"[data] dataset={len(ds)} aligned T={args.T} windows  has_bones={ds.has_bones}")

    # ---- optimizer ----
    # fused AdamW works on plain tensors (single-GPU + pure DDP); it can't mix DTensors, so
    # disable it only under FSDP sharding.
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99),
                            fused=not fsdp_sharded)

    # ---- run dir / logging ----
    out, tb = None, None
    if not args.smoke and _is_rank0(rank):
        name = args.out or f"joint7_{time.strftime('%Y%m%d_%H%M%S')}"
        out = os.path.join(RUN_ROOT, name)
        os.makedirs(out, exist_ok=True)
        json.dump({**vars(args), "trainable_M": n_train / 1e6, "world": world,
                   "active_tasks": active_tasks, "task_weights": task_weights},
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
            for key in ("T", "tasks", "objective", "motion_mrope", "coupling", "textimg_condition"):
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
        input_ids_list = [cosmos.tokenize(c) for c in captions]
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
                r = cosmos.encode_reasoner_image_text(captions[s], img)
                reasoner_inputs[s] = r
                input_ids_list[s] = r["input_ids"].view(1, -1)

        # ---- per-sample flow time/sigma (PER-MODALITY objectives) ------------------------
        # ONE t_or_sigma per sample, shared between EVERY noiser call below and
        # model.forward's t_or_sigma (INVARIANT: the SAME tensor feeds the noising and the
        # timestep embedding -- a past bug fed forward a different t than the noising used).
        # No task noises motion AND vision/camera in the same sample (task_plan.py), so each
        # sample's value carries exactly ONE semantics:
        #   motion-noised samples (text2motion/textimg2motion/video2motion): the MOTION
        #     objective's schedule -- logit-normal sigma for x0 (bs_train recipe), uniform t
        #     for velocity.
        #   gen-noised samples (fwd/inv/policy/motimg2video): ALWAYS uniform t (velocity).
        # Samples whose modality is condition-only get sigma_eff=0 per-token in the noisers,
        # so their t_sample value is inert there.
        t_sample = torch.rand(B, device=dev)
        if motion_x0:
            motion_noised = torch.tensor(
                [TP.build_task_plan(m).motion.supervised for m in modes],
                dtype=torch.bool, device=dev)
            t_sample = torch.where(
                motion_noised, flow.sample_sigma_logitnormal(B, dev), t_sample)

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
            #   x0 (default) -> add_noise_x0_masked  : t_sample rows are logit-normal sigma,
            #                   target_motion = x0 itself.
            #   velocity     -> add_noise_velocity_masked : uniform t, target = eps - x0.
            # Both take (x0, condition_mask, t_or_sigma) positionally and return the same
            # tuple structure; passing t_sample keeps the noiser and forward's t_or_sigma
            # on the SAME per-sample value (t_motion comes back == t_sample).
            motion_noiser = (flow.add_noise_x0_masked if motion_x0
                             else flow.add_noise_velocity_masked)
            x_t, t_motion, target_motion, noised_motion = motion_noiser(
                x0, cond_motion, t_sample)
            # the model's motion encoder expects noisy_frame_mask True=NOISED (loss target).
            noisy_frame_mask = noised_motion & valid           # noised AND valid

        # ---- generator modalities: noise per-sample with the resolved condition_mask -----
        # We feed the model the NOISED latents/actions + the per-token condition_mask is rebuilt
        # internally from the same plan; the target velocity (eps - x0) is kept here for the loss.
        cam_dense = batch["camera_action"]                     # [B,Tc,9] or None
        cam_pad = batch["camera_pad_mask"]                     # [B,Tc] True=pad or None
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

        # t_sample was sampled ONCE up top (per-sample, per-modality-objective semantics) and
        # already fed the motion noiser; the gen noisers below reuse the SAME tensor, and it
        # is what model.forward receives as t_or_sigma.

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
            ts = t_sample[s:s + 1]
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
                cam_dense, cond_cam, t=t_sample)
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
            t_or_sigma=t_sample,
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
        total = torch.zeros((), device=dev)
        per_task = {m: torch.zeros((), device=dev) for m in set(modes)}

        # ---- MOTION loss (feat MSE vs the objective's target + decoded joint + smooth),
        # masked to NOISED frames. target_motion is eps-x0 (velocity) or x0 itself (x0). ----
        motion_pred = out.get("motion_pred")
        if motion_pred is not None and target_motion is not None:
            # supervise only frames that were noised (motion-target tasks), and valid.
            mloss_mask = noisy_frame_mask                      # [B,Tm] True=supervised
            if mloss_mask.any():
                l_feat = masked_mse(motion_pred, target_motion, mloss_mask)
                # decoded geometric terms on x0_hat:
                #   x0 objective       -> x0_hat = the prediction itself (bs_train recipe:
                #                         geometric losses supervise the net output directly).
                #   velocity objective -> x0_hat = x_t - t*v_hat (ODE relation at t).
                l_joint = x0.new_zeros(())
                l_smooth = x0.new_zeros(())
                if args.w_joint > 0 or args.w_smooth > 0:
                    if motion_x0:
                        x0_hat = motion_pred
                    else:
                        tb_ = t_motion.view(-1, *([1] * (x_t.dim() - 1)))
                        x0_hat = x_t - tb_ * motion_pred
                    j_hat = decode_joints(x0_hat * std + mean)             # [B,T,30,3]
                    with torch.no_grad():
                        j_gt = decode_joints(x0 * std + mean).detach()
                    if torch.isfinite(j_hat).all():
                        rel_hat = j_hat - j_hat.mean(dim=2, keepdim=True)
                        rel_gt = j_gt - j_gt.mean(dim=2, keepdim=True)
                        if args.w_joint > 0:
                            l_joint = masked_mse(rel_hat, rel_gt, mloss_mask)
                        if args.w_smooth > 0:
                            vmask = mloss_mask[:, 1:] & mloss_mask[:, :-1]
                            l_smooth = masked_mse(j_hat[:, 1:] - j_hat[:, :-1],
                                                  j_gt[:, 1:] - j_gt[:, :-1], vmask)
                m_total = args.w_feat * l_feat + args.w_joint * l_joint + args.w_smooth * l_smooth
                total = total + m_total
                scalars["motion_feat"] = float(l_feat)
                scalars["motion_joint"] = float(l_joint)
                scalars["motion_smooth"] = float(l_smooth)
                # Logging only: attribute the batch-level motion loss once per active
                # motion-target mode. Do not add it once per sample, which inflates
                # loss/task/<mode> by batch size in single-task Phase-2 runs.
                motion_modes = sorted({
                    modes[s] for s in range(B)
                    if TP.build_task_plan(modes[s]).motion.supervised
                })
                for m in motion_modes:
                    per_task[m] = per_task[m] + m_total / max(1, len(motion_modes))

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
            lv = flow.vision_flow_loss(p_flat, t_flat, noised_frames.view(1, T_lat),
                                       weight=args.w_vision)
            vis_accum.append(lv)
            total = total + lv
            per_task[modes[s]] = per_task[modes[s]] + lv
        if vis_accum:
            scalars["vision"] = float(sum(vis_accum) / len(vis_accum))

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
            lc = flow.camera_flow_loss(cp.unsqueeze(0), tgt[:n].unsqueeze(0), noised_mask,
                                       weight=args.w_camera)
            cam_accum.append(lc)
            total = total + lc
            per_task[modes[s]] = per_task[modes[s]] + lc
        if cam_accum:
            scalars["camera"] = float(sum(cam_accum) / len(cam_accum))

        for m, v in per_task.items():
            scalars[f"task/{m}"] = float(v)
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
                                   "motimg2video") if t in active_tasks]

        model.train()
        ok = True
        ran = 0
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
            if plan.has_gen and no_video and batch["camera_action"] is None:
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
            try:
                model.assert_frozen_grads_zero()
                frozen_ok = True
            except AssertionError as e:
                logline(f"[smoke] '{tmode}' FROZEN-GRAD LEAK: {str(e)[:200]}")
                frozen_ok = False
            fin = bool(torch.isfinite(loss).item())
            logline(f"[smoke] {tmode:16s} loss={loss.item():.4f} finite={fin} "
                    f"train_grad={train_grad:.3f} frozen_ok={frozen_ok} "
                    f"{ {k: round(v, 3) for k, v in sc.items()} }")
            ok = ok and fin and (train_grad > 0) and frozen_ok
            opt.step()
            ran += 1
        if ran == 0:
            logline("[smoke] FAIL: ran 0 tasks")
            ok = False
        print("[smoke] PASS" if ok else "[smoke] FAIL", flush=True)
        return

    # ----------------------------------------------------------------------------------
    # in-train viz: held-out captions -> text2motion sample -> decode -> .npy + manifest
    # ----------------------------------------------------------------------------------
    viz_items = []
    if args.viz_n > 0 and out is not None and "text2motion" in active_tasks:
        try:
            # held-out split: Nymeria uses "test"; BONES uses its val pairs via train=False.
            vds = build_ds("test", train=False, cfg_dropout=0.0, max_samples=4096)
            seen = set()
            half = max(1, args.viz_n // 2)
            # BALANCE across sources so the viz always shows BOTH the Nymeria-test and BONES-test
            # sets: first pass caps each source at `half`, second pass tops up to viz_n.
            for strict in (True, False):
                for i in range(len(vds)):
                    if len(viz_items) >= args.viz_n:
                        break
                    item = vds[i]
                    if item["mode"] != "text2motion" or item.get("motion") is None:
                        continue
                    cap = item["caption"]
                    src = item.get("source", "nymeria")
                    if not cap or cap in seen:
                        continue
                    if strict and sum(v["source"] == src for v in viz_items) >= half:
                        continue
                    seen.add(cap)
                    # keep the held-out GT motion (z-scored, pad rows stripped) so do_viz can
                    # render kimodo-style GT|gen side-by-side (single panel when absent).
                    gt_m = torch.as_tensor(item["motion"]).float()
                    pm = item.get("motion_pad_mask")
                    if pm is not None:
                        gt_m = gt_m[~torch.as_tensor(pm)]
                    viz_items.append({"caption": cap, "source": src, "gt_motion": gt_m,
                                      "neutral_joints": torch.as_tensor(item["neutral_joints"]).float()})
                if len(viz_items) >= args.viz_n:
                    break
            srcs = {}
            for v in viz_items:
                srcs[v["source"]] = srcs.get(v["source"], 0) + 1
            logline(f"[viz] setup: {len(viz_items)} held-out captions by source = {srcs}")
        except Exception as e:
            logline(f"[viz] setup skipped: {type(e).__name__}: {str(e)[:120]}")

    @torch.no_grad()
    def do_viz(step):
        if not viz_items or out is None:
            return
        model.eval()
        vdir = os.path.join(out, f"viz_step{step:06d}")
        os.makedirs(vdir, exist_ok=True)
        manifest = []
        for i, it in enumerate(viz_items):
            nj_v = it["neutral_joints"].to(dev).unsqueeze(0)
            x0v = model.sample(
                caption=it["caption"], neutral_joints=nj_v, T=args.T,
                steps=args.viz_steps, guidance=args.viz_guidance,
                tokenizer=cosmos.tokenize, device=dev,
            )                                            # [1,T,283] normalized
            feat = (x0v[0].float() * std + mean)
            joints = decode_joints(feat.unsqueeze(0))[0].cpu().numpy()  # [T,30,3]
            src = it.get("source", "nymeria")
            tag = it["caption"][:28].replace(" ", "_").replace("/", "_")
            npy_path = os.path.join(vdir, f"{i}_{src}_{tag}.npy")
            np.save(npy_path, joints)
            # decode the held-out GT window (when carried) for the side-by-side left panel.
            gt_joints = gt_path = None
            gtm = it.get("gt_motion")
            if gtm is not None and gtm.numel() > 0:
                try:
                    gt_feat = gtm.to(dev) * std + mean
                    gt_joints = decode_joints(gt_feat.unsqueeze(0))[0].cpu().numpy()
                    gt_path = npy_path.replace(".npy", "_gt.npy")
                    np.save(gt_path, gt_joints)
                except Exception as e:  # noqa: BLE001 -- GT decode is best-effort
                    logline(f"[viz] GT decode skipped: {type(e).__name__}: {str(e)[:100]}")
                    gt_joints = gt_path = None
            try:  # render mp4 IN-ENV (kimodo-style renderer; GT|gen side-by-side when GT exists)
                from render_motion import render_motion_mp4
                render_motion_mp4(joints, npy_path.replace(".npy", ".mp4"), caption=it["caption"],
                                  fps=20, gt_joints=gt_joints)
            except Exception as e:  # noqa: BLE001 -- viz mp4 is best-effort; never break training
                logline(f"[viz] mp4 render skipped: {type(e).__name__}: {str(e)[:100]}")
            manifest.append({"i": i, "source": src, "caption": it["caption"], "joints_npy": npy_path,
                             "gt_joints_npy": gt_path, "T": int(joints.shape[0])})
        json.dump(manifest, open(os.path.join(vdir, "manifest.json"), "w"), indent=2)
        model.train()
        logline(f"[viz] step {step} -> {vdir} ({len(manifest)} clips, decode npy + manifest)")

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
                   "step": step, "args": vars(args), "task_weights": task_weights}
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
                loss = loss * 0.0
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

            if torch.isfinite(loss) and grads_finite(trainable):
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
            if step > 0 and step % args.viz_every == 0 and _is_rank0(rank):
                try:
                    do_viz(step)
                except Exception as e:
                    logline(f"[viz] step {step} failed: {type(e).__name__}: {str(e)[:160]}")

            step += 1
            if step >= args.steps:
                break

    save_ckpt(step)
    logline("[train] done")
    if logf is not None:
        logf.close()
    if args.ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
