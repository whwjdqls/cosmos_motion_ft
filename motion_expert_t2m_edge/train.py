"""Train the seven-layer Nano-style T2M/TI2M expert on frozen Cosmos-3 Edge."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import config
from checkpoint import (
    capture_rng_state,
    latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from data import build_phase2_dataset, collate_joint, normalize_task_weights
from flow import add_noise_x0_masked, sample_training_sigma
from losses import t2m_reconstruction_loss
from model import EdgePhase2MotionExpert
from train_visualization import (
    build_or_load_fixed_suite,
    log_suite_to_wandb,
    render_fixed_suite,
)


_RECOVERY_SIGNAL_RECEIVED = False


def _request_recovery_checkpoint(signum, _frame) -> None:
    global _RECOVERY_SIGNAL_RECEIVED
    _RECOVERY_SIGNAL_RECEIVED = True
    print(
        f"[signal] received {signal.Signals(signum).name}; "
        "will write an atomic recovery checkpoint after the current optimizer step",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--T", type=int, default=config.DEFAULT_T)
    parser.add_argument(
        "--ti2m-frames",
        type=int,
        default=config.TI2M_FRAMES,
        help="valid aligned TI2M frames, padded/loss-masked to --T",
    )
    parser.add_argument(
        "--task-weights",
        type=json.loads,
        default=None,
        help='JSON weights over text2motion/textimg2motion (default: {"text2motion":.75,"textimg2motion":.25})',
    )
    parser.add_argument(
        "--bones-frac",
        type=float,
        default=config.BONES_TEXT2M_FRAC,
        help="fraction of T2M mass routed to BONES; BONES is never used for TI2M",
    )
    parser.add_argument(
        "--reasoner-image-size", type=int, default=config.REASONER_IMAGE_SIZE
    )
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE, help="per GPU")
    parser.add_argument("--grad-accum", type=int, default=config.DEFAULT_GRAD_ACCUM)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--loss-reduction", choices=("pooled", "per_task"), default="pooled")
    parser.add_argument("--loss-guard", choices=("strict", "nano"), default="strict")
    parser.add_argument("--allow-tf32", type=int, choices=(0, 1), default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--w-feat", type=float, default=1.0)
    parser.add_argument(
        "--w-global-rot", type=float, default=0.0,
        help="additional per-coordinate weight for normalized rot6d[270:276], using the full feature-MSE denominator",
    )
    parser.add_argument(
        "--w-global-trans", type=float, default=0.0,
        help="additional per-coordinate weight for normalized translation[276:279], using the full feature-MSE denominator",
    )
    parser.add_argument("--w-joint", type=float, default=10.0)
    parser.add_argument(
        "--joint-loss-mode",
        choices=("centered", "absolute"),
        default="centered",
        help="decoded-joint MSE: remove each frame's centroid, or use absolute decoded coordinates",
    )
    parser.add_argument("--w-smooth", type=float, default=50.0)
    parser.add_argument("--w-contact", type=float, default=0.0)
    parser.add_argument("--w-foot-vel", type=float, default=0.0)
    parser.add_argument("--w-foot-height", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument(
        "--checkpoint-interval-only", action="store_true",
        help="write only regular positive save-every multiples; no rolling/signal/off-interval saves",
    )
    parser.add_argument(
        "--recovery-save-every",
        type=int,
        default=0,
        help="overwrite recovery_latest.pt at this interval; 0 disables rolling recovery",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "cosmos-motion-ft"),
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument(
        "--wandb-service-wait",
        type=float,
        default=float(os.environ.get("WANDB__SERVICE_WAIT", "300")),
        help="seconds to wait for the local wandb-core service on each init attempt",
    )
    parser.add_argument(
        "--wandb-init-attempts",
        type=int,
        default=int(os.environ.get("WANDB_INIT_ATTEMPTS", "3")),
        help="bounded wandb.init attempts using the same persisted run id",
    )
    parser.add_argument(
        "--wandb-init-retry-delay",
        type=float,
        default=float(os.environ.get("WANDB_INIT_RETRY_DELAY", "10")),
        help="seconds between failed wandb.init attempts",
    )
    parser.add_argument(
        "--wandb-tags",
        default=os.environ.get(
            "WANDB_TAGS", "cosmos3-edge,phase2,nymeria,camera-head-v1"
        ),
        help="comma-separated W&B tags",
    )
    parser.add_argument(
        "--require-wandb",
        action="store_true",
        help="fail instead of silently training if W&B initialization or upload fails",
    )
    parser.add_argument("--viz-every", type=int, default=5_000)
    parser.add_argument("--viz-samples-per-task", type=int, default=0)
    parser.add_argument(
        "--viz-bones-samples", type=int, default=2,
        help="additional fixed BONES validation T2M examples for mixed runs with Nymeria visualization; 0 disables",
    )
    parser.add_argument(
        "--viz-source", choices=("nymeria", "bones"), default="nymeria",
        help="fixed held-out suite: Nymeria test T2M/TI2M or BONES validation T2M only",
    )
    parser.add_argument("--viz-steps", type=int, default=35)
    parser.add_argument("--viz-guidance", type=float, default=2.0)
    parser.add_argument("--viz-seed", type=int, default=20260901)
    parser.add_argument("--viz-frame-stride", type=int, default=2)
    parser.add_argument("--viz-at-start", action="store_true")
    parser.add_argument("--no-viz-final", action="store_true")
    parser.add_argument(
        "--require-viz",
        action="store_true",
        help="fail after preserving the latest checkpoint if sampling/rendering fails",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return rank, world, local_rank


def lr_factor(step: int, warmup: int, total: int, minimum: float) -> float:
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def next_batch(iterator, loader, sampler, epoch: int):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def initialize_wandb(
    args: argparse.Namespace,
    *,
    out: Path,
    effective_batch: int,
    world: int,
):
    """Create or resume one run-local W&B stream on rank 0."""

    if args.wandb_mode == "disabled":
        if args.require_wandb:
            raise RuntimeError("--require-wandb cannot be used with --wandb-mode disabled")
        return None
    try:
        import wandb

        run_id_path = out / "wandb_run_id.txt"
        if run_id_path.exists():
            run_id = run_id_path.read_text().strip()
            if not run_id:
                raise RuntimeError(f"empty persisted W&B run id: {run_id_path}")
        else:
            run_id = wandb.util.generate_id()
            temporary = run_id_path.with_suffix(".txt.tmp")
            temporary.write_text(run_id + "\n")
            temporary.replace(run_id_path)
        tags = [tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()]
        wandb_config = json.loads(
            json.dumps(
                {
                    **vars(args),
                    "out": str(out),
                    "effective_batch": int(effective_batch),
                    "world_size": int(world),
                    "architecture_contract": config.architecture_contract(),
                },
                default=str,
            )
        )
        run = None
        for attempt in range(1, args.wandb_init_attempts + 1):
            try:
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity or None,
                    name=args.wandb_run_name or out.name,
                    group=args.wandb_group or None,
                    tags=tags,
                    id=run_id,
                    resume="allow",
                    mode=args.wandb_mode,
                    dir=str(out),
                    config=wandb_config,
                    settings=wandb.Settings(
                        x_service_wait=float(args.wandb_service_wait)
                    ),
                )
                break
            except Exception as error:
                if attempt >= args.wandb_init_attempts:
                    raise
                print(
                    f"[wandb] init attempt {attempt}/{args.wandb_init_attempts} "
                    f"failed: {type(error).__name__}: {str(error)[:240]}; "
                    f"retrying in {args.wandb_init_retry_delay:.1f}s",
                    flush=True,
                )
                try:
                    wandb.teardown(exit_code=1)
                except Exception as teardown_error:
                    print(
                        f"[wandb] teardown after failed init also failed: "
                        f"{type(teardown_error).__name__}: {str(teardown_error)[:160]}",
                        flush=True,
                    )
                time.sleep(args.wandb_init_retry_delay)
        if run is None:
            raise RuntimeError("wandb.init returned no run without raising an exception")
        run.define_metric("train/step")
        run.define_metric("train/*", step_metric="train/step")
        run.define_metric("visualizations/*", step_metric="train/step")
        run.summary["output_directory"] = str(out)
        run.summary["fixed_visualization_samples_per_task"] = int(
            args.viz_samples_per_task
        )
        print(
            f"[wandb] mode={args.wandb_mode} project={args.wandb_project} "
            f"entity={args.wandb_entity or '<default>'} id={run_id} url={run.url}",
            flush=True,
        )
        return run
    except Exception as error:
        if args.require_wandb:
            raise RuntimeError(
                f"required W&B initialization failed: {type(error).__name__}: {error}"
            ) from error
        print(
            f"[wandb] disabled after initialization failure: "
            f"{type(error).__name__}: {str(error)[:240]}",
            flush=True,
        )
        return None


def main() -> None:
    global _RECOVERY_SIGNAL_RECEIVED
    args = parse_args()
    signal.signal(signal.SIGUSR1, _request_recovery_checkpoint)
    rank, world, local_rank = distributed_context()
    device = torch.device("cuda", local_rank)
    if not torch.cuda.is_available():
        raise RuntimeError("Edge Phase 2 requires CUDA")
    if args.smoke:
        # Cover both default Nymeria routes; include BONES only when the
        # explicit ablation fraction is nonzero.
        args.steps = 3 if args.bones_frac > 0.0 else 2
        args.batch_size = 1
        args.grad_accum = 1
        args.num_workers = 0
        args.max_samples = 8
        args.save_every = args.steps
        args.log_every = 1
    if min(args.steps, args.batch_size, args.grad_accum) <= 0:
        raise ValueError("steps, batch-size, and grad-accum must be positive")
    if not 0.0 <= args.bones_frac <= 1.0:
        raise ValueError("bones-frac must be in [0,1]")
    if not 1 <= args.ti2m_frames <= args.T and not (args.smoke and args.ti2m_frames > args.T):
        raise ValueError("ti2m-frames must be in [1,T]")
    if args.reasoner_image_size <= 0:
        raise ValueError("reasoner-image-size must be positive")
    if args.log_every <= 0 or args.save_every <= 0:
        raise ValueError("log-every and save-every must be positive")
    for key in ("w_global_rot", "w_global_trans"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and non-negative")
    if not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("adam-beta2 must be in [0,1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be in [0,1]")
    if args.recovery_save_every < 0:
        raise ValueError("recovery-save-every must be non-negative")
    if args.checkpoint_interval_only and args.recovery_save_every:
        raise ValueError("checkpoint-interval-only requires recovery-save-every=0")
    if args.viz_samples_per_task < 0:
        raise ValueError("viz-samples-per-task must be non-negative")
    if args.viz_bones_samples < 0:
        raise ValueError("viz-bones-samples must be non-negative")
    if args.viz_samples_per_task > 0 and min(
        args.viz_every, args.viz_steps, args.viz_frame_stride
    ) <= 0:
        raise ValueError(
            "viz-every, viz-steps, and viz-frame-stride must be positive when visualization is enabled"
        )
    if args.require_viz and args.viz_samples_per_task <= 0:
        raise ValueError("--require-viz requires --viz-samples-per-task > 0")
    if args.require_wandb and args.wandb_mode == "disabled":
        raise ValueError("--require-wandb requires online or offline W&B mode")
    if args.wandb_service_wait <= 0:
        raise ValueError("--wandb-service-wait must be positive")
    if args.wandb_init_attempts <= 0:
        raise ValueError("--wandb-init-attempts must be positive")
    if args.wandb_init_retry_delay < 0:
        raise ValueError("--wandb-init-retry-delay must be non-negative")
    task_weights = normalize_task_weights(args.task_weights)
    args.task_weights = task_weights
    bones_only = args.bones_frac == 1.0 and set(task_weights) == {"text2motion"}

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    config.validate_artifacts(include_bones=args.bones_frac > 0.0)

    run_name = time.strftime("edge_phase2_7layer_%Y%m%d_%H%M%S")
    out = args.out or (config.RUN_ROOT / "motion_expert_t2m_edge" / run_name)
    out = Path(out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir(exist_ok=True)
        (out / "args.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n"
        )
        (out / "contract.json").write_text(
            json.dumps(config.architecture_contract(), indent=2, sort_keys=True) + "\n"
        )
    if world > 1:
        dist.barrier()

    resume_policy = "explicit"
    if args.resume is not None and str(args.resume) == "auto":
        resume_policy = "auto"
        args.resume = latest_checkpoint(out / "checkpoints")
        if rank == 0:
            if args.resume is None:
                print("[resume] auto: no complete checkpoint found; starting from step 0", flush=True)
            else:
                print(f"[resume] auto: selected {args.resume}", flush=True)
    if rank == 0:
        persisted_args = {
            **vars(args),
            "resume_policy": resume_policy,
            "resolved_resume": None if args.resume is None else str(args.resume),
        }
        (out / "args.json").write_text(
            json.dumps(persisted_args, indent=2, sort_keys=True, default=str) + "\n"
        )

    dataset = build_phase2_dataset(
        split="train",
        train=True,
        num_frames=args.T,
        ti2m_frames=args.ti2m_frames,
        task_weights=task_weights,
        bones_frac=args.bones_frac,
        cfg_dropout=args.cfg_dropout,
        reasoner_image_size=args.reasoner_image_size,
        max_samples=args.max_samples,
        seed=args.seed + rank,
    )
    if args.smoke:
        base_dataset = dataset

        def forced_row(index: int, mode: str, bones_frac: float) -> dict:
            base_dataset._modes = [mode]
            base_dataset._mode_w = [1.0]
            base_dataset._bones_frac = bones_frac
            return base_dataset[index]

        smoke_rows = [
            forced_row(0, "text2motion", 0.0),
            forced_row(1, "textimg2motion", 0.0),
        ]
        if args.bones_frac > 0.0:
            smoke_rows.insert(1, forced_row(2, "text2motion", 1.0))
        del forced_row, base_dataset
        dataset = smoke_rows
        smoke_contract = [(row["mode"], row["source"]) for row in dataset]
        expected = [
            ("text2motion", "nymeria"),
            ("textimg2motion", "nymeria"),
        ]
        if args.bones_frac > 0.0:
            expected.insert(1, ("text2motion", "bones"))
        ti2m_index = next(i for i, row in enumerate(dataset) if row["mode"] == "textimg2motion")
        if smoke_contract != expected or dataset[ti2m_index].get("reasoner_image") is None:
            raise RuntimeError(f"smoke data coverage mismatch: {smoke_contract}")
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
        if world > 1
        else None
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None and not args.smoke,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_joint,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=1, timeout=300)
    loader = DataLoader(**loader_kwargs)
    if len(loader) == 0:
        raise RuntimeError("training loader has zero batches")

    model = EdgePhase2MotionExpert(device=device, dtype=torch.bfloat16, verbose=rank == 0)
    parameters = list(model.trainable_parameters())
    if rank == 0 and config.MOTION_SELF_ATTENTION_PER_GAP:
        extra_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "self_attention_motion" in name
        ]
        optimizer_parameter_ids = {id(parameter) for parameter in parameters}
        if not extra_parameters or any(
            id(parameter) not in optimizer_parameter_ids for parameter in extra_parameters
        ):
            raise RuntimeError("added motion self-attention parameters are missing from the optimizer")
        print(
            f"[architecture] joint_blocks={len(config.MOTION_LAYER_INDICES)} "
            f"self_attention_blocks={len(config.MOTION_SELF_ATTENTION_AFTER_JOINT_LAYERS)} "
            f"after_joint_layers={list(config.MOTION_SELF_ATTENTION_AFTER_JOINT_LAYERS)} "
            f"added_parameters={sum(p.numel() for p in extra_parameters):,} "
            f"total_trainable_parameters={sum(p.numel() for p in parameters):,} "
            "self_attention=motion_only_bidirectional activation_checkpointing=added_blocks",
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        betas=(0.9, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )
    start_step = 0
    resume_epoch = 0
    resume_rng_state = None
    resume_guard_state = {}
    if args.resume is not None:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer)
        saved_args = payload.get("args", {})
        for key, legacy_default in (
            ("loss_reduction", "pooled"),
            ("loss_guard", "strict"),
            ("adam_beta2", 0.95),
            ("allow_tf32", 1),
        ):
            if saved_args.get(key, legacy_default) != getattr(args, key):
                raise RuntimeError(
                    f"refusing resume with training-policy drift for {key}: "
                    f"saved={saved_args.get(key, legacy_default)!r} live={getattr(args, key)!r}"
                )
        if any(tuple(group["betas"]) != (0.9, args.adam_beta2) for group in optimizer.param_groups):
            raise RuntimeError("saved AdamW betas disagree with the requested optimizer policy")
        for key in ("w_feat", "w_joint", "w_smooth", "w_contact", "w_foot_vel", "w_foot_height"):
            if key in saved_args and saved_args[key] != getattr(args, key):
                raise RuntimeError(f"refusing resume with loss-weight drift for {key}")
        for key in ("w_global_rot", "w_global_trans"):
            saved_weight = float(saved_args.get(key, 0.0))
            if saved_weight != getattr(args, key):
                raise RuntimeError(
                    f"refusing resume with global feature loss drift for {key}: "
                    f"saved={saved_weight!r} live={getattr(args, key)!r}"
                )
        # Checkpoints predating this option used centroid-relative joint MSE.
        saved_joint_loss_mode = saved_args.get("joint_loss_mode", "centered")
        if saved_joint_loss_mode != args.joint_loss_mode:
            raise RuntimeError(
                "refusing resume with joint-loss drift: "
                f"saved={saved_joint_loss_mode!r} live={args.joint_loss_mode!r}; "
                "start a separate run for a different joint-loss objective"
            )
        for key in (
            "T",
            "ti2m_frames",
            "task_weights",
            "bones_frac",
            "reasoner_image_size",
        ):
            if key in saved_args and saved_args[key] != getattr(args, key):
                raise RuntimeError(
                    f"refusing resume with data-contract drift for {key}: "
                    f"saved={saved_args[key]!r} live={getattr(args, key)!r}"
                )
        start_step = int(payload["step"])
        saved_extra = payload.get("extra", {})
        resume_epoch = int(saved_extra.get("data_epoch", 0))
        rank_rng = saved_extra.get("rng_state_by_rank")
        if rank_rng is not None:
            if len(rank_rng) != world:
                raise RuntimeError("rank-specific RNG resume requires the saved world size")
            resume_rng_state = rank_rng[rank]
        else:
            resume_rng_state = saved_extra.get("rng_state")
        resume_guard_state = saved_extra.get("loss_guard_state", {})
        if rank == 0:
            print(
                f"[resume] loaded step={start_step} data_epoch={resume_epoch} "
                f"rng_state={'yes' if resume_rng_state else 'legacy/missing'} "
                f"from {args.resume}",
                flush=True,
            )
    wrapped = (
        DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        if world > 1
        else model
    )
    model.train()
    # Track only opt-in FP32 block/interface gains, not the parity I/O heads.
    norm_names = {
        "input_norm_motion", "post_norm_motion", "q_norm_motion", "k_norm_motion",
        "reasoner_k_norm_motion",
    }
    norm_gain_parameters = [
        (name, parameter)
        for name, parameter in model.named_trainable_parameters()
        if name.startswith("layers.") and name.rsplit(".", 2)[-2] in norm_names
        and parameter.dtype == torch.float32
    ]
    norm_gain_initial = (
        torch.cat([p.detach().reshape(-1) for _, p in norm_gain_parameters]).clone()
        if norm_gain_parameters else None
    )
    if rank == 0 and norm_gain_initial is not None:
        print(
            f"[norm-policy] fp32_gain_tensors={len(norm_gain_parameters)} "
            f"fp32_gain_parameters={norm_gain_initial.numel()} "
            "change_reference=current_launch; frozen_reasoner_unchanged=True",
            flush=True,
        )

    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).to(device=device, dtype=torch.float32)
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).to(device=device, dtype=torch.float32)
    effective_batch = args.batch_size * args.grad_accum * world
    if rank == 0:
        print(
            f"[training-policy] loss_reduction={args.loss_reduction} "
            f"adam_betas=0.9/{args.adam_beta2} loss_guard={args.loss_guard} "
            f"motion_io_dtype={config.MOTION_IO_DTYPE} allow_tf32={args.allow_tf32} "
            f"lr_peak={args.lr:g} warmup={args.warmup} min_lr_ratio={args.min_lr_ratio:g}",
            flush=True,
        )
    wandb_run = None
    if rank == 0:
        print(
            f"[loss] global_rot_extra={args.w_global_rot:g} "
            f"global_trans_extra={args.w_global_trans:g} "
            f"global_coordinate_weights={args.w_feat + args.w_global_rot:g}/"
            f"{args.w_feat + args.w_global_trans:g} "
            "slices=270:276/276:279 reduction=full_283_feature_mean",
            flush=True,
        )
        wandb_run = initialize_wandb(
            args, out=out, effective_batch=effective_batch, world=world
        )
    if rank == 0:
        print(
            f"[loss] joint_loss_mode={args.joint_loss_mode} "
            f"feat/joint/smooth={args.w_feat}/{args.w_joint}/{args.w_smooth} "
            f"contact/foot_vel/foot_height={args.w_contact}/{args.w_foot_vel}/{args.w_foot_height}",
            flush=True,
        )
        print(
            f"[train] out={out} world={world} batch/gpu={args.batch_size} "
            f"accum={args.grad_accum} effective_batch={effective_batch} T={args.T} "
            f"ti2m_frames={min(args.ti2m_frames, args.T)} dataset={len(dataset)} "
            f"task_weights={task_weights} bones_frac={args.bones_frac}",
            flush=True,
        )
        total_weight = sum(task_weights.values())
        t2m_mass = task_weights.get("text2motion", 0.0) / total_weight
        print(
            "[train] effective_source_mass "
            f"nymeria_t2m={t2m_mass * (1.0-args.bones_frac):.4f} "
            f"bones_t2m={t2m_mass * args.bones_frac:.4f} "
            f"nymeria_ti2m={task_weights.get('textimg2motion', 0.0)/total_weight:.4f}; "
            "BONES is legacy motion-only UniEgo and is not camera/head-equivalent",
            flush=True,
        )

    viz_items: list[dict] = []
    viz_setup_error = ""
    if rank == 0 and args.viz_samples_per_task > 0:
        try:
            fixed_manifest, viz_items = build_or_load_fixed_suite(
                out,
                T=args.T,
                ti2m_frames=args.ti2m_frames,
                reasoner_image_size=args.reasoner_image_size,
                samples_per_task=args.viz_samples_per_task,
                seed=args.viz_seed,
                source=args.viz_source,
            )
            bones_manifest = None
            if (
                args.viz_source == "nymeria"
                and args.bones_frac > 0.0
                and task_weights.get("text2motion", 0.0) > 0.0
                and args.viz_bones_samples > 0
            ):
                # Append an independent held-out suite without changing the
                # existing Nymeria manifest or consuming training RNG state.
                viz_rng = capture_rng_state()
                try:
                    bones_manifest, bones_items = build_or_load_fixed_suite(
                        out,
                        T=args.T,
                        ti2m_frames=args.ti2m_frames,
                        reasoner_image_size=args.reasoner_image_size,
                        samples_per_task=args.viz_bones_samples,
                        seed=args.viz_seed,
                        source="bones",
                        fixed_subdir="fixed_samples_bones",
                    )
                finally:
                    restore_rng_state(viz_rng)
                viz_items.extend(bones_items)
            viz_counts = {
                f"{source}/{task}": sum(
                    item["source"] == source and item["task"] == task
                    for item in viz_items
                )
                for source, task in sorted({
                    (item["source"], item["task"]) for item in viz_items
                })
            }
            print(
                f"[viz] fixed suite ready: {len(viz_items)} samples "
                f"counts={viz_counts} manifests="
                f"{sorted({item['fixed_sample_manifest'] for item in viz_items})}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.summary["fixed_visualization_contract"] = fixed_manifest[
                    "contract"
                ]
                wandb_run.summary["fixed_visualization_counts"] = viz_counts
                if bones_manifest is not None:
                    wandb_run.summary["fixed_bones_visualization_contract"] = (
                        bones_manifest["contract"]
                    )
        except Exception as error:
            viz_setup_error = f"{type(error).__name__}: {str(error)[:500]}"
            print(f"[viz] fixed-suite setup failed: {viz_setup_error}", flush=True)
    if world > 1 and args.viz_samples_per_task > 0:
        setup_failed = torch.tensor(
            [int(bool(viz_setup_error))], dtype=torch.int32, device=device
        )
        dist.broadcast(setup_failed, src=0)
        if int(setup_failed.item()) and rank != 0:
            viz_setup_error = "fixed-suite setup failed on rank 0"
    if viz_setup_error and args.require_viz:
        raise RuntimeError(f"required visualization setup failed: {viz_setup_error}")

    completed_viz_steps = {
        int(path.parent.name.removeprefix("step_"))
        for path in (out / "visualizations").glob("step_*/manifest.json")
        if path.parent.name.removeprefix("step_").isdigit()
    }

    def run_visualization(step: int, *, reason: str) -> None:
        """Pause training coherently, render locally, then upload all media to W&B."""

        if args.viz_samples_per_task <= 0:
            return
        if step in completed_viz_steps:
            if rank == 0:
                print(
                    f"[viz] step {step} already has a complete manifest; skip duplicate ({reason})",
                    flush=True,
                )
            return
        if world > 1:
            dist.barrier()
        local_error = ""
        fatal_error = False
        if rank == 0 and viz_items:
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            optimizer.zero_grad(set_to_none=True)
            try:
                manifest_path, records = render_fixed_suite(
                    model=model,
                    items=viz_items,
                    output_root=out,
                    step=step,
                    T=args.T,
                    reasoner_image_size=args.reasoner_image_size,
                    sampling_steps=args.viz_steps,
                    guidance=args.viz_guidance,
                    frame_stride=args.viz_frame_stride,
                    device=device,
                )
                counts = {
                    f"{source}/{task}": sum(
                        record["source"] == source and record["task"] == task
                        for record in records
                    )
                    for source, task in sorted({
                        (record["source"], record["task"]) for record in records
                    })
                }
                print(
                    f"[viz] step={step} reason={reason} rendered={counts} "
                    f"manifest={manifest_path}",
                    flush=True,
                )
            except Exception as error:
                local_error = (
                    f"render {type(error).__name__}: {str(error)[:500]}"
                )
                fatal_error = bool(args.require_viz)
                records = []
                print(f"[viz] step={step} failed: {local_error}", flush=True)
            if records and wandb_run is not None:
                try:
                    log_suite_to_wandb(wandb_run, records, step=step)
                    print(
                        f"[wandb] uploaded {len(records)} visualization records at step {step}",
                        flush=True,
                    )
                except Exception as error:
                    upload_error = (
                        f"W&B media upload {type(error).__name__}: {str(error)[:500]}"
                    )
                    local_error = (
                        upload_error if not local_error else f"{local_error}; {upload_error}"
                    )
                    fatal_error = fatal_error or bool(args.require_wandb)
                    print(f"[wandb] step={step} media upload failed: {upload_error}", flush=True)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            if not local_error:
                completed_viz_steps.add(int(step))
        if world > 1:
            failure = torch.tensor(
                [int(bool(local_error)), int(bool(fatal_error))],
                dtype=torch.int32,
                device=device,
            )
            dist.broadcast(failure, src=0)
            dist.barrier()
            fatal_error = bool(int(failure[1].item()))
            if int(failure[0].item()) and rank != 0:
                local_error = "visualization or W&B upload failed on rank 0"
        if fatal_error:
            raise RuntimeError(
                f"required checkpoint visualization failed at step {step}: {local_error}"
            )

    if args.viz_at_start and start_step == 0:
        run_visualization(start_step, reason="start")
    elif args.viz_at_start and rank == 0:
        print(
            f"[viz] resume at step {start_step}; skip unscheduled start visualization",
            flush=True,
        )

    if resume_rng_state is not None:
        restored_rng = restore_rng_state(resume_rng_state)
        if rank == 0:
            print(f"[resume] RNG restore={'PASS' if restored_rng else 'SKIP'}", flush=True)
    if sampler is not None and resume_epoch > 0:
        sampler.set_epoch(resume_epoch)
    data_iterator = iter(loader)
    epoch = resume_epoch
    wall_start = time.time()
    last_rate_wall = wall_start
    last_rate_step = start_step
    seen_conditions: set[tuple[str, str]] = set()
    loss_ema = float(resume_guard_state.get("ema", 1.0))
    guarded_batches = int(resume_guard_state.get("guarded_batches", 0))
    skipped_updates = int(resume_guard_state.get("skipped_updates", 0))
    term_names = (
        "feature", "global_rot", "global_trans", "joint", "smooth",
        "contact", "foot_vel", "foot_height",
    )

    def save_training_checkpoint(path: Path, *, step: int, kind: str) -> None:
        if args.checkpoint_interval_only and (
            kind != "regular" or step <= 0 or step % args.save_every != 0
        ):
            if rank == 0:
                print(f"[checkpoint] skipped {kind} step={step}: interval-only policy", flush=True)
            return
        local_rng = capture_rng_state()
        rank_rng = None
        if args.loss_guard == "nano":
            rank_rng = [None] * world
            if world > 1:
                # Every rank enters this branch on the same checkpoint step.
                dist.all_gather_object(rank_rng, local_rng)
            else:
                rank_rng[0] = local_rng
        if rank != 0:
            return
        extra = {
            "effective_batch": effective_batch,
            "data_epoch": int(epoch),
            "rng_state": local_rng,
            "checkpoint_kind": kind,
            "loss_guard_state": {
                "ema": loss_ema,
                "guarded_batches": guarded_batches,
                "skipped_updates": skipped_updates,
            },
        }
        if rank_rng is not None:
            extra["rng_state_by_rank"] = rank_rng
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            step=step,
            args=vars(args),
            extra=extra,
        )
        print(f"[checkpoint] {kind} step={step} -> {path}", flush=True)
        if wandb_run is not None:
            wandb_run.summary["latest_checkpoint_step"] = int(step)
            wandb_run.summary["latest_checkpoint_kind"] = kind

    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = {
            key: 0.0
            for key in ("loss", *term_names)
        }
        for micro in range(args.grad_accum):
            batch, data_iterator, epoch = next_batch(data_iterator, loader, sampler, epoch)
            if bones_only and (
                any(mode != "text2motion" for mode in batch["mode"])
                or any(source != "bones" for source in batch["source"])
                or any(image is not None for image in batch["reasoner_image"])
            ):
                raise RuntimeError("BONES-only T2M run received a different source/task or an image")
            x0 = batch["motion"].to(device=device, dtype=torch.float32, non_blocking=True)
            pad = batch["motion_pad_mask"].to(device=device, non_blocking=True)
            neutral = batch["neutral_joints"].to(device=device, non_blocking=True)
            sigma = sample_training_sigma(x0.shape[0], device)
            x_sigma, sigma, target, noised = add_noise_x0_masked(x0, pad, sigma)
            reasoner_inputs = model.prepare_conditions(
                batch["caption"],
                modes=batch["mode"],
                reasoner_images=batch["reasoner_image"],
                image_size=args.reasoner_image_size,
            )
            seen_conditions.update(zip(batch["mode"], batch["source"], strict=True))
            sync_context = (
                wrapped.no_sync()
                if world > 1 and micro + 1 < args.grad_accum
                else contextlib.nullcontext()
            )
            with sync_context:
                prediction = wrapped(
                    reasoner_inputs=reasoner_inputs,
                    x_sigma=x_sigma,
                    sigma=sigma,
                    neutral_joints=neutral,
                    motion_pad_mask=pad,
                )
                loss, terms = t2m_reconstruction_loss(
                    prediction,
                    target,
                    noised,
                    mean,
                    std,
                    w_feat=args.w_feat,
                    w_global_rot=args.w_global_rot,
                    w_global_trans=args.w_global_trans,
                    w_joint=args.w_joint,
                    joint_loss_mode=args.joint_loss_mode,
                    w_smooth=args.w_smooth,
                    w_contact=args.w_contact,
                    w_foot_vel=args.w_foot_vel,
                    w_foot_height=args.w_foot_height,
                    fps=config.FPS,
                    modes=batch["mode"],
                    reduction=args.loss_reduction,
                    decoded_nonfinite=("skip" if args.loss_guard == "nano" else "raise"),
                )
                if args.loss_guard == "nano":
                    raw_loss = float(loss.detach())
                    if not math.isfinite(raw_loss) or raw_loss > max(50.0, 25.0 * loss_ema):
                        guarded_batches += 1
                        print(
                            f"[bomb-guard] rank={rank} step={step} micro={micro} "
                            f"raw_loss={raw_loss:g} ema={loss_ema:g}; zeroing rank contribution",
                            flush=True,
                        )
                        loss = (
                            loss * 0.0 if math.isfinite(raw_loss)
                            else torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
                        )
                    else:
                        loss_ema = 0.99 * loss_ema + 0.01 * raw_loss
                elif not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step={step} micro={micro}")
                (loss / args.grad_accum).backward()
            accumulated["loss"] += float(loss.detach()) / args.grad_accum
            for key in term_names:
                accumulated[key] += float(terms[key].detach()) / args.grad_accum

        if any(parameter.grad is not None for parameter in model.frozen_parameters()):
            raise RuntimeError("frozen Edge reasoner or visual tower received a gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        step_ok = bool(torch.isfinite(grad_norm))
        if args.loss_guard == "nano":
            if world > 1:
                # Couple skip decisions and USR1 saves across ranks. No rank
                # can enter a checkpoint RNG collective on its own.
                flags = torch.tensor(
                    [int(step_ok), int(not _RECOVERY_SIGNAL_RECEIVED)],
                    dtype=torch.int32, device=device,
                )
                dist.all_reduce(flags, op=dist.ReduceOp.MIN)
                step_ok = bool(flags[0].item())
                _RECOVERY_SIGNAL_RECEIVED = not bool(flags[1].item())
        elif not step_ok or float(grad_norm) <= 0.0:
            raise FloatingPointError(f"invalid motion gradient norm at step={step}: {grad_norm}")
        factor = lr_factor(step - 1, args.warmup, args.steps, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        if step_ok:
            optimizer.step()
        else:
            skipped_updates += 1
            print(
                f"[nonfinite-guard] rank={rank} step={step}; skipped optimizer update "
                f"(total={skipped_updates})", flush=True,
            )

        if rank == 0 and (step % args.log_every == 0 or step == 1):
            elapsed = max(time.time() - wall_start, 1e-6)
            peak = torch.cuda.max_memory_allocated(device) / (1024**3)
            steps_per_second = (step - start_step) / elapsed
            rate_now = time.time()
            recent_steps_per_second = (step - last_rate_step) / max(rate_now - last_rate_wall, 1e-9)
            last_rate_step = step
            last_rate_wall = rate_now
            norm_metrics = {}
            if norm_gain_initial is not None:
                norm_delta = torch.cat(
                    [p.detach().reshape(-1) for _, p in norm_gain_parameters]
                ) - norm_gain_initial
                norm_metrics = {
                    "train/norm_gain_max_change_since_launch": float(norm_delta.abs().max()),
                    "train/norm_gain_changed_fraction_since_launch": float(
                        (norm_delta != 0).float().mean()
                    ),
                }
                print(
                    f"[norm-gains] step={step} "
                    f"max_change={norm_metrics['train/norm_gain_max_change_since_launch']:.8g} "
                    f"changed_fraction={norm_metrics['train/norm_gain_changed_fraction_since_launch']:.6f}",
                    flush=True,
                )
            print(
                f"[speed] recent_steps/s={recent_steps_per_second:.4f} "
                f"overall_steps/s={steps_per_second:.4f}",
                flush=True,
            )
            print(
                f"[step {step}/{args.steps}] loss={accumulated['loss']:.6f} "
                f"feat={accumulated['feature']:.6f} joint={accumulated['joint']:.6f} "
                f"global_rot={accumulated['global_rot']:.6f} "
                f"global_trans={accumulated['global_trans']:.6f} "
                f"smooth={accumulated['smooth']:.6f} grad={float(grad_norm):.4f} "
                f"contact={accumulated['contact']:.6f} foot_vel={accumulated['foot_vel']:.6f} "
                f"foot_height={accumulated['foot_height']:.6f} "
                f"guarded_batches={guarded_batches} skipped_updates={skipped_updates} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} peak={peak:.2f}GiB "
                f"steps/s={steps_per_second:.4f} "
                f"seen={sorted(seen_conditions)}",
                flush=True,
            )
            if wandb_run is not None:
                metrics = {
                    "train/step": int(step),
                    "train/loss": accumulated["loss"],
                    "train/loss_feature": accumulated["feature"],
                    "train/loss_feature_global_rot": accumulated["global_rot"],
                    "train/loss_feature_global_trans": accumulated["global_trans"],
                    "train/loss_feature_global_extra": (
                        args.w_global_rot * accumulated["global_rot"]
                        + args.w_global_trans * accumulated["global_trans"]
                    ),
                    "train/loss_joint": accumulated["joint"],
                    "train/loss_smooth": accumulated["smooth"],
                    "train/loss_contact": accumulated["contact"],
                    "train/loss_foot_velocity": accumulated["foot_vel"],
                    "train/loss_foot_height": accumulated["foot_height"],
                    "train/guarded_batches_rank0": guarded_batches,
                    "train/skipped_updates": skipped_updates,
                    "train/grad_norm": float(grad_norm),
                    "train/grad_clip_max": float(args.grad_clip),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train/peak_allocated_gib": float(peak),
                    "train/steps_per_second": float(steps_per_second),
                    "train/effective_batch": int(effective_batch),
                    **norm_metrics,
                }
                try:
                    wandb_run.log(metrics)
                    wandb_run.summary["latest_step"] = int(step)
                    wandb_run.summary["latest_loss"] = accumulated["loss"]
                    wandb_run.summary["latest_grad_norm"] = float(grad_norm)
                except Exception as error:
                    if args.require_wandb:
                        raise RuntimeError(
                            f"required W&B scalar logging failed at step {step}: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    print(
                        f"[wandb] scalar log failed at step {step}: "
                        f"{type(error).__name__}: {str(error)[:240]}",
                        flush=True,
                    )
        regular_checkpoint_due = step % args.save_every == 0 or step == args.steps
        if regular_checkpoint_due and (rank == 0 or args.loss_guard == "nano"):
            checkpoint_path = out / "checkpoints" / f"step_{step:09d}.pt"
            save_training_checkpoint(
                checkpoint_path, step=step, kind="regular"
            )
            if args.smoke:
                load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
                print(f"[smoke] strict checkpoint reload PASS: {checkpoint_path}", flush=True)
        recovery_checkpoint_due = (
            args.recovery_save_every > 0
            and step % args.recovery_save_every == 0
        )
        signal_checkpoint_due = bool(_RECOVERY_SIGNAL_RECEIVED)
        if (rank == 0 or args.loss_guard == "nano") and (recovery_checkpoint_due or signal_checkpoint_due):
            if not regular_checkpoint_due:
                save_training_checkpoint(
                    out / "checkpoints" / "recovery_latest.pt",
                    step=step,
                    kind=("signal_recovery" if signal_checkpoint_due else "rolling_recovery"),
                )
            _RECOVERY_SIGNAL_RECEIVED = False
        if args.viz_samples_per_task > 0 and step % args.viz_every == 0:
            run_visualization(step, reason="periodic")

    if args.smoke and world == 1:
        expected_seen = {
            ("text2motion", "nymeria"),
            ("textimg2motion", "nymeria"),
        }
        if args.bones_frac > 0.0:
            expected_seen.add(("text2motion", "bones"))
        if seen_conditions != expected_seen:
            raise RuntimeError(
                f"smoke did not exercise all Phase-2 condition/source paths: {seen_conditions}"
            )
        print(f"[smoke] Phase-2 coverage PASS: {sorted(seen_conditions)}", flush=True)

    if not args.no_viz_final:
        run_visualization(args.steps, reason="final")

    if rank == 0 and wandb_run is not None:
        wandb_run.summary["training_complete"] = True
        wandb_run.finish()

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
