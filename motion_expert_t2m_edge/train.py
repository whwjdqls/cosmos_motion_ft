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
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--w-feat", type=float, default=1.0)
    parser.add_argument("--w-joint", type=float, default=10.0)
    parser.add_argument("--w-smooth", type=float, default=50.0)
    parser.add_argument("--w-contact", type=float, default=0.0)
    parser.add_argument("--w-foot-vel", type=float, default=0.0)
    parser.add_argument("--w-foot-height", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=5_000)
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
        )
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
    if args.recovery_save_every < 0:
        raise ValueError("recovery-save-every must be non-negative")
    if args.viz_samples_per_task < 0:
        raise ValueError("viz-samples-per-task must be non-negative")
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
    task_weights = normalize_task_weights(args.task_weights)
    args.task_weights = task_weights

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
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
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True,
    )
    start_step = 0
    resume_epoch = 0
    resume_rng_state = None
    if args.resume is not None:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer)
        saved_args = payload.get("args", {})
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
        resume_rng_state = saved_extra.get("rng_state")
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

    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).to(device=device, dtype=torch.float32)
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).to(device=device, dtype=torch.float32)
    effective_batch = args.batch_size * args.grad_accum * world
    wandb_run = None
    if rank == 0:
        wandb_run = initialize_wandb(
            args, out=out, effective_batch=effective_batch, world=world
        )
    if rank == 0:
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
            )
            print(
                f"[viz] fixed suite ready: {len(viz_items)} samples "
                f"({args.viz_samples_per_task} per task), "
                f"manifest={out / 'visualizations' / 'fixed_samples' / 'manifest.json'}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.summary["fixed_visualization_contract"] = fixed_manifest[
                    "contract"
                ]
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
                    task: sum(record["task"] == task for record in records)
                    for task in ("text2motion", "textimg2motion")
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
    seen_conditions: set[tuple[str, str]] = set()

    def save_training_checkpoint(path: Path, *, step: int, kind: str) -> None:
        extra = {
            "effective_batch": effective_batch,
            "data_epoch": int(epoch),
            "rng_state": capture_rng_state(),
            "checkpoint_kind": kind,
        }
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
        accumulated = {key: 0.0 for key in ("loss", "feature", "joint", "smooth")}
        for micro in range(args.grad_accum):
            batch, data_iterator, epoch = next_batch(data_iterator, loader, sampler, epoch)
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
                    w_joint=args.w_joint,
                    w_smooth=args.w_smooth,
                    w_contact=args.w_contact,
                    w_foot_vel=args.w_foot_vel,
                    w_foot_height=args.w_foot_height,
                    fps=config.FPS,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step={step} micro={micro}")
                (loss / args.grad_accum).backward()
            accumulated["loss"] += float(loss.detach()) / args.grad_accum
            for key in ("feature", "joint", "smooth"):
                accumulated[key] += float(terms[key].detach()) / args.grad_accum

        if any(parameter.grad is not None for parameter in model.frozen_parameters()):
            raise RuntimeError("frozen Edge reasoner or visual tower received a gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        if not torch.isfinite(grad_norm) or float(grad_norm) <= 0.0:
            raise FloatingPointError(f"invalid motion gradient norm at step={step}: {grad_norm}")
        factor = lr_factor(step - 1, args.warmup, args.steps, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        optimizer.step()

        if rank == 0 and (step % args.log_every == 0 or step == 1):
            elapsed = max(time.time() - wall_start, 1e-6)
            peak = torch.cuda.max_memory_allocated(device) / (1024**3)
            steps_per_second = (step - start_step) / elapsed
            print(
                f"[step {step}/{args.steps}] loss={accumulated['loss']:.6f} "
                f"feat={accumulated['feature']:.6f} joint={accumulated['joint']:.6f} "
                f"smooth={accumulated['smooth']:.6f} grad={float(grad_norm):.4f} "
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
                    "train/loss_joint": accumulated["joint"],
                    "train/loss_smooth": accumulated["smooth"],
                    "train/grad_norm": float(grad_norm),
                    "train/grad_clip_max": float(args.grad_clip),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train/peak_allocated_gib": float(peak),
                    "train/steps_per_second": float(steps_per_second),
                    "train/effective_batch": int(effective_batch),
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
        if rank == 0 and regular_checkpoint_due:
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
        if rank == 0 and (recovery_checkpoint_due or signal_checkpoint_due):
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
