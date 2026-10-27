"""Bridge-only DDP training with an explicit compute-node acceptance gate."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import config
from checkpoint import capture_rng_state, restore_rng_state
from motion_expert_phase3_edge_fourtask.data import Phase3Dataset, GlobalDrawSampler, collate, identity
from motion_expert_phase3_edge_fourtask.model import EdgePhase3
from motion_expert_phase3_edge_fourtask.objectives import noise_batch, specialist_loss
from motion_expert_phase3_edge_fourtask.settings import RECIPE, add_paths, atomic_json, digest_json, require_compute_node, sha256

STOP = False


def request_stop(_signal, _frame):
    global STOP
    STOP = True


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(cpu_tree(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def collective_check(ok, message, device):
    flag = torch.tensor(int(ok), device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise RuntimeError(message)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def all_objects(value, world):
    if world == 1:
        return [value]
    output = [None] * world
    dist.all_gather_object(output, value)
    return output


def source_identity(args, dataset, model):
    return {
        "recipe": RECIPE, "dataset": dataset.index["fingerprint"],
        "phase1": model.provenance["phase1_metadata_sha256"],
        "phase2": {**identity(args.phase2), "contract": model.provenance["phase2_contract"]},
        "implementation": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "shared_bridge": sha256(Path(__import__("modality_bridge").__file__)),
        "baseline_dependencies": {name: sha256(Path(__file__).parent.parent / "motion_expert_phase3_edge" / name)
            for name in ("data.py", "model.py", "native.py", "settings.py", "objectives.py")},
    }


def frozen_versions(model):
    return {id(p): p._version for module in (model.native, model.motion) for p in module.parameters()}


def assert_frozen(model, versions):
    for module in (model.native, model.motion):
        for p in module.parameters():
            if p.requires_grad or p.grad is not None or p._version != versions[id(p)]:
                raise RuntimeError("Frozen specialist was updated or received parameter gradients")


def new_optimizer(model):
    return torch.optim.AdamW(model.bridges.parameters(), lr=RECIPE["lr"],
                            betas=tuple(RECIPE["betas"]), weight_decay=0.0)


def lr_for_step(step):
    warmup, total = RECIPE["warmup"], RECIPE["steps"]
    if step <= warmup:
        return RECIPE["lr"] * step / warmup
    return RECIPE["lr"] * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def preflight(*args):
    import sys
    from motion_expert_phase3_edge_fourtask.acceptance import run_preflight
    return run_preflight(*args, helpers=sys.modules[__name__])


def main():
    parser = add_paths(argparse.ArgumentParser())
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--resume", default="auto")
    args = parser.parse_args()
    require_compute_node()
    rank, local_rank, world = (int(os.environ.get(k, default)) for k, default in
                              (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    if 32 % (world * args.microbatch):
        raise ValueError("microbatch * GPUs must divide global batch 32")
    if not args.preflight and world not in (2, 4):
        raise ValueError("Production uses two or four allocated GPUs")
    args.out.mkdir(parents=True, exist_ok=True)
    writer_lock = None
    if rank == 0:
        writer_lock = (args.out / ".writer.lock").open("a+")
        fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    barrier()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dataset = Phase3Dataset(args.cache, args.camera_cache)
    if not dataset.index["full_training_index"]:
        raise RuntimeError("A limited smoke cache cannot authorize production training")
    model = EdgePhase3(args.phase1, args.phase2, device=device)
    expected = dataset.index["source"]
    if expected["mean"] != model.provenance["phase2_contract"]["motion_mean"] or expected["std"] != model.provenance["phase2_contract"]["motion_std"]:
        raise RuntimeError("Prepared dataset normalization differs from the motion checkpoint")
    contract = source_identity(args, dataset, model)
    fingerprint = digest_json(contract)
    if rank == 0:
        atomic_json(args.out / "contract.json", contract)
        atomic_json(args.out / "sources.json", model.provenance)
    optimizer = new_optimizer(model)
    mean = torch.from_numpy(np.load(config.MOTION_STATS_MEAN)).to(device).float()
    std = torch.from_numpy(np.load(config.MOTION_STATS_STD)).to(device).float()
    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
    if args.preflight:
        preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint)
        barrier()
        if world > 1:
            dist.destroy_process_group()
        return
    if args.acceptance is None:
        raise RuntimeError("Production requires --acceptance from the completed GPU preflight")
    acceptance = json.loads(args.acceptance.read_text())
    if (not acceptance.get("passed") or acceptance["fingerprint"] != fingerprint
            or acceptance["world_size"] < 2):
        raise RuntimeError("Missing, stale or non-DDP acceptance gate")
    start_step = 0
    checkpoint_dir = args.out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    resume = None
    if args.resume == "auto":
        pointer = checkpoint_dir / "latest.json"
        if pointer.is_file():
            resume = checkpoint_dir / json.loads(pointer.read_text())["file"]
    elif args.resume != "none":
        resume = Path(args.resume)
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload["fingerprint"] != fingerprint:
            raise RuntimeError("Resume contract differs from specialist/data/recipe identity")
        model.bridges.load_state_dict(payload["bridges"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = payload["step"]
        if payload["next_global_draw"] != start_step * 32:
            raise RuntimeError("Invalid saved sampler cursor")
        if len(payload["rng_by_rank"]) == world:
            restore_rng_state(payload["rng_by_rank"][rank])
        else:
            torch.manual_seed(RECIPE["seed"] + start_step * 65536 + rank)
            torch.cuda.manual_seed_all(RECIPE["seed"] + start_step * 65536 + rank)
            print("[phase3/resume] topology changed: global data cursor retained; noise streams reseeded", flush=True)
        del payload
    else:
        torch.manual_seed(rank)
        torch.cuda.manual_seed_all(rank)
    sampler = GlobalDrawSampler(len(dataset), start_step, RECIPE["steps"], args.microbatch, rank, world)
    kwargs = {"persistent_workers": True, "prefetch_factor": 2, "multiprocessing_context": "spawn"} if args.workers else {}
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
                        pin_memory=True, generator=torch.Generator().manual_seed(rank), **kwargs)
    iterator = iter(loader)
    accum = 32 // (world * args.microbatch)
    model.train()
    versions = frozen_versions(model)
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, request_stop)
    if rank == 0:
        print(f"[phase3/train] fresh bridges or own resume; steps={start_step}->200000 "
              f"world={world} microbatch={args.microbatch} accumulation={accum} global_batch=32", flush=True)
    log = (args.out / "metrics.jsonl").open("a", buffering=1) if rank == 0 else None
    for step in range(start_step + 1, RECIPE["steps"] + 1):
        before = time.monotonic()
        lr = lr_for_step(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(5 + len(RECIPE["tasks"]), device=device)
        for micro in range(accum):
            raw = next(iterator)
            inputs, target = noise_batch(raw, device)
            context = ddp.no_sync() if world > 1 and micro < accum - 1 else nullcontext()
            with context:
                output = ddp(**inputs)
                loss, metrics = specialist_loss(output, target, inputs, mean, std)
                collective_check(torch.isfinite(loss).item(), "Non-finite phase-3 loss; stopping all ranks", device)
                (loss / accum).backward()
            totals[:5] += torch.stack((metrics["total"] / accum,
                                   metrics["motion"] * metrics["motion_count"],
                                   metrics["video"] * metrics["video_count"],
                                   metrics["motion_count"], metrics["video_count"]))
            totals[5:] += torch.tensor([raw["modes"].count(name) for name in RECIPE["tasks"]], device=device)
            del output, loss, inputs, target
        norm = torch.nn.utils.clip_grad_norm_(model.bridges.parameters(), 1.0)
        collective_check(torch.isfinite(norm).item(), "Non-finite bridge gradient; stopping all ranks", device)
        optimizer.step()
        assert_frozen(model, versions)
        stop_epoch = int(os.environ.get("PHASE3_STOP_EPOCH", "0"))
        stop = torch.tensor(int(STOP or (stop_epoch > 0 and time.time() >= stop_epoch)), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        save = step % 10000 == 0 or bool(stop.item()) or step == 200000
        if step % 10 == 0 or step == start_step + 1:
            if world > 1:
                dist.all_reduce(totals)
                totals[0] /= world
            torch.cuda.synchronize()
            if rank == 0:
                values = totals.tolist()
                metrics = {"step": step, "lr": lr, "loss": values[0],
                           "motion_loss": values[1] / max(values[3], 1),
                           "video_loss": values[2] / max(values[4], 1),
                           "video_weighted": 10 * values[2] / max(values[4], 1),
                           "motion_samples": values[3], "video_samples": values[4],
                           "task_samples": dict(zip(RECIPE["tasks"], values[5:], strict=True)),
                           "grad_norm": norm.item(), "seconds": time.monotonic() - before,
                           "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
                print("[phase3/train] " + json.dumps(metrics), flush=True)
                log.write(json.dumps(metrics) + "\n")
                atomic_json(args.out / "status.json", {**metrics, "job": os.environ.get("SLURM_JOB_ID"), "stage": "training"})
        if save:
            rng = all_objects(capture_rng_state(), world)
            if rank == 0:
                name = f"step_{step:09d}.pt" if step % 10000 == 0 else "recovery_latest.pt"
                atomic_checkpoint(checkpoint_dir / name, {
                    "schema": 1, "fingerprint": fingerprint, "step": step,
                    "bridges": model.bridges.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": {"step": step, "lr": lr, "recipe": RECIPE},
                    "rng_by_rank": rng, "next_global_draw": step * 32, "sources": model.provenance,
                })
                atomic_json(checkpoint_dir / "latest.json", {"file": name, "step": step})
                # Only this new run's bridge checkpoints are eligible for pruning.
                for path in checkpoint_dir.glob("step_*.pt"):
                    if path.name not in {name, "step_000100000.pt"}:
                        path.unlink()
            barrier()
        if stop.item():
            break
    if rank == 0:
        atomic_json(args.out / "status.json", {"stage": "stopped" if stop.item() else "complete", "step": step})
        log.close()
    barrier()
    if world > 1:
        dist.destroy_process_group()
    if writer_lock:
        writer_lock.close()


if __name__ == "__main__":
    main()
