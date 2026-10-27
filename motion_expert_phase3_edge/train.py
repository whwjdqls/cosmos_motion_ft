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
from motion_expert_phase3_edge.data import Phase3Dataset, GlobalDrawSampler, collate, identity
from motion_expert_phase3_edge.model import EdgePhase3
from motion_expert_phase3_edge.objectives import noise_batch, specialist_loss
from motion_expert_phase3_edge.settings import RECIPE, add_paths, atomic_json, digest_json, require_compute_node, sha256

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


def preflight(args, dataset, model, ddp, mean, std, rank, world, device, fingerprint):
    from modality_bridge import LocalModalityBridge
    from motion_expert_phase3_edge.model import bridge_meta

    checks = []
    # Exact causal-Wan interval and directional masks, independently of training.
    for mode in RECIPE["tasks"]:
        meta = bridge_meta(mode, device)
        bridge = LocalModalityBridge(8, 2, 4).to(device)
        mask = bridge._attention_mask(meta)
        assert meta.gen_source_start[64].item() == 1
        assert meta.gen_source_end[-1].item() == 96
        assert bridge._source_interval_pair_mask(meta)[-1, -1].item()
        if mode == "video2motion":
            assert not mask[:1600, 1600:].any()
        else:
            assert not mask[1600:, :1600].any()
        # Clean opposite modality cannot depend on noisy target values.
        with torch.no_grad():
            bridge.gate_g.fill_(0.1)
            bridge.gate_m.fill_(0.1)
            g = torch.randn(1600, 8, device=device)
            m = torch.randn(98, 8, device=device)
            ga, ma = bridge(g, m, meta)
            gb, mb = bridge(g, m + 1 if mode == "video2motion" else m, meta) if mode == "video2motion" else bridge(g + 1, m, meta)
            torch.testing.assert_close(ga if mode == "video2motion" else ma,
                                       gb if mode == "video2motion" else mb, rtol=0, atol=0)
        del bridge, mask
    checks.append("causal_Wan_intervals_and_clean_cross_modal_isolation")
    rows = [dataset[(i, (rank * args.microbatch + i) % len(dataset))] for i in range(args.microbatch)]
    batch = collate(rows)
    versions = frozen_versions(model)
    model.eval()
    with torch.no_grad():
        for mode in RECIPE["tasks"]:
            batch["modes"] = [mode] * len(rows)
            batch["captions"] = ["" if mode == "video2motion" else "The camera wearer walks forward."] * len(rows)
            inputs, target = noise_batch(batch, device)
            actual = model(**inputs)
            if mode == "video2motion":
                baseline = model.motion(
                    reasoner_inputs=model.motion.prepare_conditions([""] * len(rows)),
                    x_sigma=inputs["motion"], sigma=inputs["motion_sigma"],
                    neutral_joints=inputs["neutral"],
                    motion_pad_mask=torch.zeros(len(rows), 97, dtype=torch.bool, device=device))
                torch.testing.assert_close(actual["motion"], baseline, rtol=0, atol=0)
            else:
                reference = model.native(actual["packed"])["preds_vision"]
                baseline = torch.stack([v.reshape(48, 25, 16, 16) for v in reference])
                torch.testing.assert_close(actual["video"], baseline, rtol=0, atol=0)
                _, terms = specialist_loss(actual, target, inputs, mean, std)
                mask = torch.ones_like(baseline)
                mask[:, :, 0] = 0
                expected = ((baseline.float() - target["velocity"]).square() * mask).mean()
                torch.testing.assert_close(terms["video"], expected, rtol=1e-6, atol=1e-7)
            bypass = model(**inputs, bridges_enabled=False)
            torch.testing.assert_close(actual["motion"], bypass["motion"], rtol=0, atol=0)
            torch.testing.assert_close(actual["video"], bypass["video"], rtol=0, atol=0)
            del actual, baseline, bypass
    checks.append("bit_exact_zero_bridge_phase1_and_phase2_parity")
    checks.append("native_video_weight10_full_denominator")
    # Fixed mixed batch, including both tasks even for microbatch=1 across DDP ranks.
    batch["modes"] = ["video2motion" if (rank * len(rows) + i) % 2 == 0 else "motimg2video"
                      for i in range(len(rows))]
    batch["captions"] = ["" if m == "video2motion" else "The camera wearer walks forward." for m in batch["modes"]]
    inputs, target = noise_batch(batch, device)
    optimizer = new_optimizer(model)
    model.train()
    losses = []
    for step in range(12):
        optimizer.zero_grad(set_to_none=True)
        output = ddp(**inputs)
        loss, _ = specialist_loss(output, target, inputs, mean, std)
        collective_check(torch.isfinite(loss).item(), "Non-finite acceptance loss", device)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.bridges.parameters(), 1.0)
        collective_check(torch.isfinite(norm).item(), "Non-finite acceptance gradients", device)
        gates = [p for name, p in model.bridges.named_parameters() if "gate_" in name]
        core = [p for name, p in model.bridges.named_parameters() if "gate_" not in name]
        if step == 0:
            assert any(p.grad is not None and p.grad.abs().max().item() > 0 for p in gates)
            assert all(p.grad is not None and p.grad.abs().max().item() == 0 for p in core)
        if step == 11:
            assert any(p.grad is not None and p.grad.abs().max().item() > 0 for p in core)
        optimizer.step()
        assert_frozen(model, versions)
        value = loss.detach().clone()
        if world > 1:
            dist.all_reduce(value)
            value /= world
        losses.append(value.item())
        if rank == 0:
            print(f"[phase3/preflight] disposable_overfit={step+1}/12 loss={value.item():.7f}", flush=True)
        del output, loss
    if losses[-1] >= losses[0]:
        raise RuntimeError(f"Disposable overfit did not improve: {losses[0]} -> {losses[-1]}")
    checks.extend(["finite_mixed_task_DDP_backward", "gate_then_core_gradients",
                   "specialists_frozen", "disposable_fixed_batch_overfit"])
    # Save/reload with real optimizer and per-rank RNG; never publish smoke weights as production.
    rng = all_objects(capture_rng_state(), world)
    checkpoint_path = args.out / "preflight_roundtrip.pt"
    expected_state = {key: value.detach().cpu().clone() for key, value in model.bridges.state_dict().items()}
    if rank == 0:
        atomic_checkpoint(checkpoint_path, {"bridges": expected_state, "optimizer": optimizer.state_dict(), "rng": rng})
    barrier()
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    model.bridges.load_state_dict(restored["bridges"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    restore_rng_state(restored["rng"][rank])
    for key, value in model.bridges.state_dict().items():
        assert torch.equal(value.detach().cpu(), expected_state[key])
    digest = hashlib.sha256()
    for value in model.bridges.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    assert len(set(all_objects(digest.hexdigest(), world))) == 1
    del restored, expected_state
    # One real resumed optimizer update, not just deserialization.
    optimizer.zero_grad(set_to_none=True)
    output = ddp(**inputs)
    loss, _ = specialist_loss(output, target, inputs, mean, std)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.bridges.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    assert_frozen(model, versions)
    checks.append("DDP_bit_identical_bridge_parameters_and_save_reload_continue")
    barrier()
    if rank == 0:
        checkpoint_path.unlink()
        atomic_json(args.out / "acceptance.json", {
            "passed": True, "fingerprint": fingerprint, "checks": checks,
            "world_size": world, "microbatch": args.microbatch,
            "overfit_losses": losses, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "production_weights": "fresh_zero_gated_bridges_not_preflight_weights",
        })
        print("[phase3/preflight] PASSED; disposable weights are not used for production", flush=True)


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
    dataset = Phase3Dataset(args.cache)
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
        totals = torch.zeros(5, device=device)
        for micro in range(accum):
            raw = next(iterator)
            inputs, target = noise_batch(raw, device)
            context = ddp.no_sync() if world > 1 and micro < accum - 1 else nullcontext()
            with context:
                output = ddp(**inputs)
                loss, metrics = specialist_loss(output, target, inputs, mean, std)
                collective_check(torch.isfinite(loss).item(), "Non-finite phase-3 loss; stopping all ranks", device)
                (loss / accum).backward()
            totals += torch.stack((metrics["total"] / accum,
                                   metrics["motion"] * metrics["motion_count"],
                                   metrics["video"] * metrics["video_count"],
                                   metrics["motion_count"], metrics["video_count"]))
            del output, loss, inputs, target
        norm = torch.nn.utils.clip_grad_norm_(model.bridges.parameters(), 1.0)
        collective_check(torch.isfinite(norm).item(), "Non-finite bridge gradient; stopping all ranks", device)
        optimizer.step()
        assert_frozen(model, versions)
        stop = torch.tensor(int(STOP), device=device)
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
        atomic_json(args.out / "status.json", {"stage": "stopped" if STOP else "complete", "step": step})
        log.close()
    barrier()
    if world > 1:
        dist.destroy_process_group()
    if writer_lock:
        writer_lock.close()


if __name__ == "__main__":
    main()
