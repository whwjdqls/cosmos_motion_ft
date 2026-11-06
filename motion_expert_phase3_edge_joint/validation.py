"""Periodic fixed-cohort evaluation using the live weights on training GPUs.

All ranks sample disjoint cases. Rank zero scores and appends one durable event
to the training JSONL. Evaluation cannot advance the training RNG/data cursor.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from .settings import RECIPE, atomic_json, digest_json, sha256


def add_validation_options(parser):
    parser.add_argument("--validation-config", type=Path,
                        help="Pinned node-local fixed71 inputs and metric assets; omitted disables validation")
    parser.add_argument("--validation-every", type=int, default=10000)
    parser.add_argument("--validation-batch-size", type=int, default=32,
                        help="Sampling cases per GPU; halve on CUDA OOM without changing seeds")
    parser.add_argument("--validation-decode-batch-size", type=int, default=4)
    return parser


def local_path(value):
    from .site_paths import storage_path
    return storage_path(value)


def read_configuration(path):
    path = local_path(path)
    config = json.loads(path.read_text())
    file_keys = ("cases", "tmr_checkpoint", "text_cache", "body_adapter", "vae", "i3d_weights",
                 "i3d_source", "i3d_preprocess_source", "lpips_alexnet", "lpips_linear")
    identity = {}
    for name in file_keys:
        asset = local_path(config[name])
        actual = sha256(asset)
        if actual != config["sha256"][name]:
            raise ValueError(f"Validation asset checksum differs: {name}")
        config[name] = str(asset)
        identity[name] = actual
    stats = local_path(config["tmr_stats"])
    files = sorted(p for p in stats.iterdir() if p.is_file())
    actual = {p.name: sha256(p) for p in files}
    if not actual or actual != config["sha256"]["tmr_stats"]:
        raise ValueError("Validation TMR statistics differ")
    config["tmr_stats"] = str(stats)
    # DreamSim's cache contains weights plus its architecture/configuration.
    cache = local_path(config["dreamsim_cache"])
    if not config["sha256"].get("dreamsim_files"):
        raise ValueError("Pin DreamSim weights/configuration before training")
    for name, expected in config["sha256"]["dreamsim_files"].items():
        if sha256(local_path(cache / name)) != expected:
            raise ValueError(f"DreamSim asset differs: {name}")
    config["dreamsim_cache"] = str(cache)
    records = json.loads(Path(config["cases"]).read_text())
    if (len(records) != 71 or len({r["id"] for r in records}) != 71
            or len({r["uuid"] for r in records}) != 71
            or any(r.get("split") != "test" for r in records)):
        raise ValueError("Expected the fixed71 held-out monitoring cohort")
    for row in records:
        if Path(row["id"]).name != row["id"] or row["id"] in {"", ".", ".."}:
            raise ValueError("Unsafe validation case ID")
        local_path(row["bundle"])
        if not isinstance(row["seed"], int) or not isinstance(row["caption"], str):
            raise ValueError("Each case requires an integer seed and caption")
    identity.update(tmr_stats=actual, dreamsim_files=config["sha256"]["dreamsim_files"],
                    config=sha256(path))
    return config, records, identity


def configure_validation(args):
    RECIPE.pop("validation", None)
    path = getattr(args, "validation_config", None)
    if path is None:
        return
    if RECIPE.get("camera_prediction"):
        raise ValueError("This periodic protocol uses the camera-free image-joint sampler; camera runs need their camera-aware evaluator")
    for name in ("validation_every", "validation_batch_size", "validation_decode_batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    _, _, identity = read_configuration(path)
    RECIPE["validation"] = dict(config=str(local_path(path)), assets=identity,
        every=args.validation_every, sampling_batch_size=args.validation_batch_size,
        decode_batch_size=args.validation_decode_batch_size,
        modes=["video2motion", "motimg2video"], steps=35, guidance=1.0,
        motion_shift=3.0, video_shift=10.0, captions="empty", weights="raw",
        cohort="fixed71_test_monitoring_only_no_checkpoint_selection",
        m2v_scored_rgb_frames=[1, 96], fvd="I3D400_logits_six16frame_segments_per_window")


@contextmanager
def preserve_training_state(model):
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    # DDP owns only its current device. Never initialize other GPUs here (or
    # initialize CUDA during CPU-only configuration/tests).
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None
    hub_directory = torch.hub.get_dir()
    modules = list(model.modules())
    if hasattr(model, "native"):
        modules += list(model.native.modules())
    modes = [(module, module.training) for module in modules]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
        # DreamSim changes this process-global setting while loading DINO.
        # Preserve the training cache and next validation's AlexNet lookup.
        torch.hub.set_dir(hub_directory)


def collective_errors(error):
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Periodic validation failed: " + repr(errors))


def load_case(row, contract):
    from motion_expert_phase3_edge.quantitative import check_bundle
    path = local_path(row["bundle"])
    if sha256(path) != row["bundle_sha256"]:
        raise ValueError(f"Held-out bundle changed: {row['id']}")
    return check_bundle(row, contract)


def sample_cases(model, records, folder, contract, batch_size, device):
    from .sample import sample
    used = []
    for mode in ("video2motion", "motimg2video"):
        pending = [row for row in records if not (folder / mode / (row["id"] + ".npz")).exists()]
        offset, current = 0, batch_size
        while offset < len(pending):
            selected = pending[offset:offset + current]
            arrays = [load_case(row, contract) for row in selected]
            try:
                inputs = {name: torch.from_numpy(np.stack([a[i] for a in arrays])).to(device).float()
                          for i, name in enumerate(("motion", "neutral", "video"))}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = sample(model, mode=mode, caption=[""] * len(selected),
                                    seed=[r["seed"] for r in selected], steps=35, guidance=1., **inputs)
                predictions = output.float().cpu().numpy()
                del inputs, output
            except torch.cuda.OutOfMemoryError:
                # Drop traceback references before retrying; no training state,
                # persistent scheduler or RNG has been consumed by this batch.
                inputs, output = None, None
                gc.collect()
                torch.cuda.empty_cache()
                if len(selected) == 1:
                    raise
                current = max(1, len(selected) // 2)
                print(f"[phase3/validation] {mode}: OOM; retry batch={current}", flush=True)
                continue
            for row, prediction in zip(selected, predictions, strict=True):
                path = folder / mode / (row["id"] + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(f".tmp.{os.getpid()}.npz")
                np.savez(temporary, generated=prediction, seed=row["seed"])
                os.replace(temporary, path)
            used.append(len(selected))
            offset += len(selected)
            print(f"[phase3/validation] {mode}: sampled {offset}/{len(pending)} batch={len(selected)}", flush=True)
    return used


def append_event(log, path, event):
    # A restart after checkpoint-save/evaluation must neither lose nor double-log
    # a completed validation event in the append-only W&B history.
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("kind") == "validation" and row.get("validation_id") == event["validation_id"]:
                    return
    log.write(json.dumps(event, allow_nan=False) + "\n")
    log.flush()


def maybe_validate(args, model, step, fingerprint, rank, world, log):
    policy = RECIPE.get("validation")
    if not policy or not step or step % policy["every"]:
        return
    folder = args.out / "validation" / f"step_{step:09d}"
    folder.mkdir(parents=True, exist_ok=True)
    signature = digest_json(dict(policy=policy, training_fingerprint=fingerprint, step=step))
    error, config, records = None, None, None
    try:
        if (folder / "identity.json").exists():
            if json.loads((folder / "identity.json").read_text())["signature"] != signature:
                raise ValueError("Refusing to mix validation identities")
        elif rank == 0:
            atomic_json(folder / "identity.json", dict(signature=signature, step=step, policy=policy))
        config, records, identity = read_configuration(policy["config"])
        if identity != policy["assets"]:
            raise ValueError("Validation configuration/assets changed since acceptance")
    except Exception as exc:
        error = f"rank{rank}: {type(exc).__name__}: {exc}"
    collective_errors(error)
    started = time.monotonic()
    complete = folder / "COMPLETE.json"
    # All ranks must make the same decision, including resumed completed steps.
    done = complete.is_file() if rank == 0 else None
    if dist.is_initialized():
        values = [done]
        dist.broadcast_object_list(values, src=0)
        done = values[0]
    with preserve_training_state(model):
        if not done:
            error = None
            try:
                batches = sample_cases(model, records[rank::world], folder,
                    model.provenance["phase2_contract"], policy["sampling_batch_size"], next(model.parameters()).device)
                atomic_json(folder / f"rank{rank}_sampling.json", dict(batches=batches, cases=len(records[rank::world])))
            except Exception as exc:
                error = f"rank{rank}: {type(exc).__name__}: {exc}"
            collective_errors(error)
        error = None
        if rank == 0:
            try:
                if not done:
                    from .validation_metrics import score
                    torch.cuda.empty_cache()
                    metrics = score(config, records, folder, model.provenance["phase2_contract"], policy)
                    event = dict(kind="validation", validation_id=signature, step=step,
                                 validation={**metrics, "seconds": time.monotonic() - started, "windows": len(records)})
                    atomic_json(complete, event)
                event = json.loads(complete.read_text())
                if event["validation_id"] != signature:
                    raise ValueError("Completed validation signature differs")
                append_event(log, args.out / "metrics.jsonl", event)
                print("[phase3/validation] " + json.dumps(event), flush=True)
            except Exception as exc:
                error = f"rank0 scoring: {type(exc).__name__}: {exc}"
        collective_errors(error)
    gc.collect()
    torch.cuda.empty_cache()
