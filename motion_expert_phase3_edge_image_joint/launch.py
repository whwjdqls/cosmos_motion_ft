"""Explicit two-GPU Foundary launch; never queue, transfer, or stop other jobs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import importlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from motion_expert_phase3_edge_joint.adapters import add_training_options, configure_training
from motion_expert_phase3_edge_joint.settings import add_bridge_options


def gpu_rows():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    return {int(index): uuid.strip() for index, uuid in (line.split(",", 1) for line in text.splitlines())}


def require_empty(uuids):
    for uuid in uuids:
        pids = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if pids:
            raise RuntimeError(f"GPU {uuid} is occupied (PIDs {pids.replace(chr(10), ', ')}); not queued")


def freeze_sources(repo, snapshot):
    from motion_expert_phase3_edge_joint.settings import sha256
    # Include lazily loaded evaluators: later repository edits must not change
    # the definition of the 10k/20k/... points within a running experiment.
    packages = ("motion_expert_phase3_edge_joint", "motion_expert_phase3_edge_image_joint", "motion_expert_phase3_edge_mmdit",
                "motion_expert_joint_attention", "motion_expert_t2m_edge", "motion_expert_phase3_edge",
                "native_phase_training", "nymeria_world", "nymeria_baselines", "shape_aware_TMR", "migration")
    sources = sorted({p for name in packages for p in (repo / name).rglob("*.py")})
    sources += [repo / name for name in ("runtime_paths.py", "scripts/phase3_performance.py", "scripts/stream_phase3_wandb.py")]
    manifest = {str(p.relative_to(repo)): sha256(p) for p in sources}
    for path in sources:
        destination = snapshot / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return manifest


def main(experiment_package="motion_expert_phase3_edge_image_joint"):
    parser = add_training_options(add_bridge_options(argparse.ArgumentParser(description=__doc__)))
    parser.add_argument("--gpus", required=True, help="Two coordinated physical GPU indices, e.g. 2,3")
    for key in ("phase1", "phase2", "cache", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Check local inputs/resources without creating a run or using a GPU")
    parser.add_argument("--wandb-entity", default="jungbinc-upenn")
    parser.add_argument("--wandb-project", default="cosmos-motion-ft")
    args = parser.parse_args()
    host = socket.gethostname().split(".")[0]
    if host not in {"lg-gwangmo-4xh200-sep-1", "lg-gwangmo-4xh200-sep-2"}:
        raise RuntimeError(f"This launcher is for the two Foundary nodes, got {host}")
    for key in ("phase1", "phase2", "cache", "out"):
        path = getattr(args, key)
        if not path.is_absolute() or not path.resolve().is_relative_to("/mnt/local"):
            raise ValueError(f"--{key} must be an absolute node-local /mnt/local path")
        if key != "out" and not path.exists():
            raise FileNotFoundError(f"Missing node-local --{key}: {path}; stage the exact dependency on this node first")
    if not (args.cache / "index.json").is_file() or not args.phase2.is_file():
        raise ValueError("Need the validated Phase3 cache and selected Phase2 checkpoint file")
    if args.workers < 0:
        raise ValueError("--workers must be nonnegative")
    indices = [int(x) for x in args.gpus.split(",")]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError("Specify exactly two distinct coordinated GPUs")
    available = gpu_rows()
    uuids = [available[i] for i in indices]
    os.environ.update(CUDA_VISIBLE_DEVICES=",".join(uuids), FOUNDRY_GPU_UUIDS=",".join(uuids))
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    from native_phase_training.compute_access import require_compute_access
    require_compute_access()
    configure = importlib.import_module(experiment_package + ".policy").configure
    recipe = configure()
    configure_training(args)
    from .train import configure_local_paths
    path_map = configure_local_paths()
    Phase3Dataset = importlib.import_module(experiment_package + ".data").Phase3Dataset
    dataset = Phase3Dataset(args.cache)
    if not dataset.index["full_training_index"]:
        raise RuntimeError("A smoke/limited cache cannot authorize production")
    for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        row = dataset[(index, index)]
        if tuple(row["motion"].shape) != (97, 283) or tuple(row["video"].shape) != (48, 25, 16, 16):
            raise ValueError("Wrong cache representation or temporal shape")
    del dataset
    require_empty(uuids)
    plan = {"stage": "checked_not_started", "host": host, "gpus": indices, "gpu_uuids": uuids,
            "phase1": str(args.phase1), "phase2": str(args.phase2), "cache": str(args.cache),
            "out": str(args.out), "recipe": recipe, "path_map": path_map, "global_batch": 32,
            "microbatch": 16, "accumulation": 1, "gpu_acceptance": "pending"}
    if args.check_only:
        print(json.dumps(plan, indent=2))
        return
    if args.out.exists():
        raise FileExistsError("Fresh experiment requires a new --out; resume separately using its own contract and checkpoint")
    existing_parent = args.out.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < 32 * 2**30:
        raise RuntimeError("Need at least 32 GiB of free local storage for checkpoints/probes")
    with ExitStack() as stack:
        lock_root = Path(os.environ.get("FOUNDRY_LOCAL_ROOT", "/mnt/local/" + os.environ["USER"])) / "locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        for uuid in sorted(uuids):
            lock = stack.enter_context((lock_root / (uuid + ".lock")).open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_empty(uuids)
        args.out.mkdir(parents=True)
        ops = args.out / "launch"
        ops.mkdir()
        from motion_expert_phase3_edge_joint.settings import atomic_json, sha256
        atomic_json(ops / "plan.json", plan)
        repo = Path(__file__).resolve().parents[1]
        manifest = freeze_sources(repo, ops / "source_snapshot")
        atomic_json(ops / "source_manifest.json", manifest)
        snapshot = ops / "source_snapshot"
        child_environment = os.environ.copy()
        child_environment["PYTHONPATH"] = os.pathsep.join([
            str(snapshot / "motion_expert_t2m_edge"), str(snapshot / "motion_expert_joint_attention"),
            str(snapshot / "nymeria_world"), str(snapshot), child_environment.get("PYTHONPATH", "")])
        child_cwd = os.environ["COSMOS_FRAMEWORK_EDGE_ROOT"]
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
                   "-m", experiment_package + ".train"]
        common = ["--phase1", str(args.phase1), "--phase2", str(args.phase2), "--cache", str(args.cache),
                  "--microbatch", "16", "--workers", str(args.workers)]
        for name in ("phase3_lora_rank", "phase3_lora_alpha", "phase3_lora_lr", "phase3_lora_weight_decay",
                     "bridge_lr", "bridge_weight_decay", "ema_decay", "bridge_parameterization",
                     "bridge_attention_backend", "bridge_precision", "validation_config", "validation_every",
                     "validation_batch_size", "validation_decode_batch_size"):
            value = getattr(args, name)
            if value is not None:
                common += ["--" + name.replace("_", "-"), str(value)]

        def stage(name, extra):
            require_empty(uuids)
            if any(sha256(repo / relative) != expected for relative, expected in manifest.items()):
                raise RuntimeError("Source files changed during launch; use the saved snapshot")
            atomic_json(ops / "status.json", {"stage": name, "host": host})
            print(f"[image-joint/launch] {name}: {ops / (name + '.log')}", flush=True)
            with (ops / (name + ".log")).open("x") as log:
                subprocess.run(command + common + extra, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=child_environment, cwd=child_cwd)

        stage("preflight", ["--preflight", "--out", str(ops / "preflight")])
        acceptance = ["--acceptance", str(ops / "preflight/acceptance.json")]
        stage("fresh_probe", ["--out", str(ops / "fresh_probe"), "--resume", "none", "--probe-updates", "4", *acceptance])
        stage("reload_probe", ["--out", str(ops / "reload_probe"), "--resume", str(ops / "fresh_probe/checkpoints/recovery_latest.pt"),
                               "--probe-updates", "2", *acceptance])
        require_empty(uuids)
        atomic_json(ops / "status.json", {"stage": "starting_fresh_production", "host": host})
        with (ops / "training.log").open("x") as log:
            trainer = subprocess.Popen(command + common + ["--out", str(args.out), "--resume", "none", *acceptance],
                                       stdout=log, stderr=subprocess.STDOUT, env=child_environment, cwd=child_cwd)
            logger = None
            try:
                while not (args.out / "metrics.jsonl").is_file():
                    if trainer.poll() is not None:
                        raise RuntimeError(f"Trainer exited with {trainer.returncode}; see {ops / 'training.log'}")
                    time.sleep(2)
                import hashlib
                run_id = "p3-" + hashlib.sha256(str(args.out.resolve()).encode()).hexdigest()[:20]
                with (ops / "wandb.log").open("x") as wandb_log:
                    logger = subprocess.Popen([sys.executable, str(snapshot / "scripts/stream_phase3_wandb.py"),
                        "--run-dir", str(args.out), "--state-dir", str(ops / "wandb_state"),
                        "--name", args.out.name, "--poll-seconds", "2",
                        "--variant", recipe["experiment"], "--group", "edge-phase3-mmdit",
                        "--entity", args.wandb_entity, "--project", args.wandb_project], stdout=wandb_log, stderr=subprocess.STDOUT,
                        env=child_environment, cwd=child_cwd)
                    url = f"https://wandb.ai/{args.wandb_entity}/{args.wandb_project}/runs/{run_id}"
                    atomic_json(ops / "status.json", {"stage": "trainer_started", "host": host, "pid": trainer.pid, "wandb_url": url})
                    print(f"[image-joint/launch] trainer started; W&B {url} (check logger for remote sync)", flush=True)
                    code = trainer.wait()
                    if code == 0:
                        # Drain final training AND validation events before closing.
                        logger.wait(timeout=60)
                if code:
                    raise RuntimeError(f"Trainer exited with {code}; see {ops / 'training.log'}")
            finally:
                if logger is not None and logger.poll() is None:
                    logger.terminate()
                    logger.wait(timeout=60)
        atomic_json(ops / "status.json", {"stage": "trainer_exited", "host": host, "code": 0})


if __name__ == "__main__":
    main()
