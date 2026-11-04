"""Explicit one/two-GPU Foundary launch; never share or stop other jobs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import hashlib

from .execution import topology


def gpu_rows():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    return {int(index): uuid.strip() for index, uuid in (line.split(",", 1) for line in text.splitlines())}


def require_empty(uuids):
    for uuid in uuids:
        pids = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if pids:
            raise RuntimeError(f"GPU {uuid} is occupied (PIDs {pids.replace(chr(10), ', ')}); not queued")


def wait_for_release(uuids, timeout=60):
    """Allow completed probe CUDA contexts to disappear before the next stage."""
    deadline = time.monotonic() + timeout
    reported = False
    while True:
        try:
            require_empty(uuids)
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            if not reported:
                print("[camera/launch] waiting for GPU processes to release between stages", flush=True)
                reported = True
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="One or two coordinated physical GPU indices, e.g. 2 or 2,3")
    for key in ("phase1", "phase2", "cache", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--camera-cache", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Check local inputs/resources without creating a run or using a GPU")
    parser.add_argument("--wandb-entity", default="jungbinc-upenn")
    parser.add_argument("--wandb-project", default="cosmos-motion-ft")
    args = parser.parse_args()
    host = socket.gethostname().split(".")[0]
    if host not in {"lg-gwangmo-4xh200-sep-1", "lg-gwangmo-4xh200-sep-2"}:
        raise RuntimeError(f"This launcher is for the two Foundary nodes, got {host}")
    for key in ("phase1", "phase2", "cache", "out", "camera_cache", "reference_contract"):
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
    if len(indices) not in (1, 2) or len(set(indices)) != len(indices):
        raise ValueError("Specify one or two distinct coordinated GPUs")
    execution = topology(len(indices), 16)
    available = gpu_rows()
    uuids = [available[i] for i in indices]
    os.environ.update(CUDA_VISIBLE_DEVICES=",".join(uuids), FOUNDRY_GPU_UUIDS=",".join(uuids))
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    from native_phase_training.compute_access import require_compute_access
    require_compute_access()
    os.environ['PHASE3_CAMERA_PREDICTION_CACHE'] = str(args.camera_cache)
    from .policy import configure
    recipe = configure()
    reference = json.loads(args.reference_contract.read_text())
    mismatches = [key for key,value in reference['recipe'].items() if key != 'experiment' and recipe.get(key) != value]
    if mismatches:
        raise ValueError('Baseline recipe changed: '+', '.join(mismatches))
    from motion_expert_phase3_edge_joint.settings import sha256
    repo = Path(__file__).resolve().parents[1]
    for key,folder in [('implementation','motion_expert_phase3_edge_joint'),('image_joint_implementation','motion_expert_phase3_edge_image_joint')]:
        for name,expected in reference[key].items():
            if sha256(repo/folder/name) != expected:
                raise ValueError('Shared source differs from running baseline: '+folder+'/'+name)
    for key,path in [('shared_bridge','motion_expert_joint_attention/modality_bridge.py'),('execution_policy','scripts/phase3_performance.py')]:
        if sha256(repo/path) != reference[key]:
            raise ValueError('Shared source differs from running baseline: '+path)
    if sha256(args.phase1/'model/.metadata') != reference['phase1']:
        raise ValueError('Frozen Phase1 differs from baseline')
    if sha256(args.phase2) != '6a8d58d746560c0af3d340f8d711dcfe23ce163bb741912afcffa8ebfdbadc3a':
        raise ValueError('Frozen Phase2 differs from baseline')
    from .train import configure_local_paths
    path_map = configure_local_paths()
    from .data import Phase3Dataset
    dataset = Phase3Dataset(args.cache)
    if dataset.index["fingerprint"] != reference["dataset"]:
        raise ValueError("Training cohort differs from baseline")
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
            "out": str(args.out), "camera_cache":str(args.camera_cache), "reference_contract":str(args.reference_contract), "recipe": recipe, "path_map": path_map, "global_batch": 32,
            **execution, "gpu_acceptance": "pending"}
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
        sources = sorted((repo / "motion_expert_phase3_edge_joint").glob("*.py"))
        sources += sorted((repo/"motion_expert_phase3_edge_image_joint").glob("*.py"))
        sources += sorted(Path(__file__).parent.glob("*.py"))
        sources += [repo / "motion_expert_joint_attention/modality_bridge.py", repo / "scripts/phase3_performance.py"]
        manifest = {str(p.relative_to(repo)): sha256(p) for p in sources}
        for path in sources:
            destination = ops / "source_snapshot" / path.relative_to(repo)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        atomic_json(ops / "source_manifest.json", manifest)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={len(indices)}",
                   "-m", "motion_expert_phase3_edge_image_joint_camera.train"]
        common = ["--phase1", str(args.phase1), "--phase2", str(args.phase2), "--cache", str(args.cache),
                  "--microbatch", "16", "--workers", str(args.workers)]

        def stage(name, extra):
            wait_for_release(uuids)
            if any(sha256(repo / relative) != expected for relative, expected in manifest.items()):
                raise RuntimeError("Source files changed during launch; use the saved snapshot")
            atomic_json(ops / "status.json", {"stage": name, "host": host})
            print(f"[image-joint/launch] {name}: {ops / (name + '.log')}", flush=True)
            with (ops / (name + ".log")).open("x") as log:
                subprocess.run(command + common + extra, stdout=log, stderr=subprocess.STDOUT, check=True)

        stage("preflight", ["--preflight", "--out", str(ops / "preflight")])
        acceptance = ["--acceptance", str(ops / "preflight/acceptance.json")]
        stage("fresh_probe", ["--out", str(ops / "fresh_probe"), "--resume", "none", "--probe-updates", "4", *acceptance])
        stage("reload_probe", ["--out", str(ops / "reload_probe"), "--resume", str(ops / "fresh_probe/checkpoints/recovery_latest.pt"),
                               "--probe-updates", "2", *acceptance])
        wait_for_release(uuids)
        atomic_json(ops / "status.json", {"stage": "starting_fresh_production", "host": host})
        with (ops / "training.log").open("x") as log:
            trainer = subprocess.Popen(command + common + ["--out", str(args.out), "--resume", "none", *acceptance],
                                       stdout=log, stderr=subprocess.STDOUT)
            logger = None
            try:
                while not (args.out / "metrics.jsonl").is_file():
                    if trainer.poll() is not None:
                        raise RuntimeError(f"Trainer exited with {trainer.returncode}; see {ops / 'training.log'}")
                    time.sleep(2)
                run_id = "p3-" + hashlib.sha256(str(args.out.resolve()).encode()).hexdigest()[:20]
                with (ops / "wandb.log").open("x") as wandb_log:
                    logger = subprocess.Popen([sys.executable, str(repo / "scripts/stream_phase3_wandb.py"),
                        "--run-dir", str(args.out), "--state-dir", str(ops / "wandb_state"),
                        "--name", "edge-p3-imagejoint402020-camera-prediction-no-camera-loss",
                        "--variant", "native_camera_prediction_no_camera_loss", "--group", "edge-phase3-camera-prediction-ablation-20261102",
                        "--entity", args.wandb_entity, "--project", args.wandb_project], stdout=wandb_log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic()+90
                    while not (ops/'wandb_state/status.json').is_file():
                        if logger.poll() is not None:
                            raise RuntimeError('Required W&B logger exited; see launch/wandb.log')
                        if time.monotonic()>deadline:
                            raise RuntimeError('Required W&B logger did not initialize')
                        if trainer.poll() is not None:
                            raise RuntimeError('Trainer exited while W&B initialized')
                        time.sleep(2)
                    url = f"https://wandb.ai/{args.wandb_entity}/{args.wandb_project}/runs/{run_id}"
                    atomic_json(ops / "status.json", {"stage": "trainer_started", "host": host, "pid": trainer.pid, "wandb_url": url})
                    print(f"[image-joint/launch] trainer started; W&B {url} (check logger for remote sync)", flush=True)
                    code = trainer.wait()
                if code:
                    raise RuntimeError(f"Trainer exited with {code}; see {ops / 'training.log'}")
            finally:
                if trainer.poll() is None:
                    trainer.terminate()
                    trainer.wait(timeout=120)
                if logger is not None and logger.poll() is None:
                    logger.terminate()
                    logger.wait(timeout=60)
        atomic_json(ops / "status.json", {"stage": "trainer_exited", "host": host, "code": 0})


if __name__ == "__main__":
    main()
