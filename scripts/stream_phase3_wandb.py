#!/usr/bin/env python3
"""Backfill and follow Phase 3 JSONL metrics without touching the trainer.

Run on an owned compute allocation. The default is continuous streaming;
--once uploads the available history and finishes. Reuse the same state-dir,
entity and project to resume the same W&B run. Only metrics and configuration
are uploaded, not checkpoints, datasets, source code, or generated media.

W&B's internal step is the source record number. Charts use train/step, the
actual optimizer step, so even append-only histories containing a training
resume are preserved rather than silently dropping repeated optimizer steps.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def flatten_metrics(value, prefix="train"):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}"
        if isinstance(item, dict):
            result.update(flatten_metrics(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default="cosmos-motion-ft")
    parser.add_argument("--group", default="edge-phase3-fourtask-camfilm-20261021")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    hostname = socket.gethostname().split(".")[0]
    if "login" in hostname or hostname in {"cs-gpu-01", "node37"}:
        raise RuntimeError("Run the logger on an approved compute node")

    source = args.run_dir.resolve()
    metrics_path = source / "metrics.jsonl"
    contract = json.loads((source / "contract.json").read_text())
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "stream.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("A logger for this run is already active") from None

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())

    # Explicit run-local opt-in repairs inherited API-key/account mismatches
    # without restarting the trainer or changing any other run's credentials.
    credentials_source = "environment_or_sdk_default"
    if (args.state_dir / "use_netrc_credentials").is_file():
        if not (Path.home() / ".netrc").is_file():
            raise RuntimeError("Run requests existing netrc credentials, but ~/.netrc is missing")
        os.environ.pop("WANDB_API_KEY", None)
        credentials_source = "existing_netrc"

    import wandb

    run_id = "p3-" + hashlib.sha256(str(source).encode()).hexdigest()[:20]
    configuration = {
        "recipe": contract.get("recipe", {}),
        "training_contract": contract,
        "source_run_dir": str(source),
        "variant": args.variant,
        "logging": {
            "method": "external_append_only_jsonl_stream",
            "credentials_source": credentials_source,
            "chart_step": "train/step",
            "history_step": "zero_based_source_record_number",
            "training_modified": False,
            "system_telemetry": "disabled; logger host is not training telemetry",
            "media": "not collected by this logger",
        },
    }
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        id=run_id,
        name=args.name,
        tags=["phase3", "edge", "fourtask", args.variant, "jsonl-sidecar"],
        job_type="training-metrics",
        config=configuration,
        dir=str(args.state_dir),
        mode="online",
        resume="allow",
        save_code=False,
        settings=wandb.Settings(
            init_timeout=90,
            disable_git=True,
            disable_code=True,
            x_disable_stats=True,
        ),
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    next_remote_event = int(run.step)
    run.summary["source/metrics_path"] = str(metrics_path)
    run.summary["logging/host"] = hostname
    run.summary["logging/slurm_job"] = os.environ.get("SLURM_JOB_ID", "unknown")
    run.summary["logging/status"] = "backfilling" if args.once else "streaming"
    base_status = {
        "run_id": run_id,
        "url": run.url,
        "source": str(source),
        "logger_host": hostname,
        "logger_job": os.environ.get("SLURM_JOB_ID"),
        "remote_next_event_at_start": next_remote_event,
        "credentials_source": credentials_source,
    }
    print(json.dumps({"event": "WANDB_STARTED", **base_status}), flush=True)
    event_index = 0
    latest_step = None
    newly_queued = 0
    failed = False
    target = int(contract.get("recipe", {}).get("steps", 0))
    try:
        with metrics_path.open("rb") as stream:
            while not stop.is_set():
                queued_this_poll = 0
                while not stop.is_set():
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        stream.seek(start)
                        break
                    record = json.loads(line)
                    step = record.get("step")
                    if not isinstance(step, int) or isinstance(step, bool):
                        raise ValueError(f"Invalid optimizer step at record {event_index}")
                    latest_step = step
                    if event_index >= next_remote_event:
                        run.log(flatten_metrics(record), step=event_index)
                        newly_queued += 1
                        queued_this_poll += 1
                    event_index += 1

                metadata = metrics_path.stat()
                if metadata.st_ino != os.fstat(stream.fileno()).st_ino:
                    raise RuntimeError("Metrics file was replaced; refusing ambiguous replay")
                if metadata.st_size < stream.tell():
                    raise RuntimeError("Metrics file was truncated; refusing ambiguous replay")
                if event_index < next_remote_event:
                    raise RuntimeError("Source history is shorter than the existing W&B history")
                state = {
                    **base_status,
                    "status": "backfilling" if args.once else "streaming",
                    "source_records": event_index,
                    "latest_optimizer_step": latest_step,
                    "new_records_queued_this_process": newly_queued,
                    "source_age_seconds": max(0.0, time.time() - metadata.st_mtime),
                    "updated_unix": time.time(),
                }
                run.summary["source/record_count"] = event_index
                run.summary["source/latest_optimizer_step"] = latest_step
                run.summary["source/age_seconds"] = state["source_age_seconds"]
                atomic_json(args.state_dir / "status.json", state)
                if queued_this_poll:
                    print(json.dumps({"event": "METRICS_QUEUED", **state}), flush=True)
                if args.once or (target and latest_step is not None and latest_step >= target):
                    break
                stop.wait(args.poll_seconds)
        run.summary["logging/status"] = "backfill_complete" if args.once else "stream_stopped"
    except BaseException as error:
        failed = True
        atomic_json(args.state_dir / "status.json", {
            **base_status, "status": "error", "error": str(error),
            "updated_unix": time.time(),
        })
        raise
    finally:
        run.finish(exit_code=1 if failed else 0)
    final = {
        **base_status,
        "status": "backfill_synced" if args.once else "stream_flushed",
        "source_records": event_index,
        "latest_optimizer_step": latest_step,
        "new_records_queued_this_process": newly_queued,
        "updated_unix": time.time(),
    }
    atomic_json(args.state_dir / "status.json", final)
    print(json.dumps({"event": "WANDB_FLUSHED", **final}), flush=True)


if __name__ == "__main__":
    main()
