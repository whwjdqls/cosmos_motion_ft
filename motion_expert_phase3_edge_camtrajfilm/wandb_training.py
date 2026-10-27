"""Supervise the existing Phase 3 W&B sidecar alongside one training launch.

This process owns neither model tensors nor optimizer state. The trainer's
append-only metrics and actual contract remain the source of truth.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("Provide the training Python arguments after --")
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or "login" in host or host in {"cs-gpu-01", "node37"}:
        raise RuntimeError("Training and W&B logging require an approved compute allocation")

    run_dir = args.run_dir.expanduser().resolve()
    state_dir = run_dir / "wandb_stream"
    state_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "scripts" / "stream_phase3_wandb.py"
    sidecar_command = [
        sys.executable, str(script),
        "--run-dir", str(run_dir), "--state-dir", str(state_dir),
        "--entity", os.environ.get("WANDB_ENTITY", "jungbinc-upenn"),
        "--project", os.environ.get("WANDB_PROJECT", "cosmos-motion-ft"),
        "--group", os.environ.get("WANDB_RUN_GROUP", "edge-phase3-fourtask-camfilm-20261021"),
        "--name", os.environ.get("WANDB_NAME", run_dir.name),
        "--variant", "balanced5050-flooraware18d-motionnoise-camtrajfilm",
    ]
    stopped = False
    trainer = None
    sidecar = None

    def record(status, **extra):
        value = {
            "status": status, "run_dir": str(run_dir), "host": host,
            "slurm_job": os.environ["SLURM_JOB_ID"],
            "updated_unix": time.time(), **extra,
        }
        temporary = state_dir / "supervisor.json.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(state_dir / "supervisor.json")
        print("[phase3-wandb] " + json.dumps(value), flush=True)

    def forward_signal(signum, _frame):
        nonlocal stopped
        stopped = True
        if trainer is not None and trainer.poll() is None:
            try:
                os.killpg(trainer.pid, signum)
            except ProcessLookupError:
                pass

    def finish_sidecar():
        if sidecar is not None and sidecar.poll() is None:
            sidecar.terminate()
            try:
                sidecar.wait(timeout=120)
            except subprocess.TimeoutExpired:
                sidecar.kill()
                sidecar.wait()

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward_signal)

    with (state_dir / "stream.log").open("a", buffering=1) as log:
        trainer = subprocess.Popen([sys.executable, *command], start_new_session=True)
        record("waiting_for_training_metrics", trainer_pid=trainer.pid)
        retry_at = 0.0
        warned_missing_source = False
        start = time.monotonic()
        try:
            while trainer.poll() is None:
                if sidecar is not None and sidecar.poll() is not None:
                    record("logger_exited_retrying", logger_exit_code=sidecar.returncode)
                    sidecar = None
                    retry_at = time.monotonic() + 60
                ready = (run_dir / "contract.json").is_file() and (run_dir / "metrics.jsonl").is_file()
                if ready and sidecar is None and not stopped and time.monotonic() >= retry_at:
                    sidecar = subprocess.Popen(sidecar_command, stdout=log, stderr=subprocess.STDOUT)
                    record("logger_starting", logger_pid=sidecar.pid)
                elif not ready and not warned_missing_source and time.monotonic() - start > 900:
                    record("waiting_for_metrics_over_15_minutes", hint="Check PHASE3_RUN_DIR matches the trainer output")
                    warned_missing_source = True
                time.sleep(5)
            exit_code = trainer.returncode
        finally:
            finish_sidecar()

        # The streaming logger exits cooperatively on SIGTERM. A final bounded
        # backfill picks up any records written between its last poll and exit.
        if (run_dir / "contract.json").is_file() and (run_dir / "metrics.jsonl").is_file():
            try:
                result = subprocess.run(
                    [*sidecar_command, "--once"], stdout=log,
                    stderr=subprocess.STDOUT, timeout=180, check=False,
                )
                record("training_exited", trainer_exit_code=exit_code, final_sync_exit_code=result.returncode)
            except subprocess.TimeoutExpired:
                record("training_exited_sync_timeout", trainer_exit_code=exit_code,
                       hint="Local metrics remain available for backfill")
        else:
            record("training_exited_without_metrics", trainer_exit_code=exit_code)
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == "__main__":
    raise SystemExit(main())
