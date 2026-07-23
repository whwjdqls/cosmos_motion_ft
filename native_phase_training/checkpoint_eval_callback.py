"""Submit an official evaluation after a successful Phase 1 checkpoint."""

from __future__ import annotations

import subprocess
from pathlib import Path

from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback


DEFAULT_REQUIRED_INPUT_FILES = (
    "fd_input.jsonl",
    "invdyn_input.jsonl",
    "policy_input.jsonl",
    "i2v_input.jsonl",
)


def build_eval_submission_command(
    *, sbatch_script: str, checkpoint_path: Path, eval_input_dir: Path, eval_output_dir: Path
) -> list[str]:
    exports = ",".join(
        (
            "ALL",
            f"CHECKPOINT_PATH={checkpoint_path}",
            f"EVAL_INPUT_DIR={eval_input_dir}",
            f"EVAL_OUTPUT_DIR={eval_output_dir}",
        )
    )
    return ["sbatch", "--parsable", f"--export={exports}", sbatch_script]


class NativeCheckpointEvalSubmitter(Callback):
    """Queue an isolated official-inference job after each completed DCP save."""

    def __init__(
        self,
        *,
        enabled: bool,
        sbatch_script: str,
        eval_input_dir: str,
        output_subdir: str = "checkpoint_evals",
        every_n_iterations: int = 0,
        required_input_files: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.sbatch_script = str(sbatch_script)
        self.eval_input_dir = Path(eval_input_dir)
        self.output_subdir = str(output_subdir)
        self.every_n_iterations = int(every_n_iterations)
        self.required_input_files = tuple(
            DEFAULT_REQUIRED_INPUT_FILES if required_input_files is None else required_input_files
        )
        if self.every_n_iterations < 0:
            raise ValueError("every_n_iterations must be non-negative")
        if not self.required_input_files or any(not name for name in self.required_input_files):
            raise ValueError("required_input_files must contain at least one non-empty filename")

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        del elapsed_time
        if not self.enabled or not distributed.is_rank0():
            return
        if self.every_n_iterations and iteration % self.every_n_iterations:
            log.info(
                f"Skipping native checkpoint evaluation at iteration {iteration}; "
                f"cadence is every {self.every_n_iterations} iterations"
            )
            return

        run_dir = Path(str(self.config.job.path_local))
        checkpoint_path = run_dir / "checkpoints" / f"iter_{iteration:09d}"
        eval_output_dir = run_dir / self.output_subdir / f"iter_{iteration:09d}"
        marker = run_dir / self.output_subdir / "submitted" / f"iter_{iteration:09d}.job"
        manifest = eval_output_dir / "COMPLETE.json"

        if manifest.is_file():
            log.info(f"Native checkpoint evaluation already complete: {manifest}")
            return
        if marker.is_file():
            log.info(f"Native checkpoint evaluation already submitted: {marker.read_text().strip()}")
            return
        if not checkpoint_path.is_dir():
            log.error(f"Cannot submit native checkpoint evaluation; checkpoint is missing: {checkpoint_path}")
            return

        missing_inputs = [
            name
            for name in self.required_input_files
            if not (self.eval_input_dir / name).is_file() or (self.eval_input_dir / name).stat().st_size == 0
        ]
        if missing_inputs:
            log.error(
                f"Cannot submit native checkpoint evaluation; missing {missing_inputs} under {self.eval_input_dir}"
            )
            return

        command = build_eval_submission_command(
            sbatch_script=self.sbatch_script,
            checkpoint_path=checkpoint_path,
            eval_input_dir=self.eval_input_dir,
            eval_output_dir=eval_output_dir,
        )
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            stderr = getattr(error, "stderr", "")
            log.error(f"Failed to submit native checkpoint evaluation for iteration {iteration}: {error}; {stderr}")
            return

        job_id = completed.stdout.strip()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(job_id + "\n")
        log.info(
            f"Submitted native checkpoint evaluation job {job_id} for iteration {iteration}: "
            f"{eval_output_dir}"
        )
