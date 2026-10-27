"""Separate Nymeria C45 domain adaptation; select on validation, test once at end."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from nymeria_data import atomic_json, require_compute


def plot_curves(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event = EventAccumulator(str(out / "tb"), size_guidance={"scalars": 0})
    event.Reload()
    history = [json.loads(line) for line in (out / "validation_history.jsonl").read_text().splitlines() if line.strip()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for tag in ("loss/total", "loss/infonce"):
        points = event.Scalars(tag)
        axes[0, 0].plot([p.step for p in points], [p.value for p in points], label=tag)
    axes[0, 0].set(title="Training loss", ylabel="Loss")
    axes[0, 0].legend()
    steps = [r["step"] for r in history]
    means = [r["GT"]["R32_R03_mean"] for r in history]
    stds = [r["GT"]["R32_R03_std"] for r in history]
    axes[0, 1].errorbar(steps, means, yerr=stds, marker="o", label="GT validation, 3 seeds")
    axes[0, 1].axhline(85, color="red", linestyle="--", label="85% reference")
    axes[0, 1].set(title="Paired GT R@3, 32 candidates", ylabel="Percent")
    axes[0, 1].legend()
    for key in ("plain_avg_tie_R03", "R03"):
        axes[1, 0].plot(steps, [r["GT"][key] for r in history], label=key)
    axes[1, 0].set(title="GT full-cohort validation retrieval", ylabel="Percent")
    axes[1, 0].legend()
    for tag in ("train/lr", "train/text_lr"):
        points = event.Scalars(tag)
        axes[1, 1].plot([p.step for p in points], [p.value for p in points], label=tag)
    axes[1, 1].set(title="Learning rates", ylabel="LR")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("New optimizer step")
        ax.grid(alpha=.2)
    fig.savefig(out / "loss_validation_curves.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    require_compute()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".adaptation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = json.loads((args.baseline_run / "config.json").read_text())
        baseline = args.baseline_run / "best.pt"
        if not baseline.is_file():
            raise FileNotFoundError(baseline)
        here = Path(__file__).resolve().parent
        shared = [
            "--stats-path", cfg["stats_path"],
            "--text-cache", cfg["eval_text_cache"] or cfg["text_emb_cache"],
            "--data-dir", cfg["nymeria_data_dir"],
        ]
        def run(command):
            print("[adapt32 command] " + " ".join(map(str, command)), flush=True)
            subprocess.run([str(x) for x in command], check=True)
        request = {
            "baseline_run": str(args.baseline_run),
            "initial_weights": str(baseline),
            "data_splits_and_statistics": "unchanged from baseline",
            "changes": {
                "freeze_text_encoder": False, "text_lr_multiplier": 0.2,
                "queue_size": 0, "batch": args.batch, "steps": args.steps,
                "selection": "validation/R32_R03_mean, fixed seeds 0,1,2",
            },
            "hypotheses": [
                "Frozen BONES/pretrained text geometry limits Nymeria semantic adaptation.",
                "Cached negative motion embeddings become stale; joint text adaptation also invalidates cached text keys.",
                "32-candidate validation better matches the requested retrieval gate than full-pool paired rank.",
            ],
            "caution": "Multi-change adaptation experiment, not a single-factor causal ablation.",
            "test_policy": "Test is not used for checkpoint selection; evaluate validation-selected best after training.",
        }
        atomic_json(out / "adaptation_request.json", request)
        status = out / "pipeline_status.json"
        try:
            if not (out / "baseline_validation.json").is_file():
                atomic_json(status, {"status": "baseline_validation"})
                run([sys.executable, here / "nymeria_eval.py", "--ckpt", baseline,
                     *shared, "--split", "validation", "--out", out / "baseline_validation.json"])
            command = [sys.executable, here / "st_train.py",
                       "--dataset", "nymeria", "--nymeria-data-dir", cfg["nymeria_data_dir"],
                       "--out-dir", out, "--stats-path", cfg["stats_path"],
                       "--text-emb-cache", cfg["text_emb_cache"],
                       "--eval-text-cache", cfg["eval_text_cache"] or cfg["text_emb_cache"],
                       "--max-steps", args.steps, "--batch", args.batch,
                       "--queue-size", 0, "--text-lr-multiplier", 0.2,
                       "--nymeria-selection-metric", "R32_R03_mean",
                       "--ckpt-every", 5000, "--eval-every", 2000, "--eval-cases", 0]
            for key in (
                "num_workers", "warmup", "lr", "weight_decay",
                "lambda_kl_motion", "lambda_kl_text", "lambda_recon", "lambda_contrastive",
                "lambda_align", "aug_feat_noise_std", "info_nce_temp",
                "text_dup_threshold", "aug_time_jitter_sec", "fps", "data_fps",
                "latent_dim", "ff_size", "enc_layers", "dec_layers", "num_heads",
                "dropout", "grad_clip", "seed", "log_every",
            ):
                command.extend(["--" + key.replace("_", "-"), cfg[key]])
            for flag in ("text_use_mean", "motion_use_mean", "frozen_dup_filter", "cross_modal_recon"):
                if cfg.get(flag):
                    command.append("--" + flag.replace("_", "-"))
            if (out / "last.pt").is_file():
                command.extend(["--resume-from", out / "last.pt"])
            else:
                command.extend(["--init-from", baseline])
            atomic_json(status, {"status": "training", "command": list(map(str, command))})
            run(command)
            atomic_json(status, {"status": "selected_checkpoint_test"})
            run([sys.executable, here / "nymeria_eval.py", "--ckpt", out / "best.pt",
                 *shared, "--split", "test", "--out", out / "test_selected_on_validation.json"])
            plot_curves(out)
            report = json.loads((out / "test_selected_on_validation.json").read_text())
            metrics = report["GT"]
            score = metrics["R32_R03_mean"]
            completion = {
                "status": "complete", "selected_step": report["step"],
                "GT": metrics, "R32_R03_target": 85,
                "target_met": score >= 85, "test_used_for_selection": False,
                "report": str(out / "test_selected_on_validation.json"),
            }
            atomic_json(status, completion)
            print("[adapt32 complete] " + json.dumps(completion), flush=True)
        except Exception as error:
            atomic_json(status, {"status": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
