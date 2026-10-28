"""Opt-in Phase-1 previews inside the trainer, following EgoControl's EMA flow.

These fixed test71 cases are held-out monitoring, not a validation selection set.
All ranks sample; only rank zero renders and logs. Training RNG/modes are restored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def make_batch(latent, action, mode):
    """Use the same native action formatter, padding, and masks as training."""
    from native_phase_training.latent_nymeria_dataset import (
        ActionPromptJsonFormatter, _format_prompt_for_mode,
        build_sequence_plan_from_mode, pad_action_to_max_dim)
    from cosmos_framework.data.generator.action.utils.action_processing import (
        ActionProcessingRecord, make_batched_action_processing_fields)

    plan = build_sequence_plan_from_mode(mode=mode, video_length=97, action_length=96, video_temporal_downsample=4)
    sample = dict(video=torch.empty((3, 97, 1, 1), dtype=torch.uint8),
                  image_size=torch.tensor([256., 256., 256., 256.]),
                  conditioning_fps=torch.tensor(20), ai_caption="", viewpoint="ego_view",
                  action=pad_action_to_max_dim(torch.from_numpy(action).float(), 64))
    sample = _format_prompt_for_mode(sample, mode, ActionPromptJsonFormatter())
    return dict(video=[sample["video"][None]], video_latents=[latent],
                image_size=sample["image_size"][None], conditioning_fps=torch.tensor([20]),
                # Native unwrap_and_densify treats a tensor as ONE sequence,
                # not a batch to unbind. Actions must arrive as [T,D] items.
                ai_caption=[sample["ai_caption"]], action=[sample["action"]],
                domain_id=[torch.tensor(2)], sequence_plan=[plan],
                **make_batched_action_processing_fields(
                    ActionProcessingRecord(raw_action_dim=9, action_normalizer=None), 1))


class FixedPhase1Validation(Callback):
    def __init__(self, source, split_file, indices=(0, 23, 46, 70), every=5000, on_resume=True):
        super().__init__()
        self.source, self.split_file = Path(source), Path(split_file)
        self.indices, self.every = list(indices), int(every)
        self.on_resume = bool(on_resume)
        self.pending = None
        self.cached = {}
        if self.every <= 0 or not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("Positive interval and unique fixed cases required")

    def on_train_start(self, model, iteration=0):
        from native_phase_training.run_contract import load_run_contract
        from native_phase_training.compute_admission import require_owned_compute
        require_owned_compute()
        run = Path(str(self.config.job.path_local))
        contract = load_run_contract(run / "native_phase1_contract.json")
        if (contract.model_family, contract.model_resolution, contract.num_frames, contract.training_prefix_lengths) != ("edge", "256", 97, (1,)):
            raise ValueError("This fixed preview recipe requires Edge/256/T97/prefix1")
        if os.environ.get("NATIVEP1_DISABLE_CAPTIONS") != "1":
            raise ValueError("This fixed preview recipe uses the no-caption two-task setting")
        if model.config.rectified_flow_inference_config.scheduler_type != "unipc":
            raise ValueError("Native UniPC sampler required")
        source = [json.loads(line) for line in (self.source / "fd_input.jsonl").read_text().splitlines() if line.strip()]
        if len(source) != 71:
            raise ValueError("Expected the established fixed71 input set")
        split = json.loads(self.split_file.read_text())
        self.rows = []
        for index in self.indices:
            case = source[index]["name"].removesuffix("_forward_dynamics")
            folder = self.source / "samples" / case
            meta = json.loads((folder / "meta.json").read_text())
            if meta["uuid"] not in split["test"] or meta["uuid"] in split["train"]:
                raise ValueError("Preview recording is not held out: " + case)
            self.rows.append(dict(id=case, index=index, **meta))
        self.shift = contract.inference_shift
        self.output = run / "validation"
        self.output.mkdir(exist_ok=True)
        self.first = iteration + 1 if self.on_resume or iteration == 0 else -1
        self.label = contract.adaptation_mode
        config = dict(source=str(self.source), split_file=str(self.split_file), indices=self.indices, cases=self.rows,
                      every=self.every, preview_after_resume=self.on_resume, weights="EMA", sampler="native UniPC",
                      steps=30, shift=self.shift, guidance=1., seed=0, observed_frames=[0], scored_frames=[1, 96],
                      captions=False, fps=20, tasks=["forward_dynamics", "inverse_dynamics"],
                      purpose="Fixed test71 subset; monitoring only, no model selection or tuning")
        if dist.get_rank() == 0:
            path = self.output / "configuration.json"
            if path.exists() and json.loads(path.read_text()) != config:
                raise ValueError("Fixed preview configuration changed on resume")
            atomic_json(path, config)
        print(f"[phase1/validation] fixed cases={self.indices}; next={self.first}; every={self.every}; same-process EMA", flush=True)

    def _log_pending(self, iteration, commit=False):
        import wandb
        if dist.get_rank() != 0 or self.pending is None:
            return
        if wandb.run is not None and not wandb.run.disabled:
            # A resume can replay a few updates after the last durable save.
            # Keep the media until we catch up to W&B's committed history.
            if iteration + 1 < wandb.run.step:
                return
            wandb.define_metric("validation/step")
            wandb.define_metric("validation/*", step_metric="validation/step")
            wandb.define_metric("validation_preview/*", step_metric="validation/step")
            wandb.log(self.pending, step=iteration + 1, commit=commit)
            print(f"[phase1/validation] W&B previews+metrics queued: sample step={self.pending['validation/step']}; run={wandb.run.url}", flush=True)
            self.pending = None

    def on_training_step_start(self, model, data_batch, iteration=0):
        # The preceding native speed callback may already have committed its step.
        # Merge media into the upcoming training step, with a separate sample axis.
        if self.pending is None:
            return
        py, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self._log_pending(iteration)
        finally:
            random.setstate(py)
            np.random.set_state(np_state)

    def on_train_end(self, model, iteration=0):
        self._log_pending(iteration, commit=True)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration != self.first and (iteration <= 0 or iteration % self.every):
            return
        rank = dist.get_rank()
        destination = self.output / f"iter_{iteration:09d}"
        destination.mkdir(exist_ok=True)
        py, np_state = random.getstate(), np.random.get_state()
        modes = [(module, module.training) for net in (model.net, model.net_ema) for module in net.modules()]
        started = time.monotonic()
        print(f"[phase1/validation] start step={iteration} rank={rank}", flush=True)
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
                model.net.eval()
                # Same official scope as working EgoControl validation; avoid a
                # second GPU copy of the FP32 model and preserve BF16 Q/K/V.
                with model.ema_scope("phase1_validation", is_cpu=True):
                    self._generate(model, destination, iteration, rank)
        finally:
            random.setstate(py)
            np.random.set_state(np_state)
            for module, training in modes:
                module.training = training
        dist.barrier()
        print(f"[phase1/validation] complete step={iteration} rank={rank} seconds={time.monotonic()-started:.1f}", flush=True)

    def _generate(self, model, destination, iteration, rank):
        import wandb
        from native_phase_training.evaluate_inverse_forward import (
            LPIPSAlex, _frame_metrics, _summarize_frame_metrics, _read_video_rgb,
            _resize_gt_like_native, _aggregate, METRIC_KEYS_INVERSE)
        from native_phase_training.visualize_checkpoint import _make_video_comparison, _plot_camera_comparison
        from nymeria_world.eval_inverse_dynamics import eval_seq, gt_abs
        from motion_expert_joint_attention.render_motion import write_mp4

        perceptual = LPIPSAlex("cuda", batch_size=8) if rank == 0 else None
        records, camera_records, camera_gt = [], {}, {}
        log_values = {"validation/step": iteration}
        for row in self.rows:
            folder = self.source / "samples" / row["id"]
            if row["id"] not in self.cached:
                rgb = _resize_gt_like_native(_read_video_rgb(folder / "gt_clip.mp4"), 256, 256)
                if rgb.shape != (97, 256, 256, 3):
                    raise ValueError("Invalid preview RGB geometry")
                pixels = torch.from_numpy(rgb).permute(3, 0, 1, 2)[None].to(device="cuda", dtype=torch.float32) / 127.5 - 1
                latent = model.encode(pixels).detach().cpu()
                if latent.shape != (1, 48, 25, 16, 16) or not torch.isfinite(latent).all():
                    raise ValueError("Invalid encoded validation latent")
                action = np.asarray(json.loads((folder / "camera_action.json").read_text()), dtype=np.float32)
                if action.shape != (96, 9) or not np.isfinite(action).all():
                    raise ValueError("Invalid GT camera action")
                self.cached[row["id"]] = (rgb, latent, action)
                del pixels
            rgb, latent, action = self.cached[row["id"]]
            batch = make_batch(latent, action, "forward_dynamics")
            result = model.generate_samples_from_batch(batch, net=model.net, seed=[0], num_steps=30, shift=self.shift, guidance=1.)
            generated = result["vision"][0]
            reference = latent.to(generated.device)
            if generated.shape != reference.shape or not torch.isfinite(generated).all():
                raise FloatingPointError("Invalid generated video latent")
            prefix_error = float((generated[:, :, :1].float() - reference[:, :, :1].float()).abs().max())
            if prefix_error > 1e-4:
                raise RuntimeError(f"Sampler changed observed frame: {prefix_error}")
            if rank == 0:
                decoded = {}
                for name, value in (("model", generated), ("VAE_GT", reference)):
                    pixels = model.decode(value.float())
                    if pixels.shape != (1, 3, 97, 256, 256) or not torch.isfinite(pixels).all():
                        raise FloatingPointError("Invalid decoded video")
                    decoded[name] = ((pixels[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
                metrics = {}
                for name, predicted in (("model", decoded["model"]), ("VAE_GT", decoded["VAE_GT"]), ("GT", rgb)):
                    values = _frame_metrics(rgb[1:], predicted[1:], perceptual)
                    if not all(np.isfinite(v).all() for v in values.values()):
                        raise FloatingPointError("Non-finite validation video metric")
                    metrics[name] = _summarize_frame_metrics(values)
                gt_path, pred_path, pair_path = [destination / (row["id"] + suffix + ".mp4") for suffix in ("_GT", "_generated", "_comparison")]
                write_mp4(rgb, str(gt_path), fps=20)
                write_mp4(decoded["model"], str(pred_path), fps=20)
                _make_video_comparison(gt_video=gt_path, generated_video=pred_path, output=pair_path,
                                       label=f"{self.label} | image + camera | {iteration}", prefix_length=1)
                records.append(dict(id=row["id"], index=row["index"], observed_latent_max_abs_error=prefix_error, **metrics))
                log_values[f"validation_preview/forward_case_{row['index']:02d}"] = wandb.Video(str(pair_path), format="mp4",
                    caption="GT left; prediction right. Observed green, generated red. Fixed held-out test monitoring.")
                atomic_json(destination / "per_case.json", records)
            dist.barrier()
            inverse = model.generate_samples_from_batch(make_batch(latent, action, "inverse_dynamics"), net=model.net,
                                                        seed=[0], num_steps=30, shift=self.shift, guidance=1.)
            if rank == 0:
                predicted = inverse["action"][0].reshape(96, -1)[:, :9].float().cpu().numpy()
                if not np.isfinite(predicted).all():
                    raise FloatingPointError("Non-finite predicted camera")
                gt_poses = gt_abs(folder / "gt_camera_cosmos.npz")
                camera_records[row["id"]] = eval_seq(predicted, gt_poses)
                camera_gt[row["id"]] = eval_seq(action, gt_poses)
                plot = destination / (row["id"] + "_camera.png")
                _plot_camera_comparison(action=predicted, gt_camera=folder / "gt_camera_cosmos.npz", output=plot,
                                        title=f"{row['id']}: video to camera, Sim3 aligned", n_cameras=7)
                log_values[f"validation_preview/camera_case_{row['index']:02d}"] = wandb.Image(str(plot), caption="GT green; prediction red; Sim3 aligned")
            dist.barrier()
            print(f"[phase1/validation] finished case={row['index']} rank={rank}", flush=True)
        if rank == 0:
            summary = {name: {key: float(np.mean([r[name][key] for r in records])) for key in ("psnr_db", "ssim", "lpips_alex")}
                       for name in ("model", "VAE_GT", "GT")}
            for name, values in (("camera", camera_records), ("GT_camera", camera_gt)):
                summary[name] = {key: stats["mean"] for key, stats in _aggregate(values, METRIC_KEYS_INVERSE).items()}
            atomic_json(destination / "camera_metrics.json", {"model": camera_records, "GT": camera_gt})
            atomic_json(destination / "summary.json", dict(step=iteration, n=len(records), aggregate=summary,
                        observed_latent_max_abs_error=max(r["observed_latent_max_abs_error"] for r in records),
                        cohort="fixed test71 subset; monitoring only"))
            for name, metrics in summary.items():
                log_values.update({f"validation/{name}/{key}": value for key, value in metrics.items()})
            self.pending = log_values
