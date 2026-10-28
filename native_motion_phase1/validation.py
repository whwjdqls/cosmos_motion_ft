"""In-process native EMA previews and a one-time audit of each task's gradients."""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from cosmos_framework.utils.callback import Callback
from native_phase_training.validation import FixedPhase1Validation, atomic_json
from native_motion_phase1.data import TASKS, inference_batch


class TaskAudit(Callback):
    def on_train_start(self, model, iteration=0):
        self.seen = {}
        self.path = Path(self.config.job.path_local) / (f"task_audit_resume_{iteration:09d}.json" if iteration else "task_audit.json")
        if dist.get_rank() != 0:
            self.path = self.path.with_stem(f"{self.path.stem}_rank{dist.get_rank()}")
        if model.config.max_action_dim != 288 or not model.config.lora_enabled:
            raise ValueError("Motion ablation requires 283-D native action heads and LoRA")
        if self.config.trainer.grad_accum_iter != 1:
            raise ValueError("No gradient accumulation is permitted")

    def on_training_step_start(self, model, data_batch, iteration=0):
        names = data_batch["mode"]
        self.task = names[0]
        if self.task not in TASKS or any(t != self.task for t in names):
            raise ValueError("Expected homogeneous native task batch")

    def on_after_backward(self, model, iteration=0):
        if self.task in self.seen:
            return
        nonzero = dict(action2llm=False, llm2action=False, lora_B=False)
        for name, p in model.net.named_parameters():
            allowed = any(k in name for k in ("lora_A", "lora_B", "action2llm", "llm2action", "action_modality_embed"))
            if p.requires_grad != allowed:
                raise ValueError(f"Unexpected trainability: {name}")
            if p.grad is None:
                continue
            grad = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            for key in nonzero:
                if key in name and bool(grad.abs().max() > 0):
                    nonzero[key] = True
        required = ("action2llm", "lora_B") if self.task == "motion2video" else tuple(nonzero)
        if not all(nonzero[k] for k in required):
            raise RuntimeError(f"Missing task gradients: {self.task}: {nonzero}")
        self.gradients = nonzero

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if self.task in self.seen:
            return
        expected_video = self.task == "motion2video"
        vm, am = output_batch["condition_mask_vision"], output_batch["condition_mask_action"]
        assert all(int((m == 0).sum()) == (24 if expected_video else 0) for m in vm)
        assert all(int((m == 0).sum()) == (0 if expected_video else 97) for m in am)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Nonfinite loss")
        self.seen[self.task] = dict(step=iteration, loss=float(loss.detach()), gradients=self.gradients,
                                   vision_loss=float(output_batch["flow_matching_loss_vision"].detach()),
                                   motion_loss=float(output_batch.get("motion_reconstruction_loss", output_batch["flow_matching_loss_action"]).detach()),
                                   motion_objective=os.environ.get("MOTIONP1_OBJECTIVE", "native_velocity"))
        atomic_json(self.path, self.seen)
        print(f"[motion/audit] {self.task}: {self.seen[self.task]}", flush=True)


class MotionValidation(FixedPhase1Validation):
    def __init__(self):
        super().__init__(source=os.environ["MOTIONP1_PREVIEW_DATA"], split_file="", every=5000)

    def on_train_start(self, model, iteration=0):
        from native_phase_training.compute_access import require_compute_access
        require_compute_access()
        self.rows = json.loads((self.source / "cases.json").read_text())
        if len(self.rows) != 4 or [r["task"] for r in self.rows] != list(TASKS):
            raise ValueError("Four fixed held-out task previews required")
        self.output = Path(self.config.job.path_local) / "validation"
        self.output.mkdir(exist_ok=True)
        self.first = iteration + 1
        self.mean, self.std = (np.load(os.environ[k]).astype(np.float32)
                               for k in ("MOTION_STATS_MEAN", "MOTION_STATS_STD"))
        if dist.get_rank() == 0:
            atomic_json(self.output / "configuration.json", dict(cases=self.rows, every=5000,
                        sampler="native EMA UniPC30/shift10/CFG1", seed=0,
                        purpose="fixed held-out test monitoring; no model selection"))

    def _generate(self, model, destination, iteration, rank):
        import wandb
        from motion_expert_joint_attention.decode_uniego_torch import decode_joints
        from motion_expert_joint_attention.render_motion import render_sidebyside, write_mp4
        logs, metrics = {"validation/step": iteration}, {}
        for row in self.rows:
            task = row["task"]
            if task not in self.cached:
                with np.load(self.source / row["bundle"], allow_pickle=False) as a:
                    motion, neutral, rgb = a["motion"].copy(), a["neutral"].copy(), a["rgb"].copy()
                pixels = torch.from_numpy(rgb).permute(3, 0, 1, 2)[None].to("cuda", torch.float32) / 127.5 - 1
                latent = model.encode(pixels)[0].detach().cpu()
                self.cached[task] = motion, neutral, rgb, latent
            motion, neutral, rgb, latent = self.cached[task]
            batch = inference_batch(motion, neutral, latent, task, row["caption"])
            result = model.generate_samples_from_batch(batch, net=model.net, seed=[0],
                                                        num_steps=30, shift=10., guidance=1.)
            # Both replicas sample; only rank zero renders, writes and logs.
            # Keep collective order aligned while rank zero processes media.
            if rank != 0:
                dist.barrier()
                continue
            path = destination / (task + ".mp4")
            if task == "motion2video":
                predicted = result["vision"][0]
                error = float((predicted[:, :, :1].float() - latent[None, :, :1].to(predicted.device)).abs().max())
                if error > 1e-4:
                    raise RuntimeError(f"Observed image changed: {error}")
                pixels = model.decode(predicted.float())
                if pixels.shape != (1, 3, 97, 256, 256) or not torch.isfinite(pixels).all():
                    raise FloatingPointError("Invalid decoded RGB")
                generated = ((pixels[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
                mse = float(np.mean((generated[1:].astype(float) - rgb[1:].astype(float)) ** 2))
                metrics[task] = dict(psnr_db=float(10 * np.log10(255**2 / max(mse, 1e-12))), observed_error=error)
                frames = np.concatenate((rgb, generated), axis=2)
            else:
                predicted = result["action"][0].reshape(98, -1)[:, :283].float().cpu().numpy()
                if not np.isfinite(predicted).all():
                    raise FloatingPointError("Nonfinite predicted motion")
                shape_error = float(np.max(np.abs(predicted[0] - batch["action"][0][0, :283].numpy())))
                if shape_error > 1e-4:
                    raise RuntimeError(f"Observed neutral shape changed: {shape_error}")
                predicted = predicted[1:]
                gt_raw, gen_raw = motion * self.std + self.mean, predicted * self.std + self.mean
                gt = decode_joints(torch.from_numpy(gt_raw)[None])[0].numpy()
                gen = decode_joints(torch.from_numpy(gen_raw)[None])[0].numpy()
                if not np.isfinite(gen).all():
                    raise FloatingPointError("Nonfinite decoded joints")
                centered_gt, centered_gen = gt - gt[:, :1], gen - gen[:, :1]
                metrics[task] = dict(normalized_mse=float(np.mean((predicted - motion)**2)),
                    centered_mpjpe_m=float(np.linalg.norm(centered_gen - centered_gt, axis=-1).mean()),
                    observed_error=shape_error)
                frames = render_sidebyside(gt, gen, caption=f"{task} | step {iteration}", width=768, height=384)
                np.savez(destination / (task + ".npz"), generated=predicted, gt=motion, neutral=neutral)
            write_mp4(frames, str(path), fps=20)
            logs[f"validation_preview/{task}"] = wandb.Video(str(path), format="mp4", caption=f"{task}; GT left, prediction right; fixed held-out case. {row['caption']}")
            logs.update({f"validation/{task}/{k}": v for k, v in metrics[task].items()})
            atomic_json(destination / "summary.json", dict(step=iteration, metrics=metrics))
            print(f"[motion/validation] {task} complete", flush=True)
            dist.barrier()
        if rank == 0:
            self.pending = logs
