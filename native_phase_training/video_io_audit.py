"""Verify the opt-in video-projection ablation on each rank and task."""
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback
from native_phase_training.validation import atomic_json


class VideoIOAudit(Callback):
    def on_train_start(self, model, iteration=0):
        if model.adaptation_mode != "global_lora_video_io":
            raise ValueError("Video projection audit requires global_lora_video_io")
        self.seen = {}
        self.path = Path(self.config.job.path_local) / f"video_io_audit_{iteration:09d}_rank{dist.get_rank()}.json"
        self.video = {n: p for n, p in model.net.named_parameters() if n.startswith(("vae2llm.", "llm2vae."))}
        assert set(self.video) == {f"{m}.{p}" for m in ("vae2llm", "llm2vae") for p in ("weight", "bias")}
        self.trainable = {}
        for name, p in model.net.named_parameters():
            allowed = any(k in name for k in ("lora_A", "lora_B", "action2llm", "llm2action", "action_modality_embed", "vae2llm", "llm2vae"))
            if p.requires_grad != allowed:
                raise ValueError(f"Unexpected trainability after optimizer selection: {name}")
            if p.requires_grad:
                self.trainable[name] = p.numel()
        self._write()

    @staticmethod
    def _local(tensor):
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    def _write(self):
        atomic_json(self.path, dict(rank=dist.get_rank(), tasks=self.seen,
            trainable_parameters=sum(self.trainable.values()), trainable=self.trainable))

    def on_training_step_start(self, model, data_batch, iteration=0):
        modes = data_batch["mode"]
        assert modes and len(set(modes)) == 1
        self.task = modes[0]
        assert self.task in {"forward_dynamics", "inverse_dynamics"}
        if self.task not in self.seen:
            self.before = {n: self._local(p).detach().clone() for n, p in self.video.items()}

    def on_after_backward(self, model, iteration=0):
        if self.task in self.seen:
            return
        self.nonzero = set()
        for name, p in model.net.named_parameters():
            if not p.requires_grad:
                assert p.grad is None, name
            if p.grad is not None:
                grad = self._local(p.grad)
                if not torch.isfinite(grad).all():
                    raise FloatingPointError(f"Nonfinite gradient: {name}")
                if bool(grad.abs().max() > 0):
                    self.nonzero.add(name)
        required = ("vae2llm", "llm2vae", "action2llm", "lora_B") if self.task == "forward_dynamics" else ("vae2llm", "llm2action", "lora_B")
        for module in required:
            assert any(module in n for n in self.nonzero), (self.task, module)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if self.task in self.seen:
            return
        assert torch.isfinite(loss).all()
        vm, am = output_batch["condition_mask_vision"], output_batch["condition_mask_action"]
        forward = self.task == "forward_dynamics"
        assert all(int((m == 0).sum()) == (24 if forward else 0) for m in vm)
        assert all(int((m == 0).sum()) == (0 if forward else 96) for m in am)
        changes = {n: float((self._local(p).detach() - self.before[n]).abs().max()) for n, p in self.video.items()}
        if forward:
            assert all(v > 0 for v in changes.values()), changes
        self.seen[self.task] = dict(step=iteration, loss=float(loss.detach()),
            nonzero_gradients=sorted(self.nonzero), video_projection_max_update=changes)
        self.before = {}
        self._write()
        print(f"[video-io/audit] rank={dist.get_rank()} task={self.task} step={iteration} loss={float(loss.detach()):.6f} projection_updates={changes}", flush=True)
