"""Short native Phase1 memory probe; saves measurements, never checkpoints.

Invoke through the same Edge environment as run_latent_train.py. Native model,
optimizer, EMA and training steps are unchanged. NATIVEP1_ADAPTATION_MODE must
be action_only. The existing mode includes the action modality embedding.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from omegaconf import open_dict

from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.lazy_config import LazyCall
from native_phase_training import run_latent_train as native
from native_phase_training.latent_omni_model_edge import LatentOmniMoTModelEdge

ROOT = Path(os.environ["PHASE1_MEMORY_PROBE_DIR"])
RANK = int(os.environ.get("RANK", "0"))
STEPS = int(os.environ.get("PHASE1_MEMORY_PROBE_STEPS", "24"))
STATE = {"rank": RANK, "stage": "imports_complete", "steps": [],
         "adaptation": "action_only", "microbatch": int(os.environ["NATIVEP1_CLIPS_PER_GPU"])}


def memory():
    free, total = torch.cuda.mem_get_info()
    return {"allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "device_used_gib": (total - free) / 2**30}


def record(stage, **values):
    STATE.update(stage=stage, **values)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"rank{RANK}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(STATE, indent=2) + "\n")
    temp.replace(path)
    print("[action-memory] " + json.dumps({"rank": RANK, "stage": stage, **values}), flush=True)


class MemoryProbe(Callback):
    def on_train_start(self, model, iteration=0):
        named = [(n, p) for n, p in model.net.named_parameters() if p.requires_grad]
        assert named and all(any(k in n for k in model._ACTION_MODULES) for n, _ in named)
        assert not any("lora_" in n for n, _ in model.net.named_parameters())
        self.grad_seen = set()
        record("training_ready", trainable={n: p.numel() for n, p in named},
               trainable_parameters=sum(p.numel() for _, p in named),
               ema_enabled=model.config.ema.enabled, initial_memory=memory())

    def on_training_step_start(self, model, data, iteration=0):
        # For accumulation, keep the first microstep's timer/peak until update.
        if not getattr(self, "in_update", False):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start = time.perf_counter()
            self.in_update = True

    def on_after_backward(self, model, iteration=0):
        for name, parameter in model.net.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None, name
                continue
            grad = parameter.grad
            if grad is not None:
                grad = grad.to_local() if hasattr(grad, "to_local") else grad
                assert bool(torch.isfinite(grad).all()), name
                if grad.numel() and bool(grad.abs().max() > 0):
                    self.grad_seen.add(name)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        value = float(loss.detach())
        assert math.isfinite(value)
        row = {"step": iteration, "loss": value,
               "seconds": time.perf_counter() - self.start, **memory()}
        STATE["steps"].append(row)
        self.in_update = False
        record("training", latest=row, gradients_seen=sorted(self.grad_seen))
        if iteration >= STEPS:
            # FSDP can place a parameter's nonzero gradient on another rank.
            # All ranks must validate the same union before any rank exits.
            seen_by_rank = [sorted(self.grad_seen)]
            if dist.is_initialized():
                seen_by_rank = [None] * dist.get_world_size()
                dist.all_gather_object(seen_by_rank, sorted(self.grad_seen))
            global_seen = {name for names in seen_by_rank for name in names}
            assert any("action2llm" in n for n in global_seen)
            assert any("llm2action" in n for n in global_seen)
            record("complete", completed_steps=iteration,
                   global_gradients_seen=sorted(global_seen))
            # Exit before the native trainer's unconditional final full DCP save.
            # The scratch diagnostic is not a resumable experiment.
            if dist.is_initialized():
                dist.barrier()
            raise SystemExit(0)


def main():
    assert os.environ.get("NATIVEP1_ADAPTATION_MODE") == "action_only"
    assert os.environ.get("NATIVEP1_MODEL_FAMILY") == "edge"
    original_config = native.load_experiment_from_toml

    def load_config(*args, **kwargs):
        config = original_config(*args, **kwargs)
        with open_dict(config.trainer.callbacks):
            config.trainer.callbacks.action_memory_probe = LazyCall(MemoryProbe)()
        config.checkpoint.save_iter = 1000000000
        config.trainer.max_iter = STEPS + 1
        config.trainer.save_zero_checkpoint = False
        assert STATE["microbatch"] * int(os.environ["WORLD_SIZE"]) * config.trainer.grad_accum_iter == 32
        record("config_ready", global_batch=32,
               shard_degree=config.model.config.parallelism.data_parallel_shard_degree,
               replicate_degree=config.model.config.parallelism.data_parallel_replicate_degree,
               accumulation=config.trainer.grad_accum_iter)
        return config

    native.load_experiment_from_toml = load_config
    original_build = LatentOmniMoTModelEdge.build_net

    def build(model, *args, **kwargs):
        record("building_network", dtype=str(kwargs.get("dtype", args[0] if args else None)))
        net = original_build(model, *args, **kwargs)
        record("network_built", memory=memory(), parameters=sum(p.numel() for p in net.parameters()))
        return net

    LatentOmniMoTModelEdge.build_net = build
    try:
        native.main()
    except torch.OutOfMemoryError as error:
        record("oom", error=str(error), memory=memory())
        raise
    except Exception as error:
        record("failed", error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
