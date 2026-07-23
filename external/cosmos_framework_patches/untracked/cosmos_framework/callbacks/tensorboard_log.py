# SPDX-License-Identifier: OpenMDW-1.1
"""Minimal TensorBoard logging callback (the native stack only ships wandb).

Logs the per-step training loss + per-modality flow-matching losses to a TensorBoard
event file. Cadence is ``every_n`` iterations. Log dir comes from ``$TB_LOG_DIR``
(falls back to ``$IMAGINAIRE_OUTPUT_ROOT/tensorboard``).

CROSS-RANK AGGREGATION (important): with ``RankPartitionedDataLoader`` each rank is
pinned to ONE task (e.g. rank 0 -> forward_dynamics), so rank-0's per-modality losses
are NOT representative — rank 0 never has an action *target*, so its
``flow_matching_loss_action`` is structurally 0. We therefore all-reduce the
per-modality losses across ranks and log:
  - ``train/<key>``         = mean over ALL ranks (diluted by ranks where the modality
                              is only conditioning; comparable across runs)
  - ``train/<key>_active``  = mean over ACTIVE ranks only (ranks that actually had that
                              modality as a denoising target) — the TRUE per-task loss.
"""
import os

import torch
import torch.distributed as dist

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.utils import log
from cosmos_framework.utils.distributed import is_rank0

# Fixed order so the all_reduce collective is identical on every rank (else it deadlocks).
_MODALITY_KEYS = ["flow_matching_loss_vision", "flow_matching_loss_action", "flow_matching_loss_sound"]


class TensorBoardLog(EveryN):
    def __init__(self, every_n: int = 50, step_size: int = 1, log_dir: str | None = None):
        super().__init__(every_n=every_n, step_size=step_size)
        self.name = self.__class__.__name__
        self._log_dir = log_dir or os.environ.get("TB_LOG_DIR") or os.path.join(
            os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "."), "tensorboard"
        )
        self._writer = None
        self._ema = None  # smoothed loss

    def _ensure_writer(self):
        if self._writer is None and is_rank0():
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self._log_dir, exist_ok=True)
            self._writer = SummaryWriter(self._log_dir)
            log.critical(f"TensorBoardLog: writing to {self._log_dir}")

    @staticmethod
    def _scalar(v):
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            return float(v.detach().float().item())
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        # --- collective section: runs on EVERY rank (must, for all_reduce) ---
        loss_val = self._scalar(loss) or 0.0
        ob = output_batch or {}
        # per-modality value on THIS rank + an "active" indicator (dummy losses are exactly 0.0).
        vals, inds = [], []
        for k in _MODALITY_KEYS:
            s = self._scalar(ob.get(k))
            vals.append(s if s is not None else 0.0)
            inds.append(1.0 if (s is not None and abs(s) > 1e-12) else 0.0)

        dist_on = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if dist_on else 1
        # layout: [total_loss, *modality_vals, *modality_active_inds]
        agg = torch.tensor([loss_val] + vals + inds, dtype=torch.float32,
                           device=("cuda" if torch.cuda.is_available() else "cpu"))
        if dist_on and world > 1:
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        agg = agg.tolist()

        # --- write section: rank 0 only ---
        if not is_rank0():
            return
        self._ensure_writer()
        if self._writer is None:
            return
        n_mod = len(_MODALITY_KEYS)
        mean_loss = agg[0] / world
        self._ema = mean_loss if self._ema is None else 0.98 * self._ema + 0.02 * mean_loss
        self._writer.add_scalar("train/loss", mean_loss, iteration)
        self._writer.add_scalar("train/loss_ema", self._ema, iteration)
        for i, k in enumerate(_MODALITY_KEYS):
            sum_v = agg[1 + i]
            n_active = agg[1 + n_mod + i]
            self._writer.add_scalar(f"train/{k}", sum_v / world, iteration)                 # all-rank mean
            self._writer.add_scalar(f"train/{k}_active", sum_v / max(n_active, 1.0), iteration)  # per-task mean
            self._writer.add_scalar(f"train/{k}_active_ranks", n_active, iteration)
        # any other 0-dim scalars (rank-0 view; informational)
        for k, v in ob.items():
            if k not in _MODALITY_KEYS and isinstance(v, torch.Tensor) and v.numel() == 1:
                self._writer.add_scalar(f"train/{k}", float(v.detach().item()), iteration)
        self._writer.flush()
