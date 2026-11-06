# Optional EMA for the current Phase3 MMDiT recipe

EMA stays disabled by default. The active Yonsei no-caption/video-weight-one
run uses its unchanged source snapshot and remains EMA-off. These changes
apply to new accepted launches; they do not patch a running trainer.

## Enable and select weights

Inside an owned allocation, with the usual site environment and the same
specialist/cache/validation arguments, append these to the current
`motion_expert_phase3_edge_image_joint_wv1_notext.launch` command:

```bash
--ema-decay 0.999 --validation-weights ema
```

Use a new output directory. The shared launcher forwards both flags to the
real two-rank preflight, disposable fresh/reload probes and production.
`--ema-decay 0` is the unchanged default. `--validation-weights raw` remains
the default even when EMA is enabled; select `ema` explicitly to monitor it.
EMA validation without enabled EMA is rejected before loading models.

For offline sampling with this package's `sample` entry point, select
`--weights ema` or `--weights raw` (default). The inference manifest records
the choice; missing or malformed EMA cannot silently fall back to raw.
Periodic validation also records the weight selection in its identity and
saved recipe. Existing GT references and all sampling/metric settings remain.

## Scope and update rule

The seven trainable bridges are tracked in FP32. If additional Phase3 LoRA
is explicitly enabled, its tensors are tracked and selected together with
the bridges. Frozen Phase1/Phase2 weights, including the Phase1 specialist's
own pre-existing EMA weights, are outside this new EMA.

Shadows start as a clone of the fresh trainable state. Immediately after
each optimizer update, on every rank:

```python
ema = decay * ema + (1 - decay) * raw
ema_updates += 1
```

There is one EMA update per optimizer step, independent of accumulation.
Decay is constant; no implicit warmup or bias correction is applied. W&B
receives `train/ema_decay` and `train/ema_updates` through the existing logger.

## Save, resume and evaluation

Enabled checkpoints use the existing schema2 and contain raw `bridges`,
optional raw `phase3_lora`, and `ema = {decay, num_updates, state}`. The raw
optimizer, scheduler recipe, global data cursor and per-rank RNG states are
saved as before, through the same atomic checkpoint and latest-pointer path.
Periodic, graceful-stop/recovery and final saves all use this same payload.
Checkpoint retention applies to the complete paired checkpoint.

Resume with the run's own immutable source, original flags (including decay
and validation weight choice), acceptance and checkpoint. It restores raw
parameters into the optimizer model and restores EMA separately. EMA tensors
must have exactly the matching module/key/shape layout, finite FP32 values,
matching decay/scope and an integer update count equal to the checkpoint's
optimizer step. These checks happen before modifying the EMA/model state.
Legacy raw checkpoints still load with EMA disabled. Enabling EMA on a
non-EMA checkpoint is rejected; an earlier averaging history is not invented.

EMA validation swaps only trainable tensors, with the EMA shadow untouched.
All ranks use the same EMA step. Raw parameters, gradients, optimizer state,
train/eval modes and RNG streams are retained/restored before the next
training update, including on evaluation exceptions. A rank-local failure
while preparing EMA weights becomes a collective failure, so peers do not
continue sampling with inconsistent weights. Saving occurs before validation
and always records the raw model and its separate EMA.

The CPU/DDP checks cover atomic disk save/mmap load, exact continuation with
and without LoRA, raw/EMA inference selection, rejected corrupt metadata and
tensors, reversible evaluation, and the current no-caption recipe. The native
preflight additionally checks exact restored EMA tensors/counts on each rank
and one real optimizer/EMA continuation update. Check the operation receipt
under `cosmos/operations/phase3_optional_ema_20261106` for actual test results.

Verification on2026-11-06: **58 tests passed** on owned2616204 compute CPUs,
including a real two-rank disk checkpoint/resume and a rank-local EMA swap
failure that collectively restores raw weights. No full specialist GPU
preflight or new EMA production run was launched; the normal launcher will
require that native preflight before any future EMA production run.
