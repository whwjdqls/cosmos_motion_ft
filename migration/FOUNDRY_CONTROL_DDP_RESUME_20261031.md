# Phase 3 control: two-GPU continuation on sep-2

Current state as of2026-11-03: this continuation was stopped gracefully at
178363 on2026-11-02, exit0, before launching the fresh directional/bidirectional
50/50 comparison. Its full checkpoint is preserved under
`operations/phase3_bidirectional_5050_20261102T010605Z/stopped_checkpoints/control/recovery_latest.pt`
inside the node-local Cosmos storage root. The shared control backup is150k;
no control200k checkpoint is established. The following is the historical
continuation/acceptance record, not a claim that the control is still running.

The user requested stopping the one-GPU control and resuming it on two GPUs.
GPUs 1 and 7 had been released by the completed naeive-adaptation run and were
empty at launch. The prior trainer on GPU 6 received SIGUSR1, completed its
current update, saved step 133589, and exited with code 0. Its recovery checkpoint
is pinned locally before any later recovery-file replacement.

## Preserved training configuration

- Original Phase 3 control, original frozen Phase 1 EMA200k and Phase 2 selected
  200k specialists; only the seven bridges train (117469198 parameters).
- Four tasks at equal 25% probability, caption-free, original noise and losses.
- Global batch 32: two ranks, 16 clips per rank, accumulation 1.
- Original optimizer, cosine schedule, global data cursor and 200000-step horizon.
- W&B: https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-13d64c7d5322169ace91
- Scientific fingerprint:
  `5bae33e7afe7b5cf89bb14cf8358522eb0fece2b19996887595da40457c03684`.

Changing world size from one to two retains the sampler cursor but reseeds
per-rank noise according to the original trainer policy. The continuation is
not bit-identical to one-GPU execution.

## Verification and execution

The original scientific source files remain unchanged. A separately recorded
Foundary execution adapter translates historical paths to validated NVMe paths,
admits the assigned GPU UUIDs, keeps the FP32 bridge-mask cache and motion
activation checkpointing enabled, and provides bounded resume probes. TF32
remains disabled.

CPU source/cache/normalization checks and a spawned DataLoader worker passed.
Two-GPU acceptance passed all four conditioning modes, frozen-specialist and
zero-gate parity, finite DDP backward/overfit, conditioning isolation, and
bit-identical checkpoint/optimizer/RNG reload. Peak allocated GPU memory in the
gate was 15.81 GiB on rank 0. These are execution checks, not quality metrics.

Four resumed updates saved step 133593; a second complete reload executed two
more updates and saved step 133595. Both checkpoints preserve full optimizer,
scheduler and two-rank RNG state, finite tensors, original fingerprint and
`next_global_draw = step * 32`. The verified continuation was promoted to the
original run, and its probe metric records appended before restarting W&B.
Production resumes from step 133595 rather than from disposable acceptance
weights. Inspect live logs for current progress.

Operation directory:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/resume_control_2gpu_20261031T212152Z`

Its `plan.json`, `cpu_audit.json`, `preflight/acceptance.json`,
`gate_verified.json`, immutable `source_snapshots/`, `runtime_manifest.json`,
`training.log`, and `wandb_state/status.json` retain reproducible evidence.
The production tmux session is `cosmos_phase3_control_gpu1_7`.
All datasets, caches, logs and checkpoints remain on node-local NVMe.
