# EgoControl continuation on Foundary sep-1

The user requested copying the uploaded EgoControl training state to local NVMe
and resuming on two idle GPUs. The checkpoint is iteration91345 of
`egocontrol_edge_7cond_m2v_p1ema200k_phase1recipe_T97_gbs32_200k_yonsei_20261026`.
The production launcher uses GPUs0,4 on `lg-gwangmo-4xh200-sep-1` and the original
W&B run [8bw3hujd](https://wandb.ai/jungbinc-upenn/cosmos3_camera/runs/8bw3hujd).
Live logs establish current progress; this document records the migration.

## Local artifacts and provenance

Operations and execution adapters are under
`/mnt/local/jinhyung.kim/cosmos/operations/resume_egocontrol_20261030`.
`plan.json` records exact local paths. Training data, validation bundles,
normalization, frozen Phase1 EMA200k initialization, outputs and checkpoints
all resolve under `/mnt/local/jinhyung.kim/cosmos`; training does not repeatedly
read shared storage. The shared handoff remains at
`/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_data/yonsei_resume_20261030`.

The local copy of `egocontrol.tar` was verified against published SHA256
`b8d6827a2c9ad8373c5f25d76715cbe192326be56fe2c390847a8ddc510d632e`.
All17 files were extracted with PAX nanosecond mtimes preserved and conflicts
checked. `report.json` and `egocontrol_files.json` retain staging evidence.
The native checkpoint contains model/EMA, optimizer, scheduler and trainer/RNG
state. Every DCP metadata storage range was checked against actual shard sizes.

## Preserved recipe

This is a continuation of the seven-site EgoControl baseline, not a fresh
experiment. It retains first-frame plus body-motion to video only, no captions,
all112503 accepted Nymeria windows, T97/20fps/256, and global32=two ranks×16
with accumulation1. The inherited LoRA and new pose modules train; the reasoner
and other pretrained weights remain frozen. Conditioning sites are zero-based
3,7,11,15,19,23,27, with no final pose modulation.

The original FusedAdam state, betas0.9/0.99, epsilon1e-8, weight decay0.05,
clip1, native LambdaLinear schedule and200k horizon are retained. Base LR5e-5
has multiplier0.4 and a4x pose-group multiplier; the saved scheduler continues
its original decay, without a new warmup. Native noising, loss, PowerEMA,
BF16 compute/FP32 master weights, activation checkpointing and seed42 remain
unchanged. Periodic checkpoints remain every5k with rolling15-minute saves.
See `egocontrol_edge/README.md` and the archived resolved config for details.

## Isolated execution adaptation

The dedicated detached checkout is
`/mnt/local/jinhyung.kim/cosmos/repos/egocontrol_resume_20261030`, commit268fe8d.
Every EgoControl Python/TOML source hash matches the uploaded scientific
contract. The current shell launcher differs from the historical launcher due
to site support; `source_audit.json` records that distinction explicitly.

`run_foundary.sh` is an operations copy of the launcher. It uses the existing
Foundary live-parent/UID/exact-GPU admission instead of fabricating Slurm
ownership, and invokes `train_adapter.py`. `ego_runtime.py` translates runtime
copies of archived dataset/validation paths to existing local files; it does
not rewrite immutable cache indices, fingerprints or saved scientific source.
It also adds a finite-gradient/frozen-parameter resume audit. All execution
sources and hashes are captured in `execution_source.tar.gz` and
`execution_source_manifest.json`.

## Acceptance and production

The disposable two-GPU gate strictly restored91345, trained to91347, saved a
full native checkpoint, strictly reloaded91347 and updated/saved91348. Both
ranks had finite nonzero LoRA and pose gradients and matching pose-replica
signatures. Four fixed EMA validation cases0,23,46,70 completed with finite
metrics and observed-frame latent error9.54e-7, below the1e-4 tolerance.
These are monitoring examples, not model-selection scores.

`compare_config.py` compares the full model, optimizer, scheduler and training
DataLoader plus trainer settings after explicit path translation. Gate-only
iteration limits and observational callbacks are excluded. The comparison
passed before production. The production config was checked again, including
an exact comparison of every original trainer callback; only the added resume
audit is excluded (`production_callback_comparison.json` contains no differences).
Production starts from original91345 in the original
run directory, never from probe updates. Model/EMA, optimizer, scheduler and
trainer state use native strict resume with no skipped keys and no EMA-to-regular
weight replacement. Bit-identical cross-machine CUDA replay is not claimed.

`status.json`, `production.log`, `production_audit/rank*.json`, and
`running_verified.json` establish production status. The original W&B history
was ahead of the durable checkpoint (last history step91508), so replayed
updates below that point may be omitted by W&B until history catches up.
At2026-10-30 23:40UTC, production reached91563 at1.14s/update and the original
W&B run was running with newly uploaded training iteration91550. Both ranks
passed the production gradient audits; this is a launch snapshot, not current
progress. `verify_running.py` produced the machine-readable evidence.
The existing hidden-state and trajectory FiLM jobs remain separate and running.
After strict production restore and four successful gradient-audited updates on
each rank, the two disposable gate checkpoints and duplicate local archive were
removed. `gate_verified.json` retains the full DCP range audit and
`temporary_cleanup.json` records the exact paths. Original91345, the shared
source archive, all previews, reports and execution sources are preserved.
About915GiB remained free immediately after this cleanup.
