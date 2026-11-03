# Server Migration

## Foundary side-conversation handoff (2026-11-02)

Read [the consolidated handoff](FOUNDRY_SIDE_CONVERSATION_HANDOFF_20261102.md)
and [audit snapshot](FOUNDRY_SIDE_CONVERSATION_HANDOFF_20261102.snapshot.json)
for the three active sep-1 runs, camera/scratch packages published with the
2026-11-03 master integration,
completed FiLM-only50k training/evaluation, downloaded original/joint200k archives,
exact checkpoint formats, source validation, W&B links and outstanding caveats.
It is also linked at the top of `AGENTS_ALL.md` for the main conversation agent.

## Two-task prefix1 Phase1 and Phase2/3 training handoff (2026-10-31)

The additive upload is at
`gggdrive:cosmos_data/phase1_twotask_prefix1_phase23_20261031`.
Local operation: `/scratch2/whwjdqls99/cosmos/operations/phase123_drive_20261031`.
Trust its remotely verified `READY.json` for completion, not a launch note.
The new archive contains the full native two-task/prefix1/no-text200k Phase1
checkpoint and contract; the October27 shared backup had only142474 steps.
The handoff verifies and references31required existing shared archives and
adds Phase2 manifests/fixed samples, source, and validation evaluators. BONES
UniEgo and Nymeria RGB source trees use their established Drive locations,
with additive checksum copies and one-way verification. Credentials are excluded.

Phase2 uses the frozen Edge reasoner rather than Phase1 generator LoRA.
Fresh Phase3 combines the new Phase1 checkpoint with the selected Phase2
specialist. Pass both paths explicitly and run the normal loading/gradient
preflight before production. Old Phase3 bridge checkpoints remain tied to
their original Phase1 source. Read the uploaded `RESTORE.md`, `PLAN.json`,
`SHARED_ASSETS.json`, `DATA_TREES.json` and receipts for restoration details.

For the selected Phase2 recipe, explicitly set
`SELF_CROSS_TEXT_PACKING=nano_caption_only`,
`SELF_CROSS_BATCH_SIZE=128`, and `SELF_CROSS_GRAD_ACCUM=1` before calling
`motion_expert_t2m_edge/train_self_cross_knorm_ddp.sh` on two GPUs. Set a new
`SELF_CROSS_RUN_NAME` and set `FIXED_VIZ_SOURCE` to the restored selected
Phase2 run's `visualizations/fixed_samples` (included in archive002); the
launcher's historical default points to a different run. Set `WANDB_ENTITY`
for the destination account. Authenticate W&B rather than copying tokens.

For fresh baseline Phase3, use `motion_expert_phase3_edge/run.sh` with
`torch.distributed.run --standalone --nproc-per-node=2` and the entrypoint
`motion_expert_phase3_edge/train.py`. Explicitly supply `--phase1` (new
two-task200k DCP), `--phase2` (selected200k motion PT), `--cache` (restored
full Phase3 motion cache), and a new `--out`. `--microbatch 16` gives global32
with accumulation1 on two GPUs, subject to the destination memory preflight.
First run the same command with `--preflight`; production requires
`--acceptance NEW_OUT/acceptance.json` from that successful preflight.
Camera-conditioned variants use their corresponding modules and the matching
native-camera or floor-aware trajectory caches. These are restore directions,
not a claim that the new server or new specialist combination was GPU-tested.

For the October 2026 Edge/Phase3/baseline code and a new server, start with
[`SINGLE_REPO.md`](SINGLE_REPO.md). The older archive contract below remains
useful for the original datasets and Nano reference artifacts.

This directory is the authoritative restore contract for moving this project
off the July 2026 a3ultra server.

Cloud roots:

```text
GCS:   gs://mm-jinhyung_kim/jungbin_cho
Drive: data:
```

GCS is the comprehensive archive. Drive is a supplementary independently
verifiable copy of the active derived data, selected checkpoints, evaluation
models, and run metadata. The archive is intentionally scoped to reproducible
training and evaluation rather than every obsolete run or access-gated raw VRS
capture.

Drive commands require an `rclone` remote named `data` by default; override it
with `DRIVE_REMOTE`. GCS commands require authenticated `gcloud` access.

Files:

- `GCS_ARTIFACT_MANIFEST.tsv`: machine-readable local/GCS/restore path map.
- `DRIVE_ARTIFACT_MANIFEST.tsv`: exact Drive count/size/hash contract.
- `upload_required_artifacts.sh`: idempotent, non-deleting upload driver.
- `upload_drive_run_metadata.sh`: configs/logs/numeric eval upload without
  checkpoints or visualization media.
- `restore_from_gcs.sh`: restore driver with overridable destination roots.
- `restore_from_drive.sh`: Drive restore driver and alias-to-historical-path map.
- `verify_migration.py`: local and GCS inventory/hash verifier.
- `verify_drive.py`: Drive file hash and optional full-tree verifier.
- `export_native_phase1_delta.py`: compact exact Phase-1 EMA LoRA/action export
  for Phase-3 initialization.
- `cache_contracts/`: tracked contracts for large regenerable caches.
- `SERVER_MIGRATION.md`: human-readable dependency and restoration runbook.

The source repository is canonical on GitHub. The exact Cosmos Framework base
commit and load-bearing patch are vendored under
`external/cosmos_framework_patches/`.

## Usage

Upload one section or all sections:

```bash
bash migration/upload_required_artifacts.sh source
bash migration/upload_required_artifacts.sh core
bash migration/upload_required_artifacts.sh phase1
bash migration/upload_required_artifacts.sh phase3
bash migration/upload_required_artifacts.sh all
```

Restore into the historical layout:

```bash
bash migration/restore_from_gcs.sh all
bash migration/restore_from_drive.sh all
```

Restore into a new filesystem:

```bash
WEKA_ROOT=/mnt/weka/jungbin \
RUN_ROOT=/mnt/weka/jungbin/cosmos_motion_ft_runs \
HF_HOME=/home/newuser/.cache/huggingface \
bash migration/restore_from_gcs.sh all
```

Verify the cloud archive or a restored local tree:

```bash
python migration/verify_migration.py gcs
python migration/verify_migration.py local
python migration/verify_drive.py
python migration/verify_drive.py --include-trees
```

GCS uploads use `gcloud storage rsync`/`cp`; Drive uploads use `rclone copy` or
`copyto`. No command uses a deletion flag, so they never remove local data or
cloud objects.

The GCS `source` section creates verified Git bundles under
`/weka/jungbin/cosmos_motion_migration_staging/source` and uploads them to
`gs://mm-jinhyung_kim/jungbin_cho/source`. It captures committed refs; the
Cosmos Framework working-tree modifications are separately preserved in this
repo under `external/cosmos_framework_patches/`.

The source repository on GitHub is canonical after the final migration commit.
The older GCS source bundle remains a recovery copy of its upload-time commit.
