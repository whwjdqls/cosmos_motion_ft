# Server Migration

- [Random20 benchmark and defaults](NYMERIA_RANDOM20_V1_20261110.md): frozen1420-window manifest/CSV, cross-node handoff, and the portable `nymeria_baselines/random20_eval/` worker/global scorer.

## Yonsei Scratch2 retention refresh (2026-11-08)

[Access-time refresh and training-cache identity checks](SCRATCH2_RETENTION_20261108.md)
records the safe timestamp policy, audit location and permission exceptions.

## Sep-1 Phase3 camera and guidance evaluations (2026-11-07)

- [Inference-only camera prediction, matched V2M and M2V](FOUNDRY_PHASE3_INFERENCE_CAMERA_FIXED71_20261107.md): 852 predictions at80k/80k/40k EMA; paired camera-on/off controls and archived inference code.
- [V2M with and without text](FOUNDRY_PHASE3_V2M_TEXT_FIXED71_20261107.md): fixed71, whole-bridge CFG2, GT text routed only to the frozen motion expert.
- [M2V whole-bridge CFG1 versus2](FOUNDRY_PHASE3_M2V_WHOLECFG2_FIXED71_20261107.md): matching EMA checkpoints and native video metrics.

These historical comparisons use different checkpoint ages and remain separate from the standardized80k sweep. Reports link exact provenance and artifact archives; no model/default-guidance selection is made.

## Latest Phase3 motion-guidance2 quantitative results (2026-11-05)

[Verified full71 results for the four latest Phase3 snapshots](FOUNDRY_PHASE3_LATEST_M2V_MOTIONCFG2_FIXED71_20261105.md).
Image-joint130k and camera90k/50k/40k, motion/shape CFG2, all284 predictions
complete. Includes EgoControl/Naive context and the Naive FVD protocol distinction.

## EgoControl fixed71 native image-CFG2 results (2026-11-05)

[Verified quantitative results and exact sampling protocol](FOUNDRY_EGOCONTROL_FIXED71_CFG2_20261105.md).
Fresh native EMA200k, UniPC35/shift10, reference image-CFG2 mapped to Edge3,
fixed71 seeds0–70. All predictions and Notion publication complete.

## Motion-stream guidance pilot (2026-11-05)

[Verified guidance1/1.5/2 results and eight comparison videos](FOUNDRY_CAMERA_MOTIONCFG_PILOT8_20261105.md).
Eight-window pilot, three pinned camera checkpoints; experimental motion/shape
bridge guidance on video velocity, preserving the native camera update.

## Camera weight-one motion-to-video results (2026-11-04)

[Verified 71-window metrics and eight matched comparison videos](FOUNDRY_CAMERA_WV1_M2V_MATCHED71_20261104.md).
Joint weight1/10 at40k and no-joint weight1 at30k; native guidance1.
All 213 predictions and nine uploaded assets are complete and verified.

## Latest camera weight-one quantitative results (2026-11-04)

[Verified matched71 results: joint40k and no-joint30k](FOUNDRY_CAMERA_WV1_MATCHED71_20261104.md).
Both video-only and video+text are complete and published; all trainers continue.
The additional video-weight1/10 contrast is matched at40k.

## Phase1 video-I/O EMA200k download (2026-11-04)

[Verified Drive handoff and Yonsei/minsu1206 restore instructions](PHASE1_VIDEO_IO_EMA200K_DRIVE_20261104.md).
Exact camera-trainer metadata b2b3374b…; full27.14GB checkpoint, checksums and native contract.

## Bidirectional camera prediction without joint (2026-11-03)

See [the replacement run and preserved scratch checkpoint](FOUNDRY_CAMERA_BIDIR5050_WV1_20261103.md).
Scratch motion stopped at46338 with full recoverable state. Its GPUs5,6 are
reused for fresh pretrained-motion camera training with50/50 V2M/M2V, no joint,
bidirectional attention, video loss1 and no camera loss. Other runs continue.

## Camera video-loss-weight ablation (2026-11-03)

See [the interim40k results and weight-one launch note](FOUNDRY_CAMERA_VIDEO_WEIGHT1_20261103.md).
The camera40k video/video+text evaluation is published and verified on Notion.
The new fresh camera-prediction run changes video loss10→1, using GPUs1,7
after evaluation finishes. Existing runs and the exact50k camera evaluation
queue are preserved; consult operation receipts for live production status.

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
