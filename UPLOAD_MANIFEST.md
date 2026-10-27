# Archive Status

## Yonsei active-training audit and upload — 2026-10-27

**New additive backup is being launched; it is not yet a completed backup.**
The July checklist below describes historical archives, not coverage of current
Yonsei Edge training. Live progress and per-archive transfer receipts are under
`/scratch2/whwjdqls99/cosmos/operations/gdrive_training_backup_20261027/`.
`progress.json` records the current operation; `receipts/*.json` with
`upload_complete: true` record successful archive and receipt transfers.

Active scope: Phase3 floor-aware trajectory-FiLM (2608759), hidden-state FiLM
(2608760), four-task control (2608758), Phase1 no-text prefix1 (2608761),
EgoControl seven-condition M2V (2608784), and Phase1 action-only (2616216).
The prefix17 allocation now runs EgoControl; prefix17 is not an active trainer.

| Required artifact absent from documented Drive archives | Local apparent size |
| --- | ---: |
| Native T97 256-tier Wan latent cache | 71.65 GB |
| Nymeria camera/head-recanonicalized source motion | 15.67 GB |
| Phase3 packed motion/neutral cache and exact window index | 12.43 GB |
| Native camera-conditioning cache | 0.39 GB |
| Floor-aware 18D trajectory cache | 0.79 GB |
| Current camera/head mean/std and statistics provenance | 8.5 KB |
| Frozen Edge Phase1 200k DCP | 27.13 GB |
| Frozen Edge Phase2 self-cross caption-only B256 200k | 1.97 GB |
| Edge base HF assets, native DCP and Wan2.2 VAE | 21.21 GB |

These total approximately **151.24 GB**, before current camera-source copies,
metadata, fixed validation inputs and one recovery checkpoint per active run.
Drive reported **2.149 TB free** at the audit. Sizes are decimal and exclude
archive overhead. Recovery checkpoint sizes and selected steps are recorded by
the uploader when each checkpoint is pinned; future training steps are not
automatically mirrored.

The full-scope Yonsei remote is **`gggdrive:`**. `gdrive:` has only `drive.file`
scope and cannot establish absence; `ggdrive:` has no usable token. The old
documentation's `data:` alias is not configured here. Live listings of the
documented `cosmos_data`, `cosmos_ckpts` and `cosmos_models` roots established
the gaps above. Historical raw Nymeria/BONES directories exist, but were not
freshly recursively hash-verified. Unrelated Drive roots were not searched.

Uploads go to **`gggdrive:cosmos_data/yonsei_training_20261027/`**, using the
restartable worker [migration/backup_yonsei_active_20261027.py](migration/backup_yonsei_active_20261027.py).
The worker publishes `plan.json`, documentation, timestamp-preserving PAX tar
archives, per-archive SHA256/MD5 receipts, and a final report. Wan latents are
split into subject archives. The original sources and training processes are
unchanged; no Drive sync/delete is used. Native DCP checkpoints are pinned via
hardlinks before archiving so rolling retention cannot remove their bytes.

Current camera RGB source NPZs, dataset metadata, split, video manifest and
fixed validation inputs are included even where historical copies exist:
current loaders check exact source sizes and nanosecond mtimes. Restoring old
loose copies alone can fail cache identities. The camera/head statistics are
the active statistics, not the older Nano mean/std. See
[UPLOAD_PATH_MAP.md](UPLOAD_PATH_MAP.md) for restoration details.

No BONES reupload is needed for these six Nymeria trainers. Raw VRS/RGB videos,
unrelated experiments, environment directories, credentials, generated
visualizations and every intermediate checkpoint are excluded. This is a
training-artifact snapshot, not a fresh-server execution or restore test.

Execution uses a CPU-only overlapping step in retained job2608759/node33,
tmux `gdrive_training_backup_20261027`, one archive/transfer at a time, a
16 MiB/s transfer cap and low CPU priority. The worker depends on that
allocation remaining alive; the same command can resume from transfer receipts
inside another permitted owned allocation. `worker.log` records transfer rate,
errors and completion. Do not interpret this dated launch note as completion.

---

Status date: 2026-07-31.

This is the human-readable archive checklist. Exact Drive object counts, byte
totals, hashes, local paths, and restore paths are in
`migration/DRIVE_ARTIFACT_MANIFEST.tsv`. The independent GCS contract is
`migration/GCS_ARTIFACT_MANIFEST.tsv`.

## Authoritative Copies

| asset class | GitHub | Google Drive (`data:`) | GCS |
|---|---:|---:|---:|
| source, environment exports, code, small training stats/calibrations | yes | no duplicate required | source bundles, last verified 2026-07-23 |
| active derived Nymeria data except latent caches | no | verified | verified |
| 256-tier T97 Wan latent cache | no | no | verified |
| 720-tier T97 Wan latent cache | contract only | no; regenerable | no; regenerable |
| BONES proportional motion and UniEgo trees | no | verified | verified |
| BONES uniform motion | no | verified | not required by active Phase 2/3 |
| 20-fps Kimodo benchmark | no | verified | not required by active Phase 2/3 |
| evaluator fixtures/models/stats | C45 contract and code only | verified | verified |
| selected Phase-1 native DCPs A/B/D | no | verified | verified |
| original Phase-1 full resumable DCP | no | compact EMA delta on Drive | verified full DCP |
| selected Phase-2/3 checkpoints | no | verified for production/contact paths | verified retained set |
| run configuration/log/numeric evaluation metadata | docs in Git | verified archives | selected run trees verified |

No migration command deletes local or cloud data.

## Critical Small State

These files are required even though they are small:

- `train_test_split.json`
- `metadata/floor_calibration.json`
- `metadata/camera_motion_quality_filter_v1_T97.json`
- `video/manifest_video.jsonl`
- BONES `bones_pairs_{train,val}.jsonl` and `bones_index_{train,val}.json`
- `joint_attention/full71_windows.json` and `bomb_windows.json`
- `motion_expert/pairs_{train,val}.jsonl`
- all motion training/evaluator mean and standard-deviation arrays
- `motion_expert_joint_attention/head_camera_calibration_train.json`

The Nymeria pair files are intentionally ignored by Git. They are backed up as:

```text
data:cosmos_data/joint_attention/nymeria_pairs_train.jsonl
data:cosmos_data/joint_attention/nymeria_pairs_val.jsonl
```

`migration/restore_from_drive.sh data` restores them to their expected repo
filenames.

## Deliberately Regenerable

- `joint_latents_T97_720tier_640`: regenerate from archived video, manifest,
  split, and Wan VAE. Its exact contract is tracked under
  `migration/cache_contracts/`.
- Phase-1 720-tier evaluation fixtures: regenerate with
  `native_phase_training/prepare_phase1_eval_tier.py`.
- Base Cosmos Nano native DCP: restore the verified GCS copy, or reconstruct
  from the pinned materialized HF snapshot using
  `external/cosmos3_nano_dcp_convert.log`.
- Standalone visualizations and media: regenerate from checkpoints and fixed
  evaluation manifests. Numeric evaluation outputs and configs are retained.

Raw Nymeria VRS captures are access-gated and are not fully backed up. The
derived operational dataset needed by active trainers is backed up. Rebuilding
the complete derived dataset from raw captures additionally needs licensed body
models and the external SOMA/Kimodo pipeline.

## Verify

Fast Drive verification checks every critical file by exact byte size and
SHA-256:

```bash
python migration/verify_drive.py
```

Full Drive verification also traverses every completed tree:

```bash
python migration/verify_drive.py --include-trees
```

For GCS:

```bash
python migration/verify_migration.py gcs
```

The GCS archive last passed on 2026-07-23. The current server's `gcloud`
credentials require interactive reauthentication, so a fresh GCS check must be
run after `gcloud auth login`.
