# Archive Status

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
| BONES uniform motion | no | upload active | not required by active Phase 2/3 |
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
