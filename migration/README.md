# Server Migration

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
