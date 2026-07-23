# GCS Server Migration

This directory is the authoritative restore contract for moving this project off
the July 2026 a3ultra server.

Bucket root:

```text
gs://mm-jinhyung_kim/jungbin_cho
```

The archive is intentionally scoped to reproducible training and evaluation. It
contains the active derived datasets, exact splits/statistics/calibrations,
pretrained runtime weights, evaluation fixtures, selected resumable checkpoints,
all selected-run evaluations, and the latest visualization for each compact
joint-attention run. It does not claim to be a byte-for-byte backup of every
obsolete run or all access-gated raw Nymeria VRS captures.

Files:

- `GCS_ARTIFACT_MANIFEST.tsv`: machine-readable local/GCS/restore path map.
- `upload_required_artifacts.sh`: idempotent, non-deleting upload driver.
- `restore_from_gcs.sh`: restore driver with overridable destination roots.
- `verify_migration.py`: local and GCS inventory/hash verifier.
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
```

All upload commands use `gcloud storage rsync` or `cp` without deletion flags.
They never remove local data or cloud objects.

The `source` section creates verified Git bundles under
`/weka/jungbin/cosmos_motion_migration_staging/source` and uploads them to
`gs://mm-jinhyung_kim/jungbin_cho/source`. It captures committed refs; the
Cosmos Framework working-tree modifications are separately preserved in this
repo under `external/cosmos_framework_patches/`.
