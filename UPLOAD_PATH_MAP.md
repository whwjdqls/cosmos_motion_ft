# Server to Cloud Path Map

## Yonsei active-training snapshot — 2026-10-27

Yonsei uses full-access **`gggdrive:`** for this upload; historical `data:` is
an alias used on the original machine, not a configured Yonsei remote.
The new self-contained snapshot is
`gggdrive:cosmos_data/yonsei_training_20261027/`.

| Snapshot content | Original Yonsei location |
| --- | --- |
| Wan latent subject archives; recanonicalized motion; source camera/metadata | `/lustre/whwjdqls99/cosmos/weka/nymeriaplus_kimodo_proportional/` |
| Motion, native-camera and floor-aware18D caches | `/scratch2/whwjdqls99/cosmos/caches/` |
| Active camera/head normalization | `/lustre/whwjdqls99/cosmos/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats/` |
| Frozen Edge Phase1 EMA200k DCP | Lustre `cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_4gpu_bs8_camera_wearer_lora_exclude5_yonsei_20260905/checkpoints/iter_000200000` |
| Frozen Edge Phase2 cross-caption200k; active recovery snapshots | Scratch2 `cosmos/cosmos_motion_ft_runs/`; exact paths and selected steps in archive receipts |
| Edge HF/base DCP and VAE | Lustre `cosmos/weka/{Cosmos3-Edge,cosmos3_edge_dcp,wan22_vae}` |
| Fixed callback input bundles | Native full71 fixtures on Lustre and Phase3 held-out71 `configuration.json`/`inputs` on Scratch2 |

The exact source list is generated as `plan.json` in the remote root and in
`/scratch2/whwjdqls99/cosmos/operations/gdrive_training_backup_20261027/`.
Archives are under `archives/`, transfer/hash receipts under `receipts/` and
documentation under `metadata/`. The final dated `report_*.json` explicitly
distinguishes complete from failed items. A published plan is not evidence
that every listed artifact has uploaded.

Archives contain original absolute paths with the leading slash removed.
GNU PAX retains nanosecond mtimes; do not retouch or rewrite source files/cache
indices after restoration. Download the selected archive and its receipt,
compare its SHA256 with the receipt, then extract into a staging directory:

```bash
mkdir -p /path/to/restore-staging
tar --extract --file /path/to/downloaded.tar \
    --directory /path/to/restore-staging --no-same-owner
```

This produces `lustre/whwjdqls99/...` or `scratch2/whwjdqls99/...` beneath the
staging directory. Restore to the recorded logical paths with timestamps
preserved. On a different server, use deliberate path mapping; do not blindly
rewrite fingerprinted indices or substitute old camera/head statistics.
Existing `restore_from_drive.sh` handles the historical layout only; these
new PAX archives require the explicit extraction procedure above.

Recovery archives contain one checkpoint and its matching pointer/configs.
Native snapshots preserve optimizer/scheduler/trainer state along with model
state. They are immutable point-in-time snapshots, not the future latest
checkpoint. Public model archives are included for convenience; older Nano
specialist backups are not substitutes for the frozen Edge specialists.

Resume an interrupted upload inside an owned compute allocation by running
`python /home/whwjdqls99/cosmos_motion_ft/migration/backup_yonsei_active_20261027.py`.
Keep its original operations directory: successful receipts skip completed
items, staged archives/pins allow retry, and `--immutable --checksum` prevents
silently replacing conflicting Drive objects. No training restore was tested
as part of this upload. See [UPLOAD_MANIFEST.md](UPLOAD_MANIFEST.md) for scope.

---

The complete machine-readable maps are:

- Drive: `migration/DRIVE_ARTIFACT_MANIFEST.tsv`
- GCS: `migration/GCS_ARTIFACT_MANIFEST.tsv`

Drive remote: `data:`. GCS root:
`gs://mm-jinhyung_kim/jungbin_cho`.

## Drive Roots

| content | Drive root | historical restore root |
|---|---|---|
| derived datasets | `data:cosmos_data/` | `/weka/jungbin/` plus run fixtures |
| selected checkpoints | `data:cosmos_ckpts/` | `/weka/jungbin/cosmos_motion_ft_runs/` |
| DreamSim/FVD/LPIPS weights | `data:cosmos_models/evaluation/` | `/weka/jungbin/model_cache/` and `$TORCH_HOME` |
| run configs/logs/numeric eval outputs | `data:cosmos_run_metadata_archives/` | `/weka/jungbin/cosmos_motion_ft_runs/` |

Restore the historical layout with:

```bash
bash migration/restore_from_drive.sh all
```

Override roots on a new server:

```bash
WEKA_ROOT=/mnt/weka/jungbin \
RUN_ROOT=/mnt/weka/jungbin/cosmos_motion_ft_runs \
REPO_ROOT=/home/user/cosmos_motion_ft \
DRIVE_REMOTE=data: \
bash migration/restore_from_drive.sh all
```

## Checkpoint Aliases

Drive uses short aliases; the restore script maps them back to original run
directories.

| Drive path | checkpoint |
|---|---|
| `cosmos_ckpts/native_phase1_vq_A/iter_000100000` | Phase-1 A full DCP |
| `cosmos_ckpts/native_phase1_vq_B/iter_000100000` | Phase-1 B full DCP |
| `cosmos_ckpts/native_phase1_vq_D/iter_000100000` | Phase-1 D full DCP |
| `cosmos_ckpts/native_phase1_baseline/iter_000100000_ema_gen_delta.pt` | original Phase-1 EMA LoRA/action subset for Phase-3 initialization |
| `cosmos_ckpts/ja_phase2_t2m_ti2m_native/ckpt_step200000.pt` | native-schedule Phase 2 |
| `cosmos_ckpts/ja_phase2_t2m_ti2m_contact_unipc35/ckpt_step200000.pt` | contact-loss Phase 2 |
| `cosmos_ckpts/ja_phase3_bridge_native/ckpt_step200000.pt` | vanilla Phase 3 |
| `cosmos_ckpts/ja_phase3_bridge_native_headcam/ckpt_step115000.pt` | head-camera Phase 3 |
| `cosmos_ckpts/ja_phase3_bridge_native_multitask/ckpt_step065000.pt` | multitask Phase 3 |
| `cosmos_ckpts/ja_phase3_bridge_native_contact/ckpt_step035000.pt` | contact-initialized Phase 3 |

The compact original Phase-1 file is not a full native checkpoint: it cannot
resume Phase-1 training or run native DCP inference directly. It is an exact
293-tensor EMA adapter/action export accepted by the Phase-3 initialization
loader. The full original Phase-1 DCP remains in GCS.
