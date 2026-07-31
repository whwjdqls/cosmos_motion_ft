# Server to Cloud Path Map

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
