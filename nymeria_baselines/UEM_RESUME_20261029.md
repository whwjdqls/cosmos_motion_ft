# UniEgoMotion continuation on Yonsei, 2026-10-29

The user requested using the idle two-4090 allocation to finish UniEgoMotion.
The saved secondary resume plan identifies two unfinished variants; each now
uses one GPU in retained job2608785/node33. The all-task and trajectory-only
references already completed350epochs and were not restarted.

| Variant | Restored global step | Completed epochs | Per-run batch |
|---|---:|---:|---:|
| RGB-only reconstruction (`RGB_recon`) | 341420 | 215 | 64 |
| RGB + trajectory reconstruction (`recon`) | 339832 | 214 | 64 |

Both retain the original floor-clean Nymeria/SOMA cache, normalization,
97frames/20fps, frozen DINOv2 features, body230 target, native loss, AdamW,
EMA and350-epoch horizon (approximately555800updates). Native Lightning
restored model/optimizer/EMA/scheduler state and both runs advanced beyond
150new updates after startup. The checkpoint LR is3e-5, with the original
StepLR decay at epoch300. No DDP conversion or batch-size change was made.
The original TensorBoard logging and checkpoint policy remain active.

Both outputs remain under:

`/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/runs/`

- `uniegomotion-RGB_recon_nymeria_soma_b64_350ep_20261019`
- `uniegomotion-recon_nymeria_soma_b64_350ep_20261019`

Launch/verification evidence is in Scratch2 Cosmos
`operations/uem_resume_20261029`: `resume_evidence.json`, `launch.json`,
`verified_status.json`, and per-variant logs. The tmux window is
`rtx4090_spare2_20261026:uem-resume`. The allocation remains retained after
completion/failure; no other trainers were interrupted.

This is a dated startup snapshot. Read current logs and `COMPLETE.json` for
progress/completion. Existing evaluation tables still refer to their original
checkpoint steps; this resume does not create new evaluation results.

The report is published to the existing
[Video2motion comparison page](https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3);
`publication.json` records whether the remote write was verified.
