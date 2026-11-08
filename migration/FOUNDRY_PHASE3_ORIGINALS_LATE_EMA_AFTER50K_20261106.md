# Sep-1 original Phase3 models: late EMA after50k — 2026-11-06

Both original caption-free50/50 no-joint models completed their50k raw-weight evaluations, passed native acceptance for the EMA continuation, gracefully saved their complete current states, and resumed with bridge EMA0.999. Video weight remains10. The separate fresh directional/video1/EMA run on GPUs0/1 remains active and unchanged.

| Model | GPUs | EMA initialized at optimizer step | Verified step | W&B continuation |
|---|---|---:|---:|---|
| directional | 4,7 | 50,043 | 50,270 | [run](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-244621e199b8191a46c6) |
| bidirectional | 2,3 | 50,143 | 50,420 | [run](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-54fb3e983641efb7c839) |

These are continuations from the final graceful checkpoints, not fresh initializations or rollbacks to50k. The few updates after evaluation were retained. Old runs are stopped and their completed50k validation outputs and original50k checkpoint files are preserved. New W&B run IDs keep the EMA phase and its provenance distinct from the original raw-only phase.

## Preserved scientific and training state

- Same frozen Phase1 video-I/O EMA200k and selected supplied-shape Phase2 200k; identical dataset/statistics identities, shared bridge/model source, temporal grouping, normalization and QKV/O projections.
- Same directional/bidirectional policies,50%V2M /50%M2V /0%joint, empty captions at both specialists, global32=2×16, workers4/rank, grouped BF16 and no additional Phase3 LoRA.
- Video weight10, native motion/contact losses, peak LR2e-4, original200k cosine schedule with1000 warmup, AdamW betas0.9/0.99, WD0 and global clip1 retained. LR/warmup were not restarted.
- Raw bridge tensors, all AdamW optimizer tensors/parameter groups, scheduler step/LR, both rank RNG states, source identities and exact next global data cursor were compared against the original saved state after serializing/reloading each converted checkpoint.
- EMA shadows initialize as FP32 clones of the current raw bridges. Checkpoints record `start_step`, actual `num_updates` since that step and source checkpoint SHA256/fingerprint. `start_step + num_updates == optimizer_step`; no earlier EMA history is invented. Raw parameters remain the optimizer parameters.

## Paired future monitoring

Every10k, starting at60k, the same71 windows/seeds,35-step UniPC CFG1, shifts3/10, batch32/GPU and decode4 are evaluated twice. Raw results are under `validation_raw/step_...` and W&B `val_raw/*`; EMA results are under `validation/step_...` and W&B `val/*`. Both use the same evaluator assets and RGB/FVD conventions as the original run. Raw weights, optimizer and RNG are retained/restored around EMA evaluation. Raw and EMA are saved together in each future checkpoint.

## Acceptance and handoff evidence

28 focused CPU tests passed, including exact subsequent raw AdamW updates in one/two-rank controls, late-EMA save/reload, corrupted offset/provenance rejection, reversible EMA swaps, and paired validation policy restoration. Each native two-GPU preflight passed full specialist parity, weight10 loss, finite mixed-task backward, gate/core gradients, frozen experts, EMA roundtrip and nonzero-bridge poisoned-caption invariance. All362 immutable source files per continuation match their recorded manifests.

Each old50k COMPLETE receipt was required before acceptance or any stop signal. The handoff briefly held minimal CUDA contexts on the already allocated pairs to prevent an empty-GPU scheduling gap; all guards exited after the new trainers were active. No other researcher process or the fresh video1 run was stopped.

A directional post-start metadata check initially required exact equality between differently ordered floating-point expressions. It was changed to tolerance; the already running trainer was not restarted. Its W&B sidecar was then attached and replayed the saved JSONL. Historical error and recovery receipts are retained. Both remote recipes, source-parent metadata and EMA counters were verified through W&B.

## Locations and continuation

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/phase3_originals_late_ema_after50k_20261106T184453Z`. All artifacts are node-local.

### directional

Output: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_directional5050_no_text_wv10_lateema0999_from50k_T97_gbs32_sep1_20261106T184453Z`

Original: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_directional5050_temporallocal_noshapebridge_bf16_no_text_v2_T97_gbs32_sep1_20261105T225242Z`

Pinned raw50k / final raw / converted EMA initialization are under `/mnt/local/jinhyung.kim/cosmos/operations/phase3_originals_late_ema_after50k_20261106T184453Z/directional`.

### bidirectional

Output: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_bidirectional5050_no_text_wv10_lateema0999_from50k_T97_gbs32_sep1_20261106T184453Z`

Original: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_bidirectional5050_temporallocal_noshapebridge_bf16_no_text_v2_T97_gbs32_sep1_20261105T225242Z`

Pinned raw50k / final raw / converted EMA initialization are under `/mnt/local/jinhyung.kim/cosmos/operations/phase3_originals_late_ema_after50k_20261106T184453Z/bidirectional`.

Use each continuation’s **own immutable** `launch/source_snapshot` for inference and restart: it includes the explicit late-EMA counter/provenance support and paired validation. The generic older EMA loader assumes averaging began at step0 and must not reinterpret these checkpoints. The running snapshots and original snapshots were not edited.

`resume_auto_argv.json` under each per-model operation records the correct own-run continuation command, attention, GPU pair, source snapshot and path map. Only resume after the current trainer exits and resources are coordinated. An EMA-initialized recovery checkpoint and latest pointer were seeded in each new output so auto-resume is valid before the first60k save. Use the original cache translation and EMA/both-validation flags.

Per-model receipts include `ORIGINAL_50K_EVALUATION.json`, native `preflight/acceptance.json`, `STOP_REQUESTED.json`, `MIGRATION_VERIFIED.json`, `INITIAL_RECOVERY_SEEDED.json`, `PRODUCTION_VERIFIED.json` and `REMOTE_WANDB_VERIFIED.json`. The operation also retains source manifests, migration/controller/guard code, and all CPU tests.

Completion here means both EMA continuations are running and verified, not that200k training or the60k paired validation is complete.
