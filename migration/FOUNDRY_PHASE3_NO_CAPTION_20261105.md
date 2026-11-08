# Foundary sep-1 caption-free Phase3 handover — 2026-11-05

User requires no dataset captions anywhere in Phase3. Both fresh replacements started and took production updates before either old captioned run received a graceful stop signal. The user then explicitly requested stopping the old trainers immediately; signals were sent after both replacements reached step10. Both passed native acceptance and are now verified beyond step40 with remote W&B training histories. This is a completed handover, not completed200k training.

| Attention | GPUs | W&B | Local output |
|---|---|---|---|
| directional | 4, 7 | [p3-1841eb77997bc0da105e](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-1841eb77997bc0da105e) | `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_directional5050_temporallocal_noshapebridge_bf16_no_text_v2_T97_gbs32_sep1_20261105T225242Z` |
| bidirectional | 2, 3 | [p3-8012b39423adff404c6e](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-8012b39423adff404c6e) | `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_bidirectional5050_temporallocal_noshapebridge_bf16_no_text_v2_T97_gbs32_sep1_20261105T225242Z` |

## Controlled recipe

Both runs retain **50% V2M / 50% M2V, no joint objective**, global32=2×16 with accumulation1, fresh zero-gated MMDiT bridges, grouped BF16 bridge execution, per-modality normalization and Q/K/V/O, temporal locality, shape excluded from the bridge, and clean-motion timestep zero. Same frozen Phase1 video-I/O EMA200k and selected Phase2 motion200k; no additional Phase3 LoRA or EMA. LR2e-4,1000 warmup updates, cosine200k with zero floor, AdamW betas0.9/0.99, WD0, clip1, video weight10. Dataset and specialist identities match the old contracts exactly.

The only recipe differences against each old run are experiment identity, all-empty caption policy, and caption dropout0. Dataset output, noising and both specialist entrances clear captions. Fixed native FPS/duration/resolution template tokens remain. External TMR captions are metric references only, never model conditions. Captioned bridge checkpoints must not be resumed into these fresh experiments.

## Validation and evidence

- Existing every10k fixed71-window V2M/M2V monitoring is retained, with sampling batch32/GPU, decode batch4,35 UniPC steps and CFG1; metrics stream to each new W&B history.
- Native acceptance on both ranks and both tasks injects a private caption with nonzero bridge gates. Both specialist inputs remain empty and both motion/video outputs remain bit-exact under caption injection. Frozen experts remain unchanged.
- Native acceptance also checks zero-gate specialist parity, finite mixed-task DDP gradients, gates/core gradients, disposable overfit and save/reload consistency.
- Each replacement independently ran a fresh4-update probe and strict reload to6, retaining98 bridge tensors,98 optimizer states,two rank RNG states and exact data cursors. Production uses `--resume none` and starts from0.
- Relevant CPU tests:28 passed. Live losses and gradients are finite, both tasks sampled, joint count0, global sample count32. Both remote W&B recipes match actual contracts and histories advanced beyond20.
- Each production process uses its own immutable `launch/source_snapshot`, including the new native caption audit. Do not modify live snapshots.

## Old state preservation

Old directional saved/stopped at18842; old bidirectional at18878. Both GPU rank pairs exited and old W&B streams flushed. Final recovery states include98 bridge tensors,98 Adam states,two rank RNGs and exact next-draw cursors. Final recovery checkpoints and prior10k milestones are hardlinked under:

`/mnt/local/jinhyung.kim/cosmos/operations/phase3_no_caption_two_task_handover_20261105T225242Z/preserved/{directional,bidirectional}/`

Old output paths, immutable source snapshots, original10k milestones and migrated10k resume states remain intact. No shared-storage transfer/deletion was performed and no other user's jobs were stopped.

## Operation receipts

`/mnt/local/jinhyung.kim/cosmos/operations/phase3_no_caption_two_task_handover_20261105T225242Z`

Read `plan.json`, `NO_TEXT_ACCEPTANCE_VERIFIED.json`, `PRE_CUTOVER_VERIFIED.json`, `OLD_STOP_REQUESTED.json`, `OLD_RUNS_PRESERVED.json`, `RUNNING_VERIFIED.json`, `REMOTE_WANDB_VERIFIED.json` and `COMPLETE.json`. Status/steps here are snapshots: consult current local status and W&B for live training. These sep-1 50/50 runs are separate from sep-2's40/40/20 runs.
