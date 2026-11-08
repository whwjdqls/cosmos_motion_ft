# Sep-1 directional Phase3: video weight1 and EMA — 2026-11-06

A fresh matched run is active on coordinated GPUs **0,1**, with the original
caption-free directional and bidirectional runs retained on4,7 and2,3. The user
explicitly selected fresh initialization and requested both changes together.

W&B: [p3-d2d167e7b78773091498](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-d2d167e7b78773091498). Remote recipe/history verified
through optimizer step110 at2026-11-06T16:14:55Z; local production
verified through160 at2026-11-06T16:15:24Z. All values are snapshots.
The original sidecar rejected a69-character experiment tag. A replacement sidecar
uses the shorter tag `mmdit_dir_5050_no_text_wv1_ema0999`, replays saved JSONL events
into the same W&B run and retains the full experiment identity in recipe/config.
No trainer, live source snapshot, optimizer or checkpoint was changed for this fix.

## Matched recipe

- Directional50%V2M /50%M2V, no joint objective, empty captions at dataset/noising
  and both specialists, caption dropout0. Same frozen Phase1 video-I/O EMA200k
  and selected supplied-shape Phase2 200k; same full local cache and statistics.
- Fresh zero-gated temporally local seven-block MMDiT bridges; grouped BF16
  execution, per-modality normalization/QKV/O, shape excluded from bridge.
- Video loss weight **1**, previously10; native motion/contact losses unchanged.
- Trainable bridge EMA **decay0.999**, constant, FP32 shadows, once per optimizer
  step. No additional Phase3 LoRA. Frozen specialist weights are outside this EMA.
  Raw trainable weights and EMA shadows are saved together; optimizer uses raw.
- Global32=2×16, accumulation1, workers4/rank;200k updates; LR2e-4 peak,
  1000-step warmup, cosine to0, AdamW betas0.9/0.99, WD0, global clip1.
- Every10k, fixed71 V2M/M2V validation uses **EMA weights**,35 UniPC steps,
  CFG1, shifts3/10, sampling32/GPU with OOM backoff, RGB decode4. The identical
  pinned evaluator/cohort is retained; external TMR text is for scoring only.
- Dataset, specialist identities and shared-bridge source match the original
  directional run. Full recipe parity differs only in experiment name, video
  weight, trainable EMA and validation weight selection. Initial unweighted
  motion/video losses and task counts match the original first update exactly.

## Verification

17 focused CPU checks passed, including real two-rank EMA checkpoint/continuation
and collective EMA-validation failure recovery. Native two-GPU acceptance passed
specialist zero-bridge parity, weight-one full-denominator loss, finite mixed-task
backward, frozen experts, gate/core gradients, disposable overfit, checkpoint/EMA
roundtrip and nonzero-bridge poisoned-caption invariance for both tasks/ranks.

Fresh4 and exact reload to6 saved98 bridge tensors plus separate EMA, two rank RNGs,
matching EMA/optimizer step counts and exact next-draw cursors128/192. Production
then started from0 independently of all disposable probe weights. Through step
160, losses/gradients are finite, `video_weighted == video_loss`,
EMA updates equal optimizer steps, and joint sample count is0. Median of the last10
logged update timings: **1.23s/update**;
peak allocated memory **18.63GiB/rank**.385 immutable source
files were checked against the launch manifest. Both original run pairs remain alive.

## Locations and continuation

Output: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_directional5050_temporallocal_no_text_wv1_ema0999_T97_gbs32_sep1_20261106T160246Z`

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/phase3_directional_wv1_ema_20261106T160246Z`

Original directional baseline: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_directional5050_temporallocal_noshapebridge_bf16_no_text_v2_T97_gbs32_sep1_20261105T225242Z`

The working repository and its other uncommitted changes were left untouched.
A node-local source copy changes only its two-task policy identity/video weight
and the corresponding existing policy assertion. Shared EMA/validation support
comes from merged master`bf0d21e`; all launch children use the frozen source under
`launch/source_snapshot`. Do not resume this run using the repository's video10
policy or another experiment's checkpoint/fingerprint.

`REQUEST.json`, `RECIPE_PARITY.json`, `CHECK_ONLY.json`, `cpu_tests.log`,
`PROBES_VERIFIED.json`, `PRODUCTION_VERIFIED.json`, `REMOTE_WANDB_VERIFIED.json`
and `WANDB_LOGGER_RECOVERED.json` record authorization, settings and actual progress.
`resume_argv.json` records the exact own-run continuation arguments and sidecar;
resume only after this trainer has exited and coordinated resources are available.
Use the original immutable snapshot, both-rank path translations, EMA flags and
acceptance. This document is confirmation of a running experiment, not a200k result.
