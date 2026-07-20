# Shape-aware TMR retrieval debug notes

Last updated: 2026-07-07.

## Goal

This directory trains a shape-aware TMR evaluator for text-to-motion retrieval and
generation scoring. The retrieval embedding should capture motion semantics while
being mostly invariant to actor skeleton size. The decoder receives the skeleton
conditioning so reconstruction does not force body shape into the retrieval latent.

## Metric interpretation

Kimodo benchmark R@k is the published protocol metric from
`kimodo.metrics.tmr.compute_tmr_retrieval_metrics`. It de-duplicates near-identical
text prompts using text-text similarity threshold `0.99` in score space, equivalent
to cosine `> 0.98`.

`plain_R01/plain_R03/plain_MedR` in `st_inline_eval.py` are local diagnostics. They
rank only the exact diagonal pair and do no text de-duplication. These numbers are
not the NVIDIA/Kimodo benchmark metric, but they are useful for detecting over-
clustered embeddings that make protocol R@k look artificially high.

`PROTOCOL_INFLATED` is a conservative warning based on text-text cosine `> 0.96`.
It intentionally fires before the official de-duplication threshold.

## Confirmed implementation issues

- `st_eval.py` previously loaded every checkpoint as the v1 VAE architecture. This
  made standalone v3 checkpoint evaluation invalid. It now dispatches by
  `arch_version` and uses v3's `cst` contrastive head.
- `st_shape_ablation.py` and `st_ensemble_eval.py` were v1-only. They now use the
  variant-aware embedder path.
- `st_eval.py` now uses the benchmark text cache for benchmark evaluation instead
  of loading the large training text cache unnecessarily.

## Current run readout

The strongest plain retrieval results among recent runs come from the C15 two-stage
family, especially `c15_twostage_hot/step_00004000.pt`.

Full official testsuite R@3 / plain R@3:

| Checkpoint | content overview | content single | content multi | rep overview | rep single | rep multi |
|---|---:|---:|---:|---:|---:|---:|
| C6@12.5k | 78.05 / 40.50 | 76.04 / 35.45 | 70.50 / 42.66 | 92.17 / 55.56 | 89.14 / 40.55 | 78.07 / 49.91 |
| C12@30k | 70.36 / 47.75 | 70.79 / 43.22 | 65.39 / 50.06 | 92.54 / 69.25 | 87.62 / 51.03 | 77.32 / 61.14 |
| C15@4k | 70.91 / 58.18 | 72.98 / 52.41 | 68.71 / 60.54 | 98.06 / 86.18 | 94.19 / 72.34 | 87.45 / 80.77 |

C18 at temp `0.05` is not hard-collapsed by embedding std, but it is over-clustered:
protocol R@3 is high while plain R@3 is poor across all groups. Treat it as a bad
retriever despite high benchmark-style R@k.

## Likely causes of performance drops

- Low InfoNCE temperature (`0.05`) can over-sharpen the contrastive loss and produce
  clustered embeddings with tiny top-1 gaps.
- False-negative masking with `text_dup_threshold=0.9` can remove too many useful
  hard negatives for semantically similar action captions.
- v3 gives high weight to reconstructing motion from text (`rec_t`) even though the
  cached pooled caption embedding is underdetermined. Split heads help, but the
  shared transformer trunk still receives this pressure.
- Training from scratch is less reliable than two-stage adaptation from a strong v0
  retrieval geometry.

## Next experiments

Prioritize C15-style continuation:

1. Initialize from `c15_twostage_hot/step_00004000.pt`.
2. Continue with lower LR (`5e-5` or `1e-4`), temp `0.10`, and frozen duplicate
   threshold `0.95`.
3. Keep `natural_desc4_only=True`, `natural_weight=2`, and feature noise `0.02`.
4. Select checkpoints by both protocol R@3 and plain R@3. Reject checkpoints with
   high `merge_frac096` or `PROTOCOL_INFLATED`.

For v3:

1. Try lower reconstruction weight (`lambda_rec=0.1`).
2. Try disabling text-latent reconstruction or downweighting `rec_t`.
3. Keep the cst/rec split, but let the retrieval head dominate early selection.

## Operational constraint

Do not run full evaluation or training on the login node CPU. Use the GPU tmux
session or Slurm GPU allocation.

## 2026-07-07 follow-up runs

Stopped attempts:

- `c20_c15cont_lr5e5_thr095`: launched with 14 workers on a 2-CPU allocation.
  It stalled before step logs. Keep only as a failed allocation record.
- `c20_c15cont_lr5e5_thr095_nw2` and `c21_c15cont_lr1e4_thr095_nw2`: launched
  on 2-CPU allocations. Indexing completed, but training did not reach step logs
  quickly enough. Prefer proper CPU/GPU allocations for this dataset path.

Completed / evaluated:

- `c20_c15cont_lr5e5_thr095_cpu16`: launched in tmux `5` on a GPU allocation
  with `--cpus-per-task=16`, `--num-workers=12`, LR `5e-5`, temp `0.10`,
  frozen duplicate threshold `0.95`, initialized from C15@4k. It completed
  10k steps at about 3.4 steps/s. Full eval:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c20_c15cont_lr5e5_thr095_cpu16/eval_last.json`.
- `c21_c15cont_lr1e4_thr095_cpu8`: launched in tmux `0` on a GPU allocation
  with 8 CPUs, `--num-workers=6`, LR `1e-4`, temp `0.10`, frozen duplicate
  threshold `0.95`, initialized from C15@4k. It was still training while
  `step_00006000.pt` was evaluated, because step 6000 had the best inline
  plain R@3 among observed C21 checkpoints. Full eval:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c21_c15cont_lr1e4_thr095_cpu8/eval_step_00006000.json`.

Full official testsuite R@3 / plain R@3 for C20/C21:

| Checkpoint | content overview | content single | content multi | rep overview | rep single | rep multi |
|---|---:|---:|---:|---:|---:|---:|
| C20 last | 71.13 / 58.40 | 73.96 / 53.83 | 69.22 / 60.79 | 97.85 / 86.90 | 93.89 / 73.39 | 88.03 / 81.35 |
| C21@6k | 71.79 / 59.71 | 74.07 / 53.72 | 70.37 / 61.94 | 97.73 / 87.07 | 93.64 / 73.73 | 87.51 / 81.17 |

Readout: C21@6k is slightly better than C20 on content overview/multi plain R@3
and roughly tied elsewhere. Both C20/C21 improve content plain R@3 over C15@4k,
while preserving high repetition retrieval. C21@6k is the current best candidate
among these continuation runs.

## Training eval policy from 2026-07-07

Use full `content/overview` R-precision for training-time checkpoint selection.
The old inline evaluator used 500 sampled cases from overview/single/multi and
overstated progress relative to full official eval. `st_train.py`,
`st_train_v2.py`, and `st_train_v3.py` now default `--eval-cases 0`, which means
the full content overview pool only. Standalone `st_eval.py` should still be used
for the full six-group report.

## Current recommendation

C21@6k is the best anchor so far. The useful change was not longer training from
scratch; it was continuing C15@4k with `text_dup_threshold=0.95`. C6/C12/C18/C19
show that from-scratch `lr=3e-4` runs either over-cluster or lag the C15/C20/C21
plain retrieval numbers.

Next run should be a short refinement from
`c21_c15cont_lr1e4_thr095_cpu8/step_00006000.pt`, not another long from-scratch
run:

1. `lr=2e-5` or `3e-5`, warmup `200-500`, max `4k-6k` steps.
2. Keep `natural_desc4_only=True`, `natural_weight=2`, `aug_feat_noise_std=0.02`,
   `aug_time_jitter_sec=0.3`, and `frozen_dup_filter=True`.
3. Try `text_dup_threshold=0.98` to keep more hard negatives than 0.95 while still
   masking near-duplicates.
4. Also try `info_nce_temp=0.12` or `0.15`: temp `0.05` was bad, while `0.15`
   avoided the worst over-clustering in C19.
5. Select by full `content/overview` plain R@3 plus protocol R@3, then run the
   full six-group eval only for candidates.

## 2026-07-07 C22 refinement

Launched `c22b_c21s6_lr3e5_thr098_temp012_nat2_noise02_cpu16` in tmux `5` on
the A100/16-CPU allocation:

- Init: `c21_c15cont_lr1e4_thr095_cpu8/step_00006000.pt`
- `lr=3e-5`, warmup `300`, max steps `6000`
- `info_nce_temp=0.12`, `text_dup_threshold=0.98`
- Same C21 data recipe: `natural_desc4_only=True`, `natural_weight=2`,
  `aug_feat_noise_std=0.02`, `frozen_dup_filter=True`
- Training eval: full `content/overview` only, 911 usable cases

Aborted two setup attempts before training:

- `c22_c21s6_lr3e5_thr098_temp012_cpu16`: missing C21 data recipe flags.
- `c22_c21s6_lr3e5_thr098_temp012_nat2_noise02_cpu16`: correct flags, but
  interrupted during/after fresh index build. `c22b` reuses the known-good C21
  `st_train_index.json` cache.

First C22b inline full-overview eval at step 1000:

| step | content overview R@3 | plain R@3 | R@1 | R@5 | R@10 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 72.12 | 59.71 | 40.18 | 82.00 | 88.14 |

This keeps C21@6k's full-overview plain R@3 while slightly improving protocol
R@3. Let it continue and compare checkpoints at each 1000-step full-overview
eval before running full six-group evaluation.

## 2026-07-08 runs

Shape-aware target on tmux `0`:

- `c23b_c6s125_lr5e5_temp015_thr095_nat2_noise02_tmux0` started from
  C6@12.5k with fresh optimizer (`--init-from`), temp `0.15`, threshold `0.95`.
  It degraded immediately: step 1000 full `content/overview` R@3 `69.05`,
  plain R@3 `41.27`. Stopped.
- `c24_c6s125_lr1e5_temp010_thr090_nat2_noise02_tmux0` tried a smaller fresh
  optimizer LR (`1e-5`) with C6's original temp/threshold. It also degraded:
  step 500 full `content/overview` R@3 `69.70`, plain R@3 `43.58`. Stopped.
- `c25_c6resume125_to20k_tmux0` is the current shape-aware run. It uses
  `--resume-from c6_desc4_nat2/step_00012500.pt`, so it preserves C6's optimizer
  state and resumes at step 12500. First full-overview eval will be at step
  13000. This is the correct way to continue C6; fresh `--init-from` runs appear
  to damage the C6 retrieval geometry.

Shape-agnostic uniform-data baseline on tmux `5`:

- Added `st_model_agnostic.py`: same dual-encoder VAE shape as v1, but no shape
  token in the motion encoder and no shape memory in the decoder.
- Added `st_dataset_uniform_agnostic.py`: reads
  `/home/jungbin_cho/seed/soma_uniform_motions_20fps`, extracts the 30-joint
  SOMA subset from 77-joint local rotations, and uses the same 186-d
  `TMRMotionRep`.
- Added `st_train_agnostic.py`: same loss/eval policy as v1, full
  `content/overview` training eval only.
- `a1_agnostic_uniform_lr3e4_temp010_thr095_nat2_noise02_tmux5` is running from
  scratch. First eval at step 1000: protocol R@3 `60.70`, plain R@3 `4.94`,
  `PROTOCOL_INFLATED=1.0`. Let it continue briefly to see whether it de-clusters,
  but early behavior is heavily inflated.

## 2026-07-09 official TMR-SOMA-RP-v1 sanity check

Found the released NVIDIA TMR checkpoint locally at:

`/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/snapshots/e427752ae3446dedba49e928c93ddc9f0e413401`

It contains `last_weights/{motion_encoder.pt,text_encoder.pt,motion_decoder.pt}`
and 30 fps `stats/motion`. Added `official_tmr_eval.py` to evaluate this
checkpoint directly on GT benchmark motions. The evaluator uses the benchmark
LLM2Vec cache for text, upsamples the local 20 fps testsuite GT posed joints to
30 fps before encoding, and computes both the published TMR protocol metric and
our no-dedup `plain_R@3`.

Full GT eval output:

`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/official_tmr_gt_eval_full.json`

| group | n | skipped | R@3 | published GT R@3 | FID gen-GT | plain R@3 |
|---|---:|---:|---:|---:|---:|---:|
| content/overview | 917 | 0 | 88.77 | 89.09 | 2.00e-11 | 81.46 |
| content/timeline_single | 917 | 0 | 86.15 | 86.26 | 1.46e-12 | 78.19 |
| content/timeline_multi | 787 | 2 | 88.56 | 88.47 | -9.61e-12 | 84.24 |
| repetition/overview | 2380 | 0 | 93.82 | 93.91 | 3.53e-13 | 88.36 |
| repetition/timeline_single | 2380 | 0 | 89.87 | 90.13 | 1.13e-12 | 78.61 |
| repetition/timeline_multi | 1745 | 34 | 95.01 | 94.49 | 6.73e-13 | 91.06 |

Readout:

- The local official TMR checkpoint and eval path are correct. Protocol R@3 is
  within about `0.5` point of the NVIDIA published GT rows, and GT-vs-GT FID is
  effectively zero.
- Official TMR does **not** have catastrophic low `plain_R@3`, but `plain_R@3`
  is consistently below protocol R@3 by about `4-11` points. The biggest gap is
  `repetition/timeline_single`: protocol `89.87`, plain `78.61`.
- The text-merge fractions are tiny (`merge_frac096 <= 0.0009`), so the official
  model's protocol/plain gap is mostly from legitimate near-equivalent text
  handling and tie behavior, not the severe text embedding collapse seen in bad
  shape-aware runs.

## 2026-07-09 frozen official text encoder sweep

Added a training mode that loads NVIDIA TMR-SOMA-RP-v1's released
`last_weights/text_encoder.pt`, freezes the text encoder, and trains only the
motion-side model parameters. Also added `--text-use-mean` so the contrastive
target can use the deterministic text posterior mean instead of sampled text
latents. This directly tests whether our content retrieval gap is coming from
drifting the text encoder away from the official TMR text space.

Implementation notes:

- `st_train.py` flags:
  - `--pretrained-text-encoder`
  - `--freeze-text-encoder`
  - `--text-use-mean`
- Frozen parameters are excluded from the optimizer.
- When `--init-from` or `--resume-from` is also used, the pretrained text encoder
  is reloaded after checkpoint loading so the official text weights cannot be
  overwritten.
- `st_dataset.py` now builds the natural-prompt pool from
  `seed_metadata_v004.csv` durations instead of opening every `.npz`, which
  reduces startup time from minutes of file reads to seconds.

Active sweep:

| run | tmux | lambda_recon | text encoder | text latent | eval |
|---|---:|---:|---|---|---|
| `c26_offtxt_frz_mu_rec001` | 5 | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c27_offtxt_frz_mu_rec0003` | 0 | 0.003 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c28_offtxt_frz_mu_rec0001` | sbatch `10200` | 0.001 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c29_offtxt_frz_mu_rec001_klm1e5` | sbatch `10221` | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c30_offtxt_frz_mu_rec01` | sbatch `10222` | 0.1 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c31_offtxt_frz_mu_rec1` | sbatch `10223` | 1.0 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c32_offtxt_frz_mu_rec001_klm3e6` | sbatch `10267` | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c33_offtxt_frz_mu_rec001_klm1e6` | sbatch `10268` | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c34_offtxt_frz_mu_rec001_klm1e5_temp007` | sbatch `10269` | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c35_offtxt_frz_mu_rec001_klm1e5_temp012` | sbatch `10270` | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c36_offtxt_frz_mu_rec001_klm1e5_drop0` | sbatch `10271` | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c37_offtxt_frz_mu_rec001_klm1e5_nojitter` | sbatch `10272` | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c38_c32_resume20k_lr1e5` | sbatch `10274` | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |
| `c39_uniform_agnostic_offtxt_frz_mu_rec001_klm3e6_20k` | sbatch `10275` | 0.01 | official, frozen | posterior mean | full `content/overview`, 911 cases |

All runs use the current C21-style data recipe:
`natural_desc4_only=True`, `natural_weight=2`, `frozen_dup_filter=True`,
`text_dup_threshold=0.95`, `aug_feat_noise_std=0.02`, and
`info_nce_temp=0.10`.

KL sweep note:

- `C26/C27/C28`: shared `lambda_kl=1e-4`.
- `C29`: explicit `lambda_kl_motion=1e-5`, `lambda_kl_text=0.0`. This isolates
  the motion KL term now that the official text encoder is frozen.
- `C32/C33`: continue lowering motion KL to `3e-6` and `1e-6`.

Early inline full-overview results:

| run | step | content overview R@3 | plain R@3 | R@1 | R@5 | R@10 | readout |
|---|---:|---:|---:|---:|---:|---:|---|
| `c26_offtxt_frz_mu_rec001` | 1000 | 60.04 | 48.52 | 34.36 | 70.91 | 83.42 | not collapsed; protocol R@3 lags C21, but plain gap is moderate |
| `c26_offtxt_frz_mu_rec001` | 2000 | 68.72 | 58.07 | 39.74 | 78.70 | 88.91 | strong recovery after warmup; already near C21 plain R@3 |
| `c26_offtxt_frz_mu_rec001` | 10000 | 77.83 | 68.72 | 43.91 | 86.17 | 94.29 | best frozen-text sweep endpoint so far |
| `c27_offtxt_frz_mu_rec0003` | 1000 | 61.80 | 49.95 | 34.14 | 71.90 | 84.41 | slightly better than C26 at the same step; lower recon weight looks preferable early |
| `c27_offtxt_frz_mu_rec0003` | 2000 | 67.29 | 56.64 | 37.65 | 79.14 | 88.80 | slightly behind C26 at 2k; keep comparing later checkpoints |
| `c27_offtxt_frz_mu_rec0003` | 10000 | 77.17 | 68.50 | 45.01 | 86.06 | 93.63 | essentially tied with C26 on plain R@3, slightly lower protocol R@3 |
| `c28_offtxt_frz_mu_rec0001` | 10000 | 76.95 | 67.18 | 43.91 | 86.06 | 93.96 | lower recon did not improve retrieval |
| `c29_offtxt_frz_mu_rec001_klm1e5` | 10000 | 78.49 | 69.59 | 43.91 | 86.28 | 94.62 | best frozen-text endpoint so far; lowering motion KL helps modestly |
| `c30_offtxt_frz_mu_rec01` | 10000 | 70.69 | 60.15 | 41.16 | 81.23 | 90.34 | high recon hurts retrieval |
| `c31_offtxt_frz_mu_rec1` | 10000 | 38.09 | 27.66 | 21.73 | 49.18 | 63.78 | very high recon destroys retrieval geometry |
| `c32_offtxt_frz_mu_rec001_klm3e6` | 10000 | 78.05 | 69.81 | 44.24 | 86.72 | 94.40 | best plain R@3 so far; lower KL mostly tied with C29 |
| `c33_offtxt_frz_mu_rec001_klm1e6` | 10000 | 78.16 | 69.70 | 43.58 | 86.72 | 94.07 | lower KL tied with C29/C32 |
| `c34_offtxt_frz_mu_rec001_klm1e5_temp007` | 10000 | 78.38 | 68.83 | 45.01 | 85.84 | 94.18 | lower temp does not improve plain R@3 |
| `c35_offtxt_frz_mu_rec001_klm1e5_temp012` | 10000 | 78.49 | 68.72 | 43.91 | 85.73 | 94.62 | higher temp tied on protocol, worse plain R@3 |
| `c36_offtxt_frz_mu_rec001_klm1e5_drop0` | 10000 | 78.27 | 68.50 | 44.02 | 87.60 | 95.06 | no dropout improves R@5/R@10 but not plain R@3 |
| `c37_offtxt_frz_mu_rec001_klm1e5_nojitter` | 10000 | 78.70 | 68.83 | 45.33 | 86.17 | 93.96 | best protocol R@3 so far; plain below C32/C33 |
| `c38_c32_resume20k_lr1e5` | 20000 | 78.38 | 70.03 | 44.57 | 86.83 | 94.51 | continuing C32 improves plain R@3 slightly, but protocol R@3 stays flat |
| `c39_uniform_agnostic_offtxt_frz_mu_rec001_klm3e6_20k` | 20000 | 75.30 | 63.67 | 40.94 | 83.64 | 90.01 | uniform/no-shape baseline trails shape-aware C38 by a wide plain-R@3 gap |

`c28_offtxt_frz_mu_rec0001` was submitted with `sbatch` as job `10200`.
Startup log confirms official text weights loaded, text encoder frozen, and
`15.02M total / 9.23M trainable` parameters. It completed 10k steps.

`c29_offtxt_frz_mu_rec001_klm1e5` was submitted with `sbatch` as job `10221`.
Startup log confirms official text weights loaded, text encoder frozen, and
`15.02M total / 9.23M trainable` parameters.

`c30_offtxt_frz_mu_rec01` and `c31_offtxt_frz_mu_rec1` were submitted with
`sbatch` as jobs `10222` and `10223`. They extend the recon sweep upward with
`lambda_recon=0.1` and `lambda_recon=1.0`, using `lambda_kl_motion=1e-4` and
`lambda_kl_text=0.0`. Startup logs confirm official text weights loaded, text
encoder frozen, and `15.02M total / 9.23M trainable` parameters for both. Both
completed 10k steps; `0.1` is clearly worse than `0.01`, and `1.0` collapses
retrieval.

`c32_offtxt_frz_mu_rec001_klm3e6` and
`c33_offtxt_frz_mu_rec001_klm1e6` were submitted with `sbatch` as jobs `10267`
and `10268`. They keep C29's best recipe but reduce `lambda_kl_motion` to
`3e-6` and `1e-6`.

`c34_offtxt_frz_mu_rec001_klm1e5_temp007` and
`c35_offtxt_frz_mu_rec001_klm1e5_temp012` were submitted with `sbatch` as jobs
`10269` and `10270`. They keep C29's best recipe but sweep InfoNCE temperature
to `0.07` and `0.12` around the current `0.10`. Both completed 10k steps; neither
beats C29/C32 on plain R@3.

`c36_offtxt_frz_mu_rec001_klm1e5_drop0` and
`c37_offtxt_frz_mu_rec001_klm1e5_nojitter` were submitted with `sbatch` as jobs
`10271` and `10272`. They keep C29's best recipe but isolate training
augmentation: C36 sets `dropout=0.0`, C37 sets `aug_time_jitter_sec=0.0`. Both
completed 10k steps. C37 has the best protocol R@3 so far, but C32/C33 remain
better on plain R@3.

Full six-group eval for C32, selected by best full-overview plain R@3:

`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c32_offtxt_frz_mu_rec001_klm3e6/full6_eval.json`

| group | n | skipped | R@3 | plain R@3 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| content/overview | 911 | 6 | 78.05 | 69.81 | 44.24 | 86.72 | 94.40 |
| content/timeline_single | 914 | 3 | 71.33 | 60.83 | 46.61 | 79.98 | 86.98 |
| content/timeline_multi | 783 | 6 | 78.42 | 73.82 | 49.94 | 87.74 | 94.38 |
| repetition/overview | 2374 | 6 | 87.57 | 78.56 | 62.64 | 93.43 | 97.05 |
| repetition/timeline_single | 2375 | 5 | 74.19 | 63.37 | 53.14 | 79.79 | 85.39 |
| repetition/timeline_multi | 1737 | 42 | 89.98 | 84.23 | 66.55 | 94.01 | 97.29 |

`c38_c32_resume20k_lr1e5` was submitted with `sbatch` as job `10274`. It resumes
C32 from step 10k, keeps the C32 objective (`lambda_recon=0.01`,
`lambda_kl_motion=3e-6`, frozen official text, temp `0.10`), and trains to step
20k with a low continuation LR (`1e-5`) to avoid a large LR jump from extending
the cosine schedule.

`c39_uniform_agnostic_offtxt_frz_mu_rec001_klm3e6_20k` was submitted with
`sbatch` as job `10275`. It is the shape-agnostic uniform-skeleton counterpart:
same text/retrieval objective and data mix as C32/C38, no shape token/memory,
uniform 20fps motions from `/home/jungbin_cho/seed/soma_uniform_motions_20fps`,
and 20k total steps from scratch. C39 completed at `75.30 / 63.67`
full-overview R@3/plain R@3, well below C38's `78.38 / 70.03`.

## 2026-07-10 official motion initialization and 30 fps alignment

Added support for loading NVIDIA TMR-SOMA-RP-v1 motion-side weights:

- `st_train.py` and `st_train_agnostic.py` flags:
  - `--pretrained-motion-encoder`
  - `--pretrained-motion-decoder`
  - `--motion-use-mean`
- The motion loader copies only matching tensors. For shape-aware encoders this
  loads all official ACTOR motion-encoder tensors and leaves only the new
  `shape_enc` MLP random.
- `lambda_recon=0.0` now skips decoder forward, making retrieval-only training
  a real objective test instead of wasting compute on an unused decoder path.
- Added `--data-fps` separately from `--fps`. `st_dataset.py`,
  `st_inline_eval.py`, and `st_eval.py` now resample raw joint positions when
  the source files are 20 fps but the representation/model is 30 fps. This is
  required for any official-stat run; previously, passing `--fps 30` would have
  fed 20 fps samples into a 30 fps rep.

Submitted runs:

| run | job | recipe | status/readout |
|---|---:|---|---|
| `c40_offmot_offdec_rec001_20k` | `10277` | 20 fps `stats_v0`, official text frozen, official motion encoder+decoder init, C32 data mix | running |
| `c41_offmot_offdec_balanced_20k` | `10278` | C40 but `natural_weight=1` balanced source mix | running |
| `c42_offmot_retrieval_only_balanced_20k` | `10279` | C41 but retrieval-only, `lambda_recon=0`, `motion-use-mean`, no KL | running |
| `c43_agnostic_offmot_offdec_uniform_20k` | `10280` | no-shape uniform-skeleton baseline, official motion/text init | running |
| `c44_official30fps_shape_10k` | `10281` | shape-aware, official text+motion init, official 30 fps stats, 20->30 fps resampling | running |
| `eval_c44_1k` | `10282` | full six-group eval for `c44` step 1000 | done |
| `c45_official30fps_balanced_10k` | `10283` | C44 but `natural_weight=1` balanced source mix | running |
| `c46_official30fps_retrieval_only_balanced_10k` | `10284` | C45 but retrieval-only, `lambda_recon=0`, no KL | running |

Early full-overview results:

| run | step | content overview R@3 | plain R@3 | R@1 | R@5 | R@10 | readout |
|---|---:|---:|---:|---:|---:|---:|---|
| `c40_offmot_offdec_rec001_20k` | 1000 | 86.17 | 78.16 | 54.23 | 94.40 | 97.91 | official motion init alone recovers most of the gap vs C38 |
| `c41_offmot_offdec_balanced_20k` | 1000 | 86.94 | 79.14 | 56.20 | 95.61 | 98.35 | balanced source mix improves early overview too |
| `c42_offmot_retrieval_only_balanced_20k` | 1000 | 87.82 | 79.80 | 56.31 | 94.62 | 98.13 | retrieval-only is strongest 20 fps official-init early run |
| `c43_agnostic_offmot_offdec_uniform_20k` | 1000 | 85.29 | 77.39 | 50.93 | 92.43 | 96.38 | no-shape uniform also jumps with official motion init, but trails shape-aware |
| `c44_official30fps_shape_10k` | 1000 | 86.72 | 80.02 | 54.77 | 94.95 | 98.13 | first run to exceed 80 plain R@3; confirms 30 fps/stats alignment is a major missing piece |
| `c44_official30fps_shape_10k` | 2000 | 87.38 | 79.80 | 55.32 | 94.62 | 97.80 | protocol improves, plain slightly dips; keep checkpoint selection by plain R@3 |
| `c44_official30fps_shape_10k` | 3000 | 88.91 | 80.79 | 54.12 | 94.84 | 98.02 | best overview checkpoint so far; protocol is effectively at official content/overview level |
| `c45_official30fps_balanced_10k` | 1000 | 87.38 | 80.13 | 56.31 | 94.95 | 98.46 | balanced 30 fps starts slightly above C44 step 1k on plain R@3 |
| `c46_official30fps_retrieval_only_balanced_10k` | 1000 | 87.38 | 79.80 | 56.31 | 95.28 | 98.57 | retrieval-only 30 fps is strong but not above C44/C45 on plain at 1k |
| `c45_official30fps_balanced_10k` | 5000 | 89.13 | 80.90 | 52.36 | 95.06 | 97.91 | best C45 checkpoint by overview plain R@3; queued for full six-group eval |
| `c45_official30fps_balanced_10k` | 10000 | 88.69 | 80.57 | 54.67 | 94.84 | 97.91 | finished cleanly; slightly below step 5k |
| `c46_official30fps_retrieval_only_balanced_10k` | 10000 | 87.49 | 79.36 | 55.98 | 95.28 | 98.24 | finished cleanly; retrieval-only did not beat reconstruction+contrastive |

Full six-group eval for C44 step 1000:

`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c44_official30fps_shape_10k/full6_step_00001000.json`

| group | n | skipped | R@3 | plain R@3 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| content/overview | 911 | 6 | 86.72 | 80.02 | 54.77 | 94.95 | 98.13 |
| content/timeline_single | 914 | 3 | 82.17 | 74.07 | 57.66 | 87.75 | 93.33 |
| content/timeline_multi | 783 | 6 | 87.23 | 83.65 | 59.26 | 93.36 | 97.70 |
| repetition/overview | 2374 | 6 | 93.34 | 87.28 | 76.07 | 96.59 | 98.82 |
| repetition/timeline_single | 2375 | 5 | 85.98 | 73.94 | 69.47 | 89.73 | 92.59 |
| repetition/timeline_multi | 1737 | 42 | 95.22 | 91.42 | 77.37 | 97.81 | 98.73 |

Readout:

- The large retrieval drop was primarily caused by moving away from the official
  TMR motion representation/training initialization: official motion encoder init
  gives an immediate 20 fps jump, and the official 30 fps stats/resampling path
  pushes plain overview above 80 at only 1k steps.
- Shape awareness is not the cause of the low numbers. The no-shape uniform run
  also improves strongly with official motion init, while the shape-aware 30 fps
  run is currently the best evaluator.
- Timeline-single groups remain the largest residual gap vs the official
  checkpoint. Balanced-source 30 fps runs C45/C46 are intended to attack that.
- C44 step 3000 is queued for full six-group eval as job `10285`:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c44_official30fps_shape_10k/full6_step_00003000.json`

Full six-group eval for C44 step 3000:

`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c44_official30fps_shape_10k/full6_step_00003000.json`

| group | n | skipped | R@3 | plain R@3 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| content/overview | 911 | 6 | 88.91 | 80.79 | 54.12 | 94.84 | 98.02 |
| content/timeline_single | 914 | 3 | 81.84 | 73.09 | 57.55 | 87.64 | 93.33 |
| content/timeline_multi | 783 | 6 | 87.48 | 83.65 | 58.62 | 93.87 | 97.70 |
| repetition/overview | 2374 | 6 | 93.72 | 87.83 | 75.91 | 96.93 | 99.03 |
| repetition/timeline_single | 2375 | 5 | 86.91 | 75.07 | 70.74 | 90.69 | 93.77 |
| repetition/timeline_multi | 1737 | 42 | 95.51 | 91.77 | 77.03 | 97.75 | 98.85 |

Step-3000 readout:

- This is the best checkpoint so far for content/overview and most repetition
  groups. Content/overview protocol R@3 (`88.91`) is effectively tied with
  official NVIDIA TMR (`88.77` local sanity eval, `89.09` published).
- The remaining gap is concentrated in timeline-single, especially plain R@3.
  Balanced-source C45 and retrieval-only C46 should be inspected at later
  checkpoints before deciding whether to continue C44 or switch.
- C45 step 5000 is queued for full six-group eval as job `10306`:
  `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c45_official30fps_balanced_10k/full6_step_00005000.json`

Final status: all C40-C46 jobs completed cleanly and `squeue` is empty.

Final 20k endpoints for the non-30fps follow-ups:

| run | final step | content overview R@3 | plain R@3 | readout |
|---|---:|---:|---:|---|
| `c40_offmot_offdec_rec001_20k` | 20000 | 85.95 | 76.40 | degraded from early official-init peak |
| `c41_offmot_offdec_balanced_20k` | 20000 | 86.61 | 77.72 | below early C41 and far below 30fps runs |
| `c42_offmot_retrieval_only_balanced_20k` | 20000 | 86.94 | 78.38 | retrieval-only did not improve with more steps |
| `c43_agnostic_offmot_offdec_uniform_20k` | 20000 | 82.33 | 73.22 | no-shape uniform trails shape-aware by a large margin |

Full six-group eval for C45 step 5000:

`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c45_official30fps_balanced_10k/full6_step_00005000.json`

| group | n | skipped | R@3 | plain R@3 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| content/overview | 911 | 6 | 89.13 | 80.90 | 52.36 | 95.06 | 97.91 |
| content/timeline_single | 914 | 3 | 84.35 | 75.49 | 59.85 | 89.28 | 94.53 |
| content/timeline_multi | 783 | 6 | 86.46 | 83.01 | 59.90 | 93.10 | 97.70 |
| repetition/overview | 2374 | 6 | 93.93 | 88.37 | 75.95 | 97.39 | 99.03 |
| repetition/timeline_single | 2375 | 5 | 88.04 | 76.42 | 72.08 | 92.17 | 94.91 |
| repetition/timeline_multi | 1737 | 42 | 95.74 | 92.17 | 79.22 | 97.87 | 99.02 |

C45 step 5000 is the best checkpoint so far. Compared with C44 step 3000, it
improves both timeline-single groups and repetition groups, while content
timeline_multi is slightly lower. It is now the leading evaluator candidate.

## 2026-07-10 controlled shape-awareness eval matrix

Goal: test whether the C45 shape-aware evaluator is less sensitive than the
released NVIDIA TMR-SOMA-RP-v1 model to actor skeleton variation. The controlled
matrix uses the same Kimodo testsuite texts and crop windows, but resolves
`seed_motion.json` into either:

- uniform motions: `/home/jungbin_cho/seed/soma_uniform_motions_20fps`
- proportional motions: `/mnt/shared/jungbin_cho/seed/soma_proportional_uniegomotion_20fps`

Implementation changes:

- `official_tmr_eval.py` now accepts `--uniego-root`; when set, it decodes the
  selected uniego motion window instead of reading testsuite `gt_motion.npz`.
- `st_eval.py` now exposes `--testsuite` and `--uniego-root`, so C45 can be
  evaluated on uniform and proportional roots with the same benchmark pool.

Submitted `sbatch_eval_shape_awareness_matrix.sh` as job `10340` on `a2`
with one GPU, 8 CPUs, and a 2-hour time limit. It was pending for priority at
submission time. Expected outputs:

- `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/shape_awareness_matrix/official_tmr_uniform_full.json`
- `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/shape_awareness_matrix/official_tmr_proportional_full.json`
- `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c45_official30fps_balanced_10k/full6_uniform_step_00005000.json`
- `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c45_official30fps_balanced_10k/full6_proportional_step_00005000_rerun.json`

Hypothesis: if C45 is genuinely shape-aware, its uniform/proportional retrieval
metrics should be closer than the official TMR model's uniform/proportional
metrics. The already-completed C45 proportional eval is
`full6_step_00005000.json`; the rerun above records the root explicitly.

Result: job `10340` failed because the first implementation assumed all source
NPZs had proportional `features`; uniform NPZs instead store `posed_joints`.
`official_tmr_eval.py` and `st_inline_eval.py` were updated to support both
schemas. Job `10341` completed `official_tmr_uniform_full.json` then failed on
official-proportional because non-finite embeddings trigger Kimodo's rank
assertion. Both eval paths now filter non-finite embedding rows and report
`skipped_nonfinite`. Job `10342` completed the remaining matrix.

Full six-group R@3 / plain R@3:

| model/source | content ov | content single | content multi | rep ov | rep single | rep multi |
|---|---:|---:|---:|---:|---:|---:|
| official / uniform | 88.77 / 81.46 | 86.15 / 78.19 | 88.56 / 84.24 | 93.82 / 88.36 | 89.87 / 78.61 | 95.01 / 91.06 |
| official / proportional | 83.42 / 74.20 | 76.91 / 67.40 | 82.89 / 77.78 | 89.05 / 79.91 | 77.05 / 66.65 | 89.41 / 83.48 |
| C45 / uniform | 89.09 / 79.28 | 77.97 / 70.88 | 87.80 / 84.12 | 93.11 / 86.64 | 84.24 / 73.91 | 95.53 / 91.81 |
| C45 / proportional | 89.13 / 80.90 | 84.35 / 75.49 | 86.46 / 83.01 | 93.93 / 88.37 | 88.04 / 76.42 | 95.74 / 92.17 |

Non-finite rows filtered:

- official / uniform: `0` for all groups.
- official / proportional: content overview `6`, content single `3`, content
  multi `4`, repetition overview `6`, repetition single `5`, repetition multi `8`.
- C45 / uniform and C45 / proportional: `0` for all groups.

Readout: the official NVIDIA TMR evaluator clearly degrades when the same
testsuite texts/crops are resolved to proportional skeleton motions, especially
timeline-single. C45 does not show this degradation; proportional is similar or
better than uniform, and it produces no non-finite embeddings. This supports that
C45's shape conditioning is doing the intended job, though C45's
timeline-single uniform numbers are lower than its proportional numbers, so the
uniform/proportional comparison is not perfectly symmetric across all groups.

Follow-up: evaluated C45 step 10000 with the same controlled uniform/proportional
matrix using `sbatch_eval_c45_10k_shape_awareness.sh` as job `10351`.

| model/source | content ov | content single | content multi | rep ov | rep single | rep multi |
|---|---:|---:|---:|---:|---:|---:|
| C45@5k / uniform | 89.09 / 79.28 | 77.97 / 70.88 | 87.80 / 84.12 | 93.11 / 86.64 | 84.24 / 73.91 | 95.53 / 91.81 |
| C45@5k / proportional | 89.13 / 80.90 | 84.35 / 75.49 | 86.46 / 83.01 | 93.93 / 88.37 | 88.04 / 76.42 | 95.74 / 92.17 |
| C45@10k / uniform | 88.88 / 79.83 | 78.30 / 71.32 | 86.53 / 82.97 | 93.32 / 86.47 | 84.83 / 74.12 | 95.42 / 91.40 |
| C45@10k / proportional | 88.69 / 80.57 | 84.35 / 75.93 | 87.10 / 83.52 | 93.68 / 88.21 | 88.59 / 77.26 | 95.91 / 92.46 |

C45@10k has zero non-finite rows for all groups and both source roots. It is
similar to C45@5k. Step 10k slightly improves proportional timeline-single and
repetition timeline-single plain R@3, but step 5k remains marginally better on
content/overview. Keep C45@5k as the primary checkpoint unless timeline-single
plain R@3 is the selection priority.

## 2026-07-15 matched C45 checkpoint sweep: every 1k step

The original C45 selection compared inline content/overview checkpoints and only ran the full six
groups for selected steps. To answer whether step 5k was simply too early, every saved checkpoint
from 1k through 10k was evaluated with the current `st_eval.py` over all six proportional-motion
testsuite groups. C45 is an official-checkpoint fine-tune, not training from scratch: step 5k is
about 640k sample presentations at batch 128, with the official text encoder frozen and official
motion encoder/decoder initialization.

Slurm jobs `10830` (steps 1k-5k), `10829` (6k-10k), and `10831` (matched 5k rerun) completed on an
A2 GPU. Initial wrappers `10827/10828` failed before Python because `sbatch --wrap` invoked
`/bin/sh`, which rejected `set -o pipefail`; the corrected wrappers explicitly invoked Bash. All
matched evaluations have zero non-finite C45 embeddings.

The current proportional data tree resolves these fixed pools:

```text
content:    overview=911, timeline_single=914, timeline_multi=785
repetition: overview=2374, timeline_single=2375, timeline_multi=1771
```

This is slightly larger than the historical 5k file for content-multi (`783`) and
repetition-multi (`1737`). The historical result remains in `full6_step_00005000.json`; the matched
current-pool rerun is `full6_current_step_00005000.json`. New comparisons must not silently mix
those pool versions.

Content/overview full R-precision by checkpoint:

| step | R@1 | R@2 | R@3 | R@5 | R@10 | plain R@1 | plain R@3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 56.31 | 74.42 | 87.38 | 94.95 | 98.46 | 36.99 | 80.13 |
| 2k | 55.32 | 74.42 | 88.25 | 95.06 | 98.57 | 35.89 | 80.57 |
| 3k | 55.43 | 74.86 | 89.02 | 95.39 | 98.24 | 35.57 | 80.79 |
| 4k | 54.56 | 74.86 | 88.14 | 94.62 | 98.02 | 35.13 | 80.46 |
| 5k | 52.36 | 74.31 | 89.13 | 95.06 | 97.91 | 34.03 | 80.90 |
| 6k | 53.79 | 74.20 | 87.71 | 94.73 | 97.80 | 33.92 | 79.47 |
| 7k | 55.32 | 75.41 | 88.58 | 94.62 | 97.91 | 35.24 | 80.79 |
| 8k | 54.88 | 74.64 | 88.25 | 94.95 | 97.80 | 36.11 | 80.02 |
| 9k | 54.34 | 74.86 | 88.69 | 95.06 | 97.91 | 35.02 | 80.46 |
| 10k | 54.67 | 74.75 | 88.69 | 94.84 | 97.91 | 35.46 | 80.57 |

Protocol MedR is `1` and plain MedR is `2` at every step. Step 5k remains the best checkpoint under
the explicitly chosen primary selection target: content/overview protocol and plain R@3.

Full six-group `R@3 / plain R@3`; macro is the unweighted mean over the six independently scored
groups:

| step | content ov | content single | content multi | rep ov | rep single | rep multi | macro |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 87.38/80.13 | 84.25/75.38 | 87.64/83.44 | 93.64/87.83 | 86.69/74.61 | 94.81/91.36 | 89.07/82.13 |
| 2k | 88.25/80.57 | 84.14/75.27 | 87.52/83.44 | 93.56/87.91 | 87.28/75.71 | 95.60/92.38 | 89.39/82.55 |
| 3k | 89.02/80.79 | 84.35/75.38 | 87.39/83.95 | 93.81/88.29 | 88.55/76.76 | 95.43/91.93 | 89.76/82.85 |
| 4k | 88.14/80.46 | 83.70/75.05 | 87.26/83.82 | 93.89/88.67 | 88.13/76.84 | 95.82/92.15 | 89.49/82.83 |
| 5k | 89.13/80.90 | 84.35/75.49 | 86.50/83.06 | 93.93/88.37 | 88.04/76.42 | 95.60/92.09 | 89.59/82.72 |
| 6k | 87.71/79.47 | 83.81/75.49 | 86.24/82.42 | 93.89/88.50 | 88.42/77.09 | 95.71/92.26 | 89.30/82.54 |
| 7k | 88.58/80.79 | 83.70/75.27 | 86.62/83.18 | 93.81/88.08 | 89.47/77.77 | 95.71/92.09 | 89.65/82.87 |
| 8k | 88.25/80.02 | 84.14/75.60 | 87.13/83.44 | 93.81/88.33 | 88.84/77.26 | 95.71/92.38 | 89.65/82.84 |
| 9k | 88.69/80.46 | 84.35/75.93 | 87.26/83.69 | 93.68/88.16 | 88.51/77.14 | 95.77/92.38 | 89.71/82.96 |
| 10k | 88.69/80.57 | 84.35/75.93 | 87.13/83.57 | 93.68/88.21 | 88.59/77.26 | 95.77/92.38 | 89.70/82.99 |

Selection readout:

- 5k is the best content/overview checkpoint (`89.13/80.90`) and therefore remains the primary
  evaluator under the established overview selection rule.
- 3k has the best six-group macro protocol R@3 (`89.76`).
- 10k has the best six-group macro plain R@3 (`82.99`) and case-weighted plain R@3 (`83.78`).
- 7k has the best case-weighted protocol R@3 (`90.90`).
- The narrow, non-monotonic spread from 1k to 10k shows early saturation of an official-model
  fine-tune, not evidence that 5k was an undertrained from-scratch model. Calling 5k universally
  best would nevertheless be inaccurate; 10k is a reasonable alternative when aggregate plain
  retrieval across every prompt group is the deployment target.

## 2026-07-15 portable shape-aware generation evaluation bundle

The exact evaluator/runtime package used for the full-contact generation evaluation was assembled
and uploaded to:

```text
gs://mm-jinhyung_kim/jungbin/shape_aware_motion_eval_c45_20260715/
```

The remote prefix was verified against the local payload at 404 objects and 510,225,540 bytes
(486.59 MiB). It includes:

- C45 step-5k (`step_00005000.pt`) and NVIDIA official 30-fps TMR motion stats;
- the full-contact generator step-200k checkpoint and its proportional 283-D generator mean/std;
- the benchmark LLM2Vec cache, shape-aware TMR code, MotionExpert code, Kimodo evaluation code, and
  the exact Cosmos Framework 1.2.2 UniPC scheduler source;
- reference metrics/configuration, `PROVENANCE.json`, `MANIFEST.tsv`, and `SHA256SUMS`;
- runnable `run_all_text2motion.sh` and `smoke_test.sh` entry points.

The much larger datasets were not duplicated because they already exist at:

```text
gs://mm-jinhyung_kim/jungbin/Kimodo-Motion-Gen-Benchmark-20fps/
gs://mm-jinhyung_kim/jungbin/soma_proportional_uniegomotion_20fps/
```

Relocation smoke job `10824` loaded the vendored official UniPC implementation plus relocated C45
and generator checkpoints, then completed an eight-case generation/retrieval/physical/shape pass.
Checksum job `10826` removed transient bytecode, regenerated the manifest, and verified every
payload hash. GCS object/byte totals matched exactly, and downloaded `README.md`,
`PROVENANCE.json`, and `SHA256SUMS` compared byte-for-byte with the local bundle. The included
generator is the completed step-200k checkpoint; the active step-500k continuation is intentionally
not represented as complete.
