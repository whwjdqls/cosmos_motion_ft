## Sep-1 M2V at 80k/80k/40k: whole-bridge CFG 2 versus CFG 1

All 213 new M2V predictions completed: the same fixed 71 test windows for each of the three EMA checkpoints used in the preceding V2M evaluation. CFG 2 worsened DreamSim for 3/3 models and I3D FVD for 3/3 models relative to their matched CFG 1 results.

### Quantitative results

| Model / EMA step | CFG | DreamSim ↓ | I3D FVD ↓ | LPIPS-Alex ↓ | PSNR dB ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Directional, video weight 10, 80k | 1 | 0.14099 | 364.60 | 0.25322 | 19.662 | 0.61110 |
| Directional, video weight 10, 80k | 2 | 0.15810 | 403.56 | 0.25941 | 18.756 | 0.57651 |
| Bidirectional, video weight 10, 80k | 1 | 0.14063 | 369.37 | 0.25476 | 19.637 | 0.61137 |
| Bidirectional, video weight 10, 80k | 2 | 0.15897 | 397.34 | 0.25959 | 18.723 | 0.57473 |
| Directional, video weight 1, 40k | 1 | 0.14464 | 349.74 | 0.26637 | 19.234 | 0.59874 |
| Directional, video weight 1, 40k | 2 | 0.15839 | 396.75 | 0.26859 | 18.516 | 0.56533 |

### Paired changes: CFG 2 minus CFG 1

| Model | Metric | Mean change | Paired 95% interval |
|---|---|---:|---|
| Directional, video weight 10, 80k | dreamsim | +0.01711 | [+0.01210, +0.02217] |
| Directional, video weight 10, 80k | lpips_alex | +0.00619 | [+0.00247, +0.00975] |
| Directional, video weight 10, 80k | psnr_db | -0.90608 | [-1.06847, -0.74132] |
| Directional, video weight 10, 80k | ssim | -0.03459 | [-0.04037, -0.02923] |
| Bidirectional, video weight 10, 80k | dreamsim | +0.01833 | [+0.01288, +0.02364] |
| Bidirectional, video weight 10, 80k | lpips_alex | +0.00483 | [-0.00021, +0.00951] |
| Bidirectional, video weight 10, 80k | psnr_db | -0.91347 | [-1.08296, -0.73771] |
| Bidirectional, video weight 10, 80k | ssim | -0.03665 | [-0.04267, -0.03107] |
| Directional, video weight 1, 40k | dreamsim | +0.01375 | [+0.00912, +0.01815] |
| Directional, video weight 1, 40k | lpips_alex | +0.00223 | [-0.00416, +0.00815] |
| Directional, video weight 1, 40k | psnr_db | -0.71769 | [-0.94134, -0.48562] |
| Directional, video weight 1, 40k | ssim | -0.03342 | [-0.03984, -0.02685] |

Intervals use 10,000 paired resamples of the 71 windows with a fixed seed. They describe cohort variation, not training-seed variation. No interval is claimed for FVD. Lower DreamSim/LPIPS and higher PSNR/SSIM are better. The full numeric summaries and per-case values are attached.

### Sampling and conditioning

- Same EMA checkpoints as the preceding V2M evaluation: directional video weight 10 at 80k, bidirectional video weight 10 at 80k, directional video weight 1 at 40k. All use caption-free 50% V2M / 50% M2V / 0% joint training; no text is supplied during this M2V evaluation.
- Full clean GT motion in normalized 283-D space, supplied neutral body shape and one initial image condition video generation. Future video latents begin as per-case Gaussian noise. The first video latent stays fixed after every denoising update.
- Native 35-step UniPC, video shift 10, 97 frames at 20 fps and 256×256. Cases and seeds 0–70 are identical across CFG 1 and CFG 2.
- Whole-bridge CFG uses `v = v_off + 2 × (v_on − v_off)`. Both branches retain the same input tensors and empty captions; `bridges_enabled=False` bypasses all seven learned bridges, including their within-modality updates. The frozen specialists stay active. The bridge-off video reference is invariant to changing the supplied motion/shape, which is checked directly.
- Sampling batch 32 uses the identical native validation virtual ranks and batch positions: even cases 32+4, odd cases 32+3. Decode batch 4 has an OOM backoff. This reuses the matched cached native CFG 1 controls without introducing a different batching convention.

### Scoring and verification

- Scores use lossless native decoded uint8 RGB on future frames 1–96, excluding the supplied image. DreamSim, LPIPS-Alex, PSNR and SSIM reuse the pinned native training-validation evaluator and assets.
- I3D FVD follows the native training-validator convention: six 16-frame segments per window, 426 I3D-400 descriptors. This differs from the pooled 71-clip I3D FVD and VideoMAE CD-FVD in other page sections. Compare only identical definitions. GT-self FVD is checked below 1e-4.
- Each EMA tensor matches its checkpoint after loading. Frozen specialist identities, exact source snapshots, normalization/case/evaluator fingerprints, native/wrapper single-case CFG 1 parity and 32-case cached native CFG 1 parity are checked. Both parity checks are bit-exact.
- Finite samples, unchanged first latent, unchanged bridge tensors, nonzero conditioned/reference differences and zero reference bridge calls are required. All 71 cases per model are present with no skipped windows.
- Evaluation used the approved co-evaluation slots on GPUs 3 and 7 with 30% per-process memory caps and nice 10. All training processes and settings were left in place.

### EMA history and measured sampling time

| Model | EMA starts | EMA updates | Sampling batches | Sampling seconds |
|---|---:|---:|---|---:|
| Directional, video weight 10, 80k | 50,043 | 29,957 | 32, 4, 32, 3 | 256.9 |
| Bidirectional, video weight 10, 80k | 50,143 | 29,857 | 32, 4, 32, 3 | 259.7 |
| Directional, video weight 1, 40k | 0 | 40,000 | 32, 4, 32, 3 | 258.4 |

The video-weight-10 models introduced EMA after 50k; the video-weight-1 model used EMA from step 0. Their checkpoint ages, video loss weights and EMA histories differ. Within each model, CFG is the changed inference setting.

### Reproducibility

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_m2v_wholecfg2_fixed71_20261107T083035Z`. Model checkpoint selection remains the preceding V2M selection at <date start="2026-11-07" start-time="07:50" time-zone="Etc/UTC" />. Evaluation preparation UTC: `2026-11-07T08:30:35.702568+00:00`. All 213 sampled video-latent arrays remain in the node-local operation. The attached CSV and model JSON files contain the exact metrics, per-case results, cases/seeds/checksums, checkpoint identities, evaluator identities and acceptance receipts.

directional_video10 checkpoint SHA256: `15d2673475bb7a17c7d6b55befd1e39d790187f7b81143190a02a12b363e3a8a`.

bidirectional_video10 checkpoint SHA256: `fc994300e342a25a2d77316285143fdfd87abb8de3bc673d518afe324259aa79`.

directional_video1 checkpoint SHA256: `e38483ecfa72b3017bc2f5d083bf0cbec28a4d55d30d0bd16cadee0db81a3dba`.
