## Sep-1 latest EMA models — fixed71, whole-bridge CFG2 (2026-11-07)

All three selected sep-1 caption-free no-joint models were evaluated for V2M and M2V on the same 71 test windows. Every model uses EMA weights in both the CFG1 baseline and CFG2 result. The exact saved checkpoints were pinned when this evaluation was prepared; ongoing training steps are not the checkpoint steps below. All 426 CFG2 predictions completed, with zero skipped windows.

Whole-bridge CFG2 improved V2M MPJPE and FID for all three checkpoints. M2V DreamSim and FVD worsened for all three. The matched per-case intervals support the MPJPE improvements and DreamSim regressions on this cohort; this does not establish variation across training seeds. Motion acceleration errors also increased, so the reconstruction gains do not imply uniformly better motion physics.

### Selected checkpoints

| Model | Checkpoint step | Video weight | EMA decay | EMA starts at step | EMA updates in checkpoint |
|---|---:|---:|---:|---:|---:|
| [sep-1 directional](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-244621e199b8191a46c6) | 60,000 | 10 | 0.999 | 50,043 | 9,957 |
| [sep-1 bidirectional](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-54fb3e983641efb7c839) | 60,000 | 10 | 0.999 | 50,143 | 9,857 |
| [sep-1 directional](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-d2d167e7b78773091498) | 20,000 | 1 | 0.999 | 0 | 20,000 |

The60k models continue the original video-weight10 runs with EMA introduced after their50k evaluations. The20k video-weight1 model was trained fresh with EMA from step0. Checkpoint ages, video loss weights and EMA histories differ, so this table cannot isolate their individual training effects. Within each model, CFG1 versus CFG2 is checkpoint/weight/case/seed matched.

### Video → motion

| Model / step | CFG | FID ↓ | MPJPE cm ↓ | PA-MPJPE cm ↓ | Root ADE cm ↓ | R32 R@1 % ↑ |
|---|---:|---:|---:|---:|---:|---:|
| directional, videoW10, 60k | 1 | 0.3413 | 25.26 | 11.48 | 23.70 | 20.83 |
| directional, videoW10, 60k | 2 | 0.2788 | 21.29 | 9.58 | 19.74 | 27.60 |
| bidirectional, videoW10, 60k | 1 | 0.3678 | 24.50 | 11.69 | 22.83 | 17.71 |
| bidirectional, videoW10, 60k | 2 | 0.3258 | 22.07 | 10.23 | 20.20 | 29.17 |
| directional, videoW1, 20k | 1 | 0.3056 | 24.04 | 11.22 | 22.04 | 20.31 |
| directional, videoW1, 20k | 2 | 0.2298 | 19.75 | 8.79 | 18.06 | 28.65 |

### Motion → video

| Model / step | CFG | I3D FVD ↓ | DreamSim ↓ | LPIPS-Alex ↓ | PSNR dB ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| directional, videoW10, 60k | 1 | 354.74 | 0.1419 | 0.2663 | 19.24 | 0.5991 |
| directional, videoW10, 60k | 2 | 399.47 | 0.1601 | 0.2700 | 18.48 | 0.5651 |
| bidirectional, videoW10, 60k | 1 | 362.01 | 0.1413 | 0.2588 | 19.53 | 0.6079 |
| bidirectional, videoW10, 60k | 2 | 393.85 | 0.1595 | 0.2636 | 18.69 | 0.5720 |
| directional, videoW1, 20k | 1 | 360.51 | 0.1482 | 0.2840 | 18.79 | 0.5836 |
| directional, videoW1, 20k | 2 | 401.92 | 0.1615 | 0.2827 | 18.15 | 0.5494 |

I3D FVD here uses the pinned native training-validator convention: six16-frame segments per window,426 I3D400 descriptors in total. It differs from the pooled71-descriptor FVD and VideoMAE CD-FVD used in the sep-2 section of this page. Do not directly rank those different definitions. These RGB metrics use lossless native decoded uint8 pixels on future frames1–96.

### Motion physics and GT reference

| Model / step | CFG | Acceleration error m/s² ↓ | Skate cm/s | Penetration cm | Airborne % |
|---|---:|---:|---:|---:|---:|
| directional, videoW10, 60k | 1 | 1.912 | 13.96 | 0.482 | 13.45 |
| directional, videoW10, 60k | 2 | 2.597 | 14.81 | 0.633 | 17.44 |
| bidirectional, videoW10, 60k | 1 | 1.997 | 13.21 | 0.553 | 17.74 |
| bidirectional, videoW10, 60k | 2 | 2.907 | 13.46 | 0.395 | 23.49 |
| directional, videoW1, 20k | 1 | 1.851 | 14.89 | 0.543 | 18.67 |
| directional, videoW1, 20k | 2 | 2.351 | 13.04 | 0.462 | 25.87 |
| Measured GT | reference | 0.000 | 18.51 | 1.306 | 16.42 |

Measured GT retrieval R@1/R@2/R@3: 48.96% / 63.02% / 70.83%. GT-to-itself FID is0 by definition. Ground-truth physics is measured with the identical reconstruction convention; lower skating or penetration alone does not establish better reconstruction. Full retrieval seed-group results, physical metrics, matching distance and diversity are in the attached evidence.

### Paired changes: CFG2 minus CFG1

| Model | Metric | Mean change | Paired95% interval |
|---|---|---:|---|
| directional_video10 | v2m mpjpe_cm | -3.97323 | [-6.88706, -1.50089] |
| directional_video10 | m2v dreamsim | +0.01827 | [+0.01367, +0.02285] |
| directional_video10 | m2v lpips_alex | +0.00367 | [-0.00064, +0.00800] |
| bidirectional_video10 | v2m mpjpe_cm | -2.42977 | [-4.54149, -0.31146] |
| bidirectional_video10 | m2v dreamsim | +0.01821 | [+0.01340, +0.02281] |
| bidirectional_video10 | m2v lpips_alex | +0.00484 | [-0.00017, +0.00956] |
| directional_video1 | v2m mpjpe_cm | -4.29147 | [-6.60160, -2.24036] |
| directional_video1 | m2v dreamsim | +0.01332 | [+0.00840, +0.01810] |
| directional_video1 | m2v lpips_alex | -0.00127 | [-0.00797, +0.00479] |

Intervals use10000 paired resamples of the71 cases with a fixed seed. They describe cohort sampling variation, not training-seed variation. No significance or confidence interval is claimed for distribution-level FID/FVD. R32 retrieval uses the pinned three32-candidate groupings,64 cases per grouping and7 dropped; motion FID uses all71.

### Sampling, guidance and verification

- Exact fixed71 bundles/order, supplied neutral shapes and seeds0–70;97 frames at20fps,256×256. All model captions are empty; text is used only by the external retrieval evaluator.
- Native35-step UniPC; motion shift3/video shift10. Sampling batch32 with identical virtual two-rank batch positions to cached native CFG1; decode batch4. No batch16/32/64 speed benchmark is claimed.
- V2M: clean observed video and supplied neutral shape; motion starts from Gaussian noise in normalized283-D space. M2V: full GT motion, supplied neutral shape and one initial image; future video starts from Gaussian noise, and the first latent remains clamped.
- Whole-bridge CFG2 performs two evaluations each step: all seven bridges enabled versus `bridges_enabled=False`. The reference removes both bridge within-modality and cross-modal updates. It retains the frozen specialists, empty captions and supplied conditioning tensors. Guidance is `reference + 2 × (conditioned − reference)`.
- Motion x0 predictions are blended before native conversion to velocity; shared state/sigma make this algebraically equivalent to blending velocities. Video predictions are already velocity. This is inference extrapolation against a bridge-off reference; it does not assume dropped-modality null training.
- EMA selection covers the seven trainable bridges; these runs have no additional Phase3 LoRA. The frozen Phase1 video-I/O EMA200k and Phase2 200k specialists remain unchanged.
- Each EMA tensor was checked against the checkpoint after loading. The exact live run source/data/evaluator fingerprints were verified. Both modalities reproduce cached native CFG1 bit-for-bit for32 cases/model at batch32, in addition to native/wrapper single-case parity. Reference-source invariance, nonzero guidance differences, zero reference bridge calls, finite samples, unchanged bridge tensors and fixed initial video latents passed.
- All training continued. Evaluations used only the previously coordinated GPUs 3/7 with a 30% process memory cap and one evaluator per GPU; the third model was queued. All three evaluation processes have exited. No training source snapshot, run setting, optimizer or W&B training history is changed.

### Reproducibility

Checkpoint selection: <date start="2026-11-07" start-time="01:46" time-zone="Etc/UTC" />. Exact selection timestamp: `2026-11-07T01:46:33.961656+00:00`. Operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_latest_ema_wholecfg2_fixed71_20261107T014633Z`.

directional_video10, step60000: checkpoint SHA256 `4f21e314ec2214ce6aee7225b9ce81efd86832b13c1a7317ecc51d377c243bc7`.

bidirectional_video10, step60000: checkpoint SHA256 `268c246ba07de9f0ff11a4227f5f5985f038af43589baff85b52a40f84b263ef`.

directional_video1, step20000: checkpoint SHA256 `97fd379e817d8e1f06b9bd8ca7a5fbed8bfd1e50edb9ac2aa8a0c0ffbfd3b7f6`.

The attached CSV has the full numeric summaries. Each model JSON includes cases/seeds/bundle checksums, per-case CFG1/CFG2 metrics, GT references, paired changes, source/evaluator/checkpoint identities and acceptance receipts. All sampled latent arrays remain in the node-local operation.

Published with metric CSV and three per-model evidence JSON files: [Notion report](https://app.notion.com/p/binne/devugged-MMdiT-410d2a44f44b8098b1b7c7af42d2353b). See `PUBLISHED.json` in the operation for upload/write receipts. All evaluator processes exited, and all six training ranks remained active.
