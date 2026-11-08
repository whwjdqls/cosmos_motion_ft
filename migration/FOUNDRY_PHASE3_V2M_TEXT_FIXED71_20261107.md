## Sep-1 latest EMA V2M: with and without text, whole-bridge CFG2

**Adding text improved MPJPE, motion FID and R@1 retrieval for all three checkpoints.** The paired 95% intervals for MPJPE improvement exclude zero on this cohort.

All three recent sep-1 caption-free Phase3 models (50% V2M / 50% M2V / 0% joint) were evaluated on the exact fixed 71 test windows, with paired seeds 0–70. There are 426 new predictions: 71 windows × 3 models × 2 conditioning variants. Model checkpoints are pinned at 80k/80k/40k; training continued during evaluation.

The text condition is an explicit inference ablation: the ground-truth test caption enters the frozen Phase2 motion expert’s reasoner, while the video specialist still receives an empty caption. These bridges were trained without captions, so this measures zero-shot text conditioning at Phase3 inference. It does not describe text-trained Phase3 models.

### V2M reconstruction and retrieval

| Model / EMA step | Conditioning | MPJPE cm ↓ | PA-MPJPE cm ↓ | Root ADE cm ↓ | FID ↓ | R32 R@1 % ↑ | Matching distance ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Directional, video weight 10, 80k | Video | 17.12 | 7.87 | 15.37 | 0.2057 | 30.73 | 0.8757 |
| Directional, video weight 10, 80k | Video + text | 11.79 | 6.25 | 10.33 | 0.1338 | 42.71 | 0.7644 |
| Bidirectional, video weight 10, 80k | Video | 17.72 | 8.59 | 16.16 | 0.2139 | 32.81 | 0.8805 |
| Bidirectional, video weight 10, 80k | Video + text | 13.89 | 7.00 | 12.56 | 0.1364 | 41.67 | 0.7500 |
| Directional, video weight 1, 40k | Video | 14.99 | 7.16 | 13.41 | 0.1663 | 34.90 | 0.8403 |
| Directional, video weight 1, 40k | Video + text | 11.87 | 6.09 | 10.54 | 0.1234 | 45.83 | 0.7485 |

### Effect of adding text at CFG2

| Model | MPJPE change cm ↓ | Paired 95% interval cm | FID change ↓ | R@1 change pp ↑ |
|---|---:|---|---:|---:|
| Directional, video weight 10, 80k | -5.330 | [-8.769, -2.554] | -0.0719 | +11.98 |
| Bidirectional, video weight 10, 80k | -3.828 | [-7.064, -1.072] | -0.0775 | +8.85 |
| Directional, video weight 1, 40k | -3.116 | [-5.431, -1.196] | -0.0429 | +10.94 |

Negative reconstruction/FID changes favor text. Intervals use 10,000 paired resamples of the 71 cases with a fixed seed; they measure cohort variation, not training-seed variation. No FID significance interval is claimed. Ground-truth captions are additional information, so video+text is a different conditioning task.

### Motion dynamics and ground truth

| Model / step | Conditioning | Acceleration error m/s² ↓ | Skate cm/s | Penetration cm | Airborne % |
|---|---|---:|---:|---:|---:|
| Directional, video weight 10, 80k | Video | 2.506 | 12.57 | 0.480 | 22.68 |
| Directional, video weight 10, 80k | Video + text | 2.401 | 13.63 | 0.325 | 17.00 |
| Bidirectional, video weight 10, 80k | Video | 2.865 | 12.81 | 0.299 | 23.99 |
| Bidirectional, video weight 10, 80k | Video + text | 2.743 | 12.48 | 0.137 | 16.97 |
| Directional, video weight 1, 40k | Video | 2.337 | 13.66 | 0.645 | 19.05 |
| Directional, video weight 1, 40k | Video + text | 2.255 | 13.81 | 0.394 | 15.80 |
| Measured GT | Reference | 0.000 | 18.51 | 1.306 | 16.42 |

Measured GT R32 R@1/R@2/R@3: 48.96% / 63.02% / 70.83%. FID uses all 71 motion samples. R32 uses three pinned 32-candidate groupings, 64 cases per grouping and 7 dropped. Evaluator: matched Nymeria TMR SHA256 `91f9d8adec6bcaa448571dade670da0f94916e448b875a0853eda29ef44a1e21`.

### Checkpoints, sampling and verification

- EMA weights only; directional/bidirectional videoW10 are 80k with EMA introduced after 50k, while directional videoW1 is 40k with EMA from step 0. These models differ in checkpoint age, video weight and EMA history; cross-model comparisons do not isolate those effects.
- V2M uses full clean observed video latents and supplied neutral body shape. Motion starts from independent Gaussian noise in normalized 283-D space. There is no ground-truth motion prefix. Each window is 97 frames at 20 fps, 256×256.
- Native 35-step UniPC, motion shift 3. Whole-bridge CFG2 is `reference + 2 × (bridges_on − reference)`, where reference bypasses all seven learned bridges, including their within-modality updates. Frozen specialists remain active. The same GT caption is retained in both CFG passes for video+text: CFG guides bridge/video influence conditional on text; no separate text CFG is applied.
- Text routing is isolated to this operation. Native caption stripping remains in place. A scoped hook explicitly feeds GT captions to `motion.prepare_conditions`; generator packing asserts empty captions. Fixed-input checks require motion outputs to differ between empty, GT and unrelated captions, including a change in the bridge-off reference. The scope restores after normal execution and exceptions.
- Both conditioning variants use identical batch positions and seeds. Requested sampling batch 64, paired OOM backoff if needed. Actual batches/timings appear below and in receipts. Frozen text word embeddings are cached; no diffusion state or reasoner hidden state is reused.
- Blank-hook/native single-case CFG1 parity and cached native validation batch 32 parity pass bit-for-bit. Bridge-off source invariance, zero bridge calls in the reference, finite arrays and unchanged bridge tensors are checked. Each loaded EMA tensor equals its checkpoint tensor, and specialist/source/evaluator identities are verified.
- Evaluations used the explicitly approved GPUs 3/7 alongside training, with 30% per-process GPU memory caps and nice 10. All training source/settings/optimizer states remain unchanged.

### Actual batch timing

| Model | Batch sizes | Paired sampling seconds | EMA starts | EMA updates |
|---|---|---:|---:|---:|
| directional_video10 | 64, 7 | 615.2 | 50,043 | 29,957 |
| bidirectional_video10 | 64, 7 | 623.3 | 50,143 | 29,857 |
| directional_video1 | 64, 7 | 626.6 | 0 | 40,000 |

### Reproducibility

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_latest_v2m_text_fixed71_20261107T075018Z`. Checkpoint selected <date start="2026-11-07" start-time="07:50" time-zone="Etc/UTC" />. Exact UTC timestamp: `2026-11-07T07:50:18.893211+00:00`. All 426 normalized motion arrays and full per-case metrics are stored locally. Attached model JSON files contain checkpoint/source identities, exact cases/captions/seeds, token routing receipts, quantitative summaries and paired changes. The CSV contains every summary metric.

directional_video10 checkpoint SHA256: `15d2673475bb7a17c7d6b55befd1e39d790187f7b81143190a02a12b363e3a8a`.

bidirectional_video10 checkpoint SHA256: `fc994300e342a25a2d77316285143fdfd87abb8de3bc673d518afe324259aa79`.

directional_video1 checkpoint SHA256: `e38483ecfa72b3017bc2f5d083bf0cbec28a4d55d30d0bd16cadee0db81a3dba`.
