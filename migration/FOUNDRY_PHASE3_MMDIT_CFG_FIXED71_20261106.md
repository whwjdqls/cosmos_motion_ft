## Sep-1 no-joint MMDiT 10k: whole-bridge CFG comparison

This section evaluates the sep-1 caption-free directional and bidirectional models trained with 50% V2M / 50% M2V / 0% joint objective. These are different checkpoints from the sep-2 joint models elsewhere on this page. Both checkpoints are pinned at step 10,000, the latest saved checkpoints when this evaluation began. All 71 windows per task/model completed under each guidance policy. All model captions are empty.

Whole-bridge CFG2 improves directional V2M versus CFG1: MPJPE 35.32→29.84 cm and FID 0.6182→0.5390. Bidirectional MPJPE is 38.99→40.97 cm, with its paired change interval crossing zero. Whole-bridge guidance substantially improves V2M over the earlier cross-edge reference for both models, but worsens M2V DreamSim and FVD versus CFG1 for both. A single guidance reference does not improve both tasks in this experiment.

### Video → motion

| Model | Guidance | FID ↓ | MPJPE cm ↓ | PA-MPJPE cm ↓ | Root ADE cm ↓ | Skate cm/s | R32 R@1 % ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| directional | CFG 1 | 0.6182 | 35.32 | 18.14 | 32.61 | 12.74 | 13.02 |
| directional | Cross-edge CFG 2 | 1.4088 | 62.27 | 26.27 | 55.69 | 99.68 | 8.85 |
| directional | Whole-bridge CFG 2 | 0.5390 | 29.84 | 15.04 | 26.65 | 16.29 | 17.19 |
| bidirectional | CFG 1 | 0.7586 | 38.99 | 19.76 | 35.92 | 13.37 | 13.54 |
| bidirectional | Cross-edge CFG 2 | 1.1835 | 50.49 | 23.27 | 45.46 | 29.93 | 14.58 |
| bidirectional | Whole-bridge CFG 2 | 0.8172 | 40.97 | 19.38 | 37.40 | 21.81 | 13.54 |

### Motion → video

| Model | Guidance | I3D FVD ↓ | DreamSim ↓ | LPIPS-Alex ↓ | PSNR dB ↑ | SSIM ↑ |
|---|---|---:|---:|---:|---:|---:|
| directional | CFG 1 | 390.24 | 0.1548 | 0.3055 | 18.28 | 0.5692 |
| directional | Cross-edge CFG 2 | 377.02 | 0.1482 | 0.2991 | 18.16 | 0.5652 |
| directional | Whole-bridge CFG 2 | 415.55 | 0.1680 | 0.3031 | 17.62 | 0.5328 |
| bidirectional | CFG 1 | 381.74 | 0.1518 | 0.2994 | 18.39 | 0.5739 |
| bidirectional | Cross-edge CFG 2 | 361.06 | 0.1498 | 0.2953 | 18.06 | 0.5595 |
| bidirectional | Whole-bridge CFG 2 | 402.46 | 0.1669 | 0.3007 | 17.68 | 0.5322 |

FVD here reuses the native sep-1 training validator: 426 descriptors (six 16-frame segments per window). The pooled 71-descriptor FVD reported for sep-2 above uses a different aggregation and must not be directly ranked against this table. RGB metrics score future frames 1–96. Lower skating alone does not establish improved reconstruction; GT retrieval reference metrics are included in the evidence files; no new GT physics baseline is computed here.

### Paired whole-bridge changes

| Model | Reference | Metric | Whole-bridge CFG 2 minus reference | Paired 95% interval |
|---|---|---|---:|---|
| directional | CFG 1 | v2m mpjpe_cm | -5.47876 | [-8.34042, -3.03154] |
| directional | Cross-edge CFG 2 | v2m mpjpe_cm | -32.43030 | [-38.75312, -26.16037] |
| directional | CFG 1 | m2v dreamsim | +0.01320 | [+0.00697, +0.01936] |
| directional | CFG 1 | m2v lpips_alex | -0.00238 | [-0.00916, +0.00381] |
| directional | Cross-edge CFG 2 | m2v dreamsim | +0.01977 | [+0.01479, +0.02513] |
| directional | Cross-edge CFG 2 | m2v lpips_alex | +0.00397 | [-0.00305, +0.01045] |
| bidirectional | CFG 1 | v2m mpjpe_cm | +1.97558 | [-1.15299, +5.12923] |
| bidirectional | Cross-edge CFG 2 | v2m mpjpe_cm | -9.51937 | [-14.51716, -4.94307] |
| bidirectional | CFG 1 | m2v dreamsim | +0.01512 | [+0.00925, +0.02079] |
| bidirectional | CFG 1 | m2v lpips_alex | +0.00125 | [-0.00462, +0.00692] |
| bidirectional | Cross-edge CFG 2 | m2v dreamsim | +0.01710 | [+0.01262, +0.02132] |
| bidirectional | Cross-edge CFG 2 | m2v lpips_alex | +0.00539 | [-0.00045, +0.01145] |

Intervals use 10,000 paired resamples of these 71 cases with a fixed seed. They describe cohort sampling variation, not variation across training seeds. No uncertainty or significance ranking is claimed for FID/FVD. Full quantitative values and all paired changes are attached.

### Exact sampling protocol

- Same fixed71 manifests, per-case seeds 0–70, bundle order, supplied neutral shapes and pinned frozen Phase1 video-I/O EMA200k + Phase2 200k specialists across policies.
- Native 35-step UniPC; motion shift 3, video shift 10; 97 frames at 20 fps, 256×256. Batch 32 with the same virtual two-rank case/batch positions as the CFG1 validator; no batch-size benchmark is claimed.
- V2M: clean observed video plus supplied neutral shape; motion starts as Gaussian noise in normalized 283-D space. M2V: full GT motion, supplied neutral shape, one initial image; future video starts as Gaussian noise. The first video latent is clamped throughout denoising.
- Whole-bridge CFG 2: each step evaluates the frozen-specialist model with all seven learned bridges enabled, then with `bridges_enabled=False`. The reference skips bridge within-modality attention and cross-modal updates in both streams. Specialist attention and supplied conditioning tensors stay intact.
- Cross-edge CFG 2: the earlier reference removes only target-query→conditioning-key edges. It retains bridge within-modality attention and the reverse cross-modal direction. This is a separate reference policy.
- Both apply `reference + 2 × (conditioned − reference)`. For V2M the implementation blends predicted x0 before native conversion to velocity. Since state and sigma are shared, this is algebraically equivalent to blending the two velocity predictions. M2V blends velocity directly. This is non-text inference extrapolation; a dropped-modality null training distribution is not assumed.
- No caption enters either specialist. External TMR retrieval uses reference captions only for scoring, with the pinned matched evaluator and completion/statistics assets.

### Verification and provenance

Both tasks/models passed bit-exact native-versus-wrapper CFG1 checks, reference invariance to the removed source, nonzero conditioned/reference differences, finite outputs, and exact initial-video-latent preservation. Hooks assert zero bridge calls in the whole-bridge reference and seven bridge calls per sample per conditioned step. A separate batch32 reproduction matched cached CFG1 outputs bit-for-bit for the first32 cases per task/model. No optimizer update occurs; bridge parameter version counters remain unchanged. Training processes were left active.

directional checkpoint SHA256: `2a02cd3c6983f64e11505bfb4b466ab7260d3dc2420305419a2e4214c9e7d2df`; step 10,000.

bidirectional checkpoint SHA256: `0c912e014eff25fe65f51c21979b8ffb614360405ab9bd93f48e1e2de6dfdfbb`; step 10,000.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_phase3_cfg2_fixed71_20261106T032026Z/whole_bridge`. All sampling, metric and evidence outputs are node-local. The attachments contain the full tables, per-case metrics, acceptance receipts, source/evaluator fingerprints and paired changes. The complete sampled latent arrays remain in the operation directory.

## Repository evidence and publication

Published quantitative tables and per-case JSON evidence: [Notion report](https://app.notion.com/p/binne/devugged-MMdiT-410d2a44f44b8098b1b7c7af42d2353b). Existing sep-2 joint results on that page are preserved.

[Exact evaluation sources, plans, full metric CSV and paired changes](evidence/phase3_mmdit_cfg_fixed71_20261106/README.md) are archived with SHA256 checksums. Both evaluation processes exited successfully after all284 whole-bridge samples were generated and scored; the original training processes remained active. No training or production sampler source was changed by this evaluation.
