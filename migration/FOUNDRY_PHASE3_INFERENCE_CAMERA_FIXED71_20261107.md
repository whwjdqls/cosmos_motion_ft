## Sep-1 inference-only camera prediction: matched V2M and M2V ablation

**Adding camera prediction only at inference did not produce a consistent gain.** V2M MPJPE and motion FID point estimates worsened in all six comparisons. The paired MPJPE intervals exclude zero for the 80k models, while both 40k intervals include zero. M2V CFG1 DreamSim worsened for all three models. At CFG2, PSNR/SSIM point estimates improved, but all DreamSim change intervals include zero and FVD worsened for all three models.

All 852 new predictions completed: three EMA checkpoints × four settings × the same fixed 71 test windows. Each camera-on row is paired with its exact previous camera-off checkpoint, cases, seeds, text condition, target guidance and batching protocol. No training was changed.

These Phase3 bridges were trained without camera tokens. This experiment adds native camera tokens only at inference, using the frozen Phase1 camera I/O heads already present in each specialist. It tests the effect of auxiliary camera prediction on motion/video generation; camera accuracy itself is not evaluated.

### Video → motion

| Model / step | Text | Predict camera | MPJPE cm ↓ | PA-MPJPE cm ↓ | Root ADE cm ↓ | Motion FID ↓ | R32 R@1 % ↑ |
|---|---|---|---:|---:|---:|---:|---:|
| Directional, video weight 10, 80k | Empty | No | 17.120 | 7.867 | 15.372 | 0.20566 | 30.73 |
| Directional, video weight 10, 80k | Empty | Yes | 20.112 | 9.350 | 18.263 | 0.25869 | 28.65 |
| Directional, video weight 10, 80k | GT caption | No | 11.790 | 6.250 | 10.325 | 0.13379 | 42.71 |
| Directional, video weight 10, 80k | GT caption | Yes | 15.946 | 8.102 | 14.512 | 0.17269 | 38.54 |
| Bidirectional, video weight 10, 80k | Empty | No | 17.717 | 8.585 | 16.157 | 0.21388 | 32.81 |
| Bidirectional, video weight 10, 80k | Empty | Yes | 24.136 | 12.516 | 22.451 | 0.27825 | 29.17 |
| Bidirectional, video weight 10, 80k | GT caption | No | 13.890 | 7.004 | 12.560 | 0.13637 | 41.67 |
| Bidirectional, video weight 10, 80k | GT caption | Yes | 20.811 | 10.949 | 19.624 | 0.16403 | 42.71 |
| Directional, video weight 1, 40k | Empty | No | 14.987 | 7.164 | 13.413 | 0.16631 | 34.90 |
| Directional, video weight 1, 40k | Empty | Yes | 15.647 | 7.555 | 14.117 | 0.18364 | 33.85 |
| Directional, video weight 1, 40k | GT caption | No | 11.871 | 6.086 | 10.541 | 0.12342 | 45.83 |
| Directional, video weight 1, 40k | GT caption | Yes | 12.451 | 6.571 | 11.128 | 0.14127 | 42.71 |

Both V2M settings use whole-bridge CFG 2. For video+text, the same GT caption enters the frozen motion expert in both guidance branches; video-specialist captions remain empty. This is zero-shot text at Phase3 inference, matching the preceding text ablation.

### Motion → video

| Model / step | CFG | Predict camera | DreamSim ↓ | Native I3D FVD ↓ | LPIPS-Alex ↓ | PSNR dB ↑ | SSIM ↑ |
|---|---:|---|---:|---:|---:|---:|---:|
| Directional, video weight 10, 80k | 1 | No | 0.14099 | 364.60 | 0.25322 | 19.662 | 0.61110 |
| Directional, video weight 10, 80k | 1 | Yes | 0.14504 | 373.24 | 0.26284 | 19.485 | 0.60760 |
| Directional, video weight 10, 80k | 2 | No | 0.15810 | 403.56 | 0.25941 | 18.756 | 0.57651 |
| Directional, video weight 10, 80k | 2 | Yes | 0.15766 | 408.69 | 0.26510 | 18.935 | 0.58633 |
| Bidirectional, video weight 10, 80k | 1 | No | 0.14063 | 369.37 | 0.25476 | 19.637 | 0.61137 |
| Bidirectional, video weight 10, 80k | 1 | Yes | 0.14567 | 369.23 | 0.26564 | 19.398 | 0.60622 |
| Bidirectional, video weight 10, 80k | 2 | No | 0.15897 | 397.34 | 0.25959 | 18.723 | 0.57473 |
| Bidirectional, video weight 10, 80k | 2 | Yes | 0.15871 | 406.36 | 0.26473 | 18.881 | 0.58464 |
| Directional, video weight 1, 40k | 1 | No | 0.14464 | 349.74 | 0.26637 | 19.234 | 0.59874 |
| Directional, video weight 1, 40k | 1 | Yes | 0.14773 | 365.20 | 0.27236 | 19.139 | 0.59759 |
| Directional, video weight 1, 40k | 2 | No | 0.15839 | 396.75 | 0.26859 | 18.516 | 0.56533 |
| Directional, video weight 1, 40k | 2 | Yes | 0.15915 | 407.02 | 0.27321 | 18.599 | 0.57464 |

M2V captions are empty in both specialists. Scores use lossless decoded uint8 RGB on future frames 1–96, excluding the supplied first image. Native I3D FVD uses six 16-frame segments per window (426 I3D-400 descriptors), matching the camera-off controls; it differs from pooled 71-clip I3D and VideoMAE CD-FVD reported elsewhere.

### Paired camera-on minus camera-off effects

| Model | Setting | Metric | Mean change | Paired 95% interval |
|---|---|---|---:|---|
| Directional, video weight 10, 80k | v2m_cfg2 | mpjpe_cm | +2.99248 | [+1.72594, +4.33075] |
| Directional, video weight 10, 80k | v2m_text_cfg2 | mpjpe_cm | +4.15631 | [+2.89700, +5.54104] |
| Directional, video weight 10, 80k | m2v_cfg1 | dreamsim | +0.00405 | [+0.00205, +0.00611] |
| Directional, video weight 10, 80k | m2v_cfg2 | dreamsim | -0.00044 | [-0.00232, +0.00160] |
| Bidirectional, video weight 10, 80k | v2m_cfg2 | mpjpe_cm | +6.41874 | [+4.06219, +9.08355] |
| Bidirectional, video weight 10, 80k | v2m_text_cfg2 | mpjpe_cm | +6.92111 | [+4.60412, +9.59079] |
| Bidirectional, video weight 10, 80k | m2v_cfg1 | dreamsim | +0.00504 | [+0.00290, +0.00729] |
| Bidirectional, video weight 10, 80k | m2v_cfg2 | dreamsim | -0.00026 | [-0.00244, +0.00192] |
| Directional, video weight 1, 40k | v2m_cfg2 | mpjpe_cm | +0.65936 | [-0.27508, +1.56203] |
| Directional, video weight 1, 40k | v2m_text_cfg2 | mpjpe_cm | +0.57999 | [-0.17553, +1.31826] |
| Directional, video weight 1, 40k | m2v_cfg1 | dreamsim | +0.00309 | [+0.00126, +0.00487] |
| Directional, video weight 1, 40k | m2v_cfg2 | dreamsim | +0.00076 | [-0.00192, +0.00352] |

Negative changes favor camera prediction for both metrics shown. Intervals use 10,000 paired case resamples with a fixed seed and describe cohort variation, not training-seed variation. No confidence interval is claimed for distribution-level FID/FVD. Other paired frame/reconstruction metrics are in the attachments.

### Ground truth and motion physics

Measured GT R32 R@1/R@2/R@3: 48.96% / 63.02% / 70.83%. GT-to-itself motion FID is zero. Measured GT skating 18.51 cm/s, penetration 1.306 cm and airborne 16.42%. Full generated physics metrics are in the numeric evidence. Video GT identity is PSNR capped at 100, SSIM 1, LPIPS 0, DreamSim approximately 0, FVD 0; each video scorer checks GT-self FVD below 1e-4.

### Camera inference and guidance

- Add 96 native camera/action tokens, domain 2, raw 9-D backward framewise relative SE(3) padded to 64. All camera tokens start from Gaussian noise with an independent per-case seed 131071 + case_seed. No GT camera is loaded or supplied, and no camera token is marked clean.
- Camera tokens participate in native generator video/action attention. The seven MMDiT bridges still connect exactly 1600 video and 98 motion tokens per sample. There are no direct camera-motion bridge edges; camera influences motion through native generator video hidden states.
- The camera scheduler uses shift 3 with noisy motion in V2M and shift 10 with noisy video in M2V. Camera, motion and video predictions at a step all see the same pre-update states. V2M keeps the observed video latents unchanged; M2V keeps full GT motion unchanged and repins the first video latent after every update.
- Auxiliary camera velocity uses the native conditional prediction, with no direct CFG extrapolation. Camera trajectories can nevertheless change indirectly as guided motion/video states evolve. This choice holds the previous target-guidance convention fixed.
- Whole-bridge CFG remains `target_off + scale × (target_on − target_off)`. Both passes share the current video/motion/camera states. The reference disables all seven learned bridges, including their within-modality updates, while camera tokens remain in native generator attention. This is not selective cross-modal-only guidance and does not assume a trained dropped-condition null.
- Native 35-step UniPC, 97 frames / 20 fps / 256×256, fixed seeds 0–70 and supplied neutral shapes. V2M batches 64 + 7 match preceding V2M controls. M2V uses native virtual-rank batches 32 + 4 and 32 + 3; decode batch 4. Actual batch sizes and timing are recorded, with OOM backoff if needed.

### Verification and provenance

- Every loaded EMA tensor equals its checkpoint value. Frozen specialist/source/case/evaluator identities and the original caption-free recipe are checked. No camera-training recipe is substituted.
- Camera disabled in the new sampler reproduces the original sampler bit-for-bit and matches 32 cached native CFG1 cases per direction/worker. Fixed-input checks require finite camera outputs, nonzero camera-on/off and camera-state effects, and bridge-off target invariance to the removed conditioning at fixed states.
- The new coupled camera sampler matches the existing scalar native camera sampler bit-for-bit for one real case in each direction, including final motion, video and camera arrays. This check ran alongside the last video scoring stage.
- CPU checks cover independent camera RNG, per-case batching invariance and preserving conditional camera velocity under target CFG. Every sample is finite; clean inputs stay fixed. Bridge and frozen camera-projection parameter versions remain unchanged. Text/no-text outputs must differ while their initial target/camera noise hashes match.
- The first two workers stopped when a second video-scorer initialization detected that DreamSim had changed the process Torch Hub directory. All samples and completed scores were preserved. A process-local context restores the checksum-pinned LPIPS path before each video scorer and restores the previous path afterward, including errors. The original asset guard remains active. The missing CFG2 scores were recovered from the original latents with no resampling; receipts and both runner versions are archived.
- Exact existing camera forward/packing code was copied into this isolated operation with import/provenance changes and a blank-generator-caption guard. Live source snapshots, repository code and training settings were untouched. Evaluations used the previously approved GPUs 3/7 with 30% process memory caps and nice 10.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_inference_camera_fixed71_20261107T092515Z`. Same checkpoints selected at <date start="2026-11-07" start-time="07:50" time-zone="Etc/UTC" />; inference-camera experiment prepared UTC `2026-11-07T09:25:15.263178+00:00`. The video-weight-10 models are 80k with late EMA and the video-weight-1 model is 40k with EMA from step 0. Within each setting only camera inference is changed. Cross-model comparisons retain age/weight/EMA-history differences.

All sampled motion/video latents and final 96×64 camera arrays remain node-local. Attached domain JSON files contain quantitative/per-case metrics, exact cases/seeds/checksums, checkpoint/source/evaluator identities, noise receipts and acceptance checks.
