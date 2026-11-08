# sep-2 joint MMDiT10k: fixed71 whole-bridge-off CFG2

Completed evaluation and verified publication of284 predictions:71 V2M and71
M2V for each caption-free sep-2 directional/bidirectional40/40/20 joint model,
pinned at the newest complete checkpoints at selection, step10000.

[Full tables and media](https://app.notion.com/p/410d2a44f44b8098b1b7c7af42d2353b).
All13 uploaded files passed downloaded-SHA256 verification:12 fixed comparison
videos and the11479455-byte quantitative ZIP. The ZIP contains all142 normalized
sampled motion arrays, per-case scores, TMR/GT/text embeddings, ranks, paired
intervals, video feature arrays, identities and evaluator sources. All12 videos
decode to97 frames at20fps, and the ZIP passed CRC checks. Archive SHA256:
`8e65b04d065ddfb761e93fbd9ea8c15af66c65690d9b9dacdec6c5059c3a0c6f`.

| Metric | Bidirectional10k | Directional10k |
| --- | ---: | ---: |
| M2V PSNR dB | 17.585091 | 17.334494 |
| M2V SSIM | 0.531434 | 0.523437 |
| M2V LPIPS-Alex | 0.303946 | 0.321009 |
| M2V DreamSim | 0.168478 | 0.176872 |
| M2V pooled71 I3D-FVD | 361.132243 | 353.247768 |
| M2V VideoMAE CD-FVD | 331.710909 | 348.282655 |
| V2M MPJPE cm | 39.939045 | 29.944388 |
| V2M PA-MPJPE cm | 15.564899 | 13.417677 |
| V2M root ADE cm | 37.296960 | 27.055792 |
| V2M TMR FID | 0.622661 | 0.459407 |
| V2M sJPE mean | 0.531956 | 0.424821 |

These are descriptive point estimates. Full physical/GT/retrieval tables and
paired5000-bootstrap95% intervals are in the report/archive; no FVD significance
ranking is claimed. Do not compare this pooled71 FVD directly with the
training monitor's426-descriptor FVD.

The exact original71 held-out windows, one per recording, retain case order,
bundle hashes, neutral shapes and seeds0–70. No other test windows are sampled.
All model captions are empty. M2V receives full clean283-D motion, neutral shape
and the first image latent, which is clamped every denoising update. V2M receives
the full clean video and neutral shape, starting motion from Gaussian noise.
Unknown target inputs are zero placeholders before noise initialization.

Both use native35-step UniPC, motion shift3/video shift10, T97/20fps/256. Each
model/task passed bit-exact single-case CFG1 parity against native inference,
then selected batch64 from measured16/32/64 candidates. Packed BF16 batch
rounding may differ from single-case inference; per-case noise seeds remain
independent of batching. Qualitative indices0/14/28/42/56/70 were fixed before
viewing outputs. V2M panels share the GT virtual camera and scale and retain
raw predictions without new alignment, smoothing, grounding or head inpainting.

At each denoising step, the reference disables **all seven learned bridges**,
including their within-modality updates. Guidance is velocity_off plus
2×(velocity_on−velocity_off). Motion x0 predictions are converted to velocity
before mixing; mixing x0 first is algebraically equivalent at fixed state/time,
apart from rounding. Both branches retain the same observed inputs and shape.

This is the earlier whole-bridge extrapolation protocol, not text CFG or a null
trained with dropped conditioning. It is also different from the user's quoted
sep-1 no-joint/batch32 protocol that retains bridges and masks only target-query
cross-modality keys. Both model recipe and reference definition differ; report
those comparisons separately. No edge-only reference was evaluated here.

Video metrics score future RGB frames1–96, using native lossless uint8 decode
before MP4 compression. Original fixed71 LPIPS/DreamSim/I3D/VideoMAE assets and
preprocessing are retained; matmul TF32 off, evaluator convolution TF32 on.
FVD uses the CDFVD population-covariance convention (denominator71). An
equivalent low-rank nuclear-norm formulation avoids unstable square roots of
rank-deficient1408-D covariance matrices; full-rank dense parity and GT-self
checks passed. Generated pixels/features were not changed or resampled.

TMR checkpoint SHA25691f9d8adec6bcaa4… and original stats match the charts.
The same train-only Body23→SOMA30 completion and20→30fps conversion reproduce
saved GT embeddings exactly. Reference captions score retrieval only and never
enter generation. FID retains all71 samples; three32-candidate retrieval
groupings each retain64 cases/drop7 and report mean±sample SD. UPE/LPE and
DisCoRD wrist/toe jerk metrics retain the existing raw-joint definitions.

Checkpoint SHA256:

- Bidirectional: `93f2dcf3ece552439dfc9b88f9396e81435c1301a14f1549f73104e923ee50e5`.
- Directional: `c2f04565affdec386bf3be8ef03fa4681016655e9f619c577743e655dfc65d0d`.

Operation on sep-2:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_mmdit_no_text_10k_fixed71_cfg2_20261106T032131Z`.
Read `METRICS_COMPLETE.json`, `MEDIA_ARCHIVE_VALIDATED.json`,
`FVD_NUMERICAL_ACCEPTANCE.json`, `notion_publication.json` and `COMPLETE.json`.
Evaluation temporarily shared owned GPUs0/1 with explicit user approval; all
GPU evaluators exited before publication finished and training was left active.
