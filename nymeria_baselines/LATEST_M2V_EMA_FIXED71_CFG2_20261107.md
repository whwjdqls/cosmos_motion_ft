# M2V matching the latest V2M/text checkpoints, 2026-11-07

Report: https://app.notion.com/p/410d2a44f44b8098b1b7c7af42d2353b

Operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_latest_m2v_ema_fixed71_cfg2_20261107T083008Z`

The user requested M2V in addition to the completed latest V2M/text evaluation.
Same pinned EMA checkpoints B60000, D/video10-70000 and fresh D/video1-40000;
exact same71 windows, original case order and seeds0–70. All213 clips completed
with no replaced or skipped windows. These are different training ages, not a
matched-step or isolated video-loss-weight comparison.

| Model / step | PSNR dB | SSIM | LPIPS-Alex | DreamSim | I3D-FVD | VideoMAE CD-FVD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bidirectional video10 / 60000 | 18.435320 | 0.563804 | 0.269510 | 0.159872 | 327.612078 | 315.485667 |
| Directional video10 / 70000 | 18.607550 | 0.567669 | 0.267646 | 0.159007 | 336.092539 | 323.091724 |
| Directional video1 / 40000 | 18.350652 | 0.560034 | 0.273314 | 0.160690 | 333.829863 | 321.261816 |

## Sampling and conditioning

- Full clean normalized283-D motion, supplied neutral shape, one initial image
  latent and empty dataset captions in both frozen specialists.
- Native35-step UniPC, video timestep shift10, BF16 with TF32 off. Unknown video
  inputs are zero placeholders before independent per-case Gaussian sampling.
  The initial latent is clamped after every update and preserved in all213
  final saved latent arrays.
- Whole-bridge-disabled CFG2: reference +2×(conditioned−reference), with all
  seven whole learned bridges disabled in the reference, including their
  within-modality updates. Both passes retain the same observed inputs, shape
  and blank captions. These models were not trained with that reference.
- Correct frozen source/contract and strict origin-aware EMA state: B9983
  updates since50017; D19181 since50819; D/video1 40000 since0. Each checkpoint
  SHA matches its completed V2M/text evaluation exactly.
- Native CFG1 single-case bit-exact sampler parity passes for every model.
  Benchmarks32/64/71 select64 for B and71 for both directional models. No
  cross-batch bit-exact numerical identity is claimed for BF16 execution.

## Scoring and validation

All metrics exclude conditioned frame0 and score future RGB frames1–96 at
256×256/20fps. Native Wan decode is rounded to lossless uint8 before MP4
compression. Original LPIPS-Alex and official DreamSim use128-frame batches;
VAE decode uses four clips. DreamSim16-vs128-frame batching passes parity.

The original FP32 I3D400 and1408-D VideoMAE evaluators/preprocessing are retained,
matmul TF32 off and native evaluator convolution TF32 on. GT features are freshly
computed once in canonical case order, chunks4 with feature pairs2, and match
the prior matched71 reference exactly (maximum absolute difference0 for both
evaluators over all71 cases). Identical GT arrays are reused for all models;
heavy evaluator modules are loaded once. FVD pools all71 clip descriptors with
population covariance, computed using the equivalent low-rank nuclear-norm
trace. GT-self FVD is checked. This is different from the426-descriptor training
monitor.

The report/archive includes per-case metrics, temporal horizons1–32/33–64/65–96,
all12 paired frame-metric contrasts with5000 recording bootstrap intervals,
seed20261029, per-case FVD features, evaluator/source/checkpoint identities and
SHA256 manifests for all213 generated latents and lossless decoded pixel files.
Full latents and pixels remain node-local. No FVD significance claim is made.

## Publication and operational state

Six indices0/14/28/42/56/70 were fixed before viewing outputs. Each MP4 compares
GT, B60k, D70k and D/video1-40k;97 frames at20fps, checked with PyAV. GT panels
are labeled observed and predictions generated. A label-only render correction
is recorded; generated pixels, latent samples and metrics were unchanged.

The new top M2V toggle preserves the earlier V2M/text and older M2V results.
`notion_publication.json` records22 exact table rows and7 stable attachment
references. Fresh attachment size/SHA checks are in `verified_uploads/`;
`COMPLETE.json` is written only after all7 pass.

Only the user-authorized idle GPU4 was used; no shared-storage transfers and no
trainer restarts, source edits or recipe changes. All three trainers were still
running after GPU evaluation. The production B controller remains in the older
EMA evaluation operation; never terminate it during evaluation cleanup.
