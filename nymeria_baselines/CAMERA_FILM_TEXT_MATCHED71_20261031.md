# Camera-FiLM text-conditioning follow-up, 2026-10-31

Report: https://app.notion.com/p/binne/camera-FiLM-40ad2a44f44b808c8a6eedd4ab2d13c2

This follows [the completed no-text camera comparison](CAMERA_FILM_MATCHED71_20261031.md).
Status: **published and verified**. All568 new predictions,284 reused outputs,
29 summary rows and12 new videos passed checks. New remote numeric tables,
preservation of previous tables, and all13 new uploaded artifacts were verified.
GPU2 is released; all three training runs continue unchanged.

Operation on sep-1:
`/mnt/local/jinhyung.kim/cosmos/operations/eval_camera_film_text_matched71_20261031`.
Its `plan.json` pins the same hidden-state FiLM150000 and trajectory FiLM130000
checkpoint files used by the preceding evaluation. No trainer or scientific
training source is modified. GPU2 alone is used for inference/evaluation;
qualitative rendering runs on CPU. GPU1 belongs to another researcher.

## Requested inputs and coverage

Both checkpoints are evaluated with text+video, text+camera, and
text+camera+video, each at bridge scales1 and2, on the exact same71 fixed held-out
recording windows. Existing284 text+video predictions are reused unchanged.
New camera-conditioned sampling adds568 predictions:71×2models×2conditions×2scales.
All852 text-conditioned predictions use the original case caption, seeds,
neutral shapes,97frames/20fps,35-step UniPC and shift3.

The original, joint and both FiLM Phase3 checkpoints were all trained without
text. Caption-conditioned inference is the same intervention across all four,
using their frozen specialists' existing text capability. The saved caption is
injected into the frozen native generator reasoner and motion reasoner paths
at every denoising step. Native scale1 is the matched caption/no-caption
contrast for each of the three conditions. Scale2 retains the caption in both
branches and disables entire bridges in its reference branch; it is **not
text CFG**. No previous no-text camera scale2 control exists, so no caption-only
effect is inferred at that scale.

## Verification and reporting

First-case checks reproduce previous text+video and no-text camera predictions
bitwise and verify that text changes both camera-conditioned predictions.
Generator and motion reasoner caption counts are checked at35/70 calls per
prediction for scales1/2. Zeroing the motion template changes no prediction;
zeroing video also changes no text+camera prediction. Text+camera receives no
video tokens. The same frozen Phase1 EMA200k and Phase2 selected200k specialists,
raw native camera transitions, and trajectory-specific floor-aware18D features
remain pinned.

Scoring reuses the exact previous GT/text embeddings, training-only Body23→SOMA30
completion and validation-selected TMR checkpoint, with fresh parity checks.
Physical, semantic, foot and trajectory metrics include GT. Caption effects use
paired recording bootstrap intervals (5000resamples, seed20261029) at scale1.
All earlier21 rows remain in the combined29-row quantitative summary.

Twelve fixed-index qualitative videos display RGB/GT and both models under all
three text-conditioned inputs at native scale1, with the actual provided caption.
RGB is reference context only for text+camera. The full result and quantitative
ZIP are appended to the existing report. `COMPLETE.json` and
`notion_publication.json=published_verified` establish verified completion,
including remote numeric tables and uploaded artifact hashes.

## Results

Native bridge scale1, identical71 cases; lower MPJPE/FID and higher retrieval
are favorable. GT retrieval is measured rather than assumed100%.

| Model | Input | MPJPE cm | FID | R@3 % |
| --- | --- | ---: | ---: | ---: |
| GT | reference | 0.00 | 0.000000 | 70.83 |
| Hidden-state150k | text+video | 19.71 | 0.139870 | 65.62 |
| Trajectory130k | text+video | 22.52 | 0.142513 | 69.79 |
| Hidden-state150k | text+camera | 28.00 | 0.207146 | 66.15 |
| Trajectory130k | text+camera | 12.50 | 0.146361 | 62.50 |
| Hidden-state150k | text+camera+video | 14.51 | 0.124617 | 68.23 |
| Trajectory130k | text+camera+video | 11.03 | 0.120673 | 67.71 |

Adding captions to camera+video changes MPJPE by−4.39cm for hidden-state
(paired95%CI[−8.27,−1.12]) and−1.18cm for trajectory ([−2.12,−0.28]).
For camera-only, trajectory improves by−3.03cm ([−4.14,−1.92]); hidden-state's
point estimate worsens by+4.56cm, with its interval spanning zero ([−2.70,11.84]).
Text+video improves reconstruction for both models. FID and R@3 improve with
text in all six native comparisons, so semantic gains and reconstruction gains
must be distinguished for hidden-state camera-only.

At bridge scale2, trajectory text+camera+video reaches9.44cm MPJPE; hidden-state
is14.83cm. This is whole-bridge guidance with captions retained in both branches,
not text CFG. All12 text-conditioned rows and corresponding GT/foot diagnostics
are in the Notion report and combined29-row quantitative summary. GT foot skate
is18.51cm/s and penetration1.31cm using the original physical-metric convention.
