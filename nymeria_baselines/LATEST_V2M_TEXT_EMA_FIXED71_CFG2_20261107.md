# Latest sep-2 V2M and V2M+text evaluation, 2026-11-07

Report and qualitative comparisons:
https://app.notion.com/p/410d2a44f44b8098b1b7c7af42d2353b

Operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_latest_v2m_text_ema_fixed71_cfg2_20261107T074917Z`

The user requested latest checkpoints, V2M with and without text, and batched
sampling on an idle GPU. GPU4 was empty before launch and checked between GPU
stages. All data, evaluators, outputs and checkpoint hardlinks are node-local.
All three trainers continued without source, recipe or checkpoint migration.

## Results

Exactly the existing 71 held-out windows and original per-case seeds0–70;
426 generated motions across three checkpoints and two input conditions.
There are no skipped or newly selected cases. Checkpoints have different ages.

| Model | Step | V2M MPJPE cm | Video+text MPJPE cm | V2M motion FID | Video+text motion FID |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bidirectional, video weight10 | 60000 | 27.37547 | 19.79250 | 0.357617 | 0.178208 |
| Directional, video weight10 | 70000 | 18.46537 | 14.99788 | 0.226553 | 0.157289 |
| Directional, video weight1 | 40000 | 17.27264 | 12.72125 | 0.231559 | 0.132389 |

The report also contains PA-MPJPE, root ADE/FDE and relative ADE, UPE/LPE,
sJPE variants, acceleration, skate/penetration/airborne metrics, matched TMR
retrieval R@1/2/3 with grouping SD, matching distance and diversity. Full per-case
results and paired physical-metric intervals are retained in the evidence.

Paired text-minus-video MPJPE differences, 5000 recording bootstrap resamples,
seed20261029: B −7.58296cm [−12.29581,−3.01197]; D/video10 −3.46749cm
[−6.34096,−0.96468]; D/video1 −4.55139cm [−7.53590,−1.94954]. These intervals
cover this fixed cohort, not training replicas or generation-seed variation.

## Sampling and text intervention

- Native35-step UniPC, motion timestep shift3, BF16 with TF32 disabled.
- EMA bridge states: B9983 updates since50017; D19181 since50819; fresh
  D/video1 40000 updates since0. Source/contract/fingerprint and all98 bridge
  state entries are verified against the correct frozen source snapshot.
- Full clean video and supplied neutral shape condition pure per-case Gaussian
  motion noise. Unknown motion inputs are zero placeholders. No measured camera
  actions or GT motion values enter sampling.
- CFG2 uses reference +2×(conditioned−reference), with all seven whole bridges
  disabled in the reference, including within-modality bridge updates. Both
  passes retain the same caption. This reference was not used during training.
- Video-only uses blank captions. Video+text preserves each full saved caption
  at both specialist boundaries and the native V2M video packer in this
  inference process only, matching the earlier caption intervention. Training
  remains caption-free. This is reference-caption-assisted evaluation.
- Every checkpoint passes bit-exact native single-case CFG1 parity, blank-text
  intervention parity, and a nonzero caption-effect check. Both text paths
  receive the expected caption list70 times per35-step CFG2 batch.
- Benchmarks32/64/71 select64 for five combinations and71 for fresh D/video1
  with text. Inference retains independent per-case RNG and original seeds.
  No cross-batch bit-exact numerical identity is claimed for BF16 execution.

## Evaluation, media and reproducibility

Matched TMR checkpoint SHA256:
`91f9d8adec6bcaa448571dade670da0f94916e448b875a0853eda29ef44a1e21`.
The original eight normalization files, train-only Body23→SOMA30 completion and
20→30fps conversion are preserved. Recomputed complete/full GT embeddings have
zero maximum difference from the reference; final text embeddings are reused
exactly. FID uses all71 cases; R32 uses the original64-case groupings with7
dropped per seed0/1/2. Only PA-MPJPE aligns generated trajectories.

Six qualitative indices0/14/28/42/56/70 were fixed before outputs were viewed.
Each model has six videos, showing observed RGB, GT, V2M and V2M+text with shared
camera/scale within a comparison. All18 MP4s decode to97 frames at20fps; input
RGB preview crops match bit-for-bit across models. Raw motion is not grounded
or reset for display. Three model-specific ZIPs each retain all142 predictions,
identities, embeddings/groups, per-case metrics, paired intervals and eval code.

`notion_publication.json` records the40 exact table rows and21 stable attachment
references. `verified_uploads/` contains fresh Notion download size/SHA checks;
`COMPLETE.json` is written only after all21 pass. Older Notion results remain.
Two CPU aggregator issues (helper import location and a variable shadow) were
fixed and recorded in `source_revision_finalize_import.json`; all426 sampled
outputs and semantic scores were preserved without GPU resampling.
