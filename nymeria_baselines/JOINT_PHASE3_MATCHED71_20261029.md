# Joint-trained Phase3: matched V2M comparison

The user requested evaluation of the existing joint-trained Phase3 model on the
same benchmark as the Notion Video2motion comparison. This evaluates its V2M
reconstruction behavior, not its joint video/motion generation outputs.

Report: https://app.notion.com/p/binne/Video2motion-comparison-3ffd2a44f44b806ca79ec5987d6fd1f3

Yonsei operation:
`/scratch2/whwjdqls99/cosmos/operations/phase3_joint_v2m_matched_20261029`

Checkpoint:
`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_joint3_equal_T97_gbs32_wv10_yonsei_20261002/checkpoints/step_000200000.pt`

## Fixed protocol

- Reuse the exact71 cases from `operations/v2m_video_cfg_20261021/configuration.json`:
  one97-frame window per held-out recording,20fps, same caption/seed/video latent/
  neutral shape. This is not all test windows.
- Four prespecified settings: video and video+caption, each with guidance1 and2.
  UniPC35, motion shift3. No observed motion, camera trajectory, head inpainting,
  predicted-motion grounding or trajectory reset.
- Reuse the historical `GuidedPrediction` and `caption_conditioning` implementations
  with recorded source hashes. Scale1 calls native inference. Scale2 extrapolates
  against **entire bridges disabled**, with the same caption in both branches.
  This is neither a trained missing-video CFG null nor pure text CFG. Captions in
  V2M are an inference intervention for both original and joint-trained models.
- Strict bridge loading, identical frozen Phase1 EMA200k / Phase2 Self→cross
  caption-only B256200k source contracts, and generator-stat hash checks.
- First-case controls compare the native model with the scale1 wrapper and
  replace the GT motion template with zeros; predictions must match bitwise.

## Metrics and parity

`evaluate_joint_phase3_matched.py` computes unchanged per-case Body23 and SOMA30
physical metrics through the original evaluator, including measured GT. All joint
positions remain raw; only PA-MPJPE aligns predictions.

`score_joint_phase3_matched.py` uses the same validation-selected Nymeria TMR,
evaluator statistics, training-only Body23-to-SOMA30 completion,20→30fps conversion
and saved text embeddings. FID uses71 cases against common-completion GT. R@1/2/3
uses32 paired candidates with grouping seeds0/1/2:64 cases scored and7 dropped per
seed, mean and sample SD. Do not compare these scores directly to full-test Phase2
results.

Before scoring new predictions, re-embed GT and the original model's saved native
V2M predictions. On PRO6000 versus the previous4090, maximum embedding differences
were1.564621925354004e-7; GT retrieval was unchanged. Text re-embedding differed by
2.086162567138672e-7. The exact saved text embeddings are reused for all scoring;
cross-architecture bitwise identity is not claimed. Generator and evaluator
normalizations are kept separate and verified against their own contracts.

Physical differences use paired bootstrap intervals over71 recordings,5000
resamples, seed20261029. FID/retrieval differences are descriptive small-sample
diagnostics. Original training uses50:50 V2M/M2V; joint training uses equal thirds
V2M/M2V/joint. Equal200k updates do not imply equal task exposure.

## Execution and completion evidence

The inference worker is resumable, saves immutable configuration/source hashes and
checks cached prediction signatures. Four disjoint shards share spare VRAM in
owned PRO6000 allocations; inference allocator cap35% and scorer15%. Existing
training processes and checkpoints are unchanged. Never run this on a login node,
`cs-gpu-01` or `node37`.

The operation contains `worker.sh` (offset0 or2), `score.sh`, per-shard status and
acceptance files. `METRICS_COMPLETE.json` indicates scoring completion;
`notion_publication.json` must say `published_verified`; `COMPLETE.json` is written
only after remote table verification. Consult these files for current status.
Results include `comparison.csv`, `summary.json`, per-case measurements, embeddings,
retrieval groups, prediction hashes, evaluator acceptance and paired differences.

Completed2026-10-29 08:03KST: all71 cases in all four settings,284 predictions,
zero failed cases. Tables were remotely verified and the quantitative ZIP was
subsequently uploaded and SHA256-verified. Native video-only joint200k versus
original200k: MPJPE22.53 versus20.56cm, FID0.286935 versus0.259868,
R@3 45.31 versus42.71%. Caption/scale2: MPJPE13.98 versus14.13cm,
FID0.132200 versus0.140510, R@3 65.10 versus69.79%. GT FID0 and
R@3 70.83%. Results are mixed; all four paired MPJPE intervals cross zero.
Caption/scale2 airborne frames increase18.50% versus11.24%, so reduced skating
and penetration must not be described as an unqualified contact improvement.
