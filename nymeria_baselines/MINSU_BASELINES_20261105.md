# Minsu baseline training and evaluation handoff (2026-11-05)

Mutable artifacts are under `/scratch2/minsu1206/cosmos/operations`. The shared
training/evaluation implementation is `/home/whwjdqls99/cosmos_motion_ft`.
EgoEgo's separately authorized integration is `nymeria_baselines/egoego_shape`.
No checkpoints, test data or private credentials are included in this commit.

## EgoAllo final200k: camera-to-motion

Training finished at200,000 updates. Native camera plus supplied16D training-PCA
shape inference and scoring completed all11,911 annotation/window cases,
11,677 physical windows and71 held-out recordings. The exact frozen
`configuration.json`, `cases.json`, input bundles and per-case seeds were retained.
The native quadratic DDIM30 updates, weights, shape conditioning and FK were
unchanged. Both GPU replay checks were bitwise identical; shape was preserved
exactly, with coordinate/FK audits passing. No held-out body fitting or prediction
floor correction entered inference. Placement uses camera and train-only head
extrinsics. Sampling ran at approximately0.63 seconds/case/GPU on two4090s.

Common Body23/TMR window-weighted results:

| Metric | EgoAllo | Matched GT reference |
| --- | ---: | ---: |
| MPJPE cm | 12.847869 | 0 |
| Whole-clip PA-MPJPE cm | 9.439523 | approximately0 |
| Root ADE cm | 9.129298 | 0 |
| UPE14 cm | 10.067711 | 0 |
| LPE9 cm | 17.172558 | 0 |
| Four-end-effector mean sJPE | 0.452271 | 0 |
| R@3 percent | 61.920363 +/-0.400328 | 91.703069 +/-0.139897 |
| FID | 0.159336 | 0 |
| Foot skating cm/s | 47.610109 | 14.076369 |
| Mean foot penetration cm | 3.113295 | 1.048054 |

Retrieval retains the original three R32 grouping seeds,11,904 cases and seven
dropped per seed. TMR normalization, text cache and Body23-to-SOMA30 completion
are the existing frozen evaluator protocol. Both window-weighted and
recording-balanced metrics and per-case records are saved.

Operation: `egoallo_c2m_fulltest_20261105`. Source checkpoint SHA256:
`f7a197c5d132fc49a96df1d56847240bc2c975779b4c92e5e4db7aa4d608c27f`.
Case manifest SHA256:
`6d06b4c40c7ffa2d1e3236d09746fe468c0a78a87a8cd1d9509257b069e08723`.
Score summary SHA256:
`db27d0b3f18167131948256ef8cd6cb75b1f79d449c89ee1b4b437428cd6db4d`.
`COMPLETE.json` and `VERIFIED_PUBLICATION.json` establish full scoring and remote
publication verification. The separate camera-to-motion section is published at
https://app.notion.com/p/binne/Ablations-V2M-40dd2a44f44b8080aad3cda6b5fea7c8 .

## Pretrained REVIV: current common16 protocol

RGB-only, no fine-tuning; released native model/sampler weights, preprocessing
and decoding were preserved. All11,911 cases/71 recordings completed, using
prediction-only temporal stitching to97 frames/20FPS. The released weights'
recording-level training/test overlap could not be verified.

The user selected the existing conservative16-joint comparison and unavailable
markers for unsupported metrics. Joint mapping omits ambiguous neck, spine and
collar sites. Position errors use whole-clip GT rigid registration without scale;
PA-MPJPE uses independent per-frame similarity registration. UPE covers7 upper
joints and LPE9 pelvis/leg joints, not the Body23 UPE14/LPE9 definition above.
sJPE uses the unchanged shared DisCoRD jerk-magnitude normalization over both
wrists and toes, on native unaligned joints. These are distinct evaluation
conventions from the raw Body23 table; preserve their labels when comparing.

| Metric | REVIV window mean | Matched GT reference |
| --- | ---: | ---: |
| Clip-rigid MPJPE cm | 17.852278 | approximately0 |
| Per-frame PA-MPJPE cm | 11.597141 | approximately0 |
| Clip-rigid root ADE cm | 16.536392 | approximately0 |
| Clip-rigid UPE7 cm | 16.526456 | 0 |
| Clip-rigid LPE9 cm | 18.883473 | 0 |
| Four-end-effector mean sJPE | 0.484658 | 0 |
| R@3 | unavailable: no validated SOMA30 adapter | unavailable |
| FID | unavailable: no validated SOMA30 adapter | unavailable |
| Skating | unavailable: no calibrated floor reference | unavailable |
| Penetration | unavailable: no calibrated floor reference | unavailable |

REVIV poses are relative to the first camera, rather than a floor origin.
Contact-height skating and floor penetration therefore require another defined
floor/registration protocol; do not silently use a GT-aligned floor or ground
the predictions and mix those values with the raw floor metrics above.

Operation: `reviv_zeroshot_20261105`. Original complete comparison and per-case
scores: `full/scores_common16/`. Additive requested metrics and their GT self-check:
`full/scores_requested_common16/`. Original summary SHA256:
`0518ee01da12af61e71e81af93c336422aecba0732f105d8f33a2a458d7faf7b`.
Each directory has its own score configuration and hashed completion receipt.

## GPU state at this handoff

EgoEgo continues under2670587 on two4090s; completed EgoAllo2667149 retains
two idle4090s, and completed REVIV2682742 retains one idleA100. Live progress and
Slurm state supersede this snapshot. Do not restart the user-stopped UEM RGB
and RGB+trajectory reconstruction runs.

Four additional requests were authorized:2690804 and2690802 each request2xA100;
2690803 and2690805 each request2xPRO6000. Each uses16CPUs,128GiB and three days,
with a distinct tmux queue worker. They hold GPUs idle when granted; no training
was assigned to those new allocations by this request. Their logs/assignment
receipts are under `gpu_pair_requests_20261105/{a100,pro6000}_{1,2}`.
