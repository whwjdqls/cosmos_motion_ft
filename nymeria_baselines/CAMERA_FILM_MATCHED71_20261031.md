# Camera-FiLM interim checkpoints on matched test71

Report: https://app.notion.com/p/binne/camera-FiLM-40ad2a44f44b808c8a6eedd4ab2d13c2

Operation on Foundary sep-1:
`/mnt/local/jinhyung.kim/cosmos/operations/eval_camera_film_matched71_20261031`.
Completed852 predictions:71 cases × six settings × two checkpoints. Hidden-state
FiLM150k and trajectory/motion-noise FiLM130k were pinned with local hard links
before training retention could remove them. They are interim snapshots, not200k
models. No existing training process or scientific training setting was changed.

## Protocol and parity

Reused the exact71 fixed held-out97-frame/20fps windows from the original
Video2motion comparison. Every input bundle matches the earlier SHA256, and the
Body23 GT physical scores match exactly. Same supplied neutral shapes, case seeds,
UniPC35/shift3, generator normalization and original frozen Phase1 EMA200k / Phase2
Self→cross/K-norm/caption-only B256200k specialists. The newly downloaded two-task
Phase1 checkpoint was not substituted.

Each FiLM model was sampled with video and video+caption at bridge scales1/2,
plus camera-only and camera+video at native scale1. Scale2 uses the historical
whole-bridge-disabled reference with caption retained in both branches, not a
trained missing-video CFG null. Caption arms are inference interventions for
these caption-empty-trained models. Camera arms additionally observe measured
poses. The trajectory model uses its saved floor-aware18D training normalization.

Native/zero-GT-motion-template sampling matched bitwise on both models. Camera-only
outputs were unchanged when video was replaced by zeros. Disabling FiLM changed
camera-conditioned outputs; captions also changed predictions. All852 outputs
are finite and retain the exact matching GT/shape/seed records.

The validation-selected Nymeria TMR checkpoint matches SHA256
`91f9d8adec6bcaa448571dade670da0f94916e448b875a0853eda29ef44a1e21`.
The exact train-only Body23→SOMA30 completion was reconstructed using the historical
floor-clean cohort and matches SHA256
`63ab10a62392aed8d6172b6fa85561cb6f0756d14e82e743e7aeff8e6113afa8`.
All71 captions use unchanged base rows of the expanded text cache, whose original
base hash is pinned. GT retrieval and published completion diagnostics match.
Fresh cross-GPU embeddings are not claimed bit-identical. FID uses71 cases;
retrieval uses32 candidates and grouping seeds0/1/2,64 scored and seven dropped
per seed, with mean and sample SD. These are small-sample diagnostics.

## Results

| Model | Native video MPJPE cm | Camera+video MPJPE cm | Camera+video FID | Camera+video R@3 % |
| --- | ---: | ---: | ---: | ---: |
| GT reference | 0.00 | 0.00 | 0.000000 | 70.83 |
| Original200k | 20.56 | — | — | — |
| Joint200k | 22.53 | — | — | — |
| Hidden-state FiLM150k | 26.19 | 18.90 | 0.221805 | 50.52 |
| Trajectory FiLM130k | 31.00 | 12.21 | 0.252486 | 48.96 |

Within-model camera+video minus video MPJPE: hidden−7.29cm, paired95% interval
[−10.82,−4.18]; trajectory−18.79cm,[−23.51,−14.60]. Camera-only MPJPE is23.43cm
and15.53cm respectively. Measured camera input improves reconstruction in this
cohort; it is additional information rather than an input-matched architectural
win over video-only baselines. Native video-only FiLM reconstruction is worse
at these interim steps. Trajectory has lower camera-conditioned position error,
while hidden-state has better camera+video FID and slightly higher retrieval.

The complete report includes all settings, root/foot/acceleration metrics,
paired intervals versus joint200k, and12 prespecified eight-panel videos at
cases0,6,…,66. Plotting cameras track each panel's root; generated motion itself
is neither aligned nor grounded. Original/joint200k numerical rows are reused
from the completed reference report and ZIP, not newly sampled. Checkpoint
steps, training task exposure and camera information differ; no single cause
of the performance differences is isolated.

## Completion evidence

`artifacts_verified.json` validates852 predictions and12 decodable videos.
`remote_tables_verified.json` checks every quantitative table row;
`remote_media_verified.json` verifies SHA256 of all12 published MP4s and the ZIP.
The quantitative ZIP includes per-case measurements, embeddings, retrieval groups,
paired intervals, input/source hashes and acceptance records. `COMPLETE.json`
and `notion_publication.json` establish verified delivery. GPU2 was released;
EgoControl0/4, hidden-state FiLM6/7 and trajectory FiLM3/5 continued unchanged.
