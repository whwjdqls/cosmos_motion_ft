# EgoControl full Nymeria test evaluation

User-selected scope: **every eligible native aligned97-frame test window**,
after EgoControl finishes200k and the original-recipe Phase2 no-shape ablation
successfully starts. Do not replace this scope with the fixed71-case cohort.

On sep-1, 2026-11-01, EgoControl finished200k successfully and the full native
model/optimizer/scheduler/trainer checkpoint was verified. Phase2 no-shape passed
its full two-rank gate and started fresh with online W&B. The finalEMA200k
evaluation then started on GPU2 at12:18UTC, after a fresh idle-process/memory
check and exclusive admission. It preserves all other training processes and
GPU1's unrelated process. **Full-set inference/metrics are running; final results
remain pending.** All11,677 RGB input windows prepared without decode failures;
the first/middle/last fixture checksums match independent native decoding.

## Cohort

The calibrated native test index contains11,938 caption/window rows over71
held-out recordings.236 repeated caption entries have identical video, motion,
and floor inputs. Deduplicating those leaves11,702 unique windows.25 fail the
existing finite/absolute-z≤20 motion guard, leaving **11,677 windows**, still
covering all71 recordings. This uses native non-overlapping97-frame starts
within usable caption spans, rather than every possible sliding-frame start.
Recording-level train/test disjointness is asserted against the actual training
cache. Exact-window video decode failures are recorded as input exclusions;
there is no next-window replacement or metric/inference-failure exclusion.

No held-out window has a precomputed training latent. Inference encodes only
observed RGB frame0, and uses noise for future latent positions. Future GT RGB
never enters the generator. Metrics use raw GT after native spatial resizing
and padding, and lossless uint8 predictions before MP4 compression.

## Sampling and guidance

- Final native `iter_000200000` **EMA**, strict DCP inference loading, no optimizer
  state, no partial/skipped learned parameters. Source EgoControl checkout is
  `repos/egocontrol_resume_20261030`, commit268fe8d; no live trainer code changes.
- Seven pose-conditioning blocks3/7/11/15/19/23/27, no final pose modulation,
  Body23 pose plus first observed image, no captions or camera actions.
- Native Edge UniPC35, shift10, T97,20fps,256pixels, fixed seed0.
- Reference `CVG-Bonn/EgoControl@5dcfb82ef898315d415f8f04c94337c83d9a099d`
  evaluation guidance default is2. Its formula is `c + 2*(c-u)`, which equals
  Edge's standard `u + 3*(c-u)`. Use standard scale3 for that mapping.
- The evaluation-only null branch zeros the image latent at frame0, retaining
  the clean timestep/mask and pose in both branches. Disable packed templates
  so no conditional image persists in a null-branch cache. Both branches retain
  identical no-text metadata. Reference Predict2's negative-text encoder is
  absent from this caption-free Edge task; disclose this adaptation.

## Metrics

All requested metrics exclude observed RGB frame0, scoring frames1–96:
PSNR (native100dB identity cap), RGB SSIM, LPIPS AlexNetv0.1, official DreamSim
ensemble0.2.1 with normalized embeddings, and dataset-level FVD. FVD follows the
EgoControl reference PyTorch I3D400 logits and spatial preprocessing: six
contiguous16-frame suffix segments per window, short-side224/center-crop,
[-1,1] normalization, one Fréchet calculation across all segment features.
It is not an average of per-window FVDs. Public Kinetics400 RGB I3D weights:
SHA256 `2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538`.
The paper's custom checkpoint checksum is unspecified.

Three CPU tests cover channel-correct null-image masking, non-mutating inputs,
retained pose/text across branches, guidance algebra, and Fréchet identity and
known mean shift. Three real exact-window decodes/native resizes pass; DreamSim
CPU identity is0. GPU metric identity/temporal checks and an actual three-step
native inference gate remain mandatory before full inference. Persist lossless
predictions, frame metrics, I3D features, per-recording qualitative comparisons,
coverage/exclusions, and source/cohort/checkpoint hashes. Resume only matching
contracts; never report partial coverage as complete.

## Evidence and report

Operations: `/mnt/local/jinhyung.kim/cosmos/operations/eval_egocontrol_fulltest_20261101`.
Use `monitor_status.json`, `status.json`, `evaluation.log`, and
`results/COMPLETE.json` for actual progress. Readiness files mean preparation,
not completed GPU acceptance or evaluation. Isolated metric additions are in
`/mnt/local/jinhyung.kim/cosmos/envs/egocontrol_metrics`; the live training
environment was not changed.

CPU preflight against native EMA195k confirms867 EMA keys and the seven expected
pose blocks. The standalone metrics venv initially failed native framework
imports because Transformer Engine searched the venv prefix for cuDNN. The
launcher now uses the original Conda Python with only the isolated metrics
site-packages added to its import path. The same CPU config preflight passes;
GPU loading/sampling remain pending. A four-process CPU worker prepares
lossless RGB windows with the native decode/spatial transform and stores video
identities plus RGB checksums; inference validates cached inputs before use.

The EMA195k GPU compatibility probe passes strict loading, native UniPC35,
both CFG branches, prefix preservation, decoding and metric feature extraction
in `probe_195k/PASSED.json`. It is excluded from the final200k results.
Denoising takes7.09s for the checked window. Batch2/4/8 fails the predefined
batch1 output-parity tolerance; production remains batch1 for every window.

Seven CPU protocol/aggregation tests pass. They include rank-deficient400-D
Fréchet identity, known translation/scaling distances, complete-cohort coverage,
and rejecting non-input failures as exclusions. Fréchet uses the mathematically
equivalent nuclear norm of the product of covariance square roots, in float64,
to avoid negative identity distances from roundoff in the sandwich formulation.
Final aggregation reads persisted per-frame arrays, verifies96 frames and6×400
features per scored window, and writes completion only after cohort coverage
and all required metrics validate. See `FVD_NUMERICS_AUDIT.json` in operations.

Verified preparation report:
https://app.notion.com/p/40bd2a44f44b8185b567d54f0e5be3b8 .
Phase2 no-shape preparation report:
https://app.notion.com/p/40bd2a44f44b8187850ac41dd36ef75c .
