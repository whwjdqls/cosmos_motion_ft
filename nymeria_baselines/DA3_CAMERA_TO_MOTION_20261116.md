# DA3 camera-to-motion: full Nymeria test cohort

User requested replacing the measured camera input of final EgoEgo, EgoAllo and
UniEgoMotion with the uploaded DA3 camera estimates. No training, loss,
checkpoint or original sampler is modified. This is a new inference experiment;
the prior measured-camera results are preserved.

Operation on **Yonsei**:
`/scratch2/whwjdqls99/cosmos/operations/da3_c2m_fulltest_20261116`.
Download: Google Drive folder `1VUosa1zGJ3K8x37uOG9i0d4VuN_ZKpmt`, using the
already authorized `gggdrive:` remote. Fourteen archive parts and the restored
archive passed SHA256; the supplied verifier checks all23,354 camera predictions
(DA3 and Phase1),23,377 payload files and full/random20 coverage. We evaluate only
DA3 here. Both raw estimator exports remain unmodified in `handoff/`.

## Gauge policy: a necessary assumption, not recovered information

DA3NESTED-GIANT-LARGE-1.1 provides metric CV camera poses but does **not** supply
absolute gravity, floor or initial camera height. Metric scale has ALREADY been
applied in the handoff; never apply `metric_scale_factor` a second time.

We remove the arbitrary first-camera reconstruction frame and initialize using
one training-only camera orientation and height prior. For each of578 training
recordings, take the median initial camera height above its training floor over
unique training-window starts, and the SO3 mean of yaw-removed initial camera
rotations. Take the median height and equal-record SO3 mean across recordings.
The fitted initial height is approximately1.519727m; the full rotation and source
hash are in `training_camera_prior.json`. All predicted relative transforms,
translation scale, vertical motion and rotations are then retained.

No test camera, floor, body pose, initial height or GT scale fit supplies any
conditioning or prediction placement. Supplied actor shape follows the original
baseline policy. Because the initial tilt/height are assumed, raw position/floor
errors include the resulting uncertainty. Historical measured-camera baselines
have additional gravity/height information and must be labeled accordingly.
GT is used only for scoring and an independent measured-camera adapter control.

## Native interfaces

- EgoEgo: camera-only horizontal origin/yaw anchoring,9D absolute camera pose
  (position3 + first-two-ROWS6D), original training normalization; original
  centered SOMA30 neutral shape. Camera is deliberately not replaced with an
  anatomical head pose: that matches this checkpoint's actual training input.
  Original EMA0.999/final555800,FP32 cosine1000-step x0 DDPM, native draw order.
  Train-only camera/head extrinsics map the output to the existing benchmark
  head-yaw/origin frame. Predicted positions are not replaced with FK.
- UniEgoMotion: original18D floor/yaw-aware camera feature builder, first delta
  identity, original camera mean/(std+1e-6), supplied SOMA30 shape, trajectory
  condition only. Final555800/350epoch snapshot, native1000 ancestral updates,
  original CFG and state_dict policy. Decode with original motion std+1e-6.
- EgoAllo: original CV camera axes with the world converted from Y-up to Z-up;
  duplicate the first camera as the previous frame, matching identity first
  relative transform. Native conditioning computes relative transforms,
  floor-relative height and yaw-free orientation. Supplied training-PCA16 shape
  stays fixed in every denoising step and FK. Final200k, native quadratic DDIM30,
  guidance off, no hands. Existing train-only camera/head extrinsics place FK.

Each model preserves the existing supplied-shape contract; the model names above
refer to our Nymeria-SOMA-trained adapters, not untouched AMASS/SMPL checkpoints.

## Coverage, acceptance and metrics

Reuse all11,911 annotation cases /11,677 physical windows /71 held-out recordings,
97frames at20FPS and exact original per-case seeds. Join the downloaded physical
windows by **(uuid,start)**; repeated annotations reuse the corresponding camera
without dropping their original evaluation cases/seeds. The random20 manifest
is retained but not used to reduce coverage.

Six CPU geometry tests protect gauge invariance, translation scale, native
Z-up/previous-frame semantics, height, benchmark origin/floor and proper SO(3).
The measured-camera integration control checks all71 recordings against the
original EgoEgo dataset and output adapter (camera max8.35e-7; common joints
max2.39e-7m). `MEASURED_ADAPTER_PARITY.json` records this audit.

Before each model/rank samples production cases it must pass full-native-step
singleton bitwise replay and compare the actual production batch128 against
the singleton (mean joint difference<=0.1cm; maximum<=1cm). EgoAllo retains its
original singleton sampler per worker. All model weights must match the pinned
SHA256. No reduced-step sampler is used for acceptance or production.

Reuse frozen GT/text embeddings with explicit provenance and original Nymeria
TMR evaluator statistics, distinct from generator training statistics. Shared
scoring includes MPJPE, whole-clip similarity PA-MPJPE, Root ADE/FDE, UPE14/LPE9,
DisCoRD sJPE (native variants retained), acceleration, skating, penetration,
FID and R@1/2/3 (32-candidate batches,3 grouping seeds;7 cases dropped per seed
only for retrieval). GT is always included. No MPJVE is requested.

## Execution and completion evidence

Retained job2622174/node58 has2xPRO6000 and16 CPUs. Both GPUs passed an actual
CUDA kernel check; GPU1's anomalous idle100% reading alone did not prevent CUDA.
Do not cancel/reassign other training jobs. New tmux/session and launch details
are recorded in operation `LAUNCH.json`; use the current receipts/logs for status.

Run inside the owned allocation:

```bash
bash /home/whwjdqls99/cosmos_motion_ft/nymeria_baselines/run_da3_c2m_yonsei.sh
```

`PREPARED.json` proves conversion, not inference. Each rank has acceptance,
progress and completion receipts. Each model needs `INFERENCE_COMPLETE.json`
and `scores/<model>/COMPLETE.json`. Root `COMPLETE.json` is written only after
all three score summaries exist and the Notion comparison has been read back.

EgoAllo's final checkpoint initially belonged to `minsu1206` with mode600 and
was unreadable. During setup it acquired an ACL granting `whwjdqls99` read access;
an actual header read succeeded. All three models can now evaluate locally.
The pinned final200k SHA256 must still match; no older checkpoint is substituted.
The worker checks access again before EgoAllo; rerunning it skips already scored
models and resumes incomplete ranks.

Environment acceptance found that the shared EgoEgo/UEM Python3.10 environment
cannot import the Python3.12+ EgoAllo source and lacks safetensors. The corrected
launcher selects Cosmos Python3.13/Torch2.10.0+cu128 for EgoAllo, with a private
`egoallo_deps/` overlay containing the original jaxtyping0.3.2,
rotary-embedding-torch0.8.6 and wadler-lindig0.1.7. Existing environments are not
modified. The historical EgoAllo reference used Torch2.7.1+cu126 on4090s; a
same measured-camera/shape/seed full-native control passed, with mean0.002241cm
and maximum0.012164cm joint differences. `EGOALLO_FRAMEWORK_PARITY.json` records
the exact values. All three full-step native singleton replays passed bitwise;
actual batch128 checks passed for EgoEgo/UEM. This is numerical parity within
the stated tolerance across frameworks/hardware, not a claim of bitwise identity
against the older4090 prediction. Native dependency/checkpoint source hashes
also passed (`NATIVE_DEPENDENCIES_VERIFIED.json`).

To use the fixed interpreter without altering ongoing sampling, the original
evaluation step is allowed to finish and score EgoEgo, then the corrected worker
is relaunched within the SAME retained allocation, reusing saved predictions.
Original wrapper/logs are preserved as worker_v1 artifacts. Current tmux is
`da3_c2m_resume_20261116`; inspect LAUNCH.json, worker_v2.log and rank progress.
An additional idle allocation2650574/node57 (onePRO6000/8CPU) was used only for
early UEM/EgoAllo acceptance and the measured-camera framework control initially.
After those passed it was assigned the FULL native singleton EgoAllo evaluation
(all11911 cases,world1), in parallel with the two-GPU UEM evaluation. Its tmux is
`da3_egoallo_eval_20261116`, launcher `run_da3_egoallo_yonsei.sh`, and receipt
`EGOALLO_EXTERNAL_WORKER.json`. The two-GPU controller waits for that score
receipt rather than launching duplicate EgoAllo workers; it can resume locally
if the separate worker fails and releases its lock. The allocation was not
canceled or assigned training. All3 GPUs use24 allocated CPUs in total.

Local report: `RESULTS.md`, `comparison.csv/json`, `scores/`, all per-case
predictions, raw camera handoff and separately saved converted conditions.
Notion: append a uniquely named DA3 section to
https://app.notion.com/p/binne/Ablations-40dd2a44f44b8080aad3cda6b5fea7c8 .
`NOTION_VERIFIED.json` records actual readback and completed-model count;
pending rows are never reported as finished metrics.

## First completed model: EgoEgo

All11911 cases generated and scored; full results and the paired references are
readback-verified in the Notion DA3 section. UEM/EgoAllo were still running at
this documentation update; their live receipts supersede this snapshot.

| Metric | DA3 + training-prior gauge | Measured camera | GT |
| --- | ---: | ---: | ---: |
| MPJPE cm |37.7057|17.4109|0|
| Whole-clip PA-MPJPE cm |16.9094|11.8597|approximately0|
| Root ADE cm |34.3703|13.5366|0|
| UPE14 cm |36.7200|14.9090|0|
| LPE9 cm |39.2389|21.3026|0|
| sJPE mean |0.362085|0.343693|0|
| Skating cm/s |22.8102|17.6417|14.0764|
| Mean foot penetration cm |4.3596|1.0232|1.0481|
| FID |0.202993|0.062487|0|
| R@3 percent |28.2818 ±0.1436|55.3931 ±0.4585|91.7031 ±0.1399|

The degradation combines DA3 relative-trajectory error with the unobserved
initial gravity/height assumption. This is not a controlled test isolating only
relative-camera-estimator accuracy. No GT initial-pose/height alignment was
introduced to improve these scores.

## All three evaluations completed (2026-11-16)

All3 models generated and scored11911 cases/71 recordings. Notion's table and
additional audit paragraphs were read back. Root `COMPLETE.json` is now
`complete_verified`; no cases were filtered or predictions replaced.

| Model | Mean MPJPE cm | Median case MPJPE cm | Whole-clip PA cm | R@3 percent |
| --- | ---: | ---: | ---: | ---: |
| GT |0|0|approximately0|91.7031|
| EgoEgo |37.7057|34.4266|16.9094|28.2818|
| EgoAllo |35.4706|32.0350|15.5892|23.9163|
| UniEgoMotion |2938113.1159*|30.8900|15.4775|32.2917|

`*` UEM has19 cases with absolute positions exceeding10m (15 have case-mean
MPJPE exceeding10m). Several explode catastrophically: the largest absolute
joint coordinate is5.6163e8m. Its raw full-set mean is dominated by these cases
and is not a stable typical-case performance measure. Median/p95 statistics
still use ALL11911 cases, not a selected subset. `robust_case_distribution.json`
and all per-case scores preserve the complete distribution.

The scorer initially stopped because this new inference writer retained the
native decoder's float64 positions while the common TMR completion returns
float32. The shared scorer now explicitly converts positions to the original
benchmark's FP32 boundary BEFORE physics/completion, retaining the strict
completion identity check. Original prediction files remain unchanged. All11911
inputs were audited: relative component rounding<=5.961e-8. Existing EgoEgo and
EgoAllo positions were already FP32 and are unchanged. The large absolute cast
rounding error on extreme cases comes from their already enormous predictions;
it does not cause the divergence. See `UEM_SCORING_DTYPE_AUDIT.json`.

Worst-case original native SINGLETON1000-step replay, same camera/shape/seed,
also diverges: max absolute joint5.0375e8m versus saved batch1285.6163e8m.
This confirms instability exists in native inference for that case; it does
not establish identical magnitudes or whole-cohort batch parity. Do not treat
the precise catastrophic raw mean as numerically stable. Several extreme cases
have camera horizontal-step inputs around40–100 training standard deviations.
This is evidence of a large conditioning-distribution shift, an association
rather than a proven sole cause. No smoothing, clipping, output filtering or
oracle gravity/height alignment was introduced to improve the report.

Original failed scoring logs and worker exit1 are retained as history. Recovery
is proved by `uem_score_fix.exit_code=0`, score COMPLETE receipts for all3,
`UEM_EXTREME_NATIVE_REPLAY.json` and the fresh Notion verification receipt.
The original GPU allocations remain retained after evaluation; current idle/live
state must come from Slurm/NVML rather than the historical training-style names.

## User-requested exclusion of19 cases and GA-MPJPE (2026-11-16)

The user requested excluding the19 extreme UEM predictions. New filtered report
`exclude19_20261116/` preserves the original full-set results and all raw files.
The predicate is exactly `max(abs(decoded UEM global body23 coordinates))>10m`,
which selects19 annotation-case IDs, not the different15-case MPJPE threshold.
The same19 IDs are excluded from EgoEgo, EgoAllo and GT for the matched comparison.
This is explicitly labeled post-hoc UEM-output-dependent filtering.

All physical/recording-balanced means are reaggregated for11892 cases; FID and
R@1/2/3 are recomputed using the retained paired cached embeddings. R32 has11872
cases per seed and20 dropped (seeds0/1/2). No inference or normalization changed.
The historical GT per-case file lacked the later UPE/LPE/sJPE backfill, so those
GT self-errors/jerk metrics were recomputed from the frozen target cache, with
its case-manifest and content hashes checked. GT errors/FID remain zero.

| Model | MPJPE cm | Whole-clip PA/GA cm | Root ADE cm | UPE cm | LPE cm | sJPE | Skating cm/s | Penetration cm | FID | R@3 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT |0|~0|0|0|0|0|14.0433|1.0474|0|91.5263|
| EgoEgo |37.6190|16.8652|34.2868|36.6246|39.1658|.3618|22.7016|4.3585|.203222|28.0997|
| EgoAllo |35.3416|15.5442|32.4830|34.3708|36.8517|.6601|60.4059|5.4045|.456315|24.0004|
| UniEgoMotion |34.6736|15.3947|31.5877|34.0573|35.6323|.4897|21.9206|2.6587|.202988|32.3197|

The filtered table was published/readback-verified at
https://www.notion.so/40dd2a44f44b8080aad3cda6b5fea7c8#419d2a44f44b810d9db1d58ade618d11 .
`excluded_cases.json` records every excluded ID/uuid/start and source prediction
hash. `all_metrics.csv` includes all native physical, retrieval, distribution
and recording-balanced keys; the compact table is in `comparison.csv`.

The user additionally requested GA-MPJPE on measured-GT-camera inputs. Existing
`motion_expert_joint_attention/eval_motion_recon.py:joint_metrics` already fits
one Umeyama rotation/translation/scale over the entire flattened T*J point cloud.
Thus historical whole-clip PA-MPJPE has the requested GA-MPJPE definition; it
was never a per-frame fit. `sequence_ga_mpjpe.py` independently recomputes the
metric in float64 for all11911 measured-camera cases/model, verifies cached PA,
and saves one similarity transform per entire sequence. Synthetic global-fit,
time-varying-translation and forbidden-reflection tests protect the definition.
Outputs/publication receipts: `gt_camera_ga_mpjpe_20261116/` in the same operation.
The measured-camera report uses the full cohort, without the DA3-specific19-case
exclusion. No measured-camera inference is rerun.

Measured-GT-camera GA-MPJPE recomputation and publication completed for all3
models plus GT on11911 cases. The maximum per-case independent-float64 versus
legacy-PA difference is1.32e-6m for model predictions (GT3.31e-6m), confirming
the same alignment definition within floating-point precision.

| Measured camera input | Raw MPJPE cm | GA-MPJPE cm |
| --- | ---: | ---: |
| GT |0|~0|
| EgoEgo EMA555800 |17.4109|11.8597|
| EgoAllo200000 |12.8479|9.4395|
| UniEgoMotion555800 trajectory |12.7490|8.9396|

Fresh GA table readback:
https://www.notion.so/40dd2a44f44b8080aad3cda6b5fea7c8#419d2a44f44b81458d79c38ee255ad48 .
`gt_camera_ga_mpjpe_20261116/COMPLETE.json` and `NOTION_VERIFIED.json` prove
completion. Per-model `single_clip_transforms.npz` has11911 scalar scales,
11911 rotations and11911 translations: exactly one transform per T97 clip.
