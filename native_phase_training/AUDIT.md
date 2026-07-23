# Native Phase Training Audit

Initial audit: 2026-07-10. Correctness update: 2026-07-11.

This audit covers the isolated cached-latent Phase 1 path in `native_phase_training/` and was checked against the local `/home/jungbin_cho/cosmos-framework` source.

## Fixed Issues

### Finite child-loader livelock

`IterativeJointDataLoader` creates child iterators once and assumes they are effectively infinite. A finite map-style child can exhaust; the joint iterator then returns without advancing `global_id`, and the trainer can repeatedly select the same exhausted stream and silently spin.

`CyclingDataLoader` restores that invariant by cycling each finite PyTorch loader forever. It forwards the Cosmos-injected `collate_fn` and calls `DistributedSampler.set_epoch(epoch)` on each pass so the shuffle order changes. The historical pixel experiment in `cosmos-framework/.../world_camera_nymeria_nano.py` still has the finite-loader risk; this repository does not patch the framework copy.

### Train/inference prompt mismatch

The first cached-latent implementation applied legacy viewpoint/duration/resolution prose to every task. Official Cosmos action inference instead uses `ActionPromptJsonFormatter` for non-empty action prompts, while an empty inverse prompt remains exactly empty. Official generic image-to-video inference uses plain duration/resolution prose and does not add action-viewpoint prose.

The dataset now applies the contract per mode before tokenization and CFG dropout:

- forward dynamics and policy: official action JSON dictionary, serialized by `TextTokenizerTransform`;
- inverse dynamics: exactly empty text;
- image-to-video: official generic duration/FPS and resolution prose.

The image-to-video formatter is implemented explicitly because the framework training augmentor truncates `97/20` to `4.0` seconds, while official inference formats the true ratio as `4.8` seconds. A CPU contract test compares both local prompt paths directly against the official inference helpers.

### Dummy-video packing undercount

The stock joint loader counted the metadata tensor `[3,97,1,1]`, producing zero spatial patch tokens and only the two vision boundary markers. Training then used a fixed 32 samples per packed batch even though each real cached latent `[48,25,16,16]` contributes 1,600 patch tokens after 2x2 patchification. A typical 32-sample action batch therefore exceeded the model's native 45,056-token target.

`LatentAwareIterativeJointDataLoader` retains the parent count for text, EOS, vision markers, actions, and sound, then adds `25 * 8 * 8 = 1600` real vision patches per cached latent. This fixed the original undercount and remains required for token-budget runs.

As of 2026-07-12 production defaults to `NATIVEP1_CLIPS_PER_GPU=4`, which selects `max_samples_per_batch=4` and `max_sequence_length=None`. This is an intentional optimization-policy change, not a sampler-contract change. It is bounded safely because text is truncated to 4,096 tokens: four worst-case T97 action samples remain below 24k tokens, and a real resolved batch measured 7,323. `NATIVEP1_CLIPS_PER_GPU=0` restores the audited 45,056-token mode, where typical batches contain about 25-26 samples depending on prompt length. Token-budget job `2838` was cancelled at step 4,399 without a checkpoint; fixed-four replacement job `2852` was then submitted.

### Evaluation resolution/shift mismatch

The historical `nymeria_world/prep_test_eval.py` defaults to image size 480 and shift 10. That is not the current Phase 1 contract: training uses the 256 tier, whose native shift is 3, with T97/action96 at 20 FPS.

`native_phase_training/prep_test_eval.py` now pins resolution/image size 256, shift 3.0, T97, action96, and 20 FPS. It creates forward/inverse/policy/image-to-video JSONL files from one usable window per held-out UUID, keeps inverse prompts empty, and reports unavailable UUIDs instead of silently reducing the test set. Action modes use NVIDIA's 30-step/guidance-1 defaults; image-to-video uses 35 steps/guidance 6; all use the official UniPC path.

NVIDIA's bundled modality defaults contain shift 10 for the high-resolution release setting. The local evaluator uses the same official inference/UniPC implementation but overrides shift to Nano's native 256-tier value of 3, matching this run's training distribution.

Each record uses a mode-specific sample name so official inference cannot overwrite another mode's files. `visualize_checkpoint.py` requires every requested record in all four JSONL files to succeed, generates three GT/video comparisons plus inverse/policy camera plots per source sample, and only then writes a completion manifest.

## Other Parity Fixes

`LatentOmniMoTModel.get_data_and_condition` returns `raw_state_action` and `raw_state_sound`, matching native `OmniMoTModel`. The training packer consumes `x0_tokens_*`, so this was not known to affect loss, but it restores parity for visualization and slicing helpers.

`build_cached_index` logs kept, missing-latent, and candidate-window counts. Missing latent files are still skipped, but any shrinkage is visible.

## Configuration Truth

- Video training uses the `waver` timestep distribution and the resolution-256 shift of 3.
- `independent_action_schedule=false`, so action noising reuses each sample's vision/video sigma. The configured `train_time_action_distribution=logitnormal` is inactive unless independent action scheduling is enabled.
- Image-to-video uses the image `logitnormal` distribution.
- Loss weighting is uniform and action loss weight is 10.0.
- The LambdaLinear schedule has `f_start=f_max=0.4`: its first 500 steps are a flat 0.4x plateau, not an increasing warmup, followed by linear decay over the 100k cycle.
- LoRA mode trains generator `q/k/v/o_proj_moe_gen` LoRA plus `action2llm`, `llm2action`, and `action_modality_embed`. It leaves `time_embedder`, `vae2llm`, and `llm2vae` frozen.
- PowerEMA is enabled and official evaluation uses it. The generic framework builds and updates a full FP32 `net_ema`, although only LoRA/action parameters differ from their frozen-base values; this is retained for the baseline despite the avoidable memory cost.

## Versioned Camera/Motion Quality Filter

The 2026-07-21 source audit found two distinct effects. Most stored upright-RGB camera positions and decoded SOMA-Head positions share the same metric world frame and maintain the expected roughly 14 cm physical offset. Separately, sparse source discontinuities and a long tail of smooth non-rigid Head/camera disagreement remain. The latter matters when a Phase-1 camera expert will later be paired with the motion expert even though Phase 1 itself consumes only video and camera action.

`build_camera_motion_quality_filter.py` reconstructs the exact native cached-latent index: usable, captioned, non-overlapping T97 windows with an existing latent, before any motion floor filtering. It computes direct world-frame step/separation metrics and per-window best-fixed-rigid-transform residuals. The pinned artifact is:

```text
/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/camera_motion_quality_filter_v1_T97.json
SHA-256 1fd6465890cbf175068db839beb8bb220f6964090ff2c583cbf50d5001989848
```

An exact physical `(uuid,start,end)` is excluded when any of these gates fires:

- camera translation step at least 0.25 m in one 20-FPS frame;
- camera rotation step at least 30 degrees in one frame;
- decoded Head translation step at least 0.25 m;
- decoded Head rotation step at least 30 degrees;
- direct camera-versus-Head world-displacement disagreement at least 0.25 m in one step;
- direct camera-origin-to-Head-joint separation above 0.5 m;
- mean SO(3) residual above 25 degrees after fitting the best fixed Head-to-camera rotation for that T97 window;
- Head-frame lever residual RMS above 5 cm after fitting the best fixed lever for that window.

The 0.25 m translation gates correspond to at least 5 m/s in one sample and the normal direct separation is about 0.14 m, so those thresholds target gross discontinuities. The train smooth-rotation distribution is median 8.22 degrees, p95 18.35, p99 24.68, and max 81.86; 25 degrees isolates the extreme tail. The lever-residual RMS is median 3.23 mm, p99 11.49 mm, and max 2.87 m; 50 mm is more than four times p99. The 30-degree one-frame angular gates are intentionally conservative: they can include genuine fast turns and should not be interpreted as proof of a coordinate reset. Their impact is small and is retained here because this experiment explicitly tests a conservative paired video/camera/motion cleanup. Threshold sensitivity is embedded in the artifact: raising the camera angular gate from 30 to 60 degrees changes train hits from 74 to 13 rows, and the corresponding Head hits from 584 to 222.

Exact impact, including duplicate caption rows:

```text
split   input rows / unique     excluded rows / unique    kept rows / unique
train   119632 / 115583         1583 / 1524                118049 / 114059
test     12613 /  12372          113 /  113                 12500 /  12259
```

Train reason counts overlap: smooth rotation 1,098 rows, Head rotation jump 584, smooth translation 243, separation 209, cross-modal translation jump 135, Head translation jump 122, camera rotation jump 74, and camera translation jump 40. Subject-normalized train exclusion is highest for S09 at 4.00%, S04 at 3.58%, and S17 at 3.27%; this reflects concentrated audited failures rather than applying a subject blacklist. Test S09 is 14/178 (7.87%) and is explicitly visible in the artifact rather than hidden by an aggregate rate.

Three diagnostics are deliberately not filters: disagreement with one train-global extrinsic, because valid actor/session extrinsics vary; raw origin-relative trajectory RMSE, because a real lever arm rotates; and the motion floor-drop list, because native Phase 1 does not consume floor-grounded motion. Phase 2/3 keep their separate floor and normalized-feature guards.

The loader validates kind/version/T/split/duplicate keys and summary counts, removes every duplicate caption row sharing an excluded physical window, and logs retained and per-reason counts. The production launcher additionally pins the file SHA and fails before torchrun on a missing or changed artifact. Cached latents and source manifests are never rewritten.

Two controlled 100k jobs use this same artifact. The control preserves the original `40/25/20/15` four-task mix. The ablation removes only I2V, leaving raw `40/25/20` ratios (effective `47.06/29.41/23.53%`); inverse dynamics remains exactly unchanged. Both keep the base checkpoint, LoRA/action-head optimizer, native RF schedules, action loss, fixed-four global batch 32, PowerEMA, save cadence, and official inference contract identical. Automatic checkpoint-eval submission is disabled only to avoid uncontrolled one-GPU queue growth; selected checkpoints must be evaluated manually.

Production launched on 2026-07-21 as jobs `3017` (four-task, node 1) and `3018` (no-I2V, node 0). Both independently logged the exact artifact SHA, retained `118,049/119,632` train rows, initialized eight ranks, and advanced with finite losses. Queue staging canceled old jobs `3011`, `3010`, and finally `3003` one at a time; `3016` was allowed to occupy the first released node before the next cancellation, and `3003` was retained until `3017` had all ranks, model/filter loading, and GPU allocation initialized. This prevented the two-node pending job `3014` from being displaced by an accidental two-node release. Exact output and log paths are recorded in `README.md`.

The follow-up four-task LR ablation changes only optimizer base/LoRA LR from `5e-5` to `1e-5`; action-head multipliers stay 4x, giving `4e-5` effective action LR. Its pinned launcher is `sbatch_phase1_native_camera_qfilter_lr1e5.sh`, and its dry-run was checked against the same filter, task, batching, EMA, RF, scheduler, and checkpoint contracts. Job `3021` was submitted with `afterany:3016` because another user had SSH-launched processes consuming all eight GPUs on the only Slurm-idle node; bypassing the memory preflight would OOM.

The caption-subject ablation is opt-in through `NYMERIA_REPLACE_STANDALONE_C=1` and
is pinned by `sbatch_phase1_native_camera_qfilter_lr1e5_person.sh`. It is paired with
the filtered four-task `1e-5` run: raw captions replace only whole-token uppercase `C`
with `A person` before official task formatting and CFG dropout. It does not mutate the
manifest, cached latents, inverse-dynamics empty prompt, or lowercase/embedded `c` text.
Focused tests passed 14/14, the native dry-run resolved the flag on all four streams,
and the exact filtered train index audit found 168,508 replacements across 117,995 of
118,049 rows. Slurm job `3024` (`np1qf4l1p`) was submitted on 2026-07-21 and initially
waited in `PENDING (Priority)` without canceling any active run. Its output directory is
`/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora1e5_action4x_ema_100k_qfilterv1_person`.

## 2026-07-21 Video-Quality Ablation Audit

The A-D video-quality suite lowers action loss from the historical value 10 to 2,
enables active-suffix normalization, and compares global generation LoRA against
an action-interface-only model and camera-token-only K/V LoRA. Historical
checkpoints and launchers retain action weight 10.

The cached causal Wan-VAE contract makes the originally proposed RGB prefix list
`[1,8,16,32,48]` inexact. Latent 0 represents RGB frame 0; later latent frames
advance in four-frame causal groups. Conditioning an integral number of cached
latent frames therefore has exact RGB boundaries `1+4N`. The implemented list is
`[1,9,17,33,49]`, mapping to `[1,3,5,9,13]` latent frames. This is a deliberate
anti-leakage correction, not an unnoticed experimental change. Non-boundary
prefixes fail validation. Contract tests verify the clean prefix, noised suffix,
suffix-only MSE indexes, unchanged action offset, and equal per-sample active
normalization across unequal suffix lengths.

The C/D I2V concern is valid but the exact native failure mode is a no-op, not a
backward crash. `OmniMoTModel._compute_losses` connects its dummy action prediction
to a zero loss when no action exists, preserving FSDP collective consistency. In C,
the real I2V vision loss has no trainable path because only action modules require
gradients. In D, the camera K/V mask is empty, so the video prediction is also the
frozen base path. Such a batch advances optimizer/scheduler/EMA without a useful
gradient. C/D therefore remove I2V from training and fail at config import if it is
active; raw `40/25/20` becomes effective `47.06/29.41/23.53%`. All five-prefix I2V
evaluations remain mandatory so preservation of the base visual prior is measured.

Camera K/V LoRA is installed before FSDP using NVIDIA's standard injected-linear
state layout. Each packed forward maps finalized action sequence indexes into the
generation stream, including context-parallel padding/shards, and applies residuals
only at those rows. K/V LoRA B is zero initialized. Focused tests verify exact
base equality at initialization, unchanged unmasked rows, nonzero K/V adapter
gradients from a visual-attention loss, frozen base weights, packed index mapping,
and unchanged state-dict keys. Real DDP/FSDP/save/resume verification is embedded
as a mandatory two-step first-allocation preflight for B-D before their 100k command.

The compact evaluation uses a fixed five-source by five-prefix grid. The official inference
entrypoint is still authoritative; the local shim only validates and transfers
explicit contiguous latent condition indexes into action-mode sequence plans.
EMA UniPC settings remain 30/guidance-1/shift-3 for action modes and
35/guidance-6/shift-3 for I2V. Metrics are suffix-only PSNR/SSIM/LPIPS with relative
early/middle/late thirds, inverse camera metrics, and policy camera metrics over
both the full trajectory and the suffix re-anchored at RGB frame `prefix-1`.
Visualization includes every individual GT/generated pair plus one GT/five-prefix
grid per source and mode. Evaluation is submitted at 10k multiples despite 5k
checkpoint saves, and writes a top-level completion marker only after inference,
visualization, and metrics all finish.

This compact suite was initially also treated as the checkpoint's quantitative
evaluation, which made its `n=5` aggregates look comparable to the original
Phase-1 run's `n=71` benchmark. The source clips were not changed: the first five
GT clips, first frames, camera actions, camera poses, and metadata are byte-identical
to the historical suite. The protocol cardinality was wrong. Current A-E configs
therefore submit a second, separate full-71 job every 10k steps. It evaluates the
exact historical prefix-1 forward/inverse JSONLs with EMA and official shift-3
UniPC, writes under `eval_full71_inverse_forward`, and requests an exclusive
eight-GPU node. `checkpoint_evals` remains the five-source prefix/policy/I2V
diagnostic and must not be reported as the canonical benchmark.

Reactivating the full-71 path exposed a stale visualization call introduced when
the renderer gained an explicit prefix boundary: `evaluate_inverse_forward.py`
omitted `prefix_length`. It now passes `prefix_length=1`. Both the callback launcher
and the manual all-checkpoint driver resolve `native_phase1_contract.json` before
model import, so C/D/E cannot be reconstructed with the wrong adapter semantics.

Jobs A (`3025`) and B (`3026`) started before the second callback existed, so their
already-instantiated trainers cannot discover it dynamically. Canonical 70k
backfills were submitted explicitly as jobs `3052` and `3053`; their five-source
prefix jobs are the separate `3050` and `3051`. D/E were still pending when the
callback was added and will instantiate both evaluation callbacks when allocated.

Video provenance is explicit rather than inferred from appearance. Reference
tiles and clean GT-conditioning prefix frames use a green border; sampled suffix
frames use red. The generated-tile header switches from `GT CONDITION` to
`GENERATED` on RGB frame `prefix_length`. Manifest records duplicate that
contract as zero-based inclusive GT/suffix ranges. A prefix-9 real-video smoke
verified green through frame 8 and red beginning at frame 9.

The first direct A/10k evaluation exposed an input-schema integration bug before
sampling: the local prefix records include `source_name`, `rgb_prefix_length`, and
`latent_prefix_length`, while NVIDIA's Pydantic inference records forbid extra
fields. `sanitize_prefix_inference_inputs.py` now validates and removes exactly
those local fields into `${EVAL_OUTPUT_DIR}/inference_inputs`; it retains
`condition_frame_indexes_vision`, which is an accepted field and carries the real
prefix contract. Official inference reads the sanitized copies. Downstream
visualization and metrics read the original enriched records, so grouping metadata
is not lost. A focused regression covers stripping and rejects inconsistent prefix
metadata.

### Architecture-bound checkpoint evaluation

The experiment module resolves `NATIVEP1_ADAPTATION_MODE` and
`NYMERIA_DROP_MODES` at import time. Earlier automatic evaluations happened to be
correct because Slurm inherited the training environment, but a manual
`sbatch_checkpoint_eval.sh` had no durable architecture source and defaulted to
`global_lora`. Current strict DCP loading normally catches C/D/E as missing adapter
keys, but that is not an adequate semantic contract: ordinary K/V LoRA and
camera-token-masked K/V LoRA intentionally share state-dict key names while
executing different forward functions.

Training now writes a versioned, immutable `native_phase1_contract.json` before
launch. It records adaptation mode, active/dropped modes, LoRA enablement/targets,
and training prefixes; incompatible resume settings fail. Every evaluation resolves
that contract before importing the experiment, rejects conflicting inherited or
manual environment variables, and writes `resolved_run_contract.json` into the
evaluation directory. Legacy checkpoints recover the same fields from their
resolved run-level `config.yaml`; missing or ambiguous metadata fails loudly.
Evaluation completion now requires and references this resolved contract.

Verification passed 35/35 focused CPU contracts, including immutable resume,
legacy C recovery, and adaptation/drop override rejection. A real node-2 smoke
then ran C/65k with both architecture environment variables unset. The evaluator
recovered `action_only` and dropped I2V from `config.yaml`, loaded EMA with zero
LoRA tensors, and completed official UniPC inference, visualization, metrics, and
the contract-aware completion marker for all four modes. Artifacts are under
`/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_contract_eval_C65_n1_p1_20260722`.

Submission state at implementation time: A is job `3025`; B/C/D are jobs
`3026/3027/3028`. E is implemented but intentionally not submitted. This is
historical provenance only; always inspect Slurm and run-local markers for current
state.

Observed launch verification: B completed its two finite distributed preflight
updates, wrote `iter_000000002`, restored all five resumable state groups at global
iteration 2, and advanced through finite step 3. C did the same; its live audit
listed only `action_modality_embed`, `action2llm.{fc,bias}.weight`, and
`llm2action.{fc,bias}.weight` (16,914,432 parameters), no LoRA, and exactly the
three normalized streams. The initial focused contracts passed 26/26 on node 2;
after adding the inference-schema and frame-provenance regressions they pass
28/28 on node 3. D's real
camera-K/V/FSDP integration check remains its mandatory in-job gate because no
unallocated GPU node was available; failure exits before its 100k command.

Direct evaluation recovery on 2026-07-22 used node 3 GPUs 1-6 for A/B/C steps
10k and 20k after jobs `3029`-`3034` remained pending. All six runs loaded EMA
DCP weights and logged native shift-3 UniPC (30 action steps, 35 I2V steps).
Each produced 80/80 successful sample outputs, 95 visualization records, fixed
prefix groups `[1,9,17,33,49]`, `inverse_camera_metrics.json`,
`policy_camera_prefix_metrics.json`, `video_prefix_metrics.json`, and both metric
and top-level completion markers. The six background processes exited normally;
only then were pending Slurm copies `3029`-`3034` canceled. A/B/C training and
pending D were left untouched. The direct runs also exercised the newly added
schema sanitizer against the real 80-record input suite.

The next automatic jobs, A/30k `3035` and C/30k `3036`, were likewise recovered
on node 3 and canceled only after both direct evaluations completed. Each has
80 successful samples, 95 visualization records, all metric files, and a
top-level marker. These were the first production outputs rendered with explicit
frame provenance: 75 pair records and 15 grid records contain both the visual
green/red switch and machine-readable frame ranges. A/30k inverse error is
`0.4047 deg`, `6.764 mm`, and `4.102 cm` ATE; C/30k is `0.6466 deg`,
`14.628 mm`, and `11.543 cm` ATE. A/30k forward PSNR reaches `15.709` at
prefix 1 and `16.985` at prefix 49; C/30k reaches `14.587` and `15.390`.
All numbers are means over the fixed five-source diagnostic suite.

## Remaining Deliberate Deviations

Raw video remains a `[3,T,1,1]` metadata tensor; clean training vision tokens come from `video_latents`. Latent-aware packing fixes the token-budget consequence, but MFU/VAE-FLOP estimates based on raw pixels remain cosmetic. Do not enable generation callbacks until their condition-image extraction is reviewed.

The stock `EveryNDrawSample` callback is deliberately disabled. It uses the training batch rather than independent real held-out media, covers only the selected stream, does not visualize action predictions, and defaults to sampling arguments that do not match this Phase 1 action evaluation. Production uses a rank-0 post-save hook to submit a separate official-inference Slurm job after each completed checkpoint. This avoids pausing or duplicating the model inside the training process while still producing all four qualitative modes.

Cached latents may be lower precision than a fresh VAE float32 encode. This is accepted for the speed-oriented path but remains a difference from pixel-native training.

## Restart Boundary

Production job `2801` started before the prompt and latent-aware packing fixes. Its checkpoint has already learned from a different prompt representation and fixed-32 batch composition. The corrected run must start from the base Cosmos checkpoint under the new default name `native_phase1_camera_json_tokpack_lora5e5_action4x_100k`; it must not auto-resume job `2801`'s DCP state.

## Verification

Focused tests live in `native_phase_training/test_contracts.py`. The 2026-07-11 acceptance smoke completed train/save/load and official forward/inverse/policy/image-to-video UniPC sampling, with all outputs and visualizations validated. Repeat that bar after future changes because import and config checks cannot validate the distributed native model path.
