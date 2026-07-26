# Native Phase Training

This directory is the isolated native-Cosmos training path for a new Phase 1 camera/video generator run. It exists because the older `motion_expert_joint_attention/` Phase 1 used a custom wrapper/sampler path, while this run is meant to stay as close as possible to NVIDIA Cosmos-3 Nano training and official inference.

The immediate goal is:

1. train a good Nymeria camera/video generator LoRA with native Cosmos loss and timestep distributions;
2. keep it sampleable through `cosmos_framework.scripts.inference`;
3. later reuse the trained generator LoRA as the frozen video/camera expert for motion bridge experiments.

This is not the motion-expert path. There is no motion expert and no modality bridge in this directory.

Bridge compatibility note: future motion bridge work should keep this generator on the native Cosmos
training/sampling contract. For `motimg2video`, video is the noised target and should use native Cosmos
RF noising plus official inference sampling. For `video2motion`, video is clean conditioning and the
motion expert's own motion sampler is the target-side sampler. Do not evaluate this native Phase-1
generator through the older `motion_expert_joint_attention/sample.py` custom video sampler; that path is
for historical joint-attention checkpoints and does not preserve the official Cosmos sampling contract.

## Design Intent

The training path uses cached Wan2.2 VAE latents to avoid repeatedly encoding Nymeria videos, but otherwise preserves Cosmos native behavior:

- native `OmniMoTModel` packing, rectified-flow noising, loss masks, losses, and action heads;
- native Cosmos action modes: `forward_dynamics`, `inverse_dynamics`, `policy`;
- native `image2video` regularizer;
- native official inference/sampling stack for evaluation;
- generator LoRA plus camera action heads trainable;
- frozen reasoner/base generator weights.

The important compatibility point is that cached latents are a training input optimization only. Inference still goes through the official pixel/video input path and the frozen VAE, because inference samples do not pass `video_latents`.

## Files

- `latent_omni_model.py`
  - Defines `LatentOmniMoTModel(OmniMoTModel)`.
  - If `data_batch` has `video_latents`, it bypasses VAE encode and uses those latents as clean vision tokens.
  - If `video_latents` is absent, it falls back to the native parent implementation.
  - Keeps native RF noising, sequence packing, losses, sampler contract, and output saving.
  - Flattens `IterativeJointDataLoader` nested video metadata so native `_vae_pixel_shapes` and padding metadata do not crash.

- `latent_nymeria_dataset.py`
  - Builds a cached-latent Nymeria dataset from the video manifest, train/test split, and latent `.npz` files.
  - Returns dummy `video` metadata plus real `video_latents`.
  - Emits native action-SFT fields: `sequence_plan`, `action`, `raw_action_dim`, `domain_id`, `conditioning_fps`, `image_size`, tokenized text.
  - Provides `CyclingDataLoader`, an infinite wrapper over finite map-style `DataLoader` streams. This is required because native `IterativeJointDataLoader` assumes child streams do not exhaust during long training.
  - Provides `LatentAwareIterativeJointDataLoader`, which counts cached latent patches correctly when token-budget packing is selected. Production now uses a fixed four clips per GPU; `NATIVEP1_CLIPS_PER_GPU=0` restores the 45,056-token mode.
  - Formats forward/policy captions with the same `ActionPromptJsonFormatter` used by official inference, keeps inverse text exactly empty, and matches the official plain-text image-to-video duration/resolution template.
  - Strictly validates and applies an optional versioned physical-window quality filter. One excluded `(uuid,start,end)` removes every duplicate caption row for that same T97 window.
  - Validates an optional immutable latent-cache contract before DataLoader workers start and validates each loaded sample against its spatial/temporal/action geometry.

- `latent_cache_contract.py` and `validate_latent_cache.py`
  - Define the immutable cache provenance/geometry record.
  - Validate the complete expected physical-window file set and deterministic materialized samples before a production run constructs the model.

- `prepare_phase1_eval_tier.py` and `validate_eval_inputs.py`
  - Convert canonical held-out records to the released 256 or 720 model tier.
  - Reject checkpoint/input resolution, shift, or T mismatches before official inference.

- `build_camera_motion_quality_filter.py`
  - Reconstructs the exact cached-latent Phase-1 population and audits direct upright-RGB-camera versus decoded SOMA-Head continuity and rigid-pair consistency.
  - Writes a versioned exclusion artifact with thresholds, metric distributions, threshold sensitivity, per-subject impact, duplicate-row multiplicity, and exact source hashes.

- `AUDIT.md`
  - Records the 2026-07-10 native Phase 1 audit, the finite-dataloader livelock fix, parity fixes, and documented deviations from pixel-native training.

- `PHASE1_VISUAL_QUALITY_AUDIT.md`
  - Compares the historical pixel-online runs with the current cached-latent
    video-quality suite.
  - Records the resolution, sampler, model-provenance, task-mixing, batch,
    loss-weight, and cache-precision evidence.
  - Ranks the remaining root-cause hypotheses and defines the controlled
    experiments required before changing the training contract.

- `audit_cached_latent_precision.py`
  - Provides a portable CPU audit of cached latent dtype, range, fp16 spacing,
    and exact bf16 round-trip behavior.

- `experiment.py`
  - Registers Hydra experiment `world_camera_nymeria_latent_nano`.
  - Starts from `NANO_MODEL_CONFIG`.
  - Sets `resolution=256` by default through `NYMERIA_RESOLUTION`.
  - Resolves local Wan VAE from `WAN_VAE_PATH`.
  - Resolves a local text tokenizer snapshot and defaults to offline HF mode.
  - Builds four task streams through native `IterativeJointDataLoader`.

- `world_camera_nymeria_latent.toml`
  - TOML overrides for production training.
  - Loads base checkpoint from `BASE_CHECKPOINT_PATH`.
  - Saves to `IMAGINAIRE_OUTPUT_ROOT`.

- `run_latent_train.py`
  - Training entrypoint that imports `native_phase_training.experiment` before TOML resolution.
  - Assigns `trainer.callbacks.tensorboard.log_dir` to `$TB_LOG_DIR` if set, otherwise
    `${job.path_local}/tensorboard`, so TensorBoard events are per-run and easy to find.

- `run_latent_train.sh`
  - Local wrapper that sets env defaults and calls `run_latent_train.py`.

- `inference_config.py`
  - Official inference config shim.
  - Imports `native_phase_training.experiment` so the Hydra experiment exists for local DCP loading.

- `prep_test_eval.py`
  - Builds official-inference JSONL inputs for forward dynamics, inverse dynamics, policy, and image-to-video from the 71-sequence held-out split.
  - Pins the Phase 1 contract to T97/action96, 20 FPS, resolution/image size 256, and shift 3.0.
  - Uses one usable window per test sequence, preferring walking/turning clips but falling back to the first usable T97 window.

- `visualize_checkpoint.py`
  - Consumes official-inference outputs with mode-specific directory names.
  - Creates GT-versus-generated MP4s for forward dynamics, policy, and image-to-video.
  - Marks provenance per RGB frame: the reference and clean conditioning prefix have a green border, while the generated suffix has a red border. Headers switch from `GT CONDITION` to `GENERATED` at the exact prefix boundary, and each manifest record stores the corresponding zero-based inclusive ranges.
  - Creates predicted-versus-GT camera trajectory/frustum plots for inverse dynamics and policy, plus a JSON manifest only after every requested sample in all four JSONL files has been visualized successfully.

- `checkpoint_eval_callback.py` and `sbatch_checkpoint_eval.sh`
  - After a successful DCP save, rank 0 submits one isolated one-GPU Slurm evaluation job.
  - The job loads EMA weights through official inference, samples all four modes with UniPC, then runs `visualize_checkpoint.py`.
  - Submission markers prevent duplicate jobs after trainer restart; a failed submission is logged without aborting training.

- `run_contract.py`
  - Persists architecture-critical settings in `<run>/native_phase1_contract.json` and resolves them before evaluation imports the experiment config.
  - Rejects incompatible resume settings and conflicting manual evaluation overrides instead of rebuilding a different adapter graph.

- `sbatch_phase1_native_camera.sh`
  - Production Slurm launcher for the official-compatible Phase 1 run.
  - A filtered run must provide both `NYMERIA_QUALITY_FILTER` and its exact `NATIVEP1_QUALITY_FILTER_SHA256`; the launcher fails before torchrun on any mismatch.

- `sbatch_precompute_latents_720tier.sh` and `sbatch_phase1_native_camera_720tier.sh`
  - Build the 24-shard full 640-pixel cache and launch the controlled released-720-tier Phase-1 run.
  - The training launcher requires a complete cache marker, performs a two-step DCP save/resume preflight, and evaluates compact four-mode plus canonical full-71 inputs every 10k steps.

- `sbatch_phase1_native_camera_qfilter.sh` and `sbatch_phase1_native_camera_qfilter_no_i2v.sh`
  - Pinned launchers for the filtered four-task control and the filtered three-task no-I2V ablation.

## Data Contract

Default environment:

```bash
export NYMERIA_NUM_FRAMES=97
export NYMERIA_RESOLUTION=256
export NYMERIA_LATENT_ROOT=/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export BASE_CHECKPOINT_PATH=/weka/jungbin/cosmos3_nano_dcp
```

Each latent `.npz` is expected to contain:

- `latents`: `[48,25,16,16]`, Wan2.2 latent tensor for 97 RGB frames at 256p.
- `camera_action`: `[96,9]`, raw Cosmos camera action, not z-scored.
- `image_size`: usually `[256,256,256,256]`.
- metadata such as `uuid`, `start`, `T`, `fps`.

The isolated high-tier contract is:

```bash
export NYMERIA_NUM_FRAMES=97
export NYMERIA_RESOLUTION=720
export NYMERIA_LATENT_ROOT=/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97_720tier_640
export NATIVEP1_EXPECTED_IMAGE_HW=640
export NATIVEP1_EXPECTED_LATENT_HW=40
export NATIVEP1_SHIFT_OVERRIDE=10
export NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=1
```

For this pipeline, official model tier `720` does not mean a 720x720 square tensor. Nymeria preprocessing uses transform key `480`, which maps the source square clips to `640x640`; Wan then stores fp16 `[48,25,40,40]`. Do not set the training model to resolution `480`: the released Nano model/config tier is `720`, and its released resolution-adaptive shift is 10. At inference, these are separate fields: action modes use `image_size=480`, while generic I2V uses output bucket `resolution=480, aspect_ratio=1,1`; every mode explicitly sets `num_frames=97`.

The unfiltered train manifest produces 119,632 caption rows but only 115,583 unique physical `(uuid,start)` windows. Duplicate captions intentionally remain separate training examples while sharing one cache file. At about 3.67 MiB per file, the full high-tier cache is expected to occupy roughly 414 GiB.

Strict precompute validates every newly encoded file. On a resumed build it also
reopens every existing file assigned to that shard; an incompatible or truncated
artifact is regenerated atomically instead of being trusted as a successful skip.
The dependent training job additionally checks the exact expected file set and a
spread-out geometry sample before constructing the model.

Camera action convention is the same native camera contract used elsewhere in this repo:

- domain name: `camera_pose`;
- domain id: 2;
- raw dim: 9;
- channels: `[pos(3), rot6d(6)]`;
- padded to 64 only for the native action projection;
- loss/output uses only raw channels `[:9]`.

The dataset scans:

- manifest: `/weka/jungbin/nymeriaplus_kimodo_proportional/video/manifest_video.jsonl`
- split: `/weka/jungbin/nymeriaplus_kimodo_proportional/train_test_split.json`
- latent root: `NYMERIA_LATENT_ROOT`

The unfiltered T97 train index contains 119,632 cached dataset rows representing 115,583 unique physical windows. The active quality artifact is:

```text
/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/camera_motion_quality_filter_v1_T97.json
SHA-256: 1fd6465890cbf175068db839beb8bb220f6964090ff2c583cbf50d5001989848
```

It removes 1,583 train rows / 1,524 unique windows and retains 118,049 rows / 114,059 unique windows. On test it removes 113 of 12,613 rows and retains 12,500. Filtering is fail-closed: kind, schema version, T97 span, split, duplicate keys, summary counts, file presence, and launcher SHA are validated. The filter does not mutate cached latents or manifests. Full rationale and counts are in `AUDIT.md` and inside the artifact itself.

## Task Mix

The production four-stream mix is:

```text
forward_dynamics: 40%
inverse_dynamics: 25%
policy:           20%
image2video:      15%
```

`MODE_WEIGHTS` is imported from `nymeria_world/nymeria_camera_rgb_dataset.py`.

Mode behavior:

- `forward_dynamics`
  - Condition: frame-0 image/video condition plus all 96 camera actions.
  - Target: future video latents.
  - Text: Nymeria caption represented as the official action JSON prompt.

- `inverse_dynamics`
  - Condition: all 97 video frames.
  - Target: 96 camera actions.
  - Text: empty caption, matching native inverse-dynamics convention.

- `policy`
  - Condition: frame-0 image/video condition.
  - Target: future video latents and 96 camera actions.
  - Text: Nymeria caption represented as the official action JSON prompt.

- `image2video`
  - Condition: frame-0 image/video condition.
  - Target: future video latents.
  - No action fields.
  - Text: Nymeria caption plus the same duration/FPS and resolution prose appended by official image-to-video inference. No action-viewpoint prose is added.

`image2video` is included here as video regularization. This is different from the older multi-GPU joint-attention warning where action-less steps could desync custom DDP all-reduce. Native Cosmos training handles mixed native task streams.

The no-I2V ablation sets `NYMERIA_DROP_MODES=image2video`; it does not change inverse dynamics or any model/sampler/loss setting. Its raw stream ratios remain `40/25/20`, which the native joint loader normalizes to effective probabilities `47.06/29.41/23.53%`. It deliberately answers whether the I2V regularizer helps visual quality; it is not a matched task-exposure ablation because the remaining tasks receive more updates over the same 100k iterations.

The 10% CFG text dropout runs after mode-specific formatting, so it drops the entire JSON/prose prompt to the tokenizer's empty-string conditioning. Inverse-dynamics text is already exactly empty.

## Video-Quality Ablation Suite

The 2026-07-21 A-D suite tests action-loss weighting, longer visual context, and
how narrowly the generator is adapted. All runs use the pinned quality filter,
standalone `C -> A person` caption replacement, fixed four clips per GPU (global
batch 32), 100k scheduler/steps, save every 5k, LoRA/base LR `5e-5`, 4x action-head
LR, gradient clipping 1.0, and PowerEMA. The suite-specific action loss weight is
`2.0`, and vision loss is normalized by active suffix elements per sample.

The requested RGB prefixes `[1,8,16,32,48]` cannot be represented exactly by the
cached causal Wan-VAE latents. A 97-frame clip has causal groups `{0}`, `{1..4}`,
`{5..8}`, and so on, so an exact clean/noisy boundary must contain `1 + 4N` RGB
frames. The controlled suite therefore uses the nearest forward boundaries:

```text
RGB clean prefix:    1, 9, 17, 33, 49
Wan latent prefix:   1, 3,  5,  9, 13
RGB suffix starts:   1, 9, 17, 33, 49
```

Using 8/16/32/48 with cached full-clip latents would either condition on a latent
that contains a target boundary frame (leakage) or discard some requested clean
frames. `rgb_prefix_to_latent_frames` therefore rejects non-exact boundaries.
For every visual generation task, the sequence plan marks only the latent prefix
clean; RF noise and vision MSE indexes cover only the suffix. Camera indexing is
unchanged: forward dynamics conditions all 96 actions and policy predicts all 96
actions with `action_start_frame_offset=1`.

The configurations are:

| Run | Prefix | Generator adaptation | Training streams |
| --- | --- | --- | --- |
| A | 1 | generation Q/K/V/O LoRA | forward/inverse/policy/I2V (`40/25/20/15`) |
| B | uniform `1,9,17,33,49` | generation Q/K/V/O LoRA | forward/inverse/policy/I2V (`40/25/20/15`) |
| C | uniform `1,9,17,33,49` | no generator LoRA; action interface only | forward/inverse/policy (`40/25/20`) |
| D | uniform `1,9,17,33,49` | camera-token-only generation K/V LoRA | forward/inverse/policy (`40/25/20`) |
| E, optional | 1 | camera-token-only generation K/V LoRA | forward/inverse/policy (`40/25/20`) |

C and D deliberately exclude I2V from training. An I2V pack has no camera tokens:
C's video loss reaches no trainable parameter, and D's camera mask is all false.
Native Cosmos's dummy action branch would keep backward/collectives valid, but the
step would contain only zero-valued trainable dependencies and no useful update.
The experiment now fails fast if `action_only` or `camera_kv_lora` is configured
with an active I2V stream. The remaining raw `40/25/20` ratios normalize to
`47.06/29.41/23.53%`. I2V is still evaluated for C/D as a frozen-prior regression
test at every prefix.

`camera_kv_lora` keeps the ordinary DCP keys (`weight`, `lora_A.weight`, and
`lora_B.weight`) but evaluates the LoRA residual only on packed action-token rows
of `k_proj_moe_gen` and `v_proj_moe_gen`. Video and text rows receive the frozen
base projection. LoRA B is zero initialized, so the initial network output exactly
matches the base checkpoint. Video loss can still update camera K/V through visual
queries attending to camera keys/values. Generator Q/O and all base projections
remain frozen.

Launchers:

```bash
sbatch native_phase_training/sbatch_phase1_video_quality_A.sh
sbatch native_phase_training/sbatch_phase1_video_quality_B.sh
sbatch native_phase_training/sbatch_phase1_video_quality_C.sh
sbatch native_phase_training/sbatch_phase1_video_quality_D.sh
# Optional factorization only:
sbatch native_phase_training/sbatch_phase1_video_quality_E.sh
```

B-D run a two-step, eight-GPU train/save/restart preflight on first allocation.
The second invocation resumes the same DCP, including model, optimizer, scheduler,
EMA, trainer/global step, RNG, and available dataloader state, before continuing
to 100k. An existing checkpoint skips the preflight and resumes normally.

Every 5k checkpoint is saved, but the A-E callbacks submit evaluation only at
10k multiples. Two distinct jobs are submitted; their metrics must not be mixed.

The compact diagnostic uses EMA and NVIDIA's official UniPC path (action modes:
30 steps/guidance 1/shift 3; I2V: 35 steps/guidance 6/shift 3), with five held-out
sources at all five fixed prefixes. It writes GT/generated pairs, per-source
GT-plus-prefix grids, inverse/policy camera plots, suffix-only PSNR/SSIM/LPIPS
split into relative early/middle/late thirds, and full plus suffix-reanchored
policy camera metrics under:

```text
<run>/checkpoint_evals/iter_XXXXXXXXX/{viz,metrics,COMPLETE.json}
```

This directory has `n=5`; it is a qualitative/prefix diagnostic and is not the
historical Phase-1 benchmark. The separate canonical job reproduces the original
held-out protocol: one prefix-1 forward-dynamics sample and one inverse-dynamics
sample for every one of the 71 test sequences. It uses the exact historical
`native_phase1_eval_inputs_full71_256_T97_v2` inputs, EMA, UniPC, shift 3,
30 steps, and guidance 1, and writes:

```text
<run>/eval_full71_inverse_forward/iter_XXXXXXXXX/
  inference/                  # 71 forward + 71 inverse outputs
  analysis/forward_metrics.json
  analysis/invdyn_metrics.json
  analysis/dreamsim_metrics.json          # optional advanced forward metric
  analysis/cdfvd_videomae_metrics.json    # optional advanced set metric
  analysis/COMPLETE.json
  resolved_run_contract.json
  COMPLETE.json
```

Only these `n=71` metrics are directly comparable with
`native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k/eval_full71_inverse_forward`.
The optional advanced reports use official DreamSim and CVPR 2024
content-debiased FVD with VideoMAE-v2-SSv2, not the deprecated TensorFlow/I3D
implementation. Their exact setup, frame sampling, limitations, and manual
commands are documented in `native_phase_training/FORWARD_VIDEO_METRICS.md`.
The full-71 job requests one exclusive eight-GPU node; the compact diagnostic
requests one GPU. Both resolve the immutable checkpoint architecture contract
before importing the Cosmos inference configuration.

### 2026-07-24 Final A-D Comparison

The complete canonical full-71 comparison is:

| Model | Step | PSNR | SSIM | LPIPS | DreamSim | CD-FVD | Rot. deg | Dir. cos | Trans. mm | ATE cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, action weight 10 | 100k | 19.4988 | 0.612583 | 0.285336 | 0.170482 | **281.614** | **0.213036** | **0.838294** | **3.177** | **2.357** |
| A, prefix-1 global LoRA | 100k | **19.5343** | **0.614062** | **0.284598** | **0.169134** | 295.561 | 0.271635 | 0.824262 | 3.832 | 2.413 |
| B, variable-prefix global LoRA | 100k | 18.6945 | 0.586307 | 0.315056 | 0.177504 | 304.385 | 0.287409 | 0.811321 | 4.041 | 2.522 |
| D, camera-token K/V LoRA | 100k | 18.3851 | 0.576988 | 0.328101 | 0.189021 | 319.056 | 0.314777 | 0.811414 | 4.387 | 3.030 |
| Historical step 7k, current contract | 7k | 18.4828 | 0.574474 | 0.322919 | 0.175386 | 282.414 | 0.267831 | 0.770949 | 4.681 | 3.995 |

All CD-FVD values use the canonical FP32 VideoMAE-v2-SSv2 feature path. C
stopped at 65k and has no full-71 evaluation. Its latest compact five-source
macro forward PSNR/SSIM/LPIPS is `15.203/0.430/0.447`, and its inverse camera
means are `0.601 deg`, `12.99 mm`, and `9.63 cm` ATE.

The compact five-source, five-prefix forward macro results are Original
`16.955/0.497/0.368`, A `17.203/0.505/0.359`, B
`17.507/0.519/0.356`, C `15.203/0.430/0.447`, and D
`16.756/0.489/0.379`. B benefits from long prefixes, but Original remains the
best balanced prefix-1 model because it retains the strongest camera metrics
and best full-suffix CD-FVD. See `PHASE1_VISUAL_QUALITY_AUDIT.md` for policy,
I2V, per-horizon interpretation, old-checkpoint reconstruction provenance, and
the completed 256/720 resolution-tier comparisons.

Training rank 0 atomically writes `<run>/native_phase1_contract.json` before
launch. It records the adaptation mode, active/dropped tasks, LoRA enablement and
targets, and training prefix list. A resume with a different contract fails before
training. `sbatch_checkpoint_eval.sh` resolves this file before importing
`experiment.py`, exports the saved adaptation/drop settings, and records the result
as `resolved_run_contract.json`. Explicit environment values are accepted only when
they exactly match the checkpoint contract. Runs created before this file existed
are recovered from their resolved `config.yaml`; an ambiguous legacy checkpoint
fails instead of defaulting to global LoRA. `COMPLETE.json` includes the resolved
contract path alongside visualization and metric manifests.

In every pair and prefix-grid video, green denotes a frame sourced from GT
(the full reference tile or the clean condition prefix) and red denotes a
sampled suffix frame. The generated tile changes color and header on RGB frame
`prefix_length`, so prefix 9 is green on frames 0-8 and red from frame 9 onward.
`viz/manifest.json` records these ranges under `video_frame_provenance` using
zero-based inclusive indexing.

The shared compact input set is
`/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1`.
Its expected cardinality is five inverse records and 25 records for each visual
mode. `COMPLETE.json` is written only after official inference, visualization,
and quantitative metrics all succeed.

## Packing Contract

The raw `video` tensor is only `[3,97,1,1]` metadata, so the stock joint loader would count zero spatial vision patches. `LatentAwareIterativeJointDataLoader` starts with the parent count for text, EOS, vision boundary markers, action, and sound, then adds the real cached-latent patches:

```text
latent [48,25,16,16], patch 2x2 -> 25 * 8 * 8 = 1600 vision patch tokens
720-tier latent [48,25,40,40], patch 2x2 -> 25 * 20 * 20 = 10000 vision patch tokens
```

Production uses `NATIVEP1_CLIPS_PER_GPU=4`, so each rank packs exactly four samples and the eight-GPU global batch is 32 clips. Batch size is not part of the rectified-flow or official-sampler contract. The framework requires sample-count and token-count limits to be mutually exclusive, so fixed-four mode sets `max_samples_per_batch=4` and `max_sequence_length=None`. This remains below the native ceiling even at the tokenizer's 4,096-token truncation limit: four worst-case T97 action samples are below 24k tokens, versus the 45,056-token model budget. A real resolved batch on 2026-07-12 contained four samples and 7,323 tokens.

At the 720 tier, a representative 64-text-token action clip is 10,163 packed tokens; four clips are below the 45,056 native budget and five exceed it. Fixed-four mode deliberately disables the token ceiling, but the same calculation shows that four is the natural maximum for typical high-tier clips. The 8-GPU smoke reached finite losses with about 117-119 GiB used per 143.8-GiB H200, leaving roughly 24-26 GiB per GPU.

Set `NATIVEP1_CLIPS_PER_GPU=0` to reproduce the earlier token-budget mode. That mode uses `max_sequence_length=45056`, fits about 25-26 typical samples per rank, and retains the latent-aware counter above. Do not compare iteration counts across these modes without also comparing consumed clips/tokens: roughly 30k token-budget steps and 190k-200k fixed-four steps expose a similar number of clips.

## Model and Optimizer

Default model:

- Cosmos-3 Nano native config.
- `LatentOmniMoTModel`.
- `resolution='256'`.
- local Wan VAE path from `WAN_VAE_PATH`.
- local text tokenizer from one of:
  - `COSMOS_TEXT_TOKENIZER_PATH`;
  - local `nvidia/Cosmos3-Nano` HF snapshot `text_tokenizer`;
  - local Qwen3-VL snapshot;
  - remote download only if `ALLOW_HF_TOKENIZER_DOWNLOAD=1`.

Trainable parameters in LoRA mode:

- generator LoRA on:
  - `q_proj_moe_gen`
  - `k_proj_moe_gen`
  - `v_proj_moe_gen`
  - `o_proj_moe_gen`
- LoRA rank 16, alpha 32.
- action heads kept trainable:
  - `action2llm`
  - `llm2action`
  - `action_modality_embed`

Optimizer and schedule:

- FusedAdam.
- base LR `5e-5` for LoRA mode by default (`NATIVEP1_LORA_LR`).
- camera/action modules use a 4x LR multiplier by default (`NATIVEP1_ACTION_LR_MULT=4.0`):
  - `action2llm`
  - `llm2action`
  - `action_modality_embed`
- weight decay `0.05`.
- betas `[0.9, 0.99]`.
- LambdaLinear with a 500-step flat plateau and cycle length 100000. Because `f_start=f_max=0.4`, this is not an increasing warmup: the run stays at about `2e-5` effective LR for LoRA and `8e-5` for camera/action modules for 500 steps, then linearly decays toward zero.
- grad clip 1.0.
- native PowerEMA is enabled. Training gradients update the regular LoRA/action parameters; the framework then updates an FP32 EMA copy and official evaluation samples it with `--use-ema-weights`.

Native RF/loss settings are intentionally inherited from Cosmos Nano:

- video train-time distribution: `waver`;
- image train-time distribution: `logitnormal`;
- `independent_action_schedule=false`, so action noising reuses the per-sample vision/video sigma. The configured `train_time_action_distribution=logitnormal` is dormant unless independent action scheduling is enabled;
- train-time loss weight: `uniform`;
- action loss weight: `10.0`;
- shift config: `{256: 3, 480: 5, 720: 10}`.

Do not change these unless explicitly running an ablation. The point of this directory is to match the official sampler/training distribution as closely as practical while using saved latents.

The controlled high-tier run selects `resolution='720'` and shift 10 from the released Cosmos Nano config. NVIDIA's technical report describes a different pretraining table (`1/3/5` at 256p/480p/720p), but the local released checkpoint code and inference defaults use `3/5/10`. This repository follows the released checkpoint/config contract for this run, as explicitly chosen on 2026-07-26.

## Launch

Dryrun from repo root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth \
BASE_CHECKPOINT_PATH=/weka/jungbin/cosmos3_nano_dcp \
NYMERIA_RESOLUTION=256 \
bash native_phase_training/run_latent_train.sh --dryrun \
  trainer.max_iter=1 \
  checkpoint.save_iter=1000 \
  job.name=native_phase1_config_check \
  model.config.parallelism.data_parallel_replicate_degree=8 \
  model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.parallelism.context_parallel_shard_degree=1 \
  model.config.parallelism.cfg_parallel_shard_degree=1
```

## TensorBoard

Native Phase 1 uses the Cosmos `TensorBoardLog` callback. The entrypoint writes event files to:

```bash
${RUN_DIR}/tensorboard
```

For the in-flight token-budget job `2838`:

```bash
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_tokpack_lora5e5_action4x_100k/tensorboard
```

Override with `TB_LOG_DIR=/some/path` when launching if a different location is needed. The Slurm launcher
prints the resolved TensorBoard path at startup. Job `2801`, submitted before this run-local path fix, wrote
events to the older shared fallback:

```bash
/weka/jungbin/cosmos_motion_ft_runs/tensorboard
```

To view the default production run after this patch:

```bash
tensorboard --logdir /weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k/tensorboard
```

Production Slurm:

```bash
sbatch native_phase_training/sbatch_phase1_native_camera.sh
```

High-tier cache and dependent training:

```bash
cache_job=$(sbatch --parsable native_phase_training/sbatch_precompute_latents_720tier.sh)
train_job=$(sbatch --parsable --dependency=afterok:${cache_job} \
  native_phase_training/sbatch_phase1_native_camera_720tier.sh)
printf 'cache=%s train=%s\n' "${cache_job}" "${train_job}"
```

The cache launcher is a three-element exclusive-node array with eight GPUs per element, for 24 deterministic global shards. It excludes node 2. The training dependency releases only if every array element exits successfully; training then performs an exact expected-file-set check and validates 256 spread-out cache samples before model construction.

High-tier production run:

```text
run: /weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_720tier640_bs4_lora5e5_action4x_ema_100k
compact eval inputs: /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_viz5_720_T97_release_s10_v1
full-71 eval inputs: /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_720_T97_release_s10_v1
```

It is a controlled counterpart to historical Original: all four tasks, prefix 1, global Q/K/V/O LoRA, trainable camera projections, action weight 10, LR `5e-5`, action LR 4x, global batch 32, EMA, 100k scheduler horizon, and 5k checkpoints. Only the spatial tier/cache geometry and the released tier-specific shift change. Evaluation is submitted every 10k checkpoint, not every 5k save.

Production was submitted on 2026-07-26 only after commit `a1e49f4` was pushed:

```text
cache array: 3109
  launcher: native_phase_training/sbatch_precompute_latents_720tier.sh
  Slurm logs: /home/jungbin_cho/cosmos_motion_ft/slurm-p1pc720-3109_<array>.out
  shard logs: /weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97_720tier_640/_logs/
training: 3110
  dependency: afterok:3109
  launcher: native_phase_training/sbatch_phase1_native_camera_720tier.sh
  Slurm log: /home/jungbin_cho/cosmos_motion_ft/slurm-p1cam720-3110.out
```

At submission, array elements `3109_0` and `3109_1` started on nodes 1 and 3.
Element `3109_2` remained pending for resources because nodes 0 and 2 were
allocated to other users. This is expected: the launcher excludes node 2 and no
running job was canceled. Training job `3110` remains dependency-held until all
three cache elements exit successfully.

High-tier acceptance smoke completed before production submission:

```text
limited cache:
  /weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97_720tier_640_smoke_v2
training run:
  /weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_720_bs4_v2/cosmos3_camera/camera_world/native_phase1_720tier_640_bs4_contract_smoke
resumed checkpoint:
  checkpoints/iter_000000002
official inference, visualization, and metrics:
  /weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_720_bs4_v2/official_inference_4mode_iter2_T97_v2
```

The fixed-four 8-GPU update used about 117-119 GiB per H200 and produced finite
losses on every rank. Iteration 1 saved model/EMA, optimizer, scheduler, and
trainer state; exact resume restored iteration 1, advanced one finite update,
and saved iteration 2. Official EMA/UniPC inference then completed forward,
inverse, policy, and I2V at explicit T97. Every raw output is 640x640, 97 frames,
and 20 FPS. Forward, inverse, and policy carry finite `[96,9]` action payloads.
The three generative comparisons mark clean frame 0 with a lime
`GT CONDITION` border and generated frames 1-96 with a red `GENERATED` border.
Suffix-only PSNR/SSIM/LPIPS and camera metric manifests, plus the top-level
`COMPLETE.json`, were written successfully.

Current production run name:

```text
native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k
```

Expected output directory:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k
```

The launcher currently uses:

- `#SBATCH --gres=gpu:8`
- one `torchrun --nproc_per_node=8` launcher;
- `#SBATCH --exclusive`;
- GPU memory preflight requiring every GPU to have at least `NATIVEP1_MIN_FREE_MIB`, default 132000 MiB, free before training starts;
- `model.config.compile.enabled=false` to avoid first-step TorchInductor memory spikes.
- `trainer.max_iter=100000`;
- `NATIVEP1_LORA_LR=5e-5`;
- `NATIVEP1_ACTION_LR_MULT=4.0`.
- `NATIVEP1_CLIPS_PER_GPU=4` (`0` opts back into 45,056-token packing).
- native PowerEMA enabled; checkpoint evaluation uses EMA weights.
- `NATIVEP1_AUTO_EVAL=1`, with five held-out qualitative samples by default (`NATIVEP1_VIZ_N=5`).

`--exclusive` asks Slurm for exclusive allocation, but it does not kill or prevent orphan processes that are already running outside Slurm. The preflight exists to catch those before torchrun.

## Official Inference

Use the official Cosmos inference script from `/home/jungbin_cho/cosmos-framework`.

Build held-out inputs with the native-local builder, not the historical 480p/shift-10 helper:

```bash
PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework \
python native_phase_training/prep_test_eval.py \
  --out /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs \
  --n 0 --seed 0
```

`--n 0` requests one window from every available held-out sequence. The current split has 71 test UUIDs; the script reports any UUID for which it cannot build a usable T97 sample rather than silently shrinking the set.

Important config-file syntax:

- Correct: `--config-file native_phase_training/inference_config.py`
- Incorrect: absolute `.py` path such as `/home/.../inference_config.py`
- Incorrect: module name without `.py` such as `native_phase_training.inference_config`

The loader validates `.py` paths, then internally converts relative paths into importable modules.

Example:

```bash
cd /home/jungbin_cho/cosmos-framework
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export NYMERIA_RESOLUTION=256

/home/jungbin_cho/miniforge3/envs/cosmos/bin/torchrun --standalone --nproc_per_node=1 \
  -m cosmos_framework.scripts.inference \
  --checkpoint-path /path/to/checkpoints/iter_000005000 \
  --config-file native_phase_training/inference_config.py \
  --experiment world_camera_nymeria_latent_nano \
  --sampler unipc \
  --use-ema-weights \
  --parallelism-preset latency \
  --dp-shard-size 1 --dp-replicate-size 1 --cp-size 1 --cfgp-size 1 \
  --no-use-torch-compile --no-use-cuda-graphs --no-guardrails \
  -o /weka/jungbin/cosmos_motion_ft_runs/some_eval_dir \
  -i /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs/fd_input.jsonl \
     /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs/invdyn_input.jsonl \
     /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs/policy_input.jsonl \
     /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs/i2v_input.jsonl
```

For local DCP checkpoints, pass the checkpoint iteration root. The inference loader appends `/model` if needed.

Then create all four qualitative views:

```bash
python /home/jungbin_cho/cosmos_motion_ft/native_phase_training/visualize_checkpoint.py \
  --inference-root /weka/jungbin/cosmos_motion_ft_runs/some_eval_dir \
  --eval-root /weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs
```

Checkpoint saving does **not** run qualitative inference inside the training process. The stock NVIDIA `EveryNDrawSample` callback remains disabled because it samples only the currently selected stream, treats the cached path's 1x1 dummy pixels as raw GT, does not visualize predicted actions, and uses generic shift/guidance defaults rather than this run's official action recipe.

The generic production launcher defaults to `NATIVEP1_AUTO_EVAL=1`. Before training, it prepares five held-out inputs under `NATIVEP1_EVAL_INPUT_DIR`. After each completed DCP save, `NativeCheckpointEvalSubmitter` queues `sbatch_checkpoint_eval.sh`; outputs land under:

```text
${RUN_DIR}/checkpoint_evals/iter_XXXXXXXXX/{<mode-specific sample dirs>,viz/}
```

Set `NATIVEP1_AUTO_EVAL=0` for smoke tests or when checkpoint visualization jobs should not be submitted. The official inference/visualization commands above remain the manual recovery path if a submitted evaluation job fails.

The two quality-filter launchers pin `NATIVEP1_AUTO_EVAL=0` because automatic 5k checkpoint callbacks would create 40 unsupervised one-GPU Slurm jobs while other users have multi-node jobs pending. This changes only evaluation scheduling, not training. Evaluate selected checkpoints manually with the same four-mode official path.

The inference command explicitly uses NVIDIA's `cosmos_framework.scripts.inference`, `--sampler unipc`, and EMA weights (also the official default). Action modes use 30 steps and guidance 1; image-to-video uses 35 steps and guidance 6. Resolution and shift must come from `native_phase1_contract.json`: historical runs use 256/shift 3, while the new high-tier run uses 720/shift 10. These values reach `OmniMoTModel.generate_samples_from_batch`; the logged sampler shift must equal the saved contract.

The fixed-prefix JSONLs also contain local bookkeeping fields
`source_name`, `rgb_prefix_length`, and `latent_prefix_length`. NVIDIA's
Pydantic inference schema rejects these extra fields, so
`sbatch_checkpoint_eval.sh` first writes schema-clean copies under
`${EVAL_OUTPUT_DIR}/inference_inputs/`. The sanitizer removes only those three
fields, validates their agreement with the sample name and explicit
`condition_frame_indexes_vision`, and preserves the latter for the prefix shim.
Official inference consumes the clean copies; visualization and metric grouping
consume the original enriched JSONLs. Do not point the official inference CLI
directly at the enriched files.

The framework's bundled modality JSON files contain a literal shift of 10 because the release defaults target the high-resolution tier. Historical 256 runs intentionally override that value to 3. The high-tier run intentionally keeps 10. The solver, sigma construction, EMA loading, CFG implementation, and task step/guidance defaults remain NVIDIA's official path; using either shift with the other run's spatial contract is a train/evaluation mismatch.

A 2026-07-24 controlled audit of the historical step-7000 regular weights
separated resolution tier from shift on five exact forward-dynamics inputs.
At fixed shift 3, 720-tier generation reduced adjacent RGB change, temporal
second difference, and flow-compensated residual by
`17.7%/18.8%/14.0%` relative to 256-tier generation. At fixed 256, changing
shift 3 to 10 changed the same diagnostics by only
`-0.7%/-0.4%/-1.9%`. Thus shift 10 is not the flicker fix; the larger spatial
inference tier is the dominant same-checkpoint factor. The old run's dataset
config was also 256, so this is a pretrained spatial-resolution inference
effect, not evidence of high-resolution Nymeria finetuning. See
`PHASE1_VISUAL_QUALITY_AUDIT.md` before comparing recent 256-tier EMA videos
with old 640x640 outputs.

The follow-up used every Original/A/B/D 100k EMA checkpoint and all 71
canonical held-out prefix-1 forward records. Official 720-tier sampling
produced 640x640 videos with shift 10, UniPC-30, guidance 1, and seed 0. Against
each model's canonical 256/shift-3 result, second-difference and
flow-compensated residual improved by `19-22%` and `15-23%`; DreamSim also
improved for all four models. Frame-aligned PSNR/SSIM/LPIPS worsened for all
models. CD-FVD changed from `281.614 -> 308.912` (Original), `295.561 ->
301.229` (A), `304.385 -> 303.792` (B), and `319.056 -> 295.649` (D). Thus the
high tier reliably reduces the diagnosed flicker, but only D gains a large
distributional-quality improvement. Raw outputs, reports, and 25 labeled
comparison videos are under:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/
phase1_ema100k_resolution_matrix_5_20260724/full71_720/
```

See `PHASE1_VISUAL_QUALITY_AUDIT.md` for the complete table and limitations.

### 2026-07-23 Visual-Quality Root-Cause Audit

Do not infer that historical training was better from the existing MP4s alone.
The inspected old output is 640x640 and came from a manually merged-model
evaluation path, while the recent A output is 256x256 and was loaded directly
from its DCP after recovering the architecture from the saved run config. A
direct A/100k EMA prefix-1 test at 256 changed only UniPC shift from 3 to 10 and
produced essentially tied PSNR/SSIM/LPIPS, with no consistent shift-10 gain.

The fp16 cache is also a low-probability cause. Wan encode runs under bf16
autocast, and 9,830,400 values from 32 T97 cache files were all exact under a
float32-to-bf16 round trip. The portable check is:

```bash
python native_phase_training/audit_cached_latent_precision.py \
  --latent-root /weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97 \
  --max-files 32
```

If a future matched-resolution comparison still favors the old model, the
leading training hypothesis is the shared global LoRA update contract, not the
cache: historical training averaged forward/inverse/policy across different
ranks on every update with up to 128 clips, whereas current training uses one
homogeneous task across all ranks with 32 clips. Recent A additionally includes
pure I2V updates and reduces action loss from 10 to 2, so the shared LoRA gets
substantially less camera-gradient regularization and more direct Nymeria visual
reconstruction pressure. This is not proven. Exact evidence, metrics, and the
required sampling/training matrix are in `PHASE1_VISUAL_QUALITY_AUDIT.md`.

## Smoke Tests Completed

All smoke artifacts are generated outputs under `/weka`; do not edit them as source files.

### 2026-07-21 Video-Quality Suite Launch

A-D were submitted as jobs `3025`-`3028`; E was not submitted. A passed a
separate two-step real GPU smoke before submission and then advanced normally at
about 1.0-1.1 seconds per iteration. B's in-job distributed preflight completed
two finite eight-GPU updates, saved full DCP `iter_000000002`, correctly skipped
automatic evaluation at step 2, restored `dataloader/model/optim/scheduler/trainer`
with global iteration 2, and completed finite step 3. C passed the same sequence;
its runtime audit reported exactly five trainable tensors, 16,914,432 trainable
parameters, zero LoRA tensors, and only the three expected streams. D remains
guarded by the identical first-allocation preflight; its 100k command cannot run
unless camera-token K/V forward/backward, EMA/DCP save, and exact resume succeed.

Focused CPU contracts pass 35/35. They cover causal prefix conversion, clean/noisy
and loss masks, camera alignment, active normalization, token-aware projection
equality/masking/gradients/state keys, fixed-prefix inference records, evaluation
schema sanitization/cadence, immutable run contracts, legacy architecture recovery,
conflicting evaluation overrides, frame-provenance visualization, quality-filter
behavior, and fixed-four loader cycling.

### 2026-07-22 Architecture-Contract Evaluation Smoke

The post-fix evaluator was run directly on node 2 with both
`NATIVEP1_ADAPTATION_MODE` and `NYMERIA_DROP_MODES` deliberately unset. Starting
from C/65k, it recovered `action_only` plus dropped I2V from the legacy run-level
`config.yaml`, instantiated zero LoRA tensors and exactly the five action-interface
parameters, loaded EMA DCP successfully, and completed one official UniPC sample
for forward, inverse, policy, and I2V. It wrote four `sample_outputs.json` files,
seven visualization records, all metric manifests, `resolved_run_contract.json`,
and a top-level completion marker under:

```text
/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_contract_eval_C65_n1_p1_20260722
```

### 2026-07-22 Direct 10k/20k Evaluation Recovery

The automatic A/B/C evaluations for steps 10k and 20k were pending as Slurm
jobs `3029`-`3034`, so the same wrapper was run directly on spare GPUs 1-6 of
node 3. All six evaluations completed. Each output contains 80 successful
official-inference samples (25 forward, 5 inverse, 25 policy, 25 I2V), 95
visualization manifest records (the 80 individual views plus 15 five-prefix
grids), all three metric JSONs, and a top-level `COMPLETE.json`. Results are
under each A/B/C run's `checkpoint_evals/iter_000010000` and
`iter_000020000` directories. After validating every marker and status, the
redundant pending jobs `3029`-`3034` were canceled; training jobs `3025`-`3027`
and pending D job `3028` were not changed.

### 2026-07-22 A/C 30k Evaluation

A/30k and C/30k were evaluated directly on node 3 GPUs 1 and 2 because automatic
jobs `3035` and `3036` remained pending. Both have 80/80 successful official
EMA-UniPC outputs, 95 visualization records, complete five-prefix metrics, and
`COMPLETE.json`. All 75 pair-video records and 15 grid records carry explicit
green-GT/red-generated provenance. After validation, jobs `3035` and `3036` were
canceled without changing training jobs `3025`-`3028`.

On the five-source diagnostic set, A/30k inverse means are rotation `0.4047 deg`,
translation-direction cosine `0.8935`, translation error `6.764 mm`, and ATE
`4.102 cm`. Its forward suffix PSNR/SSIM/LPIPS is `15.709/0.4590/0.4439` at
prefix 1 and `16.985/0.4952/0.3570` at prefix 49, improving all six values over
A/20k. C/30k inverse means are `0.6466 deg`, `0.7533`, `14.628 mm`, and
`11.543 cm`, continuing its 10k-to-20k-to-30k improvement. C/30k forward metrics
are `14.587/0.4148/0.5001` at prefix 1 and `15.390/0.4369/0.4182` at prefix 49.
These are five-source checkpoint diagnostics, not full held-out-set estimates.

### 2026-07-21 Filtered Four-Task and No-I2V Acceptance Smoke

Artifacts are under:

```text
/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_qfilter_20260721
```

The four-task smoke deterministically selected forward dynamics, policy, and image-to-video in four finite-loss updates; inverse dynamics was prewarmed there and selected during the nine-step no-I2V smoke. The no-I2V smoke exercised all three active streams. Across the paired smokes every active training path ran, TensorBoard and PowerEMA updated, and DCP iterations 4 and 9 saved and reloaded. Both resolved the exact 118,049-row filtered train index and the expected task ratios.

Each exact smoke checkpoint was then loaded with `cosmos_framework.scripts.inference --sampler unipc --use-ema-weights`. Both generated one held-out sample for all four inference modes, including I2V for the no-I2V model to prove that removing the training stream does not break that inference surface. Every output reports success; action outputs are `96x9`; all eight raw MP4s are 256x256, 97 frames, and 20 FPS; logs report shift-3 UniPC with 30 action steps and 35 I2V steps; and both visualization manifests contain all four modes.

### 2026-07-21 Filtered Production Launch

The controlled four-task run is Slurm job `3017` (`np1qf4`) on node 1. Its output and log are:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_qfilterv1
/home/jungbin_cho/cosmos_motion_ft/slurm-np1qf4-3017.out
```

The no-I2V ablation is Slurm job `3018` (`np1qf3`) on node 0:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_qfilterv1_noi2v
/home/jungbin_cho/cosmos_motion_ft/slurm-np1qf3-3018.out
```

Launch validation confirmed all eight ranks in each job, the pinned filter SHA, `118,049` retained and `1,583` filtered train rows, PowerEMA, rank-16 generator LoRA, the action heads, and finite updates. The four-task job kept `40/25/20/15`; the no-I2V job instantiated exactly three streams at `47.1/29.4/23.5%`. After prewarm, observed updates were approximately 1.0-1.2 seconds. Automatic evaluation is disabled as documented above.

The queue was handed over one node at a time so another user's pending two-node job was never bypassed by freeing two nodes simultaneously. Old jobs `3011` and `3010` were canceled separately, with pending one-node job `3016` allowed to start between them. After `3017` was running with all eight ranks initialized, the model/filter loaded, and stable GPU allocations, old head-camera job `3003` was canceled at its saved step-115k checkpoint and `3018` was submitted on the released node. Finite updates for `3017` were confirmed after that handoff. The other user's two-node job `3014` remained pending for two simultaneous resources throughout this sequence.

### 2026-07-21 Filtered LR Ablation

`sbatch_phase1_native_camera_qfilter_lr1e5.sh` is an exact four-task filtered-control LR ablation. It changes optimizer base/LoRA LR from `5e-5` to `1e-5`; the existing 4x action-head multiplier remains, so action-head effective LR is `4e-5`. The filter and SHA, `40/25/20/15` task mix, fixed-four global batch 32, rank-16 LoRA targets, PowerEMA, native RF distributions, LambdaLinear schedule shape, 100k duration, and 5k save cadence are unchanged. The run path is:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora1e5_action4x_ema_100k_qfilterv1
```

`sbatch_phase1_native_camera_qfilter_lr1e5_person.sh` is paired with that `1e-5`
run and changes only caption wording. Before mode-specific JSON/prose formatting, every
whole-token uppercase `C` is replaced with `A person` using `(?<!\w)C(?!\w)`; lowercase
`c` and `C` embedded in words or identifiers are untouched. Inverse-dynamics text remains
exactly empty, and the existing 10% whole-prompt CFG dropout remains unchanged. The setting
is recorded as `replace_standalone_c=true` in the resolved dataset config. Its run path is:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora1e5_action4x_ema_100k_qfilterv1_person
```

It was submitted as Slurm job `3021` (`np1qf4l1`) with log `/home/jungbin_cho/cosmos_motion_ft/slurm-np1qf4l1-3021.out`. At submission, node 3 was Slurm-idle but every GPU was occupied by 21 SSH-launched processes owned by another user, while nodes 0-2 were allocated. Job `3021` therefore uses `afterany:3016` rather than starting into a guaranteed OOM; no existing filtered control was canceled. A dry-run resolved optimizer LR `1e-5`, all three action multipliers `4.0`, four streams, the exact quality-filter path, PowerEMA, and the original 100k scheduler contract before submission.

### 2026-07-11 Prompt/Packing/Four-Mode Smoke

Corrected train run:

```text
/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_contract/cosmos3_camera/camera_world/native_phase1_json_tokpack_smoke_20260711
```

It completed four finite-loss optimizer steps with the latent-aware 45,056-token packer, wrote TensorBoard events, and saved/closed this DCP checkpoint:

```text
checkpoints/iter_000000004
```

That exact checkpoint loaded through official inference with explicit UniPC and produced four separate successful outputs under:

```text
/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_contract/official_inference_4mode_v2_20260711
```

Verified results:

- forward dynamics, inverse dynamics, policy, and image-to-video each have distinct output directories and `status=success`;
- all four MP4s are 97 frames, 256x256, 20 FPS;
- all three action-mode outputs contain action arrays shaped `[96,9]`;
- logs show UniPC shift 3 with 30 action steps and 35 image-to-video steps;
- `viz/manifest.json` covers all four modes;
- GT/generated comparison MP4s exist for forward, policy, and image-to-video;
- inverse and policy camera trajectory PNGs exist.

### Dataset and Dryrun

Local dataset/tokenizer smoke verified:

- four modes load;
- `video_latents` shape `[48,25,16,16]`;
- dummy `video` shape `[3,97,1,1]`;
- `image_size` `[256,256,256,256]`;
- camera actions `[96,64]` padded with `raw_action_dim=9`;
- `image2video` has no action fields.

TOML dryrun verified:

- local tokenizer path;
- local VAE override;
- `resolution=256`;
- four task streams 40/25/20/15;
- native RF config unchanged.

### One-Step Train Smoke

Post-patch one-step training ran on `a3ultravis-a3ultranodeset-1`, GPU 1:

```text
/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/cosmos3_camera/camera_world/native_phase1_smoke_train_postpatch
```

It verified:

- config resolution and local VAE;
- DCP base checkpoint load from `/weka/jungbin/cosmos3_nano_dcp`;
- cached-latent batch path;
- forward pass;
- backward pass;
- optimizer step;
- grad clip logging;
- final DCP checkpoint save.

Key success lines:

```text
Iteration 1: Loss: 0.1946
Done with training.
Checkpoint save completed
```

Saved smoke checkpoint:

```text
/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/cosmos3_camera/camera_world/native_phase1_smoke_train_postpatch/checkpoints/iter_000000001
```

### Official Inference Smoke

The exact post-patch checkpoint above was loaded through official `cosmos_framework.scripts.inference`.

Output:

```text
/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/official_inference_postpatch
```

Three samples succeeded:

- `smoke_forward_dynamics_camera`
- `smoke_inverse_dynamics_camera`
- `smoke_policy_camera`

Each wrote:

- `sample_args.json`;
- `sample_outputs.json`;
- `vision.mp4`;
- action content shaped `96 x 9`.

Each MP4 was verified as:

```text
width=256
height=256
r_frame_rate=20/1
duration=4.850000
nb_frames=97
```

Earlier visual/metric smoke artifacts:

```text
/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/official_inference_v5/viz
```

Files:

- `action_smoke_metrics.json`
- `inverse_action_compare.png`
- `video_contact_sheet.png`

Those are shape/decode/output-path smoke artifacts only. They are not quality evidence because they used a one-step training checkpoint and `num_steps=1` sampling.

## OOM Incident on Job 2792

First production submission:

```text
job id: 2792
name: nativep1
node: a3ultravis-a3ultranodeset-1
state: FAILED
exit: 1:0
elapsed: 00:09:22
```

Failure was CUDA OOM during the first training forward pass, not a dataset or checkpoint bug.

Evidence:

```text
/home/jungbin_cho/cosmos_motion_ft/slurm-nativep1-2792.out:1552
/home/jungbin_cho/cosmos_motion_ft/slurm-nativep1-2792.out:1700
```

The key log line:

```text
CUDA out of memory. Tried to allocate 1.24 GiB. GPU 2 has a total capacity of 139.80 GiB of which 623.38 MiB is free.
Including non-PyTorch memory, this process has 123.29 GiB memory in use.
Process 2951793 has 15.88 GiB memory in use.
```

PID check on `a3ultravis-a3ultranodeset-1`:

```text
pid: 2951793
user: jmleeluck
cmd: .venv/bin/python scripts/train_latent_grpo_libero.py ... --device cuda:2
gpu bus: 00000000:96:00.0
memory: about 16266 MiB
```

GPU mapping showed:

```text
GPU 2 = bus 00000000:96:00.0
```

So the immediate OOM trigger was an external process on the same GPU. The production config also had `model.config.compile.enabled=true`, while the successful smokes had compile disabled. TorchInductor compilation added first-step memory pressure. Both issues were addressed in the launcher.

Patch after job 2792:

- added `#SBATCH --exclusive`;
- added GPU memory preflight before torchrun;
- preflight fails if any GPU has less than 132000 MiB free by default;
- added `model.config.compile.enabled=false` to production launcher.

Replacement submission:

```text
job id: 2799
name: nativep1
state at submission: PD (Resources)
submitted on: 2026-07-10
launcher state at submission: 100k max steps, default LoRA LR 2e-4, compile disabled
```

Check current state with:

```bash
squeue -j 2799 -o '%.18i %.9P %.24j %.10u %.2t %.12M %.8D %R'
```

After the job was queued, the recommended first native baseline was revised to 50k max steps and LR around `1e-4`, with evaluation at 10k/25k/50k before extending. Applying that recommendation to job `2799` requires cancelling and resubmitting it, because Slurm copied the old launcher at submission time.

Follow-up LR recipe after job `2799`: use low-LR LoRA with faster camera/action heads instead of uniform `2e-4`. The first version used base LoRA LR `5e-5`, `NATIVEP1_ACTION_LR_MULT=4.0`, 100k max steps, and run name `native_phase1_camera_latent_lora5e5_action4x_100k`. This preserves visual LoRA weights more conservatively while letting the camera action heads adapt faster.

LR-split production submission:

```text
job id: 2800
name: nativep1
submitted on: 2026-07-10
run name: native_phase1_camera_latent_lora5e5_action4x_100k
recipe: 100k steps, LoRA base LR 5e-5, action/camera head LR multiplier 4.0, compile disabled
state at submission: PD (Resources)
later state: cancelled before start after the finite-dataloader livelock audit
```

Important audit result: finite map-style streams must be wrapped with
`CyclingDataLoader`. Without it, a long `IterativeJointDataLoader` run can silently spin
after one selected task stream exhausts, because `global_id` does not advance on an empty
packed batch. The current code includes the wrapper; any future refactor must preserve
this infinite-stream invariant.

Post-audit smoke and production submission:

```text
forced-exhaustion smoke:
  run: /weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/cosmos3_camera/camera_world/native_phase1_cycling_exhaust_smoke
  env: NYMERIA_MAX_SAMPLES=8
  result: 20 training iterations completed, loss finite, DCP checkpoint saved
  checkpoint: checkpoints/iter_000000020

official inference smoke:
  output: /weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/official_inference_cycling_exhaust_smoke
  samples: forward_dynamics, inverse_dynamics, policy
  result: all status=success; each MP4 is 97 frames, 256x256, 20 fps

production job:
  job id: 2801
  name: nativep1
  submitted on: 2026-07-10
  run name: native_phase1_camera_latent_lora5e5_action4x_100k
  recipe: 100k steps, CyclingDataLoader, LoRA base LR 5e-5, action/camera head LR multiplier 4.0, compile disabled
  state at submission: PD (Priority)
```

Correctness audit on 2026-07-11 found that job `2801` still used legacy prose prompts for action tasks and fixed 32-sample packing based on the 1x1 dummy video. Its learned prompt contract and effective batch composition therefore differ from official action inference and native token-budget packing. The token-budget correction used this run name:

```text
native_phase1_camera_json_tokpack_lora5e5_action4x_100k
```

Do not resume job `2801`'s checkpoint into the corrected code. Cancel it only when intentionally replacing it, then restart the corrected run from the base Cosmos checkpoint under the new name. Keeping the new name prevents the DCP auto-resume path from loading the incompatible old training state.

Job `2838` subsequently started the corrected JSON/token-budget recipe on 2026-07-12. It was intentionally cancelled at step 4,399 before its first 5k checkpoint, so it has no checkpoint to resume. Fixed-four replacement job `2852` ran under `native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k` and completed all 100k steps on 2026-07-14.

The final 100k EMA checkpoint has a complete official-sampler evaluation under `eval_full71_inverse_forward/iter_000100000`: 71/71 inverse and 71/71 forward samples, 71 inverse camera plots, and 71 forward comparison MP4s. Inverse means are rotation `0.213036 deg`, translation-direction cosine `0.838294`, scale ratio `1.007017`, translation error `0.003177 m`, and Sim(3) ATE `0.023574 m`. Forward means, excluding conditioned frame 0, are PSNR `19.4988 dB`, SSIM `0.612583`, and LPIPS-Alex `0.285336`. This is the best fully evaluated checkpoint from 5k through 100k on the main rotation/direction/translation/ATE and PSNR/SSIM/LPIPS metrics; use the final 100k EMA DCP for native Phase-3 initialization.

## Troubleshooting

If the job fails before torchrun with a preflight message:

- some process is already using memory on the allocated node;
- do not kill another user's process without permission;
- inspect with:

```bash
squeue -w <node> -o '%.18i %.9P %.24j %.12u %.2t %.10M %.4D %R'
ssh <node> 'nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits'
```

The `squeue` command is mandatory before every SSH compute action. Never launch through SSH on a node allocated to another user, even when `nvidia-smi` shows apparently idle GPUs. Slurm exclusivity does not prevent outside SSH processes from interfering with a training job.

If inference fails to import config:

- use `--config-file native_phase_training/inference_config.py`;
- run from `/home/jungbin_cho/cosmos-framework`;
- ensure `PYTHONPATH` includes `/home/jungbin_cho/cosmos_motion_ft` and `/home/jungbin_cho/cosmos-framework`.

If VAE construction fails with missing `pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth`:

- set `WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth`;
- this env var is required for both training and official inference.

If tokenizer loading tries to go online:

- set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
- verify the local Cosmos Nano tokenizer snapshot exists;
- or set `COSMOS_TEXT_TOKENIZER_PATH` to a local tokenizer directory containing `tokenizer.json`.

If training OOMs on clean GPUs:

- first verify production still has `model.config.compile.enabled=false`;
- check whether activation checkpointing is still `full`;
- check per-rank memory in the Slurm log;
- consider sharding instead of 8-way replication only after confirming official-compatible LoRA behavior with clean GPUs.

## Agent Rules for This Directory

Before changing this path, read:

1. `/home/jungbin_cho/cosmos_motion_ft/AGENTS_ALL.md`
2. `native_phase_training/README.md`
3. `native_phase_training/latent_omni_model.py`
4. `native_phase_training/latent_nymeria_dataset.py`
5. `native_phase_training/experiment.py`
6. `native_phase_training/sbatch_phase1_native_camera.sh`
7. `native_phase_training/checkpoint_eval_callback.py`
8. `native_phase_training/sbatch_checkpoint_eval.sh`

After changes, run the smallest relevant checks:

- `PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework /home/jungbin_cho/miniforge3/envs/cosmos/bin/python -m unittest native_phase_training.test_contracts`
- `python -m py_compile native_phase_training/*.py`
- `bash -n native_phase_training/sbatch_phase1_native_camera.sh`
- `bash -n native_phase_training/sbatch_checkpoint_eval.sh`
- TOML dryrun with `--dryrun`
- one-step GPU train smoke on a node
- official inference smoke from the saved checkpoint, covering forward, inverse, policy, and image-to-video

Do not declare this path healthy after only a config import. The required bar is train, save, load, sample, output file validation, and at least a small action-shape check.
