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

- `AUDIT.md`
  - Records the 2026-07-10 native Phase 1 audit, the finite-dataloader livelock fix, parity fixes, and documented deviations from pixel-native training.

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
  - Creates predicted-versus-GT camera trajectory/frustum plots for inverse dynamics and policy, plus a JSON manifest only after every requested sample in all four JSONL files has been visualized successfully.

- `checkpoint_eval_callback.py` and `sbatch_checkpoint_eval.sh`
  - After a successful DCP save, rank 0 submits one isolated one-GPU Slurm evaluation job.
  - The job loads EMA weights through official inference, samples all four modes with UniPC, then runs `visualize_checkpoint.py`.
  - Submission markers prevent duplicate jobs after trainer restart; a failed submission is logged without aborting training.

- `sbatch_phase1_native_camera.sh`
  - Production Slurm launcher for the official-compatible Phase 1 run.

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

It keeps train windows whose latent file exists and whose manifest window is usable. On the 2026-07-10 smoke, the T97 train index found 119,632 cached windows.

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

The 10% CFG text dropout runs after mode-specific formatting, so it drops the entire JSON/prose prompt to the tokenizer's empty-string conditioning. Inverse-dynamics text is already exactly empty.

## Packing Contract

The raw `video` tensor is only `[3,97,1,1]` metadata, so the stock joint loader would count zero spatial vision patches. `LatentAwareIterativeJointDataLoader` starts with the parent count for text, EOS, vision boundary markers, action, and sound, then adds the real cached-latent patches:

```text
latent [48,25,16,16], patch 2x2 -> 25 * 8 * 8 = 1600 vision patch tokens
```

Production uses `NATIVEP1_CLIPS_PER_GPU=4`, so each rank packs exactly four samples and the eight-GPU global batch is 32 clips. Batch size is not part of the rectified-flow or official-sampler contract. The framework requires sample-count and token-count limits to be mutually exclusive, so fixed-four mode sets `max_samples_per_batch=4` and `max_sequence_length=None`. This remains below the native ceiling even at the tokenizer's 4,096-token truncation limit: four worst-case T97 action samples are below 24k tokens, versus the 45,056-token model budget. A real resolved batch on 2026-07-12 contained four samples and 7,323 tokens.

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

The production launcher instead sets `NATIVEP1_AUTO_EVAL=1`. Before training, it prepares five held-out inputs under `NATIVEP1_EVAL_INPUT_DIR`. After each completed DCP save, `NativeCheckpointEvalSubmitter` queues `sbatch_checkpoint_eval.sh`; outputs land under:

```text
${RUN_DIR}/checkpoint_evals/iter_XXXXXXXXX/{<mode-specific sample dirs>,viz/}
```

Set `NATIVEP1_AUTO_EVAL=0` for smoke tests or when checkpoint visualization jobs should not be submitted. The official inference/visualization commands above remain the manual recovery path if a submitted evaluation job fails.

The inference command explicitly uses NVIDIA's `cosmos_framework.scripts.inference`, `--sampler unipc`, and EMA weights (also the official default). Action modes use 30 steps, guidance 1, and 256-tier shift 3. Image-to-video uses 35 steps, guidance 6, and shift 3. These values reach `OmniMoTModel.generate_samples_from_batch`, whose log must report `Using sampler: UniPC (shift=3.0, num_steps=...)`.

The framework's bundled modality JSON files contain a literal shift of 10 because the release defaults target the high-resolution tier. This run intentionally overrides that one value to 3, matching both Nano's `{256:3, 480:5, 720:10}` map and this run's 256-resolution training distribution. The solver, sigma construction, EMA loading, CFG implementation, and task step/guidance defaults remain NVIDIA's official path; using the unmodified shift 10 would be a train/evaluation mismatch here.

## Smoke Tests Completed

All smoke artifacts are generated outputs under `/weka`; do not edit them as source files.

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
ssh <node> 'nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits'
```

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

- `python -m unittest native_phase_training.test_contracts`
- `python -m py_compile native_phase_training/*.py`
- `bash -n native_phase_training/sbatch_phase1_native_camera.sh`
- `bash -n native_phase_training/sbatch_checkpoint_eval.sh`
- TOML dryrun with `--dryrun`
- one-step GPU train smoke on a node
- official inference smoke from the saved checkpoint, covering forward, inverse, policy, and image-to-video

Do not declare this path healthy after only a config import. The required bar is train, save, load, sample, output file validation, and at least a small action-shape check.
