# Phase-1 Visual-Quality Audit

Last updated: 2026-07-24

This document records the investigation into why historical Nymeria camera
checkpoints can look visually better than the newer native-compatible Phase-1
video-quality runs even though the newer runs have much better metric camera
control.

It is deliberately explicit about what is measured, what remains confounded,
and which claims are hypotheses. The server paths are provenance records. The
source code and configuration contracts needed to recreate the experiments are
in this repository and in the vendored material described by `PROVENANCE.md`.

## Runs Being Compared

Historical family:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/
  world_camera_nymeria_97f_cont
```

Related historical outputs used during this audit:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/
  world_camera_nymeria_97f_hung_iter6000/checkpoints/iter_000007000
```

Current native baseline and video-quality ablations:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/
  native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k
  native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person
  native_phase1_vq_B_varprefix_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person
  native_phase1_vq_C_varprefix_action_only_aw2_bs4_lr5e5_ema100k_qfilterv1_person
  native_phase1_vq_D_varprefix_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person
```

The historical `cont` run does not contain its own stored inference videos.
Some visual evidence discussed as "old checkpoint quality" is stored under the
related `hung_iter6000` tree. Do not treat directory placement as proof of model
provenance.

## Executive Conclusion

The current evidence does not support fp16 latent caching as the main cause.
It also does not support sampler shift as the main cause.

The first issue is that the old and new visualizations were not an
apples-to-apples comparison:

1. the inspected old videos are 640x640, while the recent A videos are 256x256;
2. the old videos used UniPC shift 10, while A normally uses the correct
   256-tier shift 3;
3. at least one old evaluation loaded a manually merged model from
   `/weka/jungbin/tmp_merge/m97f_7000/model`, not the checkpoint directory under
   which its outputs were later stored.

Changing A from shift 3 to shift 10 at 256x256 did not materially improve its
metrics. A same-checkpoint comparison on 2026-07-24 then established a more
specific result: the old step-7000 weights flicker under the current 256/shift-3
contract but are smooth under their historical 720-tier/shift-10 contract.

The completed 2x2 matrix isolates the dominant factor. Five historical
`fdpolicy5` records were regenerated with the exact same merged regular model,
prompt, condition image, camera actions, seed, guidance, UniPC solver, and 30
steps. Only resolution tier and shift changed:

| inference cell | adjacent RGB MAD | temporal second difference | flow-compensated RGB MAD |
|---|---:|---:|---:|
| 256 tier, shift 3 | 0.04057 | 0.06405 | 0.01530 |
| 256 tier, shift 10 | 0.04030 | 0.06382 | 0.01502 |
| 720 tier, shift 3 | 0.03356 | 0.05228 | 0.01311 |
| 720 tier, shift 10 | 0.03312 | 0.05160 | 0.01252 |

At fixed shift 3, moving from the 256 tier to the 720 tier lowers the three
temporal-change diagnostics by `17.7%`, `18.8%`, and `14.0%`. At fixed 256,
changing shift 3 to 10 changes them by only `-0.7%`, `-0.4%`, and `-1.9%`.
The labeled comparison videos agree with the user's visual observation:
resolution tier is the dominant factor for this checkpoint; shift is secondary.
These are diagnostic temporal-change statistics, not a replacement for a
human flicker study or GT video-quality metrics. RGB MAD and temporal second
difference are measured after resizing every output to 256x256; the
flow-compensated residual uses 128x128 analysis frames.

The old run's saved dataset config also says `resolution: '256'`; only the model
default remains `resolution: '720'`. Therefore the smoother 720-tier result is
not evidence that this LoRA was trained on high-resolution Nymeria. It instead
shows that the same weights can use the pretrained model's larger spatial-token
inference path much more coherently. More latent spatial tokens, different
positional coordinates, and higher-resolution VAE decoding all change the
generation computation; this is not equivalent to resizing a completed
256x256 MP4.

If a matched-resolution, matched-sampler comparison still shows that the old
model is more realistic, the most likely training cause is the optimization
contract around the shared global generator LoRA:

- old training combined forward, inverse, and policy gradients on every
  distributed update;
- new training uses one homogeneous task on all ranks for each update;
- old training used a global batch of up to 128 clips, versus 32 clips now;
- recent A adds pure I2V updates that have no camera tokens and directly train
  the global visual Q/K/V/O LoRA;
- recent A reduced action loss weight from 10 to 2, greatly reducing the camera
  gradient reaching the shared LoRA, even though the camera heads themselves
  retain a larger LR.

This combination can let the LoRA fit low-resolution Nymeria reconstruction
more directly while perturbing the pretrained visual prior. It is more
plausible than fp16 quantization as a training-side contributor. The
same-checkpoint result shows that inference spatial tier is a separate,
high-priority contributor. A/B/Original must be compared at the same high tier
before assigning their observed 256-tier flicker entirely to training.

## Canonical Results Through 2026-07-24

The canonical benchmark uses one prefix-1 forward-dynamics record and one
inverse-dynamics record for each of the 71 held-out sequences. All rows below
use 97 RGB frames, 256 resolution, seed 0, guidance 1, and NVIDIA UniPC with 30
steps and shift 3. PSNR/SSIM and translation-direction cosine are higher-better;
LPIPS, DreamSim, VideoMAE-v2 CD-FVD, rotation, translation error, and ATE are
lower-better. Scale ratio is best at 1.

| Model | Step | PSNR | SSIM | LPIPS | DreamSim | CD-FVD | Rot. deg | Dir. cos | Scale | Trans. mm | ATE cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, action weight 10 | 100k | 19.4988 | 0.612583 | 0.285336 | 0.170482 | **281.614** | **0.213036** | **0.838294** | 1.007017 | **3.177** | **2.357** |
| A, prefix 1 global LoRA | 100k | **19.5343** | **0.614062** | **0.284598** | **0.169134** | 295.561 | 0.271635 | 0.824262 | 1.011397 | 3.832 | 2.413 |
| B, variable-prefix global LoRA | 100k | 18.6945 | 0.586307 | 0.315056 | 0.177504 | 304.385 | 0.287409 | 0.811321 | 1.048164 | 4.041 | 2.522 |
| D, camera-token K/V LoRA | 100k | 18.3851 | 0.576988 | 0.328101 | 0.189021 | 319.056 | 0.314777 | 0.811414 | 1.078215 | 4.387 | 3.030 |
| Historical step 7k, current contract | 7k | 18.4828 | 0.574474 | 0.322919 | 0.175386 | 282.414 | 0.267831 | 0.770949 | 0.922935 | 4.681 | 3.995 |

CD-FVD in this table is the canonical FP32 VideoMAE-v2-SSv2 feature path.
The old 7k checkpoint is a historical trainable-delta DCP that cannot be loaded
as a current full-model DCP. It was reconstructed with
`nymeria_world/export_merge_lora.py`, exactly as in its historical inference
path, and evaluated from the merged regular weights. The Original/A/B/D rows
use EMA weights. Therefore the old row is useful but not a strict EMA-matched
comparison. C stopped at step 65k and has no full-71 evaluation.

The smaller `checkpoint_evals` suite uses five held-out sources at every exact
causal prefix `[1,9,17,33,49]`. The video columns below are unweighted macro
means over the five prefix-level means; inverse camera metrics use the five
source windows once. These values are diagnostics, not substitutes for the
full-71 table.

| Model | Step | Forward P/S/L | Policy P/S/L | I2V P/S/L | Inverse rot / trans / ATE |
| --- | ---: | --- | --- | --- | --- |
| Original | 100k | 16.955 / 0.497 / 0.368 | 14.130 / 0.396 / 0.538 | 13.436 / 0.362 / 0.570 | 0.236 deg / 4.37 mm / 3.14 cm |
| A | 100k | 17.203 / 0.505 / 0.359 | 14.252 / 0.399 / 0.538 | 13.442 / 0.364 / 0.568 | 0.314 deg / 5.05 mm / 3.60 cm |
| B | 100k | **17.507 / 0.519 / 0.356** | 14.258 / 0.407 / 0.530 | 13.398 / 0.364 / 0.567 | 0.333 deg / 5.52 mm / 3.70 cm |
| C | 65k | 15.203 / 0.430 / 0.447 | 13.530 / 0.368 / 0.535 | 12.952 / 0.355 / 0.577 | 0.601 deg / 12.99 mm / 9.63 cm |
| D | 100k | 16.756 / 0.489 / 0.379 | 14.117 / 0.393 / 0.530 | 12.945 / 0.356 / 0.576 | 0.397 deg / 6.54 mm / 3.93 cm |

Current interpretation:

- Original remains the strongest balanced Phase-1 model. It has the best
  full-71 camera metrics and best full-suffix CD-FVD.
- A is effectively tied with Original on aligned frame metrics and DreamSim,
  but its camera metrics and CD-FVD are worse.
- B benefits from long clean prefixes in the compact suite, especially at
  prefixes 33 and 49, but loses quality on the prefix-1 full-71 benchmark.
- C confirms that the action interface alone is not an adequate video
  adaptation path. D's camera-only K/V LoRA also underperforms global LoRA.
- The historical 7k row has CD-FVD close to Original and better than A/B/D,
  consistent with some stronger distributional/temporal appearance, but its
  pixel metrics and camera translation/ATE are worse. Its current-contract
  video still visibly flickers, so this score does not establish that the old
  weights alone solve temporal consistency.

Canonical old-7k outputs and reports:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/
world_camera_nymeria_97f_hung_iter6000/checkpoints/iter_000007000/
eval_full71_current_shift3_single_gpu_shards/
```

## Exact Configuration Differences

| Property | Historical `world_camera_nymeria_97f_*` | Recent A |
| --- | --- | --- |
| Model class | stock `OmniMoTModel` | `LatentOmniMoTModel` |
| Video input during training | decoded RGB, online Wan VAE | cached Wan latents |
| Cached dtype | not applicable | fp16 on disk, restored to float32 |
| Tasks | forward / inverse / policy | forward / inverse / policy / I2V |
| Configured ratios | 40 / 25 / 20 | 40 / 25 / 20 / 15 |
| Distributed task selection | rank partitioned | one task on all ranks per step |
| Clips per GPU | up to 16 | exactly 4 |
| Approximate global batch | up to 128 | 32 |
| Generator adaptation | global Q/K/V/O LoRA | global Q/K/V/O LoRA |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| LoRA LR | `2e-4` | `5e-5` |
| Action-module LR | `2e-4` | `2e-4` through 4x multiplier |
| Action loss weight | 10 | 2 |
| Active-token normalization | false | true |
| Training model resolution | incorrectly left at 720 | 256 |
| Training shift selected | 10 | 3 |
| Quality filter | none | qfilterv1, removes about 1.3% |
| Caption subject | standalone `C` retained | standalone `C` becomes `A person` |
| EMA | PowerEMA | PowerEMA |
| Recent evaluation | not uniform/provenance-confounded | official EMA + UniPC |

The historical `cont/config.yaml` records:

```text
checkpoint.load_path=/weka/jungbin/tmp_merge/m97f_7000
checkpoint.load_training_state=false
max_samples_per_batch=16
model.config.resolution=720
action_loss_weight=10
normalize_loss_by_active=false
optimizer.lr=2e-4
```

The continuation reset optimizer/trainer state while initializing model weights
from a manually merged step-7000 model. Its local `iter_000007000` therefore
represents approximately another 7,000 updates after that merge, not a clean
single 7,000-step run from the base checkpoint.

Recent A records:

```text
max_samples_per_batch=4
model.config.resolution=256
action_loss_weight=2
normalize_loss_by_active=true
optimizer.lr=5e-5
action LR multiplier=4
```

## Resolution Means Pixel Bucket, Not Latent Width

The 256 setting is the pre-VAE image/video bucket. With Wan's 16x spatial
compression, a square 256 video becomes a 16x16 latent grid:

```text
RGB video:  [3, 97, 256, 256]
Wan latent: [48, 25, 16, 16]
```

The inspected historical generated MP4 is 640x640 because its input media is
640x640 and the old inference configuration requested the 720 tier without
upscaling the source before the aspect-preserving crop. The inspected A output
is 256x256. A 640 video will naturally look sharper and retain more recognizable
texture than a 256 video even if the underlying generation behavior is not
better.

This audit verified with `ffprobe`:

```text
old stored output: 640x640, 97 frames, 20 FPS
recent A output:    256x256, 97 frames, 20 FPS
```

## Shift-10 Test on Recent A

The A step-100k EMA checkpoint was sampled directly on node 0 with:

```text
five fixed held-out sources
prefix 1
256x256
UniPC
shift 10
30 steps for action modes
35 steps for I2V
the same prompts, seeds, and input records as the shift-3 diagnostic
```

Outputs:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/
native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/
checkpoint_evals_shift10/iter_000100000_prefix1
```

Video metrics over each complete generated suffix:

| Task | Shift | PSNR up | SSIM up | LPIPS down |
| --- | ---: | ---: | ---: | ---: |
| Forward | 3 | 16.5085 | 0.4837 | 0.4026 |
| Forward | 10 | 16.4949 | 0.4852 | 0.4058 |
| Policy | 3 | 13.9648 | 0.4013 | 0.5452 |
| Policy | 10 | 13.8631 | 0.4020 | 0.5487 |
| I2V | 3 | 13.3397 | 0.3668 | 0.5827 |
| I2V | 10 | 13.2341 | 0.3608 | 0.5765 |

These are effectively tied and show no consistent shift-10 gain. This test
isolates shift at 256. It does not test the old 640 presentation.

## Camera Accuracy Under Comparable Evaluation

An older stored step-7000 inverse report used the historical 480/720-tier,
shift-10 input contract and reported:

```text
rotation error:              1.1688 deg
translation direction cos:  0.7656
scale ratio:                 1.4339
normalized translation err: 0.005913
ATE:                         0.06428 m
```

Do not compare that report directly with current Phase-1 evaluations. The new
current-contract reconstruction of the same historical delta reports:

```text
rotation error:              0.2678 deg
translation direction cos:  0.7709
scale ratio:                 0.9229
normalized translation err: 0.004681
ATE:                         0.03995 m
```

The old model is much better than its incompatible historical report suggested,
but Original 100k remains clearly stronger on all five camera metrics. A 100k
has similar rotation to the old model and materially better direction,
translation, scale, and ATE. The old model's apparent visual advantage therefore
still does not establish better metric camera control.

## Why fp16 Cache Is Unlikely to Be the Cause

The cache writer follows the same spatial path as online training:

```text
decode RGB
resize and reflection-pad
normalize to [-1, 1]
Wan2.2 VAE encode
crop reflected padding in latent space
store latent
```

`LatentOmniMoTModel` restores cached tensors to float32 before native packing.
More importantly, the Wan VAE itself defaults to bf16 and runs encode under
bf16 autocast. It converts its output back to the float32 input dtype, but that
does not create additional precision. In the normal latent range, every bf16
value is exactly representable in fp16 because fp16 has more mantissa bits.

The reproducible CPU audit in `audit_cached_latent_precision.py` sampled 32
cache files:

```text
values checked: 9,830,400
finite: true
range: [-7.28125, 5.78125]
mean absolute value: 0.82684
all values exact after float32 -> bf16 -> float32: true
```

For an arbitrary pre-quantization float32 value, the mean half-ULP upper bound
at the cached values was `2.92e-4`, only `0.0353%` of mean latent magnitude.
The observed exact bf16 round trip is stronger evidence: these cache values are
already on the VAE's bf16 numerical lattice.

This does not prove that every offline/online path is bit-identical. A full
fresh-encode comparison remains useful for guarding preprocessing mistakes.
It does show that fp16 storage precision itself is not a credible explanation
for a large visible quality collapse.

Run the portable precision audit with:

```bash
python native_phase_training/audit_cached_latent_precision.py \
  --latent-root /path/to/joint_latents_T97 \
  --max-files 32
```

## Why Batch Size Alone Is an Incomplete Explanation

The old global batch was up to four times larger, which lowers gradient noise.
It also mixed all three tasks in every optimizer update:

```text
ranks 0-3: forward
ranks 4-5: inverse
ranks 6-7: policy
```

The recent iterative loader deterministically selects one task from
`seed + global_id`, and all ranks select the same task. A recent optimizer
update is therefore entirely forward, inverse, policy, or I2V.

This difference is more important than the number 128 versus 32 by itself.
The old update always averaged camera and video objectives. Recent A can take a
pure I2V visual update followed by a pure inverse-camera update, producing
higher variance and task-to-task oscillation in the same global LoRA.

The LR was scaled linearly with batch:

```text
old: 2e-4 / 128 = 1.5625e-6 per global-batch element
new: 5e-5 / 32  = 1.5625e-6 per global-batch element
```

The scheduler and run lengths also make the integrated LoRA LR roughly similar
for an effective old 14k-update continuation and a new 100k run. Therefore
"the new LR is too high" and "the new batch is too small" are not sufficient
standalone diagnoses. The lower-variance, mixed-objective update is the
stronger batch-related hypothesis.

## Why Action Weight 2 May Be Counterproductive for Global LoRA

The historical model looked good while using action weight 10. That directly
contradicts the simple claim that action weight 10 necessarily destroys visual
quality.

The action loss trains more than the action output head. Through joint
self-attention it also trains the global generator LoRA. Reducing the weight
from 10 to 2 while adding I2V makes the shared LoRA much more dominated by video
reconstruction gradients. The 4x action-head LR helps `action2llm` and
`llm2action`, but it does not restore the missing action gradient scale on the
LoRA tensors.

A plausible interpretation is:

```text
old global LoRA:
  strong camera objective + video objective on every update

recent A global LoRA:
  mostly Nymeria video reconstruction, including pure I2V updates
  much weaker camera regularization of the shared attention projections
```

This could explain why recent inverse metrics are excellent through the
specialized camera heads while the shared visual prior still looks less
natural. It is a hypothesis and must be tested with a controlled action-weight
and no-I2V ablation.

## Smaller Differences

- `normalize_loss_by_active=true` changes prefix-1 video scaling by only about
  `25/24`, or 4.17%, because one of 25 latent frames is clean. It is important
  for long variable prefixes but is not a likely explanation for A.
- qfilterv1 removes only about 1.3% of training rows. It is important for
  physical consistency but too small to explain a broad visual-quality change.
- Prompt formatting and tokenizer snapshots differ. They can affect semantic
  conditioning, but they are less likely than resolution/provenance and LoRA
  optimization to explain gross realism.
- The old and new checkpoint serialization scopes differ. Recent DCPs include
  much more frozen state, but optimizer logs confirm only LoRA and camera
  interface tensors require gradients. Checkpoint size is not evidence of full
  generator finetuning.
- PSNR/SSIM/LPIPS can improve while subjective realism worsens. A model can move
  toward an averaged or over-smoothed reconstruction and score better against
  one GT future while looking less like a natural sample.

## Required Controlled Experiments

Run these in order. Do not change multiple factors in one comparison.

### 1. Completed same-checkpoint sampling matrix

The exact five-sample 2x2 matrix is under:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/
world_camera_nymeria_97f_hung_iter6000/checkpoints/iter_000007000/
sampler_matrix_fdpolicy5_exact_20260724/
```

`temporal_diagnostics.json` contains aggregate and per-sample diagnostics.
`comparisons/t{0..4}_sampler_resolution_2x2.mp4` contains the labeled visual
comparisons. The model is
`/weka/jungbin/tmp_merge/m97f_7000/model`, reconstructed regular weights rather
than EMA. All four cells use the same five historical JSONL records and the
historical inference command with `torch.compile` enabled.

The result isolates spatial tier, not shift, as the dominant same-checkpoint
effect. The next controlled inference experiment is Original/A at both 256 and
720 tiers with their own EMA weights. Do not compare a merged regular old
checkpoint to a recent EMA checkpoint and call any difference a training
effect.

### 2. Test task/update composition

The highest-value training ablation keeps cached latents and all model settings
fixed, then compares:

```text
homogeneous-task global batch 32
homogeneous-task global batch 128
mixed-rank tasks with the same total global batch
```

This separates raw batch variance from mixed-objective averaging.

### 3. Test I2V and action weighting

Use prefix 1 and global LoRA:

```text
action weight 2, with I2V
action weight 2, without I2V
action weight 10, without I2V
action weight 10, with I2V
```

If resources allow only one follow-up, prioritize action weight 10 without I2V
against A. It most closely restores the old shared-LoRA gradient balance.

### 4. Test online VAE versus cached latents

Only after the higher-probability comparisons, run a short matched experiment:

```text
same samples and ordering
same task per step
same batch, LR, loss, and seed
online VAE float32 container versus cached fp16 container
```

Compare the clean latents before noising and the first several optimizer losses
and gradients. This is the decisive cache test. The precision audit predicts a
near tie.

### 5. Evaluate realism, not only GT reconstruction

Always retain:

```text
PSNR, SSIM, LPIPS
DreamSim
content-debiased FVD with VideoMAE-v2-SSv2
blind human side-by-side preference
camera inverse/forward metrics
```

Use `FORWARD_VIDEO_METRICS.md` for the advanced metric contract. Do not use the
deprecated TensorFlow-Hub I3D FVD.

## Current Best Judgment

Ranked by probability of explaining the observed visual difference:

1. **Confirmed inference spatial-tier effect:** for the same old weights and
   five exact inputs, the 720 tier is visibly and diagnostically smoother at
   both shifts. Shift 10 provides only a small change at fixed resolution.
2. **Most likely training cause after inference matching:** global
   LoRA optimization changed from large mixed-objective updates to small
   homogeneous-task updates, with pure I2V steps and much weaker camera loss on
   the shared LoRA.
3. **Possible contributor:** longer exposure to low-resolution Nymeria video and
   subjective realism diverging from reconstruction metrics.
4. **Lower probability:** prompt/tokenizer and minor filter differences.
5. **Very low probability:** fp16 latent storage precision.
6. **Disfavored by direct test:** shift 3 versus shift 10 at either fixed tier.

Do not revert to shift 10 as the training default. Shift 3 is the correct
Cosmos-3 Nano 256-tier contract. A 720-tier deployment should normally use its
native shift-10 schedule, but the controlled result shows that shift is not why
the old 720-tier videos are smooth. First evaluate recent EMA checkpoints at the
larger tier; only then decide whether higher-tier finetuning is needed.
