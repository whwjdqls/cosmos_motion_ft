# Phase-1 Visual-Quality Audit

Last updated: 2026-07-23

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

Recent ablation A:

```text
/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/
  native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person
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
metrics. Resolution and model provenance therefore remain unresolved
presentation confounds.

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
plausible than fp16 quantization, but it is still a hypothesis until the
controlled experiments at the end of this document are run.

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

## Camera Accuracy Is Better in the Recent Run

Historical step-7000 full-71 inverse metrics from the `hung_iter6000` tree:

```text
rotation error:              1.1688 deg
translation direction cos:  0.7656
scale ratio:                 1.4339
normalized translation err: 0.005913
ATE:                         0.06428 m
```

Recent A step-100k full-71 inverse metrics:

```text
rotation error:              0.2716 deg
translation direction cos:  0.8243
scale ratio:                 1.0114
normalized translation err: 0.003832
ATE:                         0.02413 m
```

The older model's apparent visual advantage therefore does not mean it learned
better camera control. The recent model is substantially better in metric
camera space.

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

### 1. Finish the matched sampling matrix

Use identical held-out sources, seeds, prompts, prefix, EMA choice, and solver:

```text
old checkpoint at 256x256, shift 3
recent A at 256x256, shift 3
old checkpoint at source 640x640, shift 10
recent A at source 640x640, shift 10
```

The first pair tests model/training quality. The second pair tests whether the
new model can recover the presentation quality seen in old videos. Record the
actual model path loaded in every output directory.

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

1. **High confidence confound:** old 640 presentation and ambiguous merged-model
   provenance versus recent 256 official evaluation.
2. **Most likely training cause if a matched comparison still differs:** global
   LoRA optimization changed from large mixed-objective updates to small
   homogeneous-task updates, with pure I2V steps and much weaker camera loss on
   the shared LoRA.
3. **Possible contributor:** longer exposure to low-resolution Nymeria video and
   subjective realism diverging from reconstruction metrics.
4. **Lower probability:** prompt/tokenizer and minor filter differences.
5. **Very low probability:** fp16 latent storage precision.
6. **Disfavored by direct test:** shift 3 versus shift 10 at 256.

Do not revert to shift 10 as the training default. Shift 3 is the correct
Cosmos-3 Nano 256-tier contract. If higher visual resolution is required, train
and evaluate a real higher-resolution bucket with its corresponding native
schedule rather than labeling a 256 training run as 720.
