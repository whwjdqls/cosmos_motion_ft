# EgoControl adapted to Cosmos3-Edge

This is an adapted-method baseline, not an exact reproduction of the paper's
Cosmos-Predict2 checkpoint or a parameter-matched Phase3 bridge ablation.
Reference: https://github.com/CVG-Bonn/EgoControl at
`5dcfb82ef898315d415f8f04c94337c83d9a099d`.

Yonsei launch, acceptance evidence, W&B and the retained allocation are recorded
in [STATUS_20261026.md](STATUS_20261026.md). Live operation logs supersede that
timestamped snapshot.

## Comparison contract

- Task: first RGB frame plus full body motion -> remaining video; no caption,
  camera-action tokens, future RGB, motion denoising, or auxiliary motion loss.
- Exact Phase3 immutable training cache, all112503 accepted windows, including
  its floor filtering and recording split. No additional actor exclusions.
- T97,20fps,256x256; existing Wan latents48x25x16x16. Frame0 is observed.
- Initialization: the same frozen-source Phase1 EMA200k generator that our
  original Phase3 used, not the prefix17 model being replaced. Optimizer and
  scheduler start fresh; this baseline trains the inherited LoRA and new pose
  modules. The reasoner and pretrained base weights remain frozen.
- All new outputs on Scratch2 on Yonsei. Site/path overrides support Grasp.
- Evaluation must use the same fixed71 or all-window cohort for both methods,
  identical first-frame/motion inputs, seeds, native35-step UniPC/shift10,
  guidance1, resolution and20fps. Exclude observed RGB frame0 from metrics.
  Compare caption-free M2V from ours; do not silently compare against caption
  conditioned or oracle-head results. Report GT references as well.
- Report trainable parameters, update count, motion-to-video example exposures,
  wall time, and upstream pretraining. Ours also had Phase2 motion pretraining;
  this baseline does not. A single-task run sees more M2V examples per update
  than a multitask Phase3 run. These differences must be disclosed.
- Original Phase3 M2V training included captions with 10% caption dropout;
  this pose-only EgoControl adaptation does not. The primary evaluation must
  omit semantic captions for both, and label any caption-assisted comparison
  separately. This does not make their historical training information equal.
- Report both equal-update and approximately equal-M2V-exposure comparisons:
  at global32, this baseline sees 6.4M M2V examples over200k updates; original
  50/50 Phase3 sees approximately3.2M. Retain the baseline100k snapshot for
  the exposure-matched comparison against original Phase3 at200k. Match the
  actual task proportions separately for the other Phase3 variants.
- The method retains EgoControl's23-body-joint pose input; ours uses its
  trained30-joint UniEgo representation. Both derive from the same SOMA
  windows, but the information representation and parameter budget differ.
  No selection or tuning will use test scores; compare fixed100k/200k points.

## Pose/control architecture

Decode the existing camera/head-aligned Uniego283 with its own saved mean/std;
do not use evaluator normalization. Preserve the SOMA body23 subset used by
our other baselines. Convert97 motion frames into96 future controls, each138D:
head local SE(3) delta, pelvis local SE(3) delta, and21 pelvis-relative body
poses. Each pose is translation plus XYZ Euler components with Rz*Ry*Rx,
in radians, matching EgoControl. No hands, raw XSens download, GT fitting,
prediction-dependent grounding, or new body-shape input is needed.
SOMA's left/right storage order is explicitly mapped to EgoControl's
right-before-left slots. The SOMA spine/neck chain is retained; its landmarks
are an anatomical adaptation, not identical XSens joint locations.

Both EgoControl conditioning paths are present:

- Per-frame138->1024/GELU/LayerNorm tokens with fixed sinusoidal positions and
  the shared1024 projection. Exactly seven Edge blocks get separate pose
  cross-attention:16 heads x128, headwise Q/K RMSNorm, no V norm and no extra
  RoPE on the pose CA. Native Edge self-attention/RoPE stays unchanged.
  Zero-based indices3,7,11,15,19,23,27 match Phase3's seven bridge sites;
  the other21 blocks keep the native class and have no pose modules.
- Global flattened96x138 control, learnable scales initialized20 for head/root
  translations and10 for their angles, plus the two4x-width MLPs producing D
  and3D. Per-block low-rank256 modulation supplies shift,scale,gate to the
  generation-stream self-attention, pose cross-attention and FFN at those same
  seven sites. There is no separate pose-conditioned final-output modulation.

The earlier28-block experiment is preserved as history. Its architecture is
available explicitly as `EGOCONTROL_CONDITIONING_LAYOUT=legacy_all28`; use its
original run name for strict resume. The default `phase3_seven` starts a new
run from Phase1 EMA200k, never from a28-block trained checkpoint. Seven sites
matches conditioning depth, not representation or total parameter count.

Backbone adaptation is explicit: Edge has pretrained RMSNorm/SwiGLU and no
Predict2 AdaLN gates. Its existing residual gates are retained as1+pose_gate;
the new cross-attention gate starts at0. Global3D output and added per-block
modulation outputs initialize tozero, preserving the initial Phase1 generator.
This initialization is an Edge adaptation, not a claim of identical original
EgoControl initialization. Understanding/text tokens never receive poses.

## Phase1 hyperparameters, not Phase3 hyperparameters

Global32=2 GPUs x16 with accumulation1;200k updates; LoRA16/alpha32 Q/K/V/O;
FusedAdam, betas0.9/0.99, epsilon1e-8, weight decay0.05, grad clip1. Base LR5e-5
times native LambdaLinear0.4 gives initial effective LoRA2e-5. New pose modules
use the same4x multiplier as Phase1's new camera heads, effective8e-5. The
500-step constant initial segment and linear decay tozero at200k are unchanged.
Native Edge waver noising, training shift3, uniform RF velocity-MSE weight10,
full-element denominator including masked frame0, native PowerEMA rate0.1,
BF16 compute/FP32 master weights, activation checkpointing, seed42,5k saves
and rolling15-minute saves are preserved. No camera loss exists for M2V-only.

```bash
COSMOS_SITE=yonsei bash egocontrol_edge/run.sh train
```

Strict resume requires `EGOCONTROL_RESUME` inside that run's checkpoint tree.
The saved native config is the resolved authority. Preparation/check logs
establish readiness; this README alone does not assert successful training.
Do not cancel the retained allocation when replacing the prefix17 process.

## W&B and generated validation samples

The default logs to `jungbinc-upenn/cosmos3_camera` with the saved CLI login.
Four fixed cases (indices0,23,46,70) from the existing held-out test71 manifest
are sampled after update1 and every5000 updates, including200k. They are
monitoring examples, not a separate validation split or model-selection set.
Recording IDs are checked against the training cache and normalization hashes
against each bundle. Grasp can set `EGOCONTROL_VALIDATION_CONFIG` to its local
copy of the same case configuration; no Yonsei path is embedded in Python.

Use the official native EMA UniPC sampler:35 steps, video shift10, CFG1,
the manifest's fixed seeds, no semantic caption, observed frame0 only. Every
rank participates in denoising; rank0 decodes and logs GT/generated MP4 pairs.
Observed/GT frames have green borders and generated frames red borders.
First-frame latent preservation is checked within1e-4 absolute FP32 solver
roundoff; the measured error is recorded without modifying sampled outputs.
Validation saves under each run's
`validation/iter_*` directory, with per-case and per-frame metrics, generated
latents and comparison videos. W&B has `validation_preview/*` videos and
`validation/{model,VAE_GT,GT}/{psnr_db,ssim,lpips_alex}` curves. Metrics use
uncompressed decoded RGB frames1..96; GT identity PSNR is capped at100dB and
the VAE-reconstructed GT reference shows the compression ceiling.

Training batches are not consumed for validation; Python/NumPy/Torch RNG and
module training modes are restored. The protocol is recorded in
`validation/configuration.json` and checked on resume.
The official `ema_scope(is_cpu=True)` swaps/restores weights while retaining
the training network's FSDP precision policy. No extra AMP wrapper is used;
direct inference on FP32 EMA storage would give inconsistent attention dtypes.
