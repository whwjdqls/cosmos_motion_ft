# Cosmos3-Edge Egocentric Inference Audit

Date: 2026-09-01

## Outcome

For this held-out Nymeria clip, replacing `C` with wording such as `the camera
wearer` is not enough to make generic image-to-video (I2V) interpret the
caption as motion of an unseen egocentric camera. A structured prompt that
separates the off-camera wearer from the person visible in the conditioning
frame produces much more viewpoint motion, but it still does not reproduce the
ground-truth camera path and hallucinates foreground hands and table objects.

When the ground-truth 96 x 9 camera-action sequence is supplied through Edge
`forward_dynamics`, the model produces trajectory-scale viewpoint motion. It
also overshoots the measured temporal motion and introduces stronger
geometry/identity distortion. Changing only the forward caption from `The
person` to `The camera wearer` has little effect once the numerical actions are
present. The action sequence, not that noun phrase, is the dominant camera
control.

Therefore:

1. Use `forward_dynamics` with a first frame and 96 x 9 camera actions when the
   camera path is known or must be controlled.
2. Use an audited, frozen egocentric structured prompt only as a qualitative
   I2V fallback when no camera actions are available.
3. Keep native-framework I2V shift 10 / 35 steps / guidance 6 as the canonical
   Phase-1 evaluation recipe. The refreshed shift 12 / 20-step recipe was
   effectively tied on this controlled sample and remains a secondary
   model-card diagnostic.
4. Do not enable online prompt upsampling without inspecting or validating its
   output. The reasoner hallucinated subject identities in this test.

This is a one-clip diagnosis, not a population-level model comparison.

## Question and fixed sample contract

The test asks whether Edge should be inferred differently for egocentric
Nymeria captions in which the grammatical subject is the unseen camera wearer,
not necessarily the person visible in the frame.

All controlled variants used:

- untouched downloaded checkpoint: `/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge`;
- clean native Cosmos Framework commit `d4599e2e43fbd06168e9884205b9b66c3902d8f6`;
- sample `t00_S07_20231013_s0_shelley_jones_act3_w23p1b`;
- the same first-frame PNG, seed 0, 256 x 256 output, T97, and 20 FPS;
- one L40S, one process, no diffusion cache, no torch compile, no CUDA graphs,
  and no guardrails;
- raw base weights with no LoRA or DCP overlay.

The inference request independently carries `num_frames=97`, `fps=20`, and
`resolution=256`. The frozen structured prompt also contains `fps: 20`, a
`256 x 256` resolution object, and `duration: 4s`. The whole-second duration
matches the structured training-side formatting behavior for T97/20, while
the actual output contract remains 97 frames at 20 FPS. FPS is also supplied
numerically to Edge's FPS-conditioned vision path; caption metadata does not
encode a camera trajectory.

## Prompt paths tested

### Plain I2V caption

The original standalone `C` placeholders were replaced sentence-sensitively
with `The person` or `the person`. This is the current generic I2V baseline.
It remained almost static.

### Literal camera-wearer caption

The same plain caption instead used `The person wearing the camera` or `the
person wearing the camera`. It remained almost as static as the person-caption
baseline, so literal noun replacement alone does not resolve the viewpoint
semantics.

### Explicit egocentric prose

The caption was prefixed with a first-person, head-mounted-camera description
and ended with an explicit statement that the viewpoint moves with the
wearer's head and body. This increased motion, but visual inspection shows the
camera approaching the person visible in the frame rather than recovering the
full ground-truth route around the island.

### Native prompt upsampling

The explicit prose was passed through Edge's native prompt reasoner. The raw
structured result increased motion, but it made unsupported assertions,
including that the wearer was visible in first-person, that a head-mounted
camera appeared at the top of the frame, and that an additional background man
was present. It also changed clothing and scene details. The output was useful
as a schema draft but unsafe as conditioning without review.

Prompt-harvest job `527328` used the isolated diagnostic framework worktree
`/mnt/projects/ll/jungbinc/cosmos-framework-edge-i2vdiag`. Its diagnostic-only
changes tolerate the reasoner's preamble/fenced JSON and log the parsed result;
they do not alter model weights. The controlled experiment below did not use
that worktree or rerun the prompt upsampler.

### Audited frozen structured JSON

The reasoner output was manually corrected and frozen at:

`native_phase_training/prompts/edge_egocentric_s07_structured.json`

SHA-256:

`920b929d87b2865c07ae485a4b9aad341db8de34c0f2a9ecdcb42373c98dd197`

The audited prompt makes these roles explicit:

- the camera wearer is off-camera;
- exactly one other person is visible in the starting kitchen frame;
- the viewpoint, rather than the visible person, translates around the table;
- the desired path includes walking parallax, mild head motion, and a settling
  endpoint;
- the wearer's right hand may enter the lower foreground near the end.

The same byte-identical JSON and refreshed Edge negative prompt were used for
both controlled I2V samplers. Prompt upsampling was disabled, eliminating a
second stochastic semantic rewrite.

### Action-conditioned forward controls

Both forward-dynamics requests used the same conditioning image, ground-truth
camera-action file, seed, and shift 10 / 30 steps / guidance 1. Only the raw
caption subject changed:

```text
The person is standing near the kitchen table. The person then walks towards
the other side of the kitchen table and positions her right hand in front of
her at stomach level.
```

```text
The camera wearer is standing near the kitchen table. The camera wearer then
walks towards the other side of the kitchen table and positions her right hand
in front of her at stomach level.
```

## Controlled run

Slurm job `527341` completed all four samples in 59 seconds. Its completion
record verifies:

- `framework_dirty: false`;
- `weights_modified: false`;
- `diffusion_cache: false`;
- framework commit `d4599e2e43fbd06168e9884205b9b66c3902d8f6`.

The four variants were:

| Mode | Conditioning | Shift / steps / guidance |
|---|---|---:|
| I2V | audited frozen JSON | 10 / 35 / 6 |
| I2V | same audited frozen JSON | 12 / 20 / 6 |
| Forward dynamics | GT 96 x 9 actions + `The person` | 10 / 30 / 1 |
| Forward dynamics | same GT actions + `The camera wearer` | 10 / 30 / 1 |

## Diffusers backend follow-up

Two follow-up runs used the pinned Hugging Face Diffusers 0.40.0
`Cosmos3OmniPipeline` at commit
`d035dcd7cc7c88e0a154609b62887d50bba9fdc2`. Both loaded the same untouched
`/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge` checkpoint and the same Nymeria
first frame as the native tests. They generated 256 x 256, T97, 20-FPS video at
shift 12 / 20 steps / guidance 6 / seed 0 with NVIDIA's bundled structured
negative prompt, resolution and duration templates disabled, and no safety
checker.

Slurm job `527757` passed the audited frozen structured JSON directly to
Diffusers. Slurm job `527758` was retained as the interactive tmux allocation
`cosmos3_edge_gpu`; inside it, a second run passed this plain text directly:

```text
A first-person egocentric video from a head-mounted camera. The camera wearer
is standing near the kitchen table. The camera wearer then walks towards the
other side of the kitchen table and positions her right hand in front of her
at stomach level. The camera viewpoint moves with the camera wearer's head and
body.
```

The plain prompt was not converted to JSON and was not passed through Edge's
reasoner. Its completion record explicitly stores `prompt_format: plain_text`
and `native_prompt_upsampling: false`. The frozen text file is
`native_phase_training/prompts/edge_egocentric_s07_camera_wearer.txt`, SHA-256
`b048647b6256a34410f6afc4fe0deb55464163e2a6e859d061cc1e6bb4be347e`.

The Diffusers results are:

| Prompt/runtime | Mean global flow | Fraction of GT | RGB adjacent delta | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| Audited JSON, native framework | 0.5796 | 24.28% | 0.02421 | 11.885 | 0.3105 |
| Audited JSON, Diffusers | 0.3729 | 15.62% | 0.01611 | 12.442 | 0.3166 |
| Plain camera-wearer text, Diffusers | 0.8330 | 34.90% | 0.02803 | 11.672 | 0.2673 |

The native and Diffusers structured-JSON outputs are not pixel-identical even
though checkpoint, prompt, negative prompt, image, seed, dimensions, FPS, and
sampler settings were held fixed. Their cross-output mean RGB absolute error is
`0.13385`, mean frame PSNR is `14.54 dB`, and mean frame SSIM is `0.3684`.
Runtime implementation is therefore an experimental variable; a sampler tuple
alone does not make the two backends numerically equivalent.

The plain-text Diffusers output has more motion, especially early in the clip,
but visual inspection shows the camera approaching and retaining the visible
woman rather than completing the ground-truth route around the island. The
audited structured prompt better separates the unseen wearer from the visible
person, although it still hallucinates a foreground hand/object and settles
early. The larger plain-text flow number is not evidence of better egocentric
control.

Artifacts:

- structured JSON Diffusers:
  `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_egocentric_diffusers_modelcard_20260901_256_T97_20fps_s12_n20_g6_seed0`;
- plain camera-wearer Diffusers:
  `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_egocentric_plain_camera_wearer_diffusers_modelcard_20260901_256_T97_20fps_s12_n20_g6_seed0`.

These are still one-clip, one-seed diagnostics, not a backend benchmark.

## Frozen 20-clip qualitative suite

The first 20 of the 71 canonical test sequences are now a versioned qualitative
cohort. `edge_qualitative20_cohort_v1.json` stores the ordered sample names and
the SHA-256 of every source task JSONL, so later checkpoint comparisons cannot
silently select a different subset or order.

Slurm allocation `527758` completed the untouched Edge checkpoint suite at:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_zeroshot_qualitative20_v1_20260901_256_T97_20fps_seed0
```

The run produced:

- 20 native forward-dynamics results at 10 / 30 / 1;
- 20 native inverse-dynamics results at 10 / 30 / 1;
- 20 native WAM/policy results at 10 / 30 / 1;
- 20 native I2V results at 12 / 20 / 6;
- 20 pinned-Diffusers I2V results at 12 / 20 / 6;
- 20 five-way GT/FD/WAM/native-I2V/Diffusers-I2V comparison MP4s;
- 60 native task-pair MP4s and 40 ID/WAM trajectory plots.

All generations use 256 x 256, T97, 20 FPS, and seed 0. The paired native and
Diffusers I2V branches consume one shared sanitized JSONL. The exact positive
prompt string is serialized the way the native framework does immediately
before tokenization; positive, negative, and conditioning-image hashes are then
checked against both backends' saved arguments. The Diffusers branch also uses
the native framework's `FlowUniPCMultistepScheduler`, not the older generic
Diffusers UniPC schedule, so both branches have the same 20 timestep values and
solver implementation. Backend-specific packing, preprocessing, and model
execution remain real differences and explain why outputs can differ.

`validation.json` passed all 100 outputs: 60 required native generated videos
and all 20 Diffusers videos are exactly 97 frames at 20 FPS and 256 x 256; all
40 ID/WAM action tensors are finite and shaped 96 x 9; and every paired I2V
input contract matches. This is structural/reproducibility validation, not a
claim that all 20 generated futures are semantically correct. The primary
artifacts for future visual review are `viz/five_way/*.mp4`, with the exact
prompts retained beside each native and Diffusers video.

## Small-template v2 ablation

The first generic Nymeria template grew too large to support interpretable
ablation. It is retained as a historical artifact, but the active inference
path restarted from two short versioned templates:

- positive v2.0: `prompts/nymeria_i2v_prompt_template_v2.json` (783 bytes);
- positive v2.1: `prompts/nymeria_i2v_prompt_template_v2_1.json` (978 bytes);
- negative v2.0: `prompts/nymeria_i2v_negative_prompt_template_v2.json`
  (871 bytes).

Positive v2.0 contains the original caption once in
`actions[0].description`, a short unseen-wearer context, camera
motion/angle/lens, style/medium, and numeric media metadata. Positive v2.1 adds
only this field:

```json
"temporal_caption": "The described action is shown through movement of the first-person head-mounted viewpoint and, when relevant, the wearer's hands entering naturally from the frame edges."
```

Negative v2.0 deliberately has no `subjects` or `actions` block. It does not
assert a person count, scene, activity, or that a static clip is wrong. It names
only external/third-person or obstructed viewpoints, unnatural external-camera
moves, and generic temporal corruption. This replaces NVIDIA's bundled
17,578-byte negative prompt, which contains failure descriptions specialized to
the model-card driving example.

All three Diffusers comparisons used the same untouched Edge checkpoint,
conditioning image, camera-wearer source caption, 256/T97/20-FPS request,
shift 12, 20 steps, guidance 6, and seed 0:

| Positive | Negative | Mean global flow | Fraction of GT |
|---|---|---:|---:|
| v2.0 | NVIDIA model-card | 0.2733 | 11.45% |
| v2.0 | Nymeria v2.0 | 0.1940 | 8.13% |
| v2.1 | Nymeria v2.0 | 0.3162 | 13.24% |

The flow values are descriptive, not semantic scores. Frame inspection is the
deciding evidence: both v2.0 runs mainly animated the existing woman. v2.1
moved the viewpoint around the island, shifted the woman primarily by parallax,
and introduced the wearer's hands naturally from the bottom edge without
placing a complete camera operator or headset in front of the camera. v2.1 is
therefore the active small-template candidate. It still requires multi-clip and
multi-seed validation before being treated as a general solution.

Run roots:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_nymeria_prompt_v2_0_neg_driving_control_20260901_256_T97_20fps_s12_n20_g6_seed0
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_nymeria_prompt_v2_0_neg_nymeria_v2_0_20260901_256_T97_20fps_s12_n20_g6_seed0
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_nymeria_prompt_v2_1_temporal_neg_nymeria_v2_0_20260901_256_T97_20fps_s12_n20_g6_seed0
```

Each run saves the effective `positive_prompt.json`, `negative_prompt.json`,
and `prompt_manifest.json` next to `edge_i2v_diffusers.mp4`. Prompt upsampling
is disabled. This entire transform is inference-only; the Phase-1 training
caption contract is unchanged.

## Quantitative diagnostics

The motion statistic is the mean over adjacent frames of the median Farneback
flow magnitude after resizing to 128 x 128. It measures how much the generated
video changes globally; it does **not** prove that the change is the correct
camera transform. Object motion, scene warping, and generation artifacts can
all increase it.

Frame-aligned PSNR and SSIM exclude conditioned frame 0. They are useful for
the action-conditioned controls, but stochastic I2V is not expected to
reconstruct the unique ground-truth future, so they are not prompt-adherence
metrics.

| Output | Mean global flow | Fraction of GT | RGB adjacent delta | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| Ground truth | 2.3870 | 100.0% | 0.04894 | -- | -- |
| Audited JSON I2V, 10/35/6 | 0.5841 | 24.47% | 0.02410 | 11.862 | 0.3101 |
| Audited JSON I2V, 12/20/6 | 0.5796 | 24.28% | 0.02421 | 11.885 | 0.3105 |
| Forward, `The person` | 3.9581 | 165.82% | 0.06184 | 13.253 | 0.3788 |
| Forward, `The camera wearer` | 3.8699 | 162.12% | 0.06084 | 13.053 | 0.3713 |

The horizon split makes the failure modes clearer:

| Output | Frames 1--32 | Frames 33--64 | Frames 65--96 |
|---|---:|---:|---:|
| Ground truth | 3.2969 | 2.4749 | 1.3892 |
| Audited JSON I2V, 10/35/6 | 0.9061 | 0.6208 | 0.2255 |
| Audited JSON I2V, 12/20/6 | 0.8975 | 0.6219 | 0.2194 |
| Forward, `The person` | 5.2266 | 3.9552 | 2.6924 |
| Forward, `The camera wearer` | 5.1493 | 3.9164 | 2.5438 |

The two I2V sampler recipes are nearly identical under fixed structured
conditioning. Both lose most motion late in the clip. The two action controls
are also close to one another and are substantially above GT in every horizon,
which is consistent with the numerical action input dominating the wording.

For context, the same analyzer was also applied to the saved exploratory I2V
videos:

| I2V prompt | Sampler | Mean global flow | Fraction of GT |
|---|---:|---:|---:|
| Plain `The person` | 10/35/6 | 0.07510 | 3.15% |
| Plain `person wearing the camera` | 10/35/6 | 0.07685 | 3.22% |
| Explicit first-person prose | 10/35/6 | 0.72018 | 30.17% |
| Raw reasoner JSON | 12/20/6 | 0.84491 | 35.40% |

The raw reasoner result has the largest I2V motion number, but its prompt and
video contain unsupported subject/scene changes. It is not the best semantic
result merely because it moves more.

## Visual audit

Frames 0, 12, 18, 24, 30, 48, 64, 80, and 96 were inspected for GT and every
variant.

- Ground truth travels around the island, shifts the visible person out of
  frame, and reaches a substantially different view of the kitchen.
- Explicit egocentric prose creates clear motion but advances toward and keeps
  the visible person central, which is the ambiguity the test was designed to
  expose.
- Raw reasoner JSON removes the visible person early and creates clean-looking
  viewpoint motion, but it hallucinates a foreground hand and changes scene
  content in line with the reasoner's unsupported JSON.
- Both audited-JSON I2V samples move toward/across the island and no longer
  treat the visible person as the moving camera wearer. However, the person
  disappears early, hands appear earlier than requested, table objects mutate,
  and the camera settles before completing the GT path. The two sampler recipes
  differ mainly in invented object details, not in camera-motion magnitude.
- Both forward-dynamics controls make a large sweep around the kitchen and are
  the only tested outputs with trajectory-scale motion throughout the clip.
  They do not exactly track GT: the visible person persists or warps, kitchen
  geometry smears during large motion, and the total motion overshoots. The two
  caption subjects are visually and quantitatively very similar.

## Inference recommendation

### Known or prescribed camera path

Use Edge `forward_dynamics` with:

- first-frame image conditioning;
- one raw, unnormalized 96 x 9 `camera_pose` delta sequence for T97;
- 20 FPS in both the request and action conversion contract;
- shift 10, 30 UniPC steps, and guidance 1;
- diffusion cache disabled for correctness/evaluation runs.

Use `The camera wearer` in human-readable captions because it is semantically
clear, but do not expect this text substitution to provide the actual camera
control. The 9D action sequence does that.

### No camera actions

Use generic I2V with an audited structured prompt that separately names:

- the unseen/off-camera wearer;
- people visible in the image;
- `cinematography.camera_motion`;
- time-indexed viewpoint actions/segments;
- FPS, duration, resolution, and aspect ratio.

Freeze the resulting JSON before sampling. Treat the output as a plausible
egocentric future, not a reconstruction of a specific trajectory. Native
10/35/6 remains the comparison contract; refreshed 12/20/6 is faster but did
not improve this sample.

### Phase-1 training and evaluation

This diagnostic does not change the Phase-1 task mix or prompt contract. The
current four-task run still uses official action JSON for forward/policy,
empty text for inverse, and generic duration/FPS/resolution-augmented text for
I2V. The audited JSON above is an inference-only diagnostic, not evidence that
all I2V training captions should be replaced.

For checkpoint evaluation:

- retain ordinary I2V as the base-video-prior comparison;
- use forward dynamics for camera-trajectory controllability;
- optionally add frozen structured I2V as a separate egocentric-semantic probe;
- do not promote online reasoner output into the training/evaluation contract
  without an offline validation pass across many clips.

## Reproduction and artifacts

The controlled launcher is:

```bash
EVAL_OUTPUT_DIR=/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/<fresh_name> \
  sbatch native_phase_training/sbatch_edge_egocentric_structured_diagnostics.sh
```

It builds two I2V records and two forward controls, loads Edge once, runs all
four samples, computes diagnostics, verifies every output, and writes a
completion manifest only after success.

Code and prompt:

- `native_phase_training/prompts/edge_egocentric_s07_structured.json`
- `native_phase_training/prepare_edge_egocentric_structured_diagnostics.py`
- `native_phase_training/analyze_edge_egocentric_diagnostics.py`
- `native_phase_training/sbatch_edge_egocentric_structured_diagnostics.sh`
- `native_phase_training/prepare_edge_i2v_diagnostics.py`
- `native_phase_training/sbatch_edge_i2v_diagnostics.sh`

Controlled result root:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_egocentric_structured_20260901_256_T97_20fps_seed0
```

Authoritative records inside that root:

- `COMPLETE.json`: job, checkpoint, framework commit/cleanliness, and hashes;
- `inference_inputs/manifest.json`: exact prompt, sampler, media, and action
  provenance;
- `metrics/egocentric_diagnostics.json`: all controlled quantitative results;
- each sample directory: `sample_args.json`, `sample_outputs.json`, and
  `vision.mp4`.

Exploratory result roots:

```text
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_i2v_camera_wearer_20260901_256_T97_20fps_s10_n35_g6_seed0
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_i2v_egocentric_viewpoint_20260901_256_T97_20fps_s10_n35_g6_seed0
/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_i2v_egocentric_upsample_harvest_20260901_256_T97_20fps_seed0
```

## Limits

- Only one held-out clip and one seed were evaluated.
- Median optical flow is not a camera-pose estimator and can reward warping.
- PSNR/SSIM cannot establish prompt adherence for stochastic I2V.
- The GT 9D action run establishes that the action mode responds strongly; it
  does not prove precise camera-action calibration from one example.
- A training change should require a multi-clip, multi-seed comparison with
  visual review and, where possible, camera-motion estimation against GT.
