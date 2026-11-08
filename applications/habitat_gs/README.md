# Video rollout in Habitat-GS

For the live browser workflow (enter a prompt, generate, estimate camera, watch
the scene replay, repeat), see [INTERACTIVE.md](INTERACTIVE.md) and `serve.sh`.

This application repeatedly renders the static Gaussian scene, generates a
97-frame clip from text plus that first image, estimates96 relative camera
transitions from the generated clip, and re-renders at the accumulated last
camera pose. The next clip starts from the new simulator observation. It does
not feed generated RGB back as the next observed image.

External sources:

- [Habitat-GS](https://github.com/zju3dv/habitat-gs), pinned to
  `cb01b2fdf850ce3cab28ad00461a68ac7719f23f` in ignored `.deps/habitat-gs`.
- [RukawaY/gs_scenes](https://huggingface.co/datasets/RukawaY/gs_scenes).
  `download_scene.py` pins the dataset revision, records local SHA256 hashes,
  verifies published LFS hashes when available, and downloads only one scene
  plus its NavMesh. The example uses `val/interior_0516_840045` (~147MB PLY).

## Evaluated inference settings

`profile.py` reads the actual sanitized `i2v_input.jsonl` and
`invdyn_input.jsonl` from a saved evaluation. It copies all sampling settings,
including the negative prompt and metadata behavior, and stores source hashes.
Raw historical fixture files with shift3 are rejected when the native checkpoint
contract says inference shift10. Training shift3 is not the inference shift.

Initial reference: the original caption-trained Edge Phase1 EMA200k, evaluated
in `checkpoint_evals_4090/iter_000200000_job2219279`. Its two modes are:

| Mode | Inputs → output | Sampler | Steps | Guidance | Shift |
| --- | --- | --- | ---: | ---: | ---: |
| `image2video` | text + rendered first image → video | UniPC |35|6|10|
| `inverse_dynamics` | generated full video, empty text → camera | UniPC |30|1|10|

Both use EMA, T97,20fps,256×256, one output, and deterministic seed
`base_seed + segment_index`. I2V uses the existing structured egocentric prompt
builder and the evaluation's fixed negative prompt. No new prompt enhancer is
introduced. `image2video` is the official API name for text+image→video here.
The official CLI runs with the same native experiment/config, latency parallelism,
one GPU, diffusion-cache default enabled, torch compile/CUDA graphs disabled,
and guardrails disabled, matching this saved evaluation launcher. Each stage
starts a fresh official process; model reloads add overhead but preserve the
evaluated model path. A future persistent worker must pass inference parity
before replacing this baseline. No text/camera CFG sweep is enabled implicitly.

## Camera contract

World poses in application files are `T_scene_from_upright_optical_camera`:
camera-to-world matrices, meters converted by explicit `scene_units_per_meter`.
The optical camera axes are right/down/forward (+X,+Y,+Z), while Habitat's GL
camera is right/up/backward (+X,+Y,−Z). Right multiplication by
`diag(1,-1,-1,1)` converts optical to GL. Habitat world remains Y-up; do not
add another Nymeria world-axis rotation or a raw device-to-RGB extrinsic.

The model outputs Cosmos `backward_framewise` actions:

```
delta[i] = inverse(T[i]) @ T[i+1]
action[i] = [translation_xyz, rotation_column0, rotation_column1]
T[i+1] = T[i] @ delta[i]
```

All96 actions are integrated, yielding97 poses. Rotation6D is orthogonalized
as in the existing evaluator; degenerate/nonfinite actions fail visibly.
Translation is not recentered, yaw-stripped, ground-snapped, or fitted using
GT/Sim3. The example explicitly assumes one scene unit per meter; confirm the
asset's physical scale before interpreting metric distances.

**Nymeria training RGB retains Aria FISHEYE624 distortion.** The preprocessing
rotates the raw image clockwise90° then resizes; matching only K/FOV cannot
reproduce its peripheral rays. The available camera sidecar saves K/model name
but not all distortion coefficients. Therefore the two explicit modes are:

1. `calibrate --sidecar ...`: scaled upright K with correct half-pixel resizing,
   explicitly labeled **pinhole approximation to unrectified Aria**. The rollout
   requires `allow_pinhole_approximation: true` to use it.
2. `calibrate --sidecar ... --vrs ...`: use the matching recording's full
   Project Aria calibration to unproject pixels and rotate rays into the upright
   frame. Export once in a Project Aria environment. VRS/sidecar intrinsics must
   match. Six Habitat perspective views are reprojected onto these rays.

Neither mode reproduces lens blur, rolling shutter or the full original
640→256 antialiasing pipeline. The matching camera ray map's source/model/hash
is recorded in every run. A simple90° Habitat viewer camera is not claimed to
match Nymeria.

## Setup and run on Yonsei

Use an owned compute allocation, never a login node, `cs-gpu-01`, or `node37`.
The simulator has its own environment (Python3.12 via conda by default;
`HABITAT_BOOTSTRAP_PYTHON=/path/to/python` uses a separate venv with an existing
supported interpreter). The current Yonsei setup uses Python3.10 from the
existing UEM interpreter to avoid a stalled conda-forge metadata fetch; its
packages are isolated from UEM. Cosmos retains its existing CUDA12.8 environment.
Static scene rendering uses upstream's native CUDA implementation with optional
PyTorch absent; its core runtime dependencies are installed explicitly. Avatar
dependencies (PyTorch/SMPLX/chumpy) are excluded by default and can be requested
separately with `HABITAT_INSTALL_AVATAR_DEPS=1`. This application uses no avatar.
Default bulk root:
`/scratch2/whwjdqls99/cosmos/applications/habitat_gs`.
The renderer and Cosmos subprocesses run sequentially, freeing VRAM between
stages. Grasp/custom sites supply explicit roots to `run.sh`; setup's CUDA
path/conda executable can be overridden. Setup targets4090 CUDA architecture89;
change the explicit CMake architecture when installing for different hardware.

```bash
export APP_STORAGE=/scratch2/whwjdqls99/cosmos/applications/habitat_gs
bash applications/habitat_gs/setup.sh
python -m applications.habitat_gs.download_scene --out "$APP_STORAGE/assets"

bash applications/habitat_gs/run.sh profile \
  --checkpoint /path/to/original/checkpoints/iter_000200000 \
  --evaluation-inputs /path/to/checkpoint_evals_4090/iter_000200000_job2219279/inference_inputs \
  --out "$APP_STORAGE/evaluation_profile.json"

bash applications/habitat_gs/run.sh calibrate \
  --sidecar /path/to/camera_rgb/Sxx/recording.npz \
  --out "$APP_STORAGE/camera_rays.npz"

cp applications/habitat_gs/rollout.example.json "$APP_STORAGE/rollout.json"
# Edit prompt, output directory, number of segments, and optional starting pose.
bash applications/habitat_gs/run.sh rollout --config "$APP_STORAGE/rollout.json"
```

An explicit start pose is a4×4 optical camera-to-world matrix. With `null`, the
renderer chooses a seeded NavMesh point, adds initial camera height (default1.5m)
and a seeded heading. This height is a starting choice, not an estimated body
height or a constraint on subsequent predictions. Raw scene rendering and pose
transport do not enforce collision-free paths; camera estimates may drift or
leave the scene, and generated interactions need not be represented by a static
GS asset. Registration/collision correction requires a separately documented
experiment and is not silently inserted.

## Outputs and checks

`configuration.json` pins the resolved config/profile, scene/camera hashes and
framework revision. Each segment saves official inputs, both native logs/output
JSONs, predicted96×9 actions,97 world poses, and the next simulator observation.
`boundary.png` places the generated last frame beside the GS image at the
predicted last pose. This measures model/scene self-consistency; it is not an
independent GT error. No GT PSNR, retrieval or contact claims are made.

`index.html` and `rollout_preview.mp4` show all segments, with green observed
conditioning frames and red generated frames. The display concatenation retains
each segment's first observation, duplicating the shared boundary instant;
physical timestamps in the pose NPZ advance by96/20=4.8seconds per segment.
Source videos stay untouched. Completed segments are hash-checked on resume;
changed config/profile/assets require a new output directory. Failed stages
stop and retain their logs; they do not invent a camera pose or skip a segment.

Run focused CPU checks on an owned compute node:

```bash
bash applications/habitat_gs/run.sh test_contracts
```

This checks GL/optical conversions, noncommuting pose integration, rotation6D,
metric scaling, pixel-center resizing, cube-face projection and sampler-profile
preservation. Simulator installation/rendering and real model inference have
separate runtime evidence under the bulk root; code existence is not evidence
that an end-to-end rollout completed.

Human rollout is implemented as `human_rollout`, `human_motion` and
`human_replay`; see [HUMAN_REPLAY.md](HUMAN_REPLAY.md) for the data schema and
coordinate/timing contract. The first successful two-prompt run uses the
original interior_0516 scene: walk forward, then sit down on a chair. Results
and the portable `.tar.gz` are under APP_STORAGE/runs/human_walk_sit_20261107.
The saved operation config is
`/scratch2/whwjdqls99/cosmos/operations/habitat_human_rollout_20261107/rollout.json`.
It keeps the original Phase1 profile and reuses the saved baseline Phase3
200k caption-conditioned V2M evaluation (UniPC35/CFG1/shift3). Camera motion
comes from decoded Head * train calibration, not inverse dynamics. No scene
collision correction, grounding or first-motion-pose inpainting is added.
Replay needs only the saved data and Habitat-GS; inference uses owned compute.

Setup recovery (2026-10-30): interrupted initial submodule downloads left
Assimp, Bullet, Magnum, and OpenEXR repositories without a valid HEAD.
The setup script now detects these unborn checkouts and fetches/checks out
exactly the parent-pinned revision before normal recursive submodule setup.
This fixes source installation only; no successful scene render or generated
rollout is implied by the geometry/unit checks. Runtime evidence is recorded
under the application storage directory in setup_venv.log and smoke.log.

The static Gaussian CUDA renderer also requires GLM headers. Setup pins GLM
1.0.1 (0af55ccecd98d4e5a8d1fad7de25ba429d60e863) under APP_STORAGE/deps/glm
and supplies its include path to both C++ and CUDA compilation. This avoids
system package installation and leaves the Cosmos environment unchanged.

Runtime verification (2026-10-30): Habitat-GS built successfully on node33.
The initial GPU render passed, producing render_probe/initial.png under the
application storage root. Renderer subprocesses require the CUDA 12.8 runtime
library path even while excluding Cosmos-specific shared libraries. The first
failed launch is retained in runs/video_rollout_cuda_path_failure. Rendering
success is separate from the still-required video-generation rollout check.

### Nymeria camera requirement and correction (2026-10-30)

Application rollouts now default to measured Aria FISHEYE624 rays, and reject
pinhole approximation. The initial pinhole run is retained separately as a
setup diagnostic; it is not the camera-matched result. The full camera-rgb
factory calibration was extracted from the first 1 MiB of the original
20230607_s0_james_johnson_act1_7xwm28 head VRS using its authorized download
manifest. No full 8.18 GB RGB recording download was needed. Projection and
unprojection use projectaria-tools 2.3.0, not a custom lens implementation.

Factory2880 calibration is cropped by (32,32) and scaled0.5 to sensor1408;
this reproduces the saved raw K and device-to-RGB transform. The actual video
pipeline rotates clockwise90 and resizes640 then256. This older sidecar's
K_rgb_upright is for the opposite rotation (maximum discrepancy2.962 pixels
at256), so the application derives upright K from K_rgb_raw and actual pixel
rotation. Original dataset files and training checkpoints are unchanged.

Corrected K at256: fx=fy=110.728671, cx=126.399192, cy=128.980968.
Full distortion coefficients are retained by Aria unprojection. Valid-ray
fraction0.79150390625; maximum raw projection roundtrip residual1.31e-11 pixels.
This matches camera ray geometry; it does not reproduce exposure, soft lens
vignetting, rolling shutter or the exact multi-stage resize filter. One actual
TRAIN recording supplies the camera: Nymeria has per-device calibrations, not
one universal K. Ten contract tests pass, including the old-rotation regression.

Evidence under APP_STORAGE: nymeria_factory_calibration.json,
calibration_header_probe.bin, camera_rays_fisheye.json/.npz, fisheye_export.log,
nymeria_reference_rgb.png. rollout_smoke.json now requires the fisheye rays.
An unrelated preview failure was also fixed: use imageio-ffmpeg's verified
binary because this compute node has no /usr/bin/ffmpeg.

### Initial rollout complete; scene63 follow-up

The corrected-fisheye interior_0516 rollout completed both segments successfully
on2026-10-30 (smoke.exit=0; runs/video_rollout_smoke/COMPLETE.json). It advances
9.6s scene time. Generated end frames and GS rerenders visibly differ: this is
an end-to-end pipeline check, not evidence of accurate camera recovery or
perfect scene continuity. No independent GT trajectory/video exists here.

User requested scene63 Apartment or scene64 Commercial next; scene63 was chosen.
run_scene63.sh in application storage downloads only val/scene63 GS+NavMesh to
assets_scene63, verifies hashes, then runs rollout_scene63.json. It preserves
checkpoint, prompt, seeds0/1, sampling settings, camera calibration and two
segments. Outputs: runs/scene63_video_rollout. Check scene63.exit and COMPLETE.json
for completion, not this launch note. Runs use retained job2608785, node33 GPU0.

### Prompt labels in previews (2026-10-30)

All future previews show the configured action prompt throughout the clip in
a wrapped footer below the image, preserving green observed/red generated
labels and 20fps. Official inference still uses the same native structured
prompt wrapper; the readable footer shows its configured action description.
Both initial and scene63 two-segment rollouts completed. Their new
rollout_preview_prompt.mp4 files are linked from their existing galleries.
The original generated clips and checkpoint/evaluation settings are unchanged.
Each combined preview was decoded successfully:194frames at20fps. Original
inference receipts are preserved; prompt_preview.json separately records source
and presentation hashes. To refresh a completed run without inference:
`bash applications/habitat_gs/run.sh refresh_previews /absolute/run/directory`.
