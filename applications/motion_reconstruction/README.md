# SOMA motion reconstruction: publication visualization

Turns saved UniEgo283 features or global body joint predictions into full-resolution
SOMA meshes, colors their surface by reconstruction error, and exports synchronized
comparisons and paper figures. No model sampling or training is performed.

The user requested indigo `#7B77FF` at low error and a muted pink/red for high
error. The default high endpoint is rose `#D8768C`, with perceptually smooth OKLab
interpolation. One fixed **0–30cm** range is shared by all models/frames; larger
errors saturate at the explicitly labeled ≥30cm endpoint. Raw errors and saturated
fractions remain available; no per-frame normalization hides large errors.

## Scientific meaning

For the common23 SOMA body joints, the default uses pelvis-relative positions:
`e_j(t) = ||(p_pred(t,j)-p_pred(t,0)) - (p_GT(t,j)-p_GT(t,0))||` in meters,
with joint0 the pelvis and subtraction performed independently at every frame.
This removes translation only; heading and scale differences remain. It is not
parent-local bone error or Procrustes alignment. The mesh heatmap is
`e_vertex(t,v) = sum_j W(v,j) * e_j(t)`, where W is the SOMA linear-blend-skinning
influence, summed from77 rig bones onto their nearest controlled body ancestor.
Finger/face/end-bone regions inherit the corresponding body-joint influence;
they are not independently scored. No spatial or temporal post-hoc smoothing
is added beyond this declared skin-weight interpolation. Set `error_reference`
to `root_relative` (default) or `world` (legacy global error). Both numerical
versions are saved, and the selected reference is labeled on figures and videos.

This is a **joint-derived surface heatmap**, not measured per-vertex GT error.
[EgoMDM's author project page](https://yohanshin.github.io/egomdm.github.io/)
describes actual per-vertex distances for its geometric-accuracy visualizations.
We borrow the comparison idea, not that metric label. An optional
`error_mode: template_vertex_distance` colors distance between our two rendered
standard-template surfaces; it still does not establish subject-specific surface
accuracy or agreement with a scanned SOMA-X mesh.

The topology is Kimodo's standard SOMA visualization skin:18,056 vertices and
36,108 triangles,77 rig joints. It is not a subject-specific SOMA-X identity.
The stock30-to77 visualization recomputes positions by standard-skeleton FK.
Here, **all23 decoded body global translations and rotations are retained**;
only missing face/finger/end bones follow the template's local bind transforms
and relaxed hands. LBS uses `T_posed @ inverse(T_bind)` with original skin
weights. Thus the surface can stretch with inconsistent predicted bones; no
IK/FK fitting, grounding, alignment or temporal filtering silently repairs them.
Body shape influences supplied joint translations, but no identity-blendshape
fit is inferred from neutral joints. This is explicitly a visualization surface.

The independent tests check bind-pose recovery, nontrivial rigid equivariance,
preservation of an intentionally displaced body joint, zero/self error, uniform
translation error, normalized skin influences and exact palette endpoints.

## Inputs and normalization

`pipeline.py` accepts two declared formats:

- `joints`: NPZ with global `joints` [T,23/30,3] in meters and `rotations`
  [T,23/30,3,3]. Alternate key names and a global-rotation declaration are
  accepted. Joint order is the existing common BODY_JOINTS subset or SOMA30.
- `uniego283`: NPY or NPZ with a specified feature key. You must explicitly
  declare `normalized: true/false`. Normalized features require the exact
  training/checkpoint `mean` and `std` paths, with optional expected SHA256
  checks. Their actual hashes are always saved. The existing evaluation decoder
  is reused; evaluator normalization is not substituted for generator statistics.

Example configuration (all paths are explicit and can be changed per server):

```json
{
  "out": "output/soma_errors",
  "skin": "/path/to/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz",
  "fps": 20,
  "maximum_cm": 30,
  "error_mode": "joint_lbs",
  "error_reference": "root_relative",
  "shadow_style": "single_back",
  "cases": [{
    "id": "example",
    "GT": {
      "path": "gt.npz", "format": "uniego283",
      "features_key": "raw", "normalized": false
    },
    "models": [{
      "label": "ours", "title": "Ours", "input": "Video → motion",
      "path": "prediction.npz", "format": "joints"
    }]
  }]
}
```

Use the original measured/calibrated world frame for both inputs. The pipeline
does not infer a coordinate registration. It uses a proper right-handed basis
change from Y-up to Blender `(X,-Z,Y)`, without X mirroring. A single camera/bounds
fits all methods and frames in each case. There is no independent root tracking
or pose centering of the actual meshes. Pelvis subtraction is used only for the
heatmap values. The floor remains at benchmark Y=0; transparent shadow catching
allows white paper backgrounds. Default `shadow_style: single_back` uses one
soft directional light on the viewer-facing side, casting its shadow away from
the camera. Ambient world illumination adds no second directional light. The
old layout remains available as `studio_three_light` for legacy replay.

## Run

Perform processing/rendering on owned compute, not a login node. Set
`COSMOS_SITE=custom` on a non-Yonsei compute server, or retain the Yonsei Slurm
environment. Use a Python environment with NumPy, Pillow, matplotlib and
imageio-ffmpeg; feature decoding additionally uses Torch. Decoded-joint inputs
need neither Torch nor a model checkpoint. Add the repository
and `motion_expert_joint_attention` to PYTHONPATH for feature decoding.

```bash
python -m applications.motion_reconstruction.pipeline --config experiment.json
python -m pytest -q applications/motion_reconstruction/test_mesh.py
```

Use Blender4.5+ with an owned Cycles CUDA/OptiX GPU, preserving
CUDA_VISIBLE_DEVICES. The Nov7 operation uses Blender4.5.15 LTS, verified against
the official archive SHA256. It does not silently fall back to CPU rendering.

```bash
blender --background --factory-startup \
  --python applications/motion_reconstruction/blender_render.py -- \
  --manifest output/soma_errors/manifest.json --case example --model ours \
  --frames all --width 384 --height 512 --samples 48 --tag video

blender --background --factory-startup \
  --python applications/motion_reconstruction/blender_render.py -- \
  --manifest output/soma_errors/manifest.json --case example --model ours \
  --frames 0,32,48,64,96 --width 768 --height 1024 --samples 128 --tag paper --export-glb
```

Repeat both commands for GT and each model label, then:

```bash
python -m applications.motion_reconstruction.compose --root output/soma_errors
```

The example composer expects T97/20FPS; mesh preparation and individual rendering
support other lengths, but the contact-sheet frame list must then be changed
explicitly. Run from a portable bundle's root when using its relative paths.

## Saved artifacts

### Sparse temporal scenes with cameras and observed RGB

`sparse.py prepare --config configuration.json` reuses an existing prepared
mesh gallery and an explicit list of matching evaluation cases. The Nov8
extension selects frames0/48/96 (0/2.4/4.8seconds). Each method is rendered as
three bodies in one scene. A recorded display translation separates timestamps;
the same translation moves body and camera together and is shared by every
method. This spacing is not physical displacement or a camera trajectory.

Camera coordinate axes show measured upright RGB poses, using the same train-only
head calibration and camera-defined yaw/horizontal origin as the evaluated
camera baselines. They are shared reference overlays, not model predictions
and not extra conditioning for the video-only method. Camera/body rotation
agreement is audited against GT; GT does not fit or move model predictions.
The default is X red/right, Y green/down, Z blue/forward in optical camera
coordinates. Arrow length18cm is for display and does not depict a field of
view. `camera_glyph: frustum` retains the older schematic green glyphs for
legacy replay. Original per-recording camera poses and absolute source frame indices
are recorded. RGB is decoded with the exact existing20FPS evaluation reader.

Render each case/method with Blender's `blender_sparse.py -- --manifest
gallery/manifest.json --case CASE --model MODEL`, then run
`python -m applications.motion_reconstruction.sparse compose --root gallery`.
This produces three matched layouts: `clean`, `cameras`, and `cameras_video`.
The last repeats the same observed RGB frames above the corresponding poses
in each method's panel; these are not generated views of predicted bodies.
Both PNG and PDF versions are saved. All layouts retain pelvis-relative
heatmaps, a shared0–30cm scale and single rear shadows.

The axes/caption extension includes six cases: the original three plus the
first sorted eligible captions for kneeling/squatting, sitting/standing
transition, and walking/turning in three other recordings. Selection precedes
rendering. Five methods use a three-column layout: GT, Ours video only, Ours
video+caption, EgoEgo and EgoAllo. The unused sixth panel shows the caption.
Both Ours variants are sampled afresh with identical seeds, checkpoint, cached
video, supplied shape, stats and native35-step/shift3/guidance1 sampler. The
existing inference-only intervention supplies text to both reasoner paths;
caption-conditioned V2M was not part of this bridge checkpoint's training.

### Individual frames and video comparisons

- `manifest.json` / case JSONs: source hashes, exact raw error policy, palette,
  clipping fractions, camera/bounds and input conditions.
- Per-model NPZ: full surface vertices, original controlled positions/rotations,
  selected23-joint errors, both global/pelvis-relative joint and LBS surface
  errors, optional template-vertex
  distances and timestamps. Vertex errors are in meters, not normalized colors.
- `renders/<model>/video`:97 transparent PNGs; `paper`: five high-resolution
  PNGs and a colored GLB of the last selected frame96. Receipt records that frame.
- `comparison.mp4`: synchronized four-method example,97 frames at20FPS; white
  compositing, method/input labels and a common error legend.
- `paper/frame_000048.{png,pdf}` and `temporal_contact_sheet.{png,pdf}`:300DPI
  raster figures and PDFs with vector labels/colorbar; source mesh PNGs remain
  available for alternative paper layouts.
- `error_legend.{svg,pdf,png}`: reusable vector/raster color legend.
- `index.html`: local review gallery. Serve locally with `python -m http.server`.

Transparent renderer PNGs can appear black in viewers that ignore alpha; composite
them onto white for review, as the provided composer does. Error colors are
material albedos under neutral studio lighting, so shaded pixel RGB varies with
surface orientation. The color map and stored numeric errors define the legend.

Render REQUEST/COMPLETE receipts pin source, manifest and style settings; changed
recipes require a new output/tag. Geometry preparation snapshots its sources;
composition separately snapshots its actual source to preserve later style edits.
Neither process edits source predictions, checkpoints or quantitative reports.

The Nov7 global-error/three-light gallery is retained as version1. Nov8 version2
uses pelvis-relative colors and a single rear shadow on the same deterministic
first/middle/last cases from the sorted
whole-test list, chosen before heatmap inspection. It compares GT, original
Phase3 baseline200k (video), EgoEgo EMA555.8k (camera+shape), and EgoAllo200k
(camera+shape). Inputs and training steps differ; the gallery demonstrates the
visualization pipeline rather than a controlled equal-information model ranking.
