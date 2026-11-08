# Human-motion rollout: demo replay contract

Start with `manifest.json`, then `segment_000/replay.json` and
`segment_001/replay.json`. Replaying needs Habitat-GS and the archived scene;
it does not need Cosmos checkpoints, diffusion inference or training datasets.
`inventory.json` lists every retained artifact with SHA256 and byte length.

## Portable assets

- `assets/scene.gs.ply`, `assets/scene.navmesh`: the original first scene,
  interior_0516_840045. Load it with identity world transform and scale1.
- `assets/camera_rays.npz`: measured Nymeria upright FISHEYE624 rays [256,256,3]
  and valid mask [256,256]. Metadata and source calibration are in
  `camera_calibration_metadata.json`. Use the same six-view reprojection as
  `applications.habitat_gs.render.Renderer.render`; a perspective camera with
  matching focal length alone does not reproduce these distorted images.
- `assets/skeleton.json`: all30 SOMA joint names and parents, root0, Head6,
  four foot indices. Body poses are decoded joint transforms, not bone-local
  FK rotations. Draw a bone from parent to child in scene/world coordinates.
- `assets/neutral_shape.npy`: centered30x3 mean training-cache neutral joints,
  used as the motion model's supplied shape. Provenance lists the recordings
  and cache identity. It is not an animation or an observed test skeleton.
  `training_shape_inputs.npz` retains all642 contributing neutral shapes and
  their recording IDs, so the mean can be reconstructed without the cache.
- `assets/head_camera_calibration_train.json`: measured train-only rigid
  Head-to-upright-optical-camera rotation and lever arm. No test-actor oracle.
- `assets/phase1_evaluation_profile.json`, `phase3_evaluation_reference.json`,
  `caption_conditioning_reference.py`: model/sampler provenance. Absolute model
  paths in these references describe generation; they are not required to replay.

## Per-segment files

| File | Meaning |
| --- | --- |
| `conditioning.png`, `start.pose.json` | Overlay-free initial GS image and its optical camera-to-world pose |
| `generated.mp4` | Native text+image-to-video output,97 frames,20fps,256x256 |
| `generated_rgb.npy` | Exact RGB uint8 [97,256,256,3] decoded from that output and supplied to the VAE |
| `video_latents.npy` | Evaluated Wan encode of those pixels, float16 [48,25,16,16] |
| `motion_arrays.npz` | Exact normalized/unnormalized motion features, decoder outputs, supplied shape, scene joints/transforms and anchor |
| `motion_stats_mean.npy`, `motion_stats_std.npy` | Exact283-D training normalization used to invert the model's predictions |
| `replay.json` | Web-readable camera matrices and world joints/transforms for all97 frames, prompt, seed, local/global timestamps |
| `camera_trajectory.npz` | The same97 optical camera-to-world poses; `frame_times_s` here is segment-local |
| `end.pose.json`, `next_condition.png` | Final body-derived camera pose and its overlay-free GS render; input to the next segment |
| `next_condition_independent.png` | Independent final-pose render verified pixel-identical to the replay's final frame |
| `inference.json` | Checkpoint identity, native sampler, caption-call audit, mapping formula and rotation-repair diagnostics |
| `rendered/generated_rgb/*.png` | Lossless per-frame copies of the generated pixels |
| `rendered/gs_egocentric/*.png` | Overlay-free GS images at every reconstructed camera pose |
| `rendered/gs_body_camera/*.png` | Third-person GS views with depth-tested orange skeleton, magenta camera and cyan path |
| `rendered/*.mp4` | Synchronized egocentric, scene inspection and comparison videos |
| `inference/`, `image2video.jsonl`, `*.log` | Original official inputs, native outputs and stage logs |

The generated video is synthetic, not GT. The GS replay is a render of the
estimated motion-derived camera, not independent GT either. Skeletons are
body stick figures; no full avatar mesh or hand model was introduced.

### Motion array keys

| Key in `motion_arrays.npz` | Shape | Meaning |
| --- | --- | --- |
| `features_normalized` |97x283| Exact model output in its training z-score space |
| `features_unnormalized` |97x283| Output multiplied by saved std and added to saved mean |
| `decoded_joint_transforms` |97x30x4x4| Exact evaluated float32 decoder result in the decoded Y-up frame |
| `neutral_joints` |30x3| Supplied centered shape in meters |
| `joint_poses_scene` |97x30x4x4| Global joint poses in the GS scene; not parent-local bone transforms |
| `joints_scene` |97x30x3| Translations of the preceding matrices; draw bones directly from these |
| `scene_from_decoded` |4x4| One rigid anchor applied after converting decoded translations to scene units |
| `head_to_camera_meters` |4x4| Train-only head/camera rigid offset before scene-unit conversion |
| `frame_times_local_s` |97| Local sample times0..4.8s |

UniEgo channels0:270 are30 groups of9: two rotation columns (6D), then local
XYZ translation. Channels270:279 hold the incremental canonical transform
in the same6D+XYZ layout; channels279:283 are continuous predicted foot-contact
features. Use the archived decoder rather than treating those local feature
positions as world coordinates. Replay requires only the already decoded
scene poses; no second normalization or decoding is needed.

The finalized camera NPZ also has `frame_times_global_s`,
`relative_transforms_scene_units` [96,4,4] and `relative_actions_meters` [96,9].
Relative delta[t] is inverse(C[t]) @ C[t+1]. The action layout is translation
XYZ in meters FIRST, followed by rotation columns0 and1 (6D). These actions
are derived from reconstructed head motion, not predicted by the Phase1
camera head. `timeline_unique193.npz` collects both segments without the
duplicate boundary sample; the matching JSON maps every sample back to its
source segment/frame and global timestamp.

## Coordinate transforms

All matrices are nested ROW-major4x4 arrays with COLUMN-vector multiplication:
`p_world = T_world_local @ [p_local,1]`. Habitat scene is right-handed,+Y up.
Position units are scene units; `manifest.scene_units_per_meter` gives scale.
Do not negateX or apply the Matplotlib visualizer's axis remapping.
If your graphics API expects a column-major flat matrix buffer, transpose the
serialized row-major layout during import. For a hierarchical skeleton,
derive each child local matrix as inverse(parent world) @ child world;
assigning these global matrices as local bone matrices would transform twice.

Optical camera local axes are +X right,+Y down,+Z forward. For Habitat's GL
camera (right/up/backward), use exactly:

```python
T_world_GL = T_world_optical @ diag(1, -1, -1, 1)
```

These poses already refer to the upright RGB optical camera. Do not apply
another Aria-to-Kimodo basis rotation or device-to-RGB extrinsic.

The generated UniEgo features are decoded by the evaluated torch decoder.
`decoded_joint_transforms` retains that original float32 output. Only replay
rotation matrices are projected to SO(3), compensating the decoder's small
epsilon normalization error. Joint positions and raw features are unchanged.
`inference.json` records the maximum rotation projection correction.

LetH_t be the reconstructed Head pose, X the train Head-to-optical transform,
andC_start the accepted simulator camera. After scaling translations into
scene units, a single constant rigid anchor is applied to the WHOLE body:

```python
A = C_start @ inverse(H_0 @ X)
T_scene_joint[t,j] = A @ T_decoded_joint[t,j]
C_scene[t] = T_scene_joint[t,6] @ X
```

This preserves the first camera pose exactly and attaches the camera to the
moving Head throughout the clip. It is not camera inverse dynamics, yaw-only
alignment, floor snapping, Sim3 fitting or collision correction. Full pose
anchoring can tilt the model's decoded up axis if its initial head orientation
disagrees with the rendered viewpoint; that angle is saved explicitly.
The calibration was fitted for relative head/camera motion, not a guarantee
of exact subject-specific camera mounting or floor registration.

## Synchronize and stitch

Each segment has97 samples at20Hz, local times0..4.8seconds. Segmenti has
global time `i*4.8 + frame/20`. The final camera of one segment equals the
first camera of the next, and the next conditioning image is a fresh GS
render there, not the previous generated video's last frame.

For unique-time playback, keep all97 frames from segment0 and frames1..96
from segment1:193 samples covering0..9.6seconds. The presentation MP4 instead
retains both green conditioning frames and has194 frames; it intentionally
duplicates the boundary time. Use JSON timestamps, not concatenated video
frame index, as the canonical animation clock.

Bodies are inferred independently from each generated video+caption. No first
motion-pose inpainting was added. The camera is continuous at the boundary;
the reconstructed body can jump. Preserve raw segment data and label any demo
interpolation as a separate presentation option.

## Rebuild the scene replay

Use the archived application source or this repo with the Habitat environment:

```bash
python -m applications.habitat_gs.human_replay \
  --config runtime_config.json --segment /absolute/bundle/segment_000
```

After moving a bundle, update `runtime_config.json` scene/navmesh/camera/
skeleton paths to the local `assets/` files. The replay JSON and top-level
manifest use relative references for browser consumption. This renderer does
not invoke any model. To build a browser demo, bind a shared timestamp to the
generated-video player, skeleton world transforms, camera glyph and GS camera.
Render a third-person observer independently of the animated model camera.
Observed/conditioned frame borders are green; generated-video borders are red.
