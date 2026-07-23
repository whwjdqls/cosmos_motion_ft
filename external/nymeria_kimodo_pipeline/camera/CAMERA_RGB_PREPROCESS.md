# Camera-motion preprocessing: device → upright-RGB → relative actions

The camera motion that conditions the egocentric **RGB video** must be in the **RGB camera frame the
model actually sees**, not the raw Aria device/head frame. This pipeline does that conversion and saves
the result. Script: [`preprocess_camera_rgb.py`](preprocess_camera_rgb.py) (run in the `nymeria_plus` env).

```
Nymeria SLAM device trajectory  T_world_device
  → RGB camera trajectory        T_world_rgb      = T_world_device · T_device_rgb
  → upright/OpenCV video frame    T_world_upright  = T_world_rgb   · T_rgb_upright
  → relative camera actions       A_t = inv(T_t) · T_{t+k}   ([tx,ty,tz, rot6d])
```

> **Rule:** never feed raw `T_world_device` as the camera motion for the RGB video. Convert to
> `T_world_upright_rgb` first, then take relative actions from that.

## Why each step

| step | transform | why |
|---|---|---|
| device → rgb | `T_device_rgb` from VRS `get_transform_device_sensor("camera-rgb")` | the RGB sensor is mounted ~39° tilted + 1.4 cm off the device frame; static per sequence, ~globally constant (see [CALIBRATION_CONSISTENCY.md](CALIBRATION_CONSISTENCY.md)) |
| rgb → upright | `T_rgb_upright = Rz(−90°)` | the training video is rotated −90° to be upright; the camera frame must match the pixels. Validated: this frame best matches the pretrained model's predicted camera direction (dir-cosine 0.43→0.69) |
| abs → relative | `A_t = inv(T_t)·T_{t+k}`, rot6d, backward_framewise, scales=1 | Cosmos-exact action (`pose_abs_to_rel`); this is what the camera_pose action head consumes |

Time-alignment (SLAM pose ↔ video frame) is already done upstream: `extract_camera_trajectory.py`
samples `T_world_device` at each 20 fps video-frame timecode, so frame `i` ↔ pose `i` ↔ image `i`.

## Output

`/weka/jungbin/nymeriaplus_kimodo_proportional/camera_rgb/{Sxx}/{seq}.npz` — one per sequence, full length:

| key | shape | meaning |
|---|---|---|
| `cam_world_pos_upright` | (T,3) | **final** T_world_upright_rgb translation |
| `cam_world_rot_upright` | (T,3,3) | **final** R_world_upright_rgb (OpenCV optical, upright) |
| `cam_action_upright_k1` | (T−1,9) | **final** per-frame relative action `[tx,ty,tz, rot6d]` |
| `camera_step_translation_m`, `camera_step_rotation_deg` | (T−1,) | source-continuity diagnostics written by the 2026-07-21+ preprocessor |
| `camera_step_implausible` | (T−1,) bool | `translation>=0.25m OR rotation>=30deg` in one 20-FPS step; exclude intersecting train windows |
| `preprocess_version` | scalar | sidecar schema/version; version 2 has corrected upright K and continuity fields |
| `T_device_rgb` | (4,4) | static per-sequence RGB extrinsic |
| `T_rgb_upright` | (4,4) | fixed Rz(−90°) upright rotation |
| `K_rgb_raw`, `K_rgb_upright` | (3,3) | intrinsics before / after the −90° image rotation (FISHEYE624; pinhole K is approximate) |
| `image_size_hw`, `camera_model` | — | 1408×1408, `CameraModelType.FISHEYE624` |
| `cam_world_pos_device`, `cam_world_rot_device` | (T,3),(T,3,3) | raw device trajectory (reference) |
| `timestamps_us`, `fps` | (T,), () | sync with motion/video (20 fps) |

Existing active sidecars were produced before the three diagnostic arrays were added; use
`cosmos_motion_ft/motion_expert_joint_attention/audit_nymeria_camera_motion.py` and its saved
`details_all.jsonl` for their equivalent masks. Regenerating a sidecar adds diagnostics but does not
repair an upstream MPS/body jump.

The model should use **`video_rgb_upright` + `cam_action_upright_*`** — never the device-frame action.

## Source continuity gate

Timestamp equality and MPS `quality_score` do not guarantee a continuous trajectory. The full
2026-07-21 audit found 47/728 camera-bearing sequences with at least one >=0.25 m or >=30° step at
20 FPS. Twelve contain a >=0.25 m translation step and 39 trip the >=30° angular component; the
latter is deliberately conservative and can include genuine fast turns. Four train sequences have
unambiguous 1.54–3.34 m jumps with rotations up to 168.8°. The preprocessor now records and prints
those transitions. A versioned training-manifest build must drop any T97
window intersecting a flagged transition; do not clip or interpolate through it silently. The
complete unfiltered Phase-1 T97 exclusion list is
`/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/affected_aligned_T97.jsonl`;
the list after the existing Phase-2/3 floor filter is
`affected_phase23_aligned_T97_floor_filtered.jsonl` in the same directory.

The 2026-07-21 update also corrected `K_rgb_upright` for the actual Pillow clockwise
`rotate(-90)` mapping: `(u_new,v_new)=(H-1-v_old,u_old)`. This changes only stored diagnostic
intrinsics (unused by current training); `T_rgb_upright=Rz(-90)` and all camera actions were already
correct. Existing version-0/1 sidecars retain the old diagnostic K until regenerated. The current
`--skip_existing` check requires `preprocess_version>=2`, matching timestamps, and an output newer
than its raw input, so an explicit rebuild updates those old sidecars rather than silently skipping
them.

## Action stride (k) — important

`cam_action_upright_k1` is the **per-frame (k=1)** action. At 20 fps the per-frame translation is small
(median ~0.02 m, see `_summary.json`), which is low-SNR and far below the pretrained camera head's
native step (~7–9 frames). Two separate facts (see `cosmos_motion_ft/nymeria_world/README.md` §4):

- The `T_device_rgb` correction fixes the **frame/direction**, *not* the translation-scale gap.
- The active official-compatible Phase-1 run keeps the pretrained Cosmos camera action heads and
  fine-tunes them (at 4x the generator-LoRA learning rate) on raw metric `k=1` actions. It does **not**
  re-initialize those heads. A coarser stride `k=2–4` remains a possible future ablation, derived as
  `A_t = inv(T_t)·T_{t+k}` from the stored absolute `cam_world_*_upright` poses, but it is not the
  contract of the completed Phase-1 checkpoint.

## Motion-head alignment use

The Phase-3 head-camera bridge variant does not feed absolute camera poses to the motion expert. It
uses a train-split rigid approximation to convert UniEgo SOMA-Head **relative** transforms to the same
`cam_action_upright_k1` convention. For motion-to-video, this derived action is a clean condition for
the frozen Phase-1 camera pathway. For video-to-motion, synchronized camera action is a training-only
auxiliary target and is never an input. See `CALIBRATION_CONSISTENCY.md` for the distinction between
the physical device-camera extrinsic and the approximate SOMA-head-camera mapping.

## Verification

Bit-exact against the validated zero-shot baseline frame (`gt_camera_cosmos.npz`): S01/S02 full-sequence
match 0.0; S03 window[200:297] matches 0.0 after frame-offset. So this is the same transform, now
materialized for the whole dataset.
