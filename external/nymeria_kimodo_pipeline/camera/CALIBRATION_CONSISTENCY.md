# NymeriaPlus device-to-camera extrinsic and body-head alignment

Findings for the egocentric-camera world-model work — answers two questions that matter before
converting the whole dataset to the RGB-camera (OpenCV) frame:

1. **Is the device→camera transform consistent across all NymeriaPlus recordings?**
2. **Is the device → camera online calibration stable within a recording?**

Here "device" means the physical Aria device frame. It is not the fitted SOMA-30
`Head` joint frame used by the UniEgo motion representation. The latter is a separate,
approximate relation discussed below; treating those two frames as identical is incorrect.

Checker script: [`../../cosmos_motion_ft/nymeria_world/check_calib_consistency.py`](../../cosmos_motion_ft/nymeria_world/check_calib_consistency.py)
(run in the `nymeria_plus` env). Raw output: `/weka/jungbin/cosmos_motion_ft_runs/nymeria_calib_check/`.

---

## TL;DR (full 736-recording scan)

- The **device→RGB-camera extrinsic is near-constant across the whole dataset**: all 736 recordings
  agree to within **~1.6° rotation and <0.7 mm translation**. Optical tilt (device-z vs cam-z) is a
  tight **continuous** spread **38.05°–39.78°** (mean **38.75°**, std **0.28°**) — not a few discrete
  units, just small per-recording variation centred on ~38.8°.
- The extrinsic is **constant within a recording** (online calibration drifts **max 0.12° / sub-mm**
  over a full ~12-min sequence).
- **Per-subject it varies only ~0.25°** (within-subject std 0.21–0.34° across that subject's
  recordings) — i.e. essentially one device per participant, with tiny per-session online-calib jitter.

**Practical implication:** for the camera-action pipeline, extract `T_device_rgb` **per sequence**
from that recording's VRS (cheap, exact). A single global average would introduce at most ~1.6° error
(negligible), but per-sequence is exact and trivial. **No time-varying / per-frame extrinsic correction
is needed.**

## 2026-07-21 correction: source alignment is usually shared, but not uniformly clean

The static **device→RGB extrinsic** findings above remain valid. A separate end-to-end audit of the
stored proportional motion, UniEgo, raw MPS trajectory, upright-RGB trajectory, source SMPL, and
video sidecars found two additional facts that must not be collapsed into extrinsic calibration:

1. **Normal position/timing alignment is real.** Across all 728 camera-bearing sequences, all stored
   timestamps match exactly, every ±5-frame step-lag search selects lag 0, median MPS sampling error
   is ~0.246 ms, and the per-sequence median distance from decoded SOMA Head to RGB-camera origin is
   tightly centered at 0.139 m. The clean streams normally share the same metric world frame.
2. **There are both sparse source jumps and broad Head-orientation label noise.** Forty-two sequences
   contain an interval with direct Head-camera separation >0.5 m, and 47 trip the conservative
   >=0.25 m or >=30° camera gate in one 20-FPS frame. Only 12 have >=0.25 m translation jumps; 39
   trip the angular component, which can include genuine fast turns. The worst camera jumps are
   1.5–3.34 m / up to 168.8°. Separately,
   a fixed SOMA-Head→camera rotation has 15.06° mean residual over full sequences and 8.80° mean even
   inside non-overlapping 97-frame windows; horizontal-heading residuals are 13.05° and 7.81°.

The orientation problem is upstream of the proportional converter: on all 71 held-out sequences,
raw source-SMPL Head→camera residual is 14.53°, converted SOMA Head→camera is 14.42°, and SMPL→SOMA
Head disagreement is only 0.923° mean. One held-out source-jump example
`S17/20230918_s0_kevin_shaw_act2_5g4k0z` has a 1.644 m camera jump and simultaneous 0.935 m raw-SMPL
Head jump at transition 1519→1520 even though MPS reports quality 1.0 and sub-ms sampling error.

The unfiltered Phase-1-style T97 index has 722/119,632 affected train windows (0.6035%) in the
conservative camera-or-motion union, but only 95 (0.0794%) contain a bad camera step relevant to
Phase-1 camera targets (40 translation, 74 rotation, with overlap). After the floor-quality filter
already used by Phase 2/3, 539/112,937 train windows remain affected (0.4773%): 89 camera-gate
windows (38 translation/70 rotation), 457 motion-Head-gate windows (70 translation/452 rotation),
84 direct cross-modal step disagreements, and 93 >0.5 m separation windows. Existing floor
filtering therefore helps but is not a complete source-continuity gate. The strict
translation/cross-modal/separation union is smaller: 120/112,937 train windows (0.1063%); the
remaining 419 conservative exclusions are rotation-gate-only.
None of the currently selected motion-clean71/replacement-five benchmark windows intersects these
catastrophic masks. Audit code and complete affected-window lists live in
`cosmos_motion_ft/motion_expert_joint_attention/{audit_nymeria_camera_motion.py,summarize_nymeria_alignment_quality.py}`;
artifacts are under `/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/`.
Future training manifests should exclude the appropriate Phase-1 or Phase-2/3 versioned rows. Do
not interpret MPS `quality_score` as a continuity guarantee.

---

## Q1 — across recordings (consistency of the matrix)

Source: factory `T_device_sensor("camera-rgb")` from each recording's VRS device calibration.
**ALL 741 recordings (736 readable, 5 skipped), all 19 subjects** (`run_full.log`):

```
optical tilt (device-z vs cam-z): mean 38.75°  std 0.28°  min 38.05°  max 39.78°
rotation angle vs reference     : mean 0.83°   max 1.64°
translation diff vs reference   : mean 0.26 mm max 0.68 mm
tilt histogram (0.1° bins)      : 38.1°:6  38.4°:248  38.8°:195  38.9°:113  39.1°:172  39.8°:2
```

Per-subject mean tilt ± std (n recordings): all 19 subjects land in **38.62°–38.92°** with
**std 0.21–0.34°**, e.g. S01 38.92±0.23 (n=29), S02 38.68±0.25 (n=98), S07 38.74±0.27 (n=90),
S12 38.75±0.29 (n=86), S16 38.68±0.29 (n=79). So within a subject the extrinsic varies only ~0.25°
(per-session online-calib jitter), and subjects barely differ from each other (<0.3° between means).

**Conclusion:** consistent to ~1.6°/<0.7 mm across the entire dataset; a tight continuous spread, not
discrete device clusters. (`get_serial_number()` is not exposed by this VRS, so devices can't be
grouped by serial — but the per-subject tightness implies one Aria unit per participant.)
Note: the earlier 3-per-subject quick sample looked like two discrete clusters (38.4°/39.1°) only
because it hit same-session recordings; the full scan shows the true continuous spread.

## Q2 — within a recording (device→camera stability over time)

MPS provides per-frame **online calibration**:
`recording_head/mps/slam/online_calibration.jsonl` (`CameraCalibrations[].T_Device_Camera` for
camera-rgb / slam-left / slam-right). For one head recording (14,753 timestamps over the full clip):

```
camera-rgb  T_Device_Camera  translation drift : std 0.04–0.14 mm,  peak-to-peak < 0.4 mm
camera-rgb  T_Device_Camera  rotation drift     : max 0.12°,  mean 0.07°   (vs t0)
```

Matches the factory extrinsic (`T ≈ [-0.0037, -0.0124, -0.0047] m`, ~39° optical tilt, 1.4 cm lever
arm). **The device→camera transform is static within a sequence** — online calibration changes it
negligibly. So the Aria device SLAM trajectory `T_world_device` + a single per-sequence
`T_device_rgb` gives a consistent RGB-camera trajectory `T_world_rgb = T_world_device · T_device_rgb`.

This result does not establish a fixed transform from the fitted SOMA `Head` joint to the
device. SOMA head orientation is inferred from the body fit and varies relative to the worn
device because of fit noise and actual head/device motion.

---

## How this feeds the pipeline

- Camera npz (`extract_camera_trajectory.py`) stores **raw `T_world_device`** (Aria Z-up world).
- For the Cosmos camera-action / world-model work we want the **RGB optical frame** (OpenCV):
  `T_world_rgb = T_world_device · T_device_rgb`, then a **−90° optical-axis rotation** to match the
  upright training video (we rotate raw RGB −90°). See
  [`../../cosmos_motion_ft/nymeria_world/extract_camera_opencv.py`](../../cosmos_motion_ft/nymeria_world/extract_camera_opencv.py)
  and `prep_fd_cosmos_frame.py`.
- Because the extrinsic is per-sequence-static and globally near-constant, that extraction is exact and
  cheap: one VRS open per sequence to read `get_transform_device_sensor("camera-rgb")`.
- Reminder: the device→camera transform is a **rotation+lever arm**. The rotational lever-arm term
  can change each relative action's translation as well as its direction and rotation. It does not
  explain the much larger pretrained-camera action scale mismatch, which remains primarily a
  temporal-step/training-distribution issue (see `nymeria_world/README.md` §4).

## SOMA Head → upright RGB camera for Phase 3

The modality-bridge experiment in `cosmos_motion_ft/motion_expert_joint_attention` needs a relation
between UniEgo's SOMA `Head` joint (joint 6) and `cam_action_upright_k1`. It therefore estimates a
separate robust global rigid approximation from the **642 training sequences only**:

`cosmos_motion_ft/motion_expert_joint_attention/head_camera_calibration_train.json`

The rotation is fitted from world orientations after the Aria Z-up → Kimodo Y-up basis change. The
camera-origin lever arm is fitted from synchronized relative actions. Absolute translations are not
used by this historical production fit because source discontinuity intervals had not yet been
filtered. The later source audit shows that clean intervals normally do share a metric origin and a
stable lever, while 42 sequences contain gross registration intervals. For `X = T_head_camera`, the
relative mapping is:

```
R_camera = R_X^T R_head R_X
t_camera = R_X^T (t_head + (R_head - I) r_X)
```

Train fit over 61,632 frame transitions has median/mean translation error 1.54/3.09 mm and
median/mean rotation error 1.44/1.58 degrees. On all 71 held-out sequences, after applying the same
floor and `|z|` validity guards as training, 11,892 T97 windows have median/mean translation error
2.61/3.57 mm and median/mean rotation error 1.28/1.38 degrees. The learned lever arm improves the
held-out median translation error from 4.80 mm. These numbers validate an approximate **relative**
mapping; they do not make the absolute SOMA head and camera trajectories interchangeable.

### Test-actor GT oracle (diagnostic only)

To measure whether a fixed actor-specific SOMA-Head→camera transform explains the remaining error,
`cosmos_motion_ft/motion_expert_joint_attention/estimate_test_actor_head_camera_calibration.py`
fits one six-parameter rigid transform for each of the 17 actors represented in the motion-clean71
benchmark. The fit uses the exact 71 held-out GT motion/camera windows and minimizes the normalized
Phase-3 relative translation/rotation objective. It is therefore deliberately test-label-derived
and in-sample: it is not a deployable calibration or a leakage-free model metric.

On GT motion, the train-global calibration gives 4.445 mm translation and 1.235° rotation mean
error; the test-actor oracle gives 4.056 mm and 1.205°. The 8.76%/2.46% reductions show that actor
identity explains only a modest part of the residual. Applying the GT-fitted actor transforms to
the Phase-3 V2M predictions changes 6.394 mm/1.281° to 6.709 mm/1.259°, so the predicted-motion
translation error is not fixed by actor calibration. The full per-actor table, metric definitions,
leakage contract, artifact path, and backfill provenance are canonical in
`cosmos_motion_ft/AGENTS_ALL.md` under "Bridge Hypotheses and Representation Risks."
