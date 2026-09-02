# Camera-aligned Head re-canonicalization (`camhead_v1`)

## Status

The versioned corpus is built and validated. It covers every train and held-out test
sequence, and it does not overwrite the historical UniEgo corpus.

| Artifact | Path |
|---|---|
| Original corpus | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/uniego_rep` |
| Camera-aligned corpus | `/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1` |
| Build manifest | `uniego_rep_camhead_v1/camera_head_recanonicalization_manifest.json` |
| Historical train-only relative calibration | `motion_expert_joint_attention/head_camera_calibration_train.json` |
| Matching statistics | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats` |
| Quantitative report | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/quantitative/comparison_report.json` |
| Qualitative gallery | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/qualitative/gallery_manifest.json` |
| Standard-SOMA mesh gallery | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/qualitative_soma_mesh/gallery_manifest.json` |
| Experimental absolute-lever report | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/absolute_lever_refit/absolute_lever_report.json` |
| Experimental train-only absolute lever | `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/absolute_lever_refit/head_camera_calibration_train_absolute_lever_v1.json` |

Build coverage is 728 converted sequences and 13,553,128 frames (15,669,209,032
bytes): all 642 train and all 71 test sequences are present. Four auxiliary source
sequences have no upright RGB-camera sidecar and are intentionally absent; none belongs
to train or test.

## Transform contract

The RGB camera and SOMA Head have different local axis conventions, so their rotation
matrices must not be copied directly. The rigid relation is

```text
T_world_camera = T_world_head @ T_head_camera
```

The camera sidecar is in Aria's Z-up world basis. With

```text
B_aria_to_kimodo = [[1, 0,  0],
                     [0, 0,  1],
                     [0,-1,  0]]
```

the corrected Head rotation is

```text
R_world_head_new = (B_aria_to_kimodo @ R_world_camera_upright) @ R_head_camera.T
```

`R_head_camera` was estimated using only the 642 training sequences. The measured
camera rotation for each sequence is then used as the preprocessing target. Consequently,
the near-zero post-correction rotation error is a construction/integrity check, not a
claim that a learned model generalizes with zero error.

UniEgo is Head-canonical. Overwriting only the six Head channels would corrupt the
meaning of every other joint. The converter instead:

1. decodes the complete sequence to Kimodo Y-up world transforms;
2. replaces only SOMA Head joint 6's world rotation;
3. derives a new yaw-only canonical frame from the corrected Head; and
4. re-encodes all unchanged world transforms in that frame.

All world joint positions, all non-Head world rotations, foot-contact labels,
timestamps, neutral joints, and other NPZ members are preserved. Only float32
re-encoding roundoff remains.

## Full-corpus results

The exhaustive comparison uses all 728 camera-covered sequences, 13,553,128 frames,
and 13,552,400 transitions. Exact means use every value; report percentiles use a
deterministic every-20th sample.

### Camera alignment

| Metric | Original | `camhead_v1` |
|---|---:|---:|
| Absolute camera rotation error | 20.128875 deg | 0.002040 deg |
| Relative camera-action rotation error | 1.460197 deg | 0 deg |
| Head/camera angular-step magnitude mismatch | 0.819550 deg | 0.000252 deg |
| Relative camera-action translation error | 0.004354 m | 0.002608 m |
| Absolute camera-origin translation error | 0.015463 m | 0.036794 m |

The last row is an important limitation. This version corrects rotation while keeping
the fitted Head position fixed. Rotating the calibrated Head-to-camera lever therefore
increases absolute camera-origin error. Relative translation improves, but absolute
translation is not solved by `camhead_v1`; correcting it would require a separate Head
position/skeleton-fitting decision.

### Absolute camera-origin lever refit experiment (2026-08-31)

This experiment tested the narrowest possible follow-up: retain the corrected
`camhead_v1` Head rotation and unchanged Head position, but refit the constant lever
`r` in

```text
p_world_camera = p_world_head + R_world_head_camhead_v1 @ r
```

The experiment is calibration-only. It does not rewrite the motion corpus, camera
sidecars, statistics, contact labels, joint positions, feet, or rotations. The
historical calibration remains unchanged.

#### Population and leakage contract

`estimate_camera_head_absolute_lever.py` reconstructs the exact
`camera_motion_quality_filter_v1_T97.json` population. Duplicate caption rows for one
physical window are unioned so overlapping windows do not count a frame twice.

- Fit: all 642 training sequences, 10,614,277 clean union frames. The robust global
  lever uses a deterministic every-20th sample, or 531,028 training frames.
- Held-out evaluation: all 71 test sequences, 1,156,017 clean union frames and
  1,144,376 clean consecutive-frame transitions.
- The selected global lever uses no test labels. Per-actor candidates are also fit
  only on training sequences for that actor and are report-only.
- The per-test-sequence lever uses the same sequence's camera positions. It is an
  explicitly leaky diagnostic floor, never a deployable calibration or model-selection
  result.

The selected train-frame geometric median is

```text
historical relative lever  [-0.0114628803,  0.0554012135, 0.1139408424] m
absolute-fit lever         [-0.0221435396,  0.0547012496, 0.1222988217] m
difference                 [-0.0106806593, -0.0006999639, 0.0083579793] m
```

The change has norm 1.358 cm. The exact train-frame mean and the frame- and
sequence-geometric medians agree closely; all candidates and convergence metadata are
in the report.

#### Same-population held-out result

Absolute and rotation values are frame-weighted; relative translation is
transition-weighted. `Original` means the historical UniEgo Head and historical lever.
All corrected rows use `camhead_v1` Head rotation.

| Representation and lever | Absolute optical-center error | Relative translation error | Camera rotation error |
|---|---:|---:|---:|
| Original Head + historical relative lever | 1.4437 cm | 3.5762 mm | 20.1067 deg |
| Corrected Head + historical relative lever | 3.7000 cm | **2.4304 mm** | **0.002043 deg** |
| Corrected Head + selected train-global absolute lever | **3.4082 cm** | 2.4531 mm | **0.002043 deg** |
| Corrected Head + train-actor absolute lever | 3.3072 cm | 2.4523 mm | **0.002043 deg** |
| Corrected Head + test-sequence oracle (leaky) | 2.5710 cm | 2.4240 mm | **0.002043 deg** |

The train-global absolute lever improves corrected absolute error by 2.918 mm, or
7.89%, while worsening relative translation by only 0.0227 mm, or 0.93%. Rotation is
unchanged because the lever does not affect orientation. This is a real but modest
improvement: 3.408 cm remains 2.36 times the original representation's 1.444 cm.

The remaining error is not primarily a wrong global lever. A train-actor lever reaches
only 3.307 cm, and even the test-label-leaking per-sequence oracle reaches only
2.571 cm, still 1.78 times the original absolute error. Train sequence lever centers
are also dispersed around the selected value: median distance 1.765 cm and p90
3.398 cm. Therefore the corrected camera-aligned orientation and the fitted Head-joint
position do not support one constant rigid optical-center offset. A global lever can
remove bias, but it cannot remove the pose/session-dependent residual.

There is no single winning representation in this table. The original Head preserves
absolute camera-origin position better but has a roughly 20-degree orientation error
and worse relative translation. `camhead_v1` fixes the cross-modal rotation target and
improves relative translation, but its Head position is not an equally accurate anchor
for absolute camera origin after the rotation replacement.

#### Qualitative and motion-preservation result

The new trajectory sheet overlays measured camera, original Head + historical lever,
corrected Head + historical lever, and corrected Head + absolute-fit lever on the same
12 diverse held-out cases used by the inspected gallery. The refit improves all eight
automatically selected normal cases, but the gain varies and three known difficult
diagnostics worsen; the source-jump negative control remains essentially unchanged.
This is consistent with the aggregate result: a global shift helps average bias but
does not solve local/session-dependent position mismatch.

The existing human-motion keyframes and 12 H.264 videos remain the qualitative motion
check. A lever calibration is used only to draw or derive a virtual camera origin; it
cannot change the stored human motion. Consequently the exhaustive skating, floating,
penetration, joint-position, and foot-contact preservation results below remain exactly
applicable.

Artifacts:

- `absolute_lever_refit/absolute_lever_report.json`: complete fit/evaluation contract,
  exact metrics, per-artifact hashes, and links.
- `absolute_lever_refit/head_camera_calibration_train_absolute_lever_v1.json`:
  experimental opt-in calibration; not for historical checkpoints.
- `absolute_lever_refit/per_sequence_metrics.jsonl`: all 713 sequence rows.
- `absolute_lever_refit/diverse_absolute_lever_trajectory_comparison.png`: the 12-case
  trajectory overlay.
- `absolute_lever_refit/heldout_clean_absolute_vs_relative.png`,
  `heldout_clean_absolute_translation_cdf.png`, and `sequence_lever_centers.png`:
  aggregate diagnostics.

Recommendation: do not replace `head_camera_calibration_train.json` by default. Keep
the historical lever for existing relative-action checkpoints and keep this absolute
lever as an opt-in diagnostic. If absolute optical-center position becomes a training
target, the next experiment should model or repair the positional anchor itself—for
example a train-only pose-dependent offset using Head position plus neck-to-Head and
upper-torso geometry—while independently checking held-out absolute pose, relative
actions, and whole-body preservation. A neck-to-Head vector can inform position, but
does not by itself define the missing full 3-D orientation.

### Whole-body preservation

| Check | Maximum old/new decoded difference |
|---|---:|
| Any joint position | 0.046752 mm |
| Any foot position | 0.046664 mm |
| Foot height | `9.429e-12` m |
| Any bone length | `1.256e-7` m |
| Any non-Head rotation | `4.353e-6` deg |
| Corrected Head vs target | 0.022378 deg |

There are no metadata or contact-label preservation failures.

### Foot skating, floating, and floor interaction

These metrics use 133,702 usable caption windows after the same floor-calibration drop
map and offsets as the training loader. The skating definitions match the Kimodo
evaluation metrics: four-foot 3D speed under stored contacts, toe speed below 5 cm,
and the fraction of below-5-cm toe transitions faster than 0.2 m/s.

| Metric | Original | `camhead_v1` | Change |
|---|---:|---:|---:|
| Contact-foot skate | 0.051786182101 m/s | 0.051786182124 m/s | `+2.29e-11` m/s |
| Height-based toe skate | 0.146423370509 m/s | 0.146423370427 m/s | `-8.21e-11` m/s |
| Foot-skate ratio | 0.118494157525 | 0.118494157525 | 0 |
| Mean contact-foot absolute height | 0.058982344180 m | 0.058982344179 m | `-6.97e-13` m |
| Contact-labelled floating above 10 cm | 0.122250684246 | 0.122250684246 | 0 |
| Contact penetration below -5 cm | 0.053576035912 | 0.053576035912 | 0 |
| All feet above 15 cm (descriptive) | 0.011998780087 | 0.011998780087 | 0 |

The conversion therefore neither introduces nor removes skating, floating, or
penetration. These non-zero original values are existing source-data quality issues,
not endorsements of their quality. In particular, the rendered source-jump negative
control retains its original trajectory discontinuity and floor penetration exactly.

Plots:

- `quantitative/aggregate_old_vs_new_cdf.png`
- `quantitative/preservation_checks.png`
- `quantitative/physical_motion_old_vs_new.png`

## Qualitative review

The gallery ranked 11,702 caption-aligned windows from all 71 held-out sequences and
selected eight distinct automatic cases: straight locomotion, high angular turning,
long travel, high articulation, high contact-skate stress, worst clean rotation,
low motion, and a typical median case. Four known diagnostics were added, including a
source-jump negative control.

The output contains two inspected contact sheets and twelve playable H.264/yuv420p
videos at 1400x900 and 10 fps. Every video shows the human skeleton, measured and
Head-implied camera axes/trajectories, rotation error, calibrated minimum foot height,
and contact-foot speed. Old and new skeleton/foot traces visually overlap, while the
camera-facing axis aligns after correction. Existing translation and floor defects
remain visible rather than being hidden.

### Standard SOMA mesh impact gallery (2026-08-31)

The second gallery renders the same twelve cases with Kimodo's restored standard
SOMA visualization surface:

`/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/qualitative_soma_mesh/`

This is a controlled common-skin comparison, not a subject-identity reconstruction.
The restored asset has 18,056 vertices and 36,108 faces. Subject-specific SOMA-X
geometry is not present on this server. For interactive-rate MP4 rendering, the script
vertex-clusters the standard surface at 1.2 cm while averaging and renormalizing LBS
weights, producing 6,706 vertices and 13,576 faces. The source asset is unchanged.
Its SHA-256 is
`90ee2cdf50f168382a7dd7e9c88b118298aba0aca4b72022080c89c4bab0ceb2`.

The skinning contract mirrors `kimodo.viz.soma_skin.SOMASkin`: convert UniEgo's 30
global rotations to local rotations, expand by joint name to SOMA-77 with the standard
relaxed-hand rotations, run FK on the standard SOMA skeleton using the UniEgo pelvis
translation, then apply the standard linear-blend-skinning rig. It is implemented
narrowly in `visualize_camera_head_aligned_soma_mesh.py` because importing the full
restored Kimodo package also initializes unrelated model/Transformer Engine modules
whose binary requires a newer glibc than this compute node.

Each 1920x1080 video contains synchronized original and `camhead_v1` full-body meshes,
two head/neck close-ups, measured-versus-Head-implied camera forward/up axes, full
SO(3) error, world trajectory, derived surface displacement, and foot/floor traces.
Orange is original, green is `camhead_v1`, and blue is the measured camera. Thick
arrows show camera +Z and thin arrows show camera +Y, so roll is visible as well as
heading. Peak-impact posters, a 12-case poster sheet, and an aggregate metric figure
are included next to the videos.

Across the exact 1,217 source frames in the gallery:

| Check | Result |
|---|---:|
| Frame-weighted mean camera SO(3) error | 32.726843 deg -> 0.002231 deg |
| Mean Head rotation replacement, frame-weighted | 32.726846 deg |
| Case-mean whole-surface displacement on the standard skin | 5.8438 mm |
| Case-mean p95 displacement among Head-subtree-weighted vertices | 73.8588 mm |
| Per-case Head-surface p95 range | 29.8295-148.2190 mm |
| Largest derived surface displacement | 228.7014 mm (source-jump negative control) |
| Largest decoded joint-position change | 0.003328 mm |
| Largest decoded non-Head world-rotation change | `2.700e-6` deg |
| Largest non-Head-weighted standard-surface change | 0.031471 mm |
| Largest old/new minimum-foot-height difference | `5.436e-10` mm |
| Largest absolute contact-skate change | `1.711e-5` cm/s |

The centimeter-scale surface response is expected and is the visual impact being
requested: rotating a skinning joint moves vertices weighted to that joint and its
small Head subtree. It does **not** mean `camhead_v1` changed the decoded joint-position
channels. Those remain equal to floating-point reconstruction tolerance, as shown
separately in the table. Likewise, the body away from Head and the feet remain
unchanged. Existing below-floor feet in some source cases and the known source jump
remain visible in both panels; the correction neither hides nor repairs them.

All twelve outputs were decoded end to end after rendering. Eleven T97 videos contain
49 frames and last 4.9 seconds; the T150 negative control contains 76 frames and lasts
7.6 seconds. Every file is H.264, progressive yuv420p, 1920x1080, and 10 fps. The
machine-readable per-case and aggregate values are in
`qualitative_soma_mesh/gallery_manifest.json`.

The canonical render was Slurm job `527067`: exit code 0, 1 minute 14 seconds
elapsed, eight allocated CPUs, and about 3.24 GiB peak step RSS. Rendering is
CPU-bound; no GPU was requested.

## Reproduction

Run CPU-heavy commands through Slurm in `tmux 0` on this server:

```bash
cd /home/jungbinc/cosmos_motion_ft
source restored_env.sh

srun --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --time=04:00:00 bash -lc '
    cd /home/jungbinc/cosmos_motion_ft
    source restored_env.sh
    "$COSMOS_PYTHON" motion_expert_joint_attention/build_camera_head_aligned_uniego.py \
      --workers 8'

srun --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --time=04:00:00 bash -lc '
    cd /home/jungbinc/cosmos_motion_ft
    source restored_env.sh
    "$COSMOS_PYTHON" motion_expert_joint_attention/compare_camera_head_aligned_uniego.py \
      --workers 8 --sample-stride 20'

srun --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G \
  --time=02:00:00 bash -lc '
    cd /home/jungbinc/cosmos_motion_ft
    source restored_env.sh
    "$COSMOS_PYTHON" motion_expert_joint_attention/audit_motion_stats.py \
      --uniego-root /mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1 \
      --out-dir /mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats \
      --max-old-z 0'

srun --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --time=02:00:00 bash -lc '
    cd /home/jungbinc/cosmos_motion_ft
    source restored_env.sh
    "$COSMOS_PYTHON" motion_expert_joint_attention/visualize_camera_head_aligned_uniego.py \
      --render-workers 4 --frame-stride 2'

srun --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --time=01:00:00 --job-name=camhead-soma-viz bash -lc '
    cd /home/jungbinc/cosmos_motion_ft
    source restored_env.sh
    "$COSMOS_PYTHON" \
      motion_expert_joint_attention/visualize_camera_head_aligned_soma_mesh.py \
      --render-workers 4 --frame-stride 2'

srun --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --time=04:00:00 --job-name=camhead-abslever bash -lc '
    cd /home/jungbinc/cosmos_motion_ft
    source restored_env.sh
    "$COSMOS_PYTHON" \
      motion_expert_joint_attention/estimate_camera_head_absolute_lever.py \
      --workers 8 --fit-stride 20 --percentile-stride 20 --overwrite'
```

The comparator writes resumable per-sequence shards. Do not pass `--overwrite-shards`
unless the metric schema or source data changed.

The re-canonicalization, absolute-lever worker/metric/robust-fit/preservation/plotting,
and standard-SOMA skinning contracts have thirteen focused tests:

```bash
"$COSMOS_PYTHON" -m unittest -v \
  motion_expert_joint_attention.test_camera_head_absolute_lever \
  motion_expert_joint_attention.test_camera_head_recanonicalization \
  motion_expert_joint_attention.test_camera_head_soma_mesh
```

The final documented absolute-lever run was Slurm job `526812`: completed with exit
code 0 in 30 seconds using eight CPUs; peak step RSS was about 1.03 GiB.

## Opt-in training policy

The historical representation and statistics remain the defaults. For a new model:

```bash
cd /home/jungbinc/cosmos_motion_ft
source restored_env.sh
source motion_expert_joint_attention/use_camera_head_v1.sh
```

This sets `NYMERIA_UNIEGO_ROOT`, `MOTION_STATS_MEAN`, and `MOTION_STATS_STD` together.
Use `--bones_frac 0` initially: BONES has no synchronized egocentric RGB camera and
cannot supply the same Head semantics. Existing checkpoints were trained with the old
representation/statistics and must not be evaluated or resumed under the `camhead_v1`
environment without an explicit migration experiment.

Statistics were computed from 120,929 floor-filtered training windows and 11,888,119
window frames. SHA-256:

```text
mean  4043893ec7ba2004f90dbc614a081e2f383b2798cdc4a39dce4d4ea6d47101c5
std   684e9b354f60f4cd1f8e251a1f8fcf3541d76887b2bcb9570773ce460dd69439
```
