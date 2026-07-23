# NymeriaPlus shape-aware (proportional) kimodo pipeline

How the per-actor, shape-aware kimodo motion data is produced from NymeriaPlus,
parallel to the BONES-SEED proportional set, for **shape-aware** training
alongside it.

- Source SMPL→SOMA intermediates: `/weka/jungbin/nymeriaplus/{Sxx}/{seq}/body/xdata_soma.npz`
- **Shape-aware output: `/weka/jungbin/nymeriaplus_kimodo_proportional/{Sxx}/{seq}.npz`**
- Shape-*un*aware output (pre-existing, **also affected by the bug below**): `/weka/jungbin/nymeriaplus_kimodo/motions/{Sxx}/{seq}.npz`

Converter: `soma_to_kimodo_proportional.py` (env `soma`). Each `Sxx` mixes many
named actors; identity is **per-sequence** (`identity_coeffs (1,10)`), so neutrals
are stored per sequence.

---

## Why a new converter (the original soma_to_kimodo was broken for kimodo)

The original `soma_to_kimodo_single.py` baked SOMA-X's joint orient via
`apply_joint_orient_local`. Those local rotations are **not** in kimodo's
local-rotation convention, and the capture world is **Z-up** while kimodo is
**Y-up**. Two bugs resulted, found by visual inspection + this measurement
(mean head-above-feet over a clip; a standing body is ~1.2 m up):

| source | head-foot | up axis | verdict |
|---|---|---|---|
| SOMA-X ground truth (real human) | 1.20 m | +Z | standing ✓ (world is Z-up) |
| BONES-SEED proportional, kimodo FK | 1.24 m | +Y | standing ✓ (reference) |
| **old Nymeria conversion, kimodo FK** | **0.13 m** | – | **crumpled ✗** |

So the old uniform+proportional Nymeria kimodo data does not decode to a standing
human under kimodo FK. The BONES-SEED check confirms kimodo FK + the renderer are
correct; the fault is the Nymeria rotation conversion.

---

## Corrected conversion (per sequence)

Run entirely in the `soma` env (kimodo's `global_rots_to_local_rots` is
reimplemented inline so no kimodo import is needed):

1. **True world joint transforms.** Pose the SOMA identity and read the skinning
   `T_world` (Z-up): animated `R_posed[f,j]`, T-pose `R_rest[j]`, and rest
   positions `N[j]`. (Captured by patching `batched_skinning.pose(return_transforms=True)`.)
2. **Rest→posed global rotation:** `G_rel = R_posed @ R_rest^T`.
3. **Up-axis fix Z→Y:** `G_rel_y = R_z2y @ G_rel`, with
   `R_z2y = [[1,0,0],[0,0,1],[0,-1,0]]`  i.e. `(x,y,z)->(x,z,-y)`.
4. **kimodo local rotations:** `local[j] = G_rel_y[parent]^T @ G_rel_y[j]`
   (root: parent=I). This is exactly `kimodo.skeleton.transforms.global_rots_to_local_rots`.
   kimodo FK reconstructs `G_rel_y` and applies it to the rest bones.
5. **neutral_joints** = `N` (rest positions, pelvis-centered, Y-up). Exact actor
   bone lengths; for the uniform body it matches kimodo's canonical `joints.p`.
6. **root_positions** = `R_z2y @ hips_world_pos`.
7. **Height / grounding.** The capture floor sits at an arbitrary world height
   (e.g. feet ~1.8 m below y=0 for one actor). `floor_offset` = whole-sequence
   min foot/toe height (Y-up) is computed and **always saved in the NPZ**.
   - **Current stored data = `--no-ground` = raw rotated SLAM-world height**
     (`grounded=False`); root Y is NOT shifted. To ground downstream:
     `root_positions[:,1] -= floor_offset` (lands the lowest foot at y≈0, the
     BONES-SEED convention). Per-sequence grounding only anchors the *lowest*
     floor in multi-floor captures, so deferring it (and using the proper
     per-sequence floor height from the dataset's `floor/` metadata) is cleaner.
   - With the default `ground=True` the converter subtracts `floor_offset` from
     stored Y (`grounded=True`) — feet rest at y≈0, matching BONES-SEED.
   - **Preferred grounding = GT floor per atomic-action slice** (not the min-foot
     `floor_offset`, which mis-aligns sitting/lying). See
     `floor/SLICE_FLOOR_README.md`:
     `nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action_floor.jsonl` gives a
     per-slice `ground_offset_y`; ground with `root_positions[:,1] -= ground_offset_y`.
     (Metadata lives under the proportional tree as of 2026-06-15; the old
     `nymeriaplus_kimodo/metadata/` was deleted — it covered only 7 subjects. All 19
     subjects / 732 sequences are now covered.)

**Validation (asserted per sequence, activity-independent):** kimodo FK of the
result must reproduce the SOMA true joint positions (the `T_world` translations)
rotated Z→Y and grounded, to `< 0.05 m`. Typical `geo_err ≈ 2–3e-6 m` (fp32
floor). This is robust to what the actor does — standing, sitting, leaning, or
floor work all pass, because it checks geometric fidelity, not an "is-standing"
heuristic. (An earlier mean-head-above-feet gate wrongly rejected 16/732
bent-over / floor-activity sequences even though their conversion was perfect;
the geometric check passes all 732.) Confirmed: corrected jeffery_bryant renders
as a standing/walking human, feet on the floor, real-time 20 fps; FK head-foot
1.22 m, matching the SOMA-X ground truth.

### Source-quality limitation discovered 2026-07-21

The FK round-trip assertion proves that this converter preserves its SOMA input; it does **not**
prove that every source SMPL/SOMA frame is physically continuous or rigidly aligned to head-camera
MPS. The full camera/motion audit found >=0.25 m SOMA-Head translation steps in 51/728
camera-bearing sequences, while 227 trip a conservative >=30°/frame Head-rotation gate; the
angular count can include genuine fast turns and is not by itself proof of a coordinate reset.
Gross direct Head-camera separation intervals occur in 42 sequences. The severe cases are already
present in the source fit: on the 71-sequence test split, source-SMPL Head→camera rotational residual is 14.53°,
converted SOMA Head→camera is 14.42°, and SMPL→SOMA differs by only 0.923° mean. The conversion is
faithful, but it faithfully preserves upstream Head-orientation noise and occasional jumps.

At the T97 level, the unfiltered Phase-1-style index has 722/119,632 affected train windows
(0.6035%). The existing Phase-2/3 floor filter removes some of them, but 539/112,937 aligned train
windows remain affected (0.4773%), including 70 with >=0.25 m Head translations and 452 with
>=30° Head rotations (with overlap). The stricter translation/cross-modal/separation union contains
120 windows (0.1063%); 419 are rotation-gate-only. Rotation jumps are bounded
in the 6D feature representation and may not trip the downstream `|z|max` guard. Future
motion/bridge training must consume
`/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/affected_phase23_aligned_T97_floor_filtered.jsonl`
instead of assuming floor filtering or geometric round-trip validation is a complete data-quality
filter. The unfiltered Phase-1 list remains `affected_aligned_T97.jsonl` in the same directory.

> Note on viz: render **consecutive** frames at `fps=20` (subsampling across the
> whole clip plays ~100× too fast). The data is already grounded, so no
> render-side floor shift is needed. See `_verify_render.py`.

### Why not the simpler candidates (rejected)
- *Raw SOMA rest joints as neutrals + old local_rot_mats:* crumples (above).
- *orient^T frame-walk neutrals:* self-consistent only with a bare FK; 1.0 m off
  kimodo's canonical buffer; wrong for the real `SOMASkeleton` (which bakes
  orientation via `rest_local_rots`).
- *Matching `SOMALayer.pose()` joint positions as the oracle:* wrong — that path
  uses LBS + bind-reposing, not rigid parent-local FK; disagrees by >1 m.

---

## NPZ schema

| key | shape / dtype | notes |
|---|---|---|
| `local_rot_mats` | f32 `(T,77,3,3)` | kimodo joint-parent-local rotations (FK-decodable, Y-up) |
| `root_positions` | f32 `(T,3)` | world meters, Y-up; xz absolute (world), Y = raw unless `grounded` |
| `floor_offset` | f32 `()` | whole-seq min foot/toe Y; subtract from `root_positions[:,1]` to ground |
| `grounded` | bool `()` | whether `floor_offset` was already subtracted from stored Y (current data: False) |
| `timestamps_us` | i64 `(T,)` | 20 fps grid (source is 240 Hz; timestamp-aware subsample) |
| `fps` | i64 `()` | 20 |
| `source_seq`, `source_subject` | str | provenance |
| `neutral_joints` | f32 `(77,3)` | actor rest skeleton, kimodo Y-up frame |
| `identity_coeffs` | f32 `(1,10)` | SMPL betas, provenance |

30-joint slice: `idx30 = [SOMASkeleton77().bone_order_names.index(n) for n in SOMASkeleton30().bone_order_names]`.

---

## Running it

Env `soma`, **on a compute node, not login**:
```bash
SOMA_PY=/home/jungbin_cho/miniforge3/envs/soma/bin/python
for sh in 0 1 2 3; do
  $SOMA_PY soma_to_kimodo_proportional.py --num-shards 4 --shard-id $sh --device cpu --overwrite &
done; wait
# single sequence (debug, skips the slow discovery walk): --single /weka/jungbin/nymeriaplus/Sxx/<seq>
```
Defaults: in `/weka/jungbin/nymeriaplus`, out `/weka/jungbin/nymeriaplus_kimodo_proportional`,
20 fps. 732 SOMA sequences across S01–S20.

Visualize / verify (env `kimodo`):
```bash
PYTHONPATH=/home/jungbin_cho/kimodo_open python _verify_render.py <npz> <start_frame>
```

Helper buffers: `/weka/jungbin/kimodo_caches/_somaskel77_buffers.npz` (kimodo
SOMASkeleton77 joint_parents / canonical neutrals, dumped once from the kimodo env)
— used by the converter for the joint hierarchy.

---

## Visualization

`viz_with_text.py` (env `kimodo`) renders a sequence with the **atomic_action
text** and the **foot-skating value** (seq + window, cm/s) burned on top, root
canonicalized to start at `(x,z)=(0,0)`, per-window grounded, real-time 20 fps.
`_verify_render.py` is the minimal renderer. Always render **consecutive** frames
at `fps=20` (subsampling across the whole clip plays ~100× too fast).

**Root convention.** The stored `root_positions` are **absolute world coords**
(Y-up, grounded) — NOT origin-canonicalized like BONES-SEED (whose `root[0]` xz ≈
0). This is intentional: it preserves the true cross-room trajectory, and the
training dataloader re-canonicalizes **per segment** (`to_canonicalize=True`), so
training parity holds. The viz canonicalizes per window for display. (Verified:
our stored root matches the SOMA source hips trajectory to ~3 cm.)

**"Floating feet" in some clips is real, not a bug.** e.g. marie_vasquez act2
keeps feet >10 cm off the floor for 100% of the clip (footstool / feet-up
activity); the SOMA source shows the same. Multi-floor captures (e.g. kirk on an
upper floor) put feet at ~2 m — also correct (grounding sets the lowest floor to
y=0). Foot skating analysis: see `FOOT_SKATING_ANALYSIS.md` (skating is inherent
to the SMPL source, 4.07 cm/s, matched by our data at 4.05).

## Diagnostic scripts (provenance)

`validate_proportional.py`, `diagnose_frame.py`, `_diag_*.py`, `_proto_fix_*.py`
established the diagnosis and the fix (kept for reference). The decisive checks:
BONES-SEED FK stands (1.24 m) while old Nymeria FK crumples (0.13 m); the
corrected path reproduces the SOMA-X ground-truth standing pose (1.22 m, Y-up).

---

## Using it for shape-aware training

Schema-compatible with the BONES-SEED proportional set
(`local_rot_mats` + `root_positions` + `neutral_joints (77,3)`, Y-up, grounded).
Point a shape-aware dataset/pack at both trees, slice `neutral_joints` to 30
joints, and reuse the `ShapeEncoder` / FK-loss wiring from
`kimodo/scripts/train_skel_aware.py`. A combined pack + stats recompute (mirroring
`pack_bones_seed_motions_proportional.py` / `compute_motion_stats_proportional.py`)
is the next step.

> The pre-existing **uniform** `nymeriaplus_kimodo/motions/` has the same crumple
> + Z-up bug and should be regenerated with the corrected math (drop the
> neutral_joints/grounding if a shape-unaware variant is wanted) before use.
