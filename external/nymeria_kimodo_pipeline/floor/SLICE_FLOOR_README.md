# Per-slice GT floor height (`extract_slice_floor.py`)

Assigns each **atomic-action slice** the **GT floor height** of the room/level it is on,
so motion is grounded to the real floor — not the per-sequence min-foot joint. This
matters for **sitting / lying / bent** motions: min-foot grounding floats the whole body
up to the lowest joint, mis-aligning the motion against the egocentric video; GT-floor
keeps the body at the true room-floor height.

## Inputs
- Slices: `nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action.jsonl`
  (`filename, subject, start_frame, end_frame, text`), built by `build_metadata.py` from
  the proportional NPZ timestamps (so frame indices match the NPZs read here).
- Floor geometry: `nymeriaplus_kimodo/floor/all_floor.json` (from `extract_floor.py`; per-floor-instance
  `floor_z`, `half_thickness`, `footprint [xmin,xmax,ymin,ymax]`, **Z-up SLAM frame**).
- Root trajectory: the kimodo NPZ `root_positions`
  (`nymeriaplus_kimodo_proportional/{subj}/{seq}.npz`), converted back to Z-up.
  (We use the kimodo NPZ root, NOT `extract_floor._load_roots`, so frame indices match
  the slice `start_frame`/`end_frame`.)

## Coordinate frame (Z-up ↔ Y-up)
The kimodo converter applied `R_z2y=[[1,0,0],[0,0,1],[0,-1,0]]`, so
**kimodo `(x,y,z)=(soma_x, soma_z, −soma_y)`**, inverse **`soma=(kx, −kz, ky)`**, and
**kimodo Y == SOMA Z**. Floor *assignment* (footprint/below tests) runs in Z-up via
`roots_soma = column_stack([rx, −rz, ky])`. The standing surface
`floor_surface_z = floor_z + half_thickness` (Z-up) is **directly** the kimodo-Y grounding
offset, because kimodo Y = SOMA Z. (Valid because the stored NPZs are `grounded=False`,
raw height.)

## Algorithm (per slice `[sf,ef)`)
Reuses `extract_floor.py`: `_per_frame_floor` (nearest floor below the hip, within
`SUPPORT_TOL=0.30 m`, whose XY footprint contains the root). For each grounded frame the
hip is assigned to the floor **box** directly under it.

**Ambiguity is decided per physical LEVEL, not per box.** The dataset tiles one floor
with several boxes at the *same height*, so counting boxes falsely flagged single-floor
slices as ambiguous (62 % of the old flags — see "Per-level fix"). Boxes whose surface
heights are within `LEVEL_TOL=0.15 m` are merged into one **level**; a level is "present"
if it supports `≥ AMBIG_FRAC=0.20` of grounded frames; `ambiguous = (#present levels > 1)`.
The slice is grounded to the **majority level** (its highest-fraction box is the
representative for `floor_uid`/`floor_z`/`half_thickness`). Multi-level (e.g. stairs)
slices are flagged `ambiguous` and grounded to that majority level.

> The flag is **frame-weighted by where the hip is**, not by the narration text. A
> "walks down the stairs" slice whose frames are 88 % over the lower floor reads as
> single-level (not ambiguous) — correct, because the body is mostly already on one floor.

## Output: `nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action_floor.jsonl` (+ `metadata_per_subject_atomic_action_floor/`)
One row per slice (mirrors the atomic_action metadata):
```
{ filename, subject, start_frame, end_frame, text,
  floor_status: "ok" | "no_floor" | "no_support" | "no_motion",   # how the GT-floor attempt went
  floor_uid, floor_z, half_thickness, floor_surface_z,            # GT-floor rows only (majority LEVEL)
  ground_offset_y,            # kimodo: root_positions[:,1] -= this  (GT or estimated)
  ambiguous,                  # >=2 distinct floor LEVELS each support >=20% of frames
  n_floors_in_slice,          # = number of present LEVELS (not boxes)
  support_frac,
  floors: [{uid, floor_surface_z, frac_frames}, ...],             # per-box detail
  levels: [{surface_z, frac_frames, uids:[...]}, ...],            # height-grouped (ambiguity is on these)

  # --- added by fallback_floor_and_skating.py ---
  floor_source: "gt" | "estimated",   # GT box, or per-sequence foot estimate
  usable: bool,                        # false => do NOT train on this slice
  est_ambiguous: bool,                 # estimated floor + large horizontal travel => flagged unusable
  est_floor_surface_z, horiz_travel_m, # estimated rows only
  foot_skating_cms, n_contact_frames } # contact-frame foot speed vs the slice's floor
```
- `floor_status`: GT-floor provenance — `no_floor` = sequence has no `objects/boxy` floor
  (e.g. all of S17); `no_support` = actor never over a floor footprint in the slice;
  `no_motion` = no NPZ. **Authoritative use fields are `floor_source` + `usable`**, not
  `floor_status` (a `no_floor`/`no_support` slice can still be `usable` via an estimated floor).

## Grounding convention
`root_positions[:,1] -= ground_offset_y` (Y only; xz untouched). Correct for
sitting/lying because it's the room floor, not the lowest joint. Note: for crouching/
bent poses the SMPL leg fit can place feet *below* the GT floor (penetration) — GT-floor
correctly does NOT compensate for this (min-foot was masking it by floating the body).

## Run
```bash
python floor/extract_slice_floor.py        # all subjects (defaults wired)
python floor/extract_slice_floor.py --subjects S02   # one subject
```

## Last run (2026-06-16, all subjects, per-level ambiguity)
146,665 atomic-action slices over 717 sequences (**19 subjects**, S01–S20 except the
non-existent S18): `ok=127,927`, `no_floor=16,017` (seqs with no `objects/boxy` floor,
incl. all of S17), `no_support=2,721`, `ambiguous(among ok)=1,095`. Verified:
single-floor cross-check **0/35,675** mismatches vs the `all_floor` primary; multi-floor
footprint assignment correct.

### Per-level fix (2026-06-16)
The original `ambiguous` counted floor *boxes*, so the **4,679** flags it produced were
**62 % spurious**: the dataset tiles one physical floor with several boxes at the *same*
height, and any slice spanning ≥2 of them was flagged (e.g. someone standing still over a
2-box floor). Grouping boxes by height (`LEVEL_TOL=0.15 m`) before counting drops these:
**ambiguous 4,679 → 1,095** (true stairs / multi-level), the 3,584 false positives become
single-level. `ground_offset_y` is unchanged for all but **83** `ok` slices — genuine
multi-level slices that now ground to the majority *level* (its highest-fraction box)
instead of the single highest-fraction box; their skating was recomputed. The new
`levels[]` field carries the height-grouped fractions; `floors[]` keeps the per-box detail.

> Supersedes the earlier 55,087-slice / 7-subject run. That run used the deleted
> `nymeriaplus_kimodo/metadata/`, which was built from the stale `nymeriaplus_kimodo/motions/`
> dir (only 7 subjects, and frame indices that didn't match the proportional NPZs).

## Foot skating + fallback floor (`floor/fallback_floor_and_skating.py`)
A second pass (env `kimodo`, needs the `kimodo_open` package on `PYTHONPATH`) FK's each
sequence once and **enriches the jsonl in place**. Three jobs:

### 1. Foot skating for `ok` slices (relative to the GT floor)
Mean foot speed (cm/s) over **contact** frames, where contact = foot height `< 0.10 m`
above the slice's GT floor **and** speed `< 0.15 m/s` (`foot_detect_from_pos_and_vel`,
same gate as the viz). Stored per slice as `foot_skating_cms` + `n_contact_frames`.

| pool | n slices (w/ contact) | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| `ok` (GT floor) | 124,425 / 127,927 | **5.12** | 4.99 | 2.79 | 7.65 |
| estimated-usable (fallback floor) | 11,088 / 14,448 | **4.90** | 4.68 | 2.82 | 7.37 |

(cm/s. ~3.5 k `ok` slices have zero contact frames — feet never planted, e.g. sitting.
Skating is inherent to the raw SMPL leg fit — see `FOOT_SKATING_ANALYSIS.md`.)

### 2. Calibrate the floor↔foot gap (from `ok` slices)
How far the GT floor sits below the lowest foot joints, per `ok` slice
(`min over slice frames` of the joint height, minus `floor_surface`):

| joint | median (m) | p10 | p90 |
|---|---|---|---|
| toe (`ToeBase`) − floor | −0.055 | −0.181 | +0.013 |
| ankle (`Foot`, "foot wrist") − floor | −0.005 | −0.138 | +0.065 |
| min foot − floor | −0.056 | −0.182 | +0.012 |

→ the **ankle rests ≈ on the floor**; the **toe dips ~5.5 cm below** it (forefoot
penetration from the SMPL crouch fit). So the floor is ≈ a low percentile of the foot
height, plus a small offset.

### 3. Estimate a floor for `no_floor` / `no_support` sequences (no GT box)
Per **sequence** (not per slice — one floor for the whole take), from the whole-sequence
per-frame **min foot height**:
```
est_floor_surface_z = percentile_5(per-frame min-foot height) + 0.083 m
```
Percentile (5) and offset (+0.083) were **chosen to fit the GT floor** on the 284
single-floor `ok` sequences (min-MAD over candidate percentiles); residual **MAD = 4.6 cm**.

**Discard slices that move too far horizontally.** With a single estimated floor per
sequence, a slice whose root sweeps a large horizontal distance probably changed level
(like a stairs / multi-floor slice) so the single floor is wrong. Horizontal travel =
xz bounding-box diagonal of the root over the slice. Threshold **T = 2.85 m** = the 95th
percentile of `ok` *non-ambiguous* slice travel (for reference, `ok` *ambiguous* slices
have p10 = 0.97 m, p50 = 2.10 m). Slice travel `> T` → `usable=false`,
`est_ambiguous=true` (no `ground_offset_y` written). Otherwise the estimated floor is
written to `ground_offset_y` and skating is computed against it.

> Caveat: horizontal travel is a *proxy* for level changes (the user's chosen signal). A
> long flat-floor walk can be flagged (false positive); a short in-place stair step may
> slip through (false negative). It is deliberately conservative (only the most mobile 5 %
> of single-floor slices' worth of travel triggers it).

### Result (all 18,738 `no_floor`+`no_support` slices)
`estimated usable = 14,448` (got a fallback floor + skating); `est_ambiguous flagged =
4,290` (`usable=false`). Final tallies: `usable=142,375` (`gt` 127,927 + `estimated`
14,448), `usable=false` 4,290. Run: `python floor/fallback_floor_and_skating.py`
(defaults wired; ~minutes, CPU).

## The deliverable
The only **data** product is `metadata_atomic_action_floor.jsonl` (+ the per-subject
mirror), built by `extract_slice_floor.py` then enriched by `fallback_floor_and_skating.py`.
Everything below is viz tooling. Downstream use:
- **Filter to `usable == true`** (drops the 4,290 `est_ambiguous` slices).
- **Ground**: `root_positions[:,1] -= ground_offset_y` (Y only; GT for `floor_source=="gt"`,
  estimated for `"estimated"`).
- `foot_skating_cms` is a per-slice quality signal (higher = more foot sliding).

## Viz (`viz_with_text.py --mode segments --gt-floor`)
- Grounds each segment by its `ground_offset_y` → the GT floor sits at **y=0**, which is
  exactly where the renderer draws the floor plane (so floor = GT surface).
- **Opaque floor** (`floor_alpha=0.9`, vs the 0.55 default) so the figure reads as above
  the floor. Plumbed through the shared `render_soma.render_single(floor_alpha=…)` →
  `render_hml3d._draw_floor_and_grid` (default 0.55 unchanged).
- **Skeleton + projected pelvis trail render ON TOP of the opaque floor**: `render_single`
  sets `ax.computed_zorder = False` so matplotlib honors artist zorder
  (floor 0 < grid 1 < pelvis-trail 2 < bones 4 < joints 5 < frame-marker 6) instead of
  depth-sorting the floor over the skeleton.
- **Foot skating relative to the GT floor**: `skating_pool(posed30, ground_offset=ground_offset_y)`
  — the contact-height gate (foot < 0.10 m) is measured against the GT floor, not the
  min-foot. Overlaid alongside the text + `ambiguous` flag + per-floor frame fractions.
- Ambiguous (stairs) slices ground to the majority floor and are labeled `AMBIGUOUS`;
  mid-staircase the GT-floor skating reads `n/a` (feet are between floor levels).

Demos: `_demo_gtfloor.py` (standing/bent), `_demo_ambiguous.py` (stairs).
