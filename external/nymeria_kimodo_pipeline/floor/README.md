# Floor-height extraction (NymeriaPlus 3D object boxes)

Recovers a per-sequence **floor height** in the SLAM world frame from
NymeriaPlus's 3D object bounding-box annotations, validated against the motion
root trajectory. Useful for grounding motion (foot contact, root-height-above-floor).

## Files
| file | what |
|------|------|
| `fetch_objects.py` | Download + sha1-verify + extract the `object_bounding_box` zips for ONE subject (thin wrapper over the official `nymeriaplus.downloader.DownloadManager`). |
| `fetch_objects_all.py` | Subject-aware **sharded** downloader for ALL subjects: same official `DownloadLink` (stream/sha1/extract/idempotent) but routes each seq into its correct `<Sxx>/<seq>/` dir via `_subject_map.json`. `--shard i/N` for parallel runs across nodes. |
| `extract_floor.py` | Parse the boxes, resolve the Floor instance(s), pick the supporting floor, validate vs the root trajectory, emit JSON+CSV (per subject). |
| `extract_floor_all.py` | Thin driver: run `extract_floor.process_seq` over every subject under `--nymeria-root`; writes per-subject files + combined `all_floor.{json,csv}`. |

## Pipeline

### 1. Download the boxes

**One subject** — the subject's signed URL file
(`nymeria_plus_download_urls_<Sxx>_objects.json`) lists one `object_bounding_box`
zip per sequence (~100–130 MB; ~2.6 GB / subject).
```
python fetch_objects.py \
    --url-json /weka/jungbin/nymeriaplus/nymeria_plus_download_urls_S04_objects.json \
    --out-root /weka/jungbin/nymeriaplus/S04
```

**All subjects (sharded)** — use the all-subjects `object_bounding_box` URL file
(`nymeria_plus_download_urls_all_object_bbox.json`, 635 seqs, ~81 GB). Run one shard
per process; `/weka` is shared so shards on different nodes never collide:
```
# launch 6 shards (e.g. 2 per node on a3ultravis-a3ultranodeset-{1,2,3}):
python fetch_objects_all.py \
    --url-json /weka/jungbin/nymeriaplus/nymeria_plus_download_urls_all_object_bbox.json \
    --shard 0/6        # ... and 1/6 .. 5/6
```
It routes each seq into `<Sxx>/<seq>/` via `/weka/jungbin/nymeriaplus/_subject_map.json`
(subject is a download bucket, NOT derivable from the seq name). Idempotent: per-artifact
flags let re-runs retry only failures; per-shard summaries land in `<root>/.objbb_logs/`.

> **Gotcha:** `nymeria_plus_download_urls_bbox_all.json` is **mislabeled** — every entry
> is `object_mesh` (`objects/shaper/*.ply` geometry, ~26 GB), **not** the bbox CSVs.
> Use `..._all_object_bbox.json` for floor work.

Each zip extracts to `<seq>/objects/boxy/`:
- `instances.json` — object_uid → `{category, instance_name, ...}`
- `scene_objects.csv` — `object_uid, ts, t_wo_{xyz}, q_wo_{wxyz}` (object pose in world; static → ts `-1`)
- `3dbb.csv` — `object_uid, ts, p_local_obj_{xyz}{min,max}` (box extents in the object's local frame)
- `2dbb_recording_*.csv` — 2D boxes projected into each camera (unused here)

Idempotent: re-runs skip via flags in `<out-root>/.download_logs/`; summary in
`<out-root>/download_summary.json`.

### 2. Extract floor height

**All subjects:**
```
python extract_floor_all.py \
    --nymeria-root /weka/jungbin/nymeriaplus \
    --out-dir      /weka/jungbin/nymeriaplus_kimodo/floor
```
Writes per-subject `<Sxx>_floor.{json,csv}` plus combined `all_floor.{json,csv}`.

**One subject:**
```
python extract_floor.py \
    --objects-root /weka/jungbin/nymeriaplus/S04 \
    --out-dir      /weka/jungbin/nymeriaplus_kimodo/floor
```

The root trajectory for validation is read from each sequence's **co-located SOMA
fit** `<seq>/body/xdata_soma.npz` (`transl`, subsampled to ~20 fps). This is the same
SLAM world frame as the boxes — verified identical to the kimodo `root_positions`,
which is just a 20-fps copy of this `transl`. Sequences lacking a SOMA fit fall back to
the kimodo motion npz (`--motion-root`) if given, else validation is `null`.

## How floor height is computed

**World up = +z.** The NymeriaPlus SLAM world frame is gravity-aligned with
+z up. Verified empirically: every Floor box's plane normal lies along world z
(`normal_along_z ≈ 1.0`), and the body root (Hips) trajectory sits *above* the
floor in z. So height = world z and `floor_z = floor box center z`. The slab is
~0.12 m thick, so `floor_surface_z = floor_z + half_thickness`.

**Resolving the floor.** The floor is **not always `object_uid 0`** — it's found
via `instances.json` where `category == "Floor"`. Each Floor box is placed into
the world by `t_wo`/`q_wo` (from `scene_objects.csv`) applied to its local extents
(`3dbb.csv`); the box center's z is the floor height and its 8 corners give the
world-XY footprint.

**Multi-floor sequences.** Some captures span multiple levels and annotate
several Floor instances at different heights. `extract_floor.py` selects the
**supporting floor** by majority vote over a per-frame assignment: for each
frame the supporting floor is the *nearest floor below the hip* — the highest
`floor_z` with `root_z ≥ floor_z − 0.30 m` whose XY footprint contains the root
— and the primary floor is the one supporting the most frames.

This is the part that matters for **stacked floors** (same XY footprint, only z
differs, e.g. a two-storey home): XY overlap can't disambiguate them, so the
choice is driven by the hip's *height*. A higher level only wins on frames where
the hip is genuinely up on it (`root_z` near that level), so the vote tracks the
level the actor is actually on even if they crouch or sit. Fallback when no
motion is available: the largest footprint. *All* Floor instances are still
emitted in the JSON so callers can do per-frame, per-level assignment if needed.

**Validation.** Two metrics, both `dz = hip_z − floor_z`:
- **single-floor** (`root_dz_*`, `validation_ok`): `dz` against the one primary
  `floor_z`. `validation_ok` requires `normal_along_z > 0.9`, `mean(dz) ∈ [0.20, 1.20]`,
  `min(dz) > −0.40`.
- **per-frame** (`perframe_dz_*`, `ok_perframe`): `dz` against the *nearest floor below
  the hip each frame* (same assignment as the primary vote). This is the metric to trust
  for **multi-level captures**: a single scalar floor can't fit frames where the actor
  changes storeys, so single-floor `dz` looks bad while the floors are actually correct.

Across all subjects the per-frame metric passes **633/635** sequences vs **495/635**
single-floor — the gap is entirely multi-level captures, not bad floor data.

## Output schema

`<Sxx>_floor.json` — list of records:
```json
{
  "seq": "...", "status": "ok", "up_axis": "+z",
  "n_floor_instances": 4,
  "floor_instances": [ {"uid","floor_z","half_thickness","footprint":[xmin,xmax,ymin,ymax],
                        "footprint_area","normal_along_z"}, ... ],
  "primary_floor_uid": "102",
  "floor_z": -1.567, "floor_surface_z": -1.505, "normal_along_z": 1.0,
  "validation": {"n_frames","root_dz_min","root_dz_mean","root_dz_max","ok",
                 "n_frames_grounded","perframe_dz_min","perframe_dz_mean",
                 "perframe_dz_max","ok_perframe"}
}
```
`status` ∈ `ok | no_objects | no_floor_instance`. `validation` is `null` when the
sequence has neither a SOMA fit nor a kimodo motion NPZ.

`<Sxx>_floor.csv` / `all_floor.csv` — one row per sequence: `subject, seq, status,
n_floor_instances, primary_floor_uid, floor_z, floor_surface_z,
root_dz_{min,mean,max}, validation_ok, perframe_dz_{min,mean,max}, ok_perframe`.

## All-subjects result (current)

**18 subjects, 636 sequences — every sequence has a floor (636/636).** 448 are
multi-floor (multi-level captures). 635 sequences have a SOMA fit for validation
(1 — `S04/jacob_webb` — does not, so its floor is reported without validation):
- **per-frame validation OK: 633/635** (the trustworthy metric).
- single-floor validation OK: 495/635 (lower only because one scalar floor can't
  fit level-traversal — the floors themselves are correct).
- 2 per-frame failures, both genuine data edge cases, not bugs:
  - `S09/justin_heath_act3` — only the **ground** floor is annotated (1 instance) but
    the capture is multi-storey, so the hip sits ~2.8 m above it on upper levels
    (missing upper-floor annotation; `floor_z` for the ground level is still valid).
  - `S19/jason_smith_act3` — a genuinely elevated-activity capture, `dz_mean` 1.41 just
    over the 1.20 band; floor is correct.

## Caveats
- Floor height is a single scalar per sequence (the supporting level). For
  motion that genuinely traverses levels, use the full `floor_instances` list and
  the per-frame metric.
- A sequence annotating only one level of a multi-storey space (e.g.
  `S09/justin_heath_act3`) will show a large per-frame `dz` — the floor is right, the
  annotation is just incomplete. The per-frame failure surfaces this.
- The slab gives ~6 cm ambiguity between center and top surface; both are emitted.
- Up-axis (+z) and the validation bands are the only assumptions; both are
  checked per sequence via `normal_along_z` and `dz` and surfaced in the output.
