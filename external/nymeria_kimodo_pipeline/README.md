# NymeriaPlus → kimodo preprocessing pipeline

Turns raw NymeriaPlus captures into the inputs a kimodo-style model consumes:
**human motion** (SOMA 77-joint, 20 fps), **egocentric RGB** (224×224 WebP), and
**text segments** (atomic-action narration aligned to motion frames). Built to be
**re-run as more data arrives** — every stage is idempotent.

## Data locations

| | path |
|---|---|
| Raw input  | `/weka/jungbin/nymeriaplus/{Sxx}/{seq}/` — `body/`, `narration/`, `recording_head/data/data.vrs`, `metadata.json` |
| Motion out (shape-unaware) | `/weka/jungbin/nymeriaplus_kimodo/motions/{Sxx}/{seq}.npz` |
| Motion out (**shape-aware**) | `/weka/jungbin/nymeriaplus_kimodo_proportional/{Sxx}/{seq}.npz` — adds `neutral_joints (77,3)`. See `NYMERIA_PROPORTIONAL_PIPELINE.md`. |
| Ego video out | `/weka/jungbin/nymeriaplus_kimodo_proportional/video/{Sxx}/{seq}.mp4` (Cosmos-ready, 640² @20fps, frame-aligned to motion). See `video/VIDEO_README.md`. The old 224² per-frame webp under `nymeriaplus_kimodo/images/` was **deleted**. |
| Metadata   | `/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/metadata_atomic_action.jsonl` (+ per-subject, coverage, floor) |

> **Metadata moved (2026-06-15).** Metadata now lives under the **proportional**
> tree (`nymeriaplus_kimodo_proportional/metadata/`) and is built from those NPZs.
> The old `/weka/jungbin/nymeriaplus_kimodo/metadata/` was **deleted**: it was built
> from the stale `nymeriaplus_kimodo/motions/` dir, which only held **7 subjects**
> (S02/04/05/07/10/11/17), so its frame indices also didn't match the proportional
> NPZs the floor pass reads. The rebuild covers all **19 subjects / 732 sequences**
> (146,665 atomic-action slices) with frame indices consistent across metadata,
> floor, and motion. The floor *geometry* source `nymeriaplus_kimodo/floor/all_floor.json`
> is unaffected.

## Stages (run in order)

| # | script | env | in → out |
|---|--------|-----|----------|
| 1 | `process_nymeriaplus.py` | `soma` | SMPL `body/xdata_smpl_neutral.npz` → SOMA `body/xdata_soma.npz` (+ per-vertex error stats). GPU, idempotent, shardable (`--shard i/N`). |
| 2 | `soma_to_kimodo_batch.py` | `soma` | every `xdata_soma.npz` → kimodo motion NPZ (`local_rot_mats (T,77,3,3)` + `root_positions` @ 20 fps). Single-seq core: `soma_to_kimodo_single.py`. |
| 2′ | `soma_to_kimodo_proportional.py` | `soma` | **shape-aware** variant of stage 2: same NPZ **plus `neutral_joints (77,3)`** (actor rest skeleton from `identity_coeffs`). Output tree `nymeriaplus_kimodo_proportional/`. Validate with `validate_proportional_kimodo.py` (env `kimodo`). Full rationale + validation in `NYMERIA_PROPORTIONAL_PIPELINE.md`. |
| 3 | `build_metadata.py` | any (numpy) | every narration CSV → 20-fps-aligned JSONL rows, reading the **proportional** NPZ timestamps. Emits `metadata_all.jsonl` (all sources) + `metadata_atomic_action.jsonl` subset (+ per-subject dirs + coverage) under `nymeriaplus_kimodo_proportional/metadata/`. |
| 3′ | `floor/extract_slice_floor.py` | any | **per-atomic-action-slice GT floor**: assigns each slice the room-floor height (`ground_offset_y`) from the 3D boxes, flags multi-floor (stairs) slices. Output `metadata_atomic_action_floor.jsonl` (same dir). Preferred grounding (vs min-foot) for sitting/lying. See `floor/SLICE_FLOOR_README.md`. |
| 3″ | `floor/fallback_floor_and_skating.py` | `kimodo` | enriches that jsonl: **foot-skating** per `ok` slice (vs GT floor), a **fallback floor** for `no_floor`/`no_support` sequences (estimated from the foot trajectory, with skating), and a `usable` flag that discards large-horizontal-travel estimated slices. Adds `foot_skating_cms`, `floor_source`, `usable`, `est_ambiguous`. |
| 4 | `video/extract_ego_video.py` | `nymeria_plus` | head VRS RGB sampled at the 20-fps body timestamps → rotate −90°, resize **square 640**, **per-seq mp4** (libx264/yuv420p) frame-aligned to the motion NPZ — the format the **Cosmos 3** video path wants (raw mp4 + JSONL windows; VAE encodes on-the-fly). Then `video/build_video_manifest.py` (env `kimodo`) joins video⟷camera⟷motion⟷text into `video/manifest_video.jsonl`. See `video/VIDEO_README.md`. *(supersedes the deprecated `cache_images.py` 224² webp.)* |
| 4′ | `camera/extract_camera_trajectory.py` | `nymeria_plus` | sample the head Aria `T_world_device` (MPS closed-loop trajectory) at the 20-fps body frame times → sidecar `camera/{Sxx}/{seq}.npz` (Aria Z-up world). Camera frame i == motion frame i == ego mp4 frame i. |
| 4″ | `camera/render_skeleton_camera_ego.py` | `kimodo` | render **skeleton + camera trajectory** (Z-up→Y-up + same per-slice floor grounding as the body) **side-by-side with the egocentric video**. Output mp4s under `nymeriaplus_kimodo_proportional/visualization/`. See `camera/CAMERA_VIZ_README.md`. |
| 5 | `validate_motion.py` | `soma`+`kimodo` | FK round-trip check: kimodo FK on the stored rotmats == SOMA-X global rotations (bit-exact). Run on a sample after stage 2. |

`fetch_narration.py` (env `nymeria_plus`) re-downloads only missing `narration/`
zips; run before stage 3 if narration is incomplete.

**Floor height** (for grounding motion): the `floor/` subdir downloads the 3D
object bounding boxes and extracts a per-**sequence** floor height in the world
frame (`extract_floor.py`, see `floor/README.md`), then a per-**slice** GT floor
(`extract_slice_floor.py`, see `floor/SLICE_FLOOR_README.md`). Ground motion with
the per-slice `ground_offset_y` rather than the min-foot `floor_offset` — correct
for sitting/lying where the lowest joint isn't the floor.

Envs: `soma` = `/home/jungbin_cho/miniforge3/envs/soma/bin/python` (torch cu128).
`nymeria_plus` conda env has `projectaria_tools` for VRS decode.

## Key decisions

- **Text source = `atomic_action` only** (per project scope). `build_metadata.py`
  also emits `metadata_all.jsonl` with the other narration types as a bonus.
- **20 fps** target throughout; timestamp-aware subsample (searchsorted), not naive stride.
- **Motion contract**: joint-parent-local rotation matrices, 77 joints (Hips=root,
  Root dropped via `[1:]`), consumed by kimodo `from_SOMASkeleton77()`.
- **transl recenter (v1.1)**: outdoor captures (e.g. S17) place world transl at
  ±300 m, where fp32 loses ~1 cm and corrupts the sub-cm SOMA pose-inversion fit.
  `process_nymeriaplus.py` recenters transl before fitting and adds the offset back.

## Known gaps

- **Ego video + camera (stage 4) done for all 732** (job 2474, 2026-06-18, ~3.4 h on
  one a3ultra node): **728 mp4s + 728 camera sidecars**; manifest = **713 sequences /
  141,589 windows** (the other 15 video seqs have no atomic-action slices). **4
  sequences are unrecoverable** — `20230728_s0_lauren_mayer_act3`, `20230818_s0_amy_padilla_act3`,
  `20230829_s0_ray_humphrey_act2`, `20230928_s1_barbara_sandoval_act0` — partial
  head-VRS downloads with no RGB stream / no pose. Re-run `video/sbatch_full_extraction.sh`
  (idempotent) once those VRS files finish. ~277 GB on weka.
- **94 narration zips are empty upstream** (22-byte ZIPs) — genuinely absent, not a
  download error; those sequences get no text rows.

## Note

`process_nymeriaplus.py` lives here but is symlinked from `~/process_nymeriaplus.py`
because the ingestion launchers (`~/_post_download_chain.sh`, `~/_soma_launch_remote.sh`)
reference that path. Keep the symlink if those launchers are still in use.
