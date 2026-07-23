# Egocentric video processing for Cosmos finetuning

Processes the NymeriaPlus **egocentric RGB video** into the format the Cosmos 3
generator's video data path expects, frame-aligned 1:1 with the kimodo **motion**
(369-d) and the head **camera trajectory**, so the three can be trained jointly:
text + ego-video + camera-pose + floor-grounded human motion.

The motion is "floor-height-removed" downstream via the per-window `ground_offset_y`
(`root_y -= ground_offset_y`), the same GT/estimated floor offset from the slice-floor
stage — see `../floor/SLICE_FLOOR_README.md`.

---

## What Cosmos actually expects (the contract we matched)

From `cosmos-framework` (`cosmos_framework/data/vfm/local_datasets/sft_dataset.py`,
`model/vfm/tokenizers/wan2pt2_vae_4x16x16.py`, `data/vfm/utils.py`):

| Concern | Cosmos contract | What we produce |
| --- | --- | --- |
| On-disk video | **raw mp4 + JSONL metadata** with frame windows (`t2w_windows`). NOT pre-encoded latents, NOT per-frame images. | one mp4 per sequence + `manifest_video.jsonl` |
| In-memory video | uint8 `[3,T,H,W]`, **RGB**, decoded at load time | mp4 decodes to exactly this |
| Pixel range | model does `pixel/127.5 - 1.0` → `[-1,1]` **inside the model** (`omni_mot_model._normalize_video_databatch_inplace`) | we store ordinary RGB; **no normalization here** |
| VAE | Wan2.2 `4×16×16` (temporal 4×, spatial 16×), encodes **on-the-fly in the training step** | nothing to precompute |
| Temporal | training truncates a window to `T = 4N+1` frames | we store full sequences; windows are truncated by the loader |
| Resolution | square buckets via resize-to-cover + center-crop: 256p=`(256,256)`, 480p=`(640,640)`, 720p=`(960,960)` (`VIDEO_RES_SIZE_INFO`) | we store **square 640** so 256p & 480p downsample cleanly (720p needs a re-extract at `--size 960`) |
| FPS | 10–30 supported | 20 fps (the motion rate) |
| Caption | raw text or structured `caption_json`, tokenized at load time (Qwen3-VL reasoner, **not** T5/embeddings) | raw atomic-action `text` per window |

Aria native RGB is `1408×1408` uint8 RGB, rotated 90° from upright → we `rotate(-90)`
then resize to the square edge.

---

## Layout produced

```
/weka/jungbin/nymeriaplus_kimodo_proportional/
  video/
    {Sxx}/{seq}.mp4            # 20fps RGB, square 640, frame i == motion frame i == camera frame i
    {Sxx}/{seq}.json           # per-seq sidecar: dims, fps, nb_frames, valid_start/valid_end, n_invalid
    {Sxx}/_done                # sentinel (idempotent skip)
    manifest_video.jsonl       # Cosmos-SFT shape: 1 record/seq with t2w_windows[]
    manifest_video_slices.jsonl# flat: 1 record/usable window (convenience)
    manifest_video_stats.json
    _video_summary.json        # Stage-A batch tally
```

### `manifest_video.jsonl` record (per sequence)

```json
{
  "uuid": "S02/20231006_s1_kirk_flowers_act0_hfjvo9",
  "subject": "S02", "filename": "20231006_s1_kirk_flowers_act0_hfjvo9",
  "vision_path": ".../video/S02/<seq>.mp4",
  "width": 640, "height": 640, "framerate": 20, "nb_frames": 19583,
  "duration": 979.15, "valid_start": 0, "valid_end": 19582, "n_invalid": 0,
  "camera_path": ".../camera/S02/<seq>.npz",   // null if not yet extracted
  "motion_path": ".../S02/<seq>.npz",
  "t2w_windows": [
    {"start_frame": 500, "end_frame": 600, "caption": "C walks ...",
     "ground_offset_y": -0.155, "floor_source": "gt", "floor_status": "ok",
     "usable": true, "ambiguous": false, "est_ambiguous": false,
     "n_floors_in_slice": 1, "foot_skating_cms": 4.78}
  ]
}
```

A window resolves to three aligned streams over frames `[start_frame, end_frame)`:
- **video** — `mp4[start:end]` (Cosmos VAE encodes these)
- **motion** — `motion_path` NPZ → FK → 369-d kimodo, grounded by `root_y -= ground_offset_y`
- **camera** — `camera_path` NPZ `cam_world_pos/rot[start:end]`, Z→Y rotation + same `ground_offset_y` (see `../camera/CAMERA_VIZ_README.md`)

Only `usable==true` slices become windows; each is clipped to the video's
`valid_start..valid_end` (a few motion frames at a sequence's ends can fall outside
the VRS time span; those frames repeat the nearest valid frame and are excluded from
windows).

---

## Running it

**Stage A — extract mp4s** (env `nymeria_plus`; VRS decode is CPU-bound; uses system
`/usr/bin/ffmpeg` via a raw-rgb24 pipe):

```bash
# one node, fill the 224 CPUs (~80 min for all 732 sequences):
srun -p a3ultra -N1 --cpus-per-task=224 --time=6:00:00 --pty bash
N=96 SIZE=640 bash /home/jungbin_cho/nymeria_kimodo_pipeline/video/run_extract_video_node.sh

# or a subset / single sequence:
python extract_ego_video.py --seqs S02/20231006_s1_kirk_flowers_act0_hfjvo9 --workers 1
```

Idempotent: a `_done` sentinel + mp4 + json skips a finished sequence, so the job
resumes after interruption.

**Stage B — build the manifest** (numpy only; run in `kimodo`):

```bash
/home/jungbin_cho/miniforge3/envs/kimodo/bin/python \
  /home/jungbin_cho/nymeria_kimodo_pipeline/video/build_video_manifest.py
```

Re-run Stage B any time after more videos or camera sidecars appear; it only reads
sidecars + the floor metadata.

---

## Run results (2026-06-18, job 2474, one a3ultra node, ~3.4 h)

`sbatch_full_extraction.sh` (camera → video → manifest) processed all 732 sequences:
- **728 ego mp4s** + **728 camera sidecars** (640² @20fps, ~277 GB total).
- **`manifest_video.jsonl` = 713 sequences / 141,589 windows**, all with aligned
  video + camera + motion + text. (15 of the 728 video seqs have no atomic-action
  slices → no windows; still usable for video-only.)
- **4 sequences failed and are unrecoverable** (no RGB stream / no pose — partial
  head-VRS downloads): `20230728_s0_lauren_mayer_act3`, `20230818_s0_amy_padilla_act3`,
  `20230829_s0_ray_humphrey_act2`, `20230928_s1_barbara_sandoval_act0`. Re-run the
  sbatch (idempotent) after those VRS files finish downloading.

## Companion stages

- **Camera trajectories** (`../camera/extract_camera_trajectory.py`): head Aria pose
  sampled at the same `timestamps_us`. Extract for all 732 to populate every
  `camera_path` (only a handful done so far). Same Z-up Aria world as the motion.
- **Floor / grounding** (`../floor/`): supplies `ground_offset_y`, `usable`,
  `ambiguous`, `foot_skating_cms` per window.
- **Motion** (`../soma_to_kimodo_proportional.py`): the 369-d kimodo NPZs.

## Notes

- **The old `/weka/jungbin/nymeriaplus_kimodo/images` (224×224 per-frame webp) is
  superseded by this and was removed.** It was too small for any Cosmos bucket
  (≥256) and per-frame webp is the wrong on-disk shape. Re-extract from VRS via
  Stage A if egocentric frames are ever needed again.
- Storage ≈ 0.3–0.5 GB per sequence at 640/crf18 (~16 min clips) → **~277 GB** for the
  728 extracted. Drop `--size 256` (and `--crf` up) for a much lighter 256p-only set.
- 720p training needs `--size 960` (re-extract); 640 cannot upscale to the 960 bucket.
