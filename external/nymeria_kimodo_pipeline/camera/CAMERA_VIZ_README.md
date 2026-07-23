# Egocentric camera trajectory + skeleton + ego-video visualization

Adds the **head Aria camera trajectory** to the kimodo motion world and renders it next
to the **egocentric video**, so the skeleton, the camera that recorded it, and the RGB it
saw are all in one clip and one coordinate frame.

## Where the camera comes from
`recording_head/mps/slam/closed_loop_trajectory.csv` — Project Aria MPS device pose
(`T_world_device`) in the **Aria SLAM world**. That world is **Z-up** (gravity =
`(0,0,-9.81)`) and is intended to align with the world used by the body SOMA/SMPL fit and
the `objects/boxy` floor boxes. On the `kirk_flowers` demo, the sampled device sits
**+0.64 m above** and **0.22 m in front of** the pelvis with **timesync < 0.3 ms**. This is
a single visualization sanity check, not a dataset-wide calibration of the SOMA `Head`
joint to the Aria device frame.

## Time alignment (critical)
Body motion NPZs are timestamped in the **TIME_CODE** domain (same as the cached ego
images). The MPS trajectory is indexed by device time. The loader
`RecordingLoader.sample_trajectory_at_timecodes(ts_us*1000)` converts TIME_CODE→device
internally and returns one `T_world_device` per body frame, so **camera frame i == motion
frame i == ego image `frame_{i:06d}.webp`**. Median sampling error is ~0.25 ms.

## Coordinate change + floor grounding (same recipe as the skeleton)
The camera is put into kimodo coords and grounded with the **same** per-slice transform as
the body, so their relative geometry is preserved:
1. **Z-up → Y-up**: `kimodo (x,y,z) = (world_x, world_z, -world_y)` (the converter's
   `R_z2y`). Applied to camera positions *and* to the device rotation's axes (for the triad).
2. **Horizontal canonicalization**: subtract the skeleton's start-frame root xz
   (`p[sf,ROOT,0]`, `p[sf,ROOT,2]`) — the *same* offset used for the skeleton.
3. **Vertical grounding**: subtract the slice's `ground_offset_y`
   (GT or estimated, from `metadata_atomic_action_floor.jsonl`) so the GT floor is at y=0.
4. Display negates X (kimodo render convention) for both.

The 2026-07-21 full audit established that clean proportional-UniEgo and camera streams normally do
preserve a shared metric world: the per-sequence median camera-origin-to-SOMA-Head distance is
tightly centered at 0.139 m. The exception is a sparse set of upstream registration discontinuities,
including metre-scale jumps, which must be masked rather than interpreted as ordinary calibration
error. Learning code uses synchronized relative transforms and a train-only robust approximation
primarily because SOMA Head **orientation** is not a rigid camera proxy, and secondarily to avoid
depending on unfiltered absolute discontinuities. See `CALIBRATION_CONSISTENCY.md`.

## Two stages
| stage | script | env | does |
|---|---|---|---|
| A | `extract_camera_trajectory.py` | `nymeria_plus` | sample `T_world_device` at body frame times → sidecar `camera/{Sxx}/{seq}.npz` (`cam_world_pos (T,3)`, `cam_world_rot (T,3,3)`, `tdiff_ns`, `timestamps_us`, and per-step continuity diagnostics). Aria-world Z-up; no transform applied yet. |
| B | `render_skeleton_camera_ego.py` | `kimodo` | FK skeleton, transform+ground camera, render **left**: skeleton + cyan camera path/dot + device-axis triad on the opaque GT floor; **right**: the ego `frame_{idx:06d}.webp`; stacked side-by-side with a caption bar. |

(Stages are split because `nymeria_plus` has projectaria_tools but not kimodo, and vice
versa. The camera env is needed only once, to write the sidecar.)

## Run
```bash
# A: extract camera sidecar (defaults to the demo sequence; or pass subj:seq pairs)
conda run -n nymeria_plus python camera/extract_camera_trajectory.py \
    --seqs S02:20231006_s1_kirk_flowers_act0_hfjvo9
# B: render the demo slices (needs the sidecar + cached ego images for that seq)
conda run -n kimodo python camera/render_skeleton_camera_ego.py --demo
# or a single slice:
conda run -n kimodo python camera/render_skeleton_camera_ego.py \
    --slice S02 20231006_s1_kirk_flowers_act0_hfjvo9 500 600 walk_bedroom
```

## What's drawn (left panel)
- **Red** skeleton + red pelvis trail, on the **opaque GT floor** (`floor_alpha=0.9`,
  `computed_zorder=False` so skeleton/trail/camera render on top).
- **Cyan** line = the full head-camera path over the slice; **cyan dot** = camera now.
- Small **R/G/B triad** at the camera = the device frame axes (`R_world_device` columns)
  in kimodo coords — shows head orientation honestly without asserting one optical axis.
- Caption bar: the atomic-action text + `floor_source`, `ground_offset_y`, an `AMBIGUOUS`
  tag for stairs/multi-floor slices, and the legend.

## Output
`/weka/jungbin/nymeriaplus_kimodo_proportional/visualization/{seq}__seg{sf}_{label}.mp4`
(20 fps, 1408×768). Demo slices (all `kirk_flowers`, S02):
`walk_bedroom` (500), `kneel_under_table` (900), `up_stairs_ambiguous` (2040),
`down_stairs_landing_ambiguous` (2640), `walk_pick_hourglass` (4679). The two stairs
slices are GT-`ambiguous` (majority-floor grounded) and show the camera climbing while the
floor stays at the majority level — exactly the case the floor metadata flags.

## Scaling to more sequences
`extract_camera_trajectory.py --seqs Sxx:seq ...` works for any sequence with
`recording_head` MPS poses; `render_skeleton_camera_ego.py --slice ...` for any slice that
has a camera sidecar **and** cached ego images (`images/{Sxx}/{seq}/_done`). Sequences
without cached images render a black right panel.
