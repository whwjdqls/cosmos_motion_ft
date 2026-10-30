# GT motion and observed scene geometry

**Approved default (2026-10-30):** the user confirmed the final legacy-filter
point cloud and requested using it from now on. For all future scene/body
visualizations, use `review_scene_viewer.py` with `scene_gl_renderer.py`, the
SDK-matched legacy confidence filter and every retained raw head-map point.
Preserve the current depth-buffered layout with RGB, two perspectives and a
top view. No point-count cap or extra spatial/ceiling filter. The accepted
reference is `visual_review_legacy_allpoints_sdk`, linked in the
[Notion report](https://app.notion.com/p/409d2a44f44b81839675f2890d0245c6).
The scorer described below is historical and its collision metrics remain
provisional; it is not the default visualization pipeline.

**Collision-body default:** exclude both thighs, shins and feet, as requested
by the user. Hips, torso, head and arms remain (16 active capsules). The six
excluded capsules do not contribute to union depths, connected-cluster support,
frame flags, depth summaries or the body-part/frame denominator. The skeleton
can still show the legs for context. `--collision-body full_body` explicitly
reproduces the previous22-capsule protocol; current CLIs default to
`--collision-body exclude_thigh_shin_foot` and reject mixed-policy cached scores.

`evaluate_gt_scene_collision.py` evaluates the exact saved 71 held-out Phase3
windows. All paths are CLI arguments; compute runs in an owned allocation.
The original source features, canonicalization, saved floor offset and exact
normalization must reproduce each saved GT bundle. The world/window transform
is retained, and a geometric round trip must agree within 0.1mm.

The head VRS converts TIME_CODE into the SLAM device clock through
`scene_timecode.py`, which may use a separate projectaria-tools environment.
Nearest SLAM poses must reproduce the saved device-camera sidecar within
2cm/2 degrees and 25ms, and the window must remain in one SLAM graph. The scene
is transformed from Aria Z-up to Kimodo Y-up, grounded and canonicalized with
the same fixed transform as motion. No registration is fitted to improve scores.

This is a point-cloud/body-capsule intrusion proxy. Confidence filters are
`inv_dist_std <= 0.005`, `dist_std <= 0.01m`. Scene points within the motion's
bounding box plus 1.5m are voxelized at 2cm; scoring uses all retained voxels.
At least three neighboring voxel centers must be more than 2cm inside the
union of fixed height-scaled body capsules to flag a frame. Contacts within
that tolerance do not trigger the proxy. Sensitivities use 1/2/3cm margins and
0.8/1.0/1.2 radius scales. Face/finger primitives are omitted; wrists remain.

Report measured GT collision-frame/body-part rates, capsule intrusion severity,
per-frame evidence, cloud support and failures. Capsule intrusion depth is not
physical scene penetration depth. Missing geometry can hide collisions and
capsule anatomy or dynamic-map artifacts can create false GT intersections.
Floor contact is not semantically separated from furniture in this raw score.

Previews show original RGB and two views of the same canonicalized scene, with
body capsules, red intrusion points and fixed frame-zero head axes. Four fixed
first cohort windows are rendered, without selecting cases by their scores.

## Alignment review, 2026-10-30

Bulk scoring is paused. Collision percentages from the initial operation are
provisional: camera/SLAM agreement alone does not validate source-body anatomy
or establish that all map points represent static background. In case03 frame0,
the independent original Aria CSV and ungrounded body reproduce the same
intersections, so the added window/floor transform is not their cause.

The initial display `(X,Z,Y)` was a reflection. The corrected display is
`(X,-Z,Y)`, a proper rotation with determinant +1. Both scene and body use it;
distance-based measurements were invariant to the earlier reflection. Review
initial review videos show 3000 points, RGB, two perspectives, and a top view with measured
camera, head-derived camera, and torso forward arrows. Review clouds use the
official Nymeria loader defaults `inv_dist_std <=0.0004`, `dist_std <=0.02m`.
These are distinct from the initial scorer's generic Aria thresholds.

The review operation stores camera/head turn curves. Their agreement is expected
because corrected Head rotations were constructed from camera rotations; it
checks decoding and visualization, not independent torso orientation accuracy.
Temporal map observations or independent semantics are needed to investigate
people/moved-object contamination. Removing points solely because they overlap
GT would make the GT baseline circular.

## Legacy viewer comparison and uncapped display

The legacy reference is pinned to
`facebookresearch/nymeria_dataset@47ef97cc7db00c239a5edbf887f5da1b5a3bbea1`.
`nymeria/data_viewer.py` uses Rerun, right-handed Z-up coordinates, gray
RGBA(128,128,128,128) points and a point radius of 0.008m. It logs all available
device clouds. Its confidence filtering is strictly `inv_dist_std <0.0004`
and `dist_std <0.02m`, through Project Aria. It has no additional dynamic-person
removal or geometric outlier-removal step. Its screenshots still show scattered
points. Different body models, devices, scenes and viewpoints also affect the
apparent cleanliness; a cleaner picture does not validate physical collisions.
Confidence fields must first round to SDK float32 storage, then promote to
Python float for these comparisons. Comparing CSV float64 directly changes
threshold-boundary membership (case03:343701 versus SDK344796). The corrected
loader preserves SDK behavior and records the confidence-storage convention.

The provider requests 100000 points per device, although the default fresh-cache
path calls the recording loader without forwarding the count and uses its
50000 default. Cached clouds may therefore have different counts. Our uncapped
review deliberately omits this random downsampling at the user's request.

`review_scene_viewer.py` renders every finite confidence-filtered head-map point
in the matching graph, preserving raw positions. There is no point-count cap,
spatial crop, ceiling filter, voxel averaging or GT-overlap removal for display.
The fixed two perspective views and top view use `scene_gl_renderer.py`, with
a real OpenGL depth buffer and 0.008m translucent gray point sprites. It keeps
the measured camera, head-derived camera and torso directions, frame-zero head
axes and original RGB. This is a separate offscreen renderer, not Rerun. Standard
alpha blending and sprite projection need not match Rerun's appearance exactly.
The scene is transformed with the already-audited fixed world/window mapping;
camera look-at matrices preserve handedness. Views naturally clip at their
frustums, but every point is submitted to each draw call.

The metric path remains separate: all retained 2cm voxel centroids in the body
neighborhood are used, never a visualization sample. Earlier generic scorer
defaults remain unchanged for reproducibility. The new review explicitly sets
the legacy filter. Red markers are provisional capsule overlaps, not validated
physical penetration. The historical bulk scoring run stays paused; the user's
subsequent request is evaluated in a fresh legacy-filter run described below.

Yonsei artifacts: Scratch2 Cosmos
`operations/gt_scene_collision_test71_20261030/visual_review_legacy_allpoints_sdk`.
The isolated ModernGL dependencies are in the operation's `opengl_modules`,
without changes to training environments. Rendering uses one GPU from the
retained allocation; source, protocol and Notion hash receipts are saved.
Trust `COMPLETE.json` and verified publication receipts for completion.

## GT metrics with the approved cloud

The user requested a complete GT rerun after approving the point cloud. New
results are in the operation's `metrics_legacy_test71`, with explicit
`--cloud-filter nymeria_legacy_sdk --previews 0`. The four approved videos are
reused. Historical cached results are rejected if their filter differs.
The CLI defaults to the approved filter; `--cloud-filter historical` exists
only for explicit reproduction of earlier generic-Aria results.

The metric definition stays fixed: the full filtered head cloud is represented
by all retained 2cm voxel centroids within the conservative body-neighborhood
crop, with no random point-count sampling. A primary frame flag requires at
least three connected voxel centroids more than2cm inside the subject-scaled
capsule union. No registration is fitted, no points are removed using GT
overlap, and radius/margin sensitivities remain0.8/1.0/1.2 and1/2/3cm. Report
the frame rate, body-part/frame rate, affected-window rate, capsule-intrusion
depth summaries, per-case values and sensitivity grid. Depth is measured
inside the body proxy, not as signed scene penetration.

Every window must pass source/statistics/canonicalization parity, original
headset clock conversion, SLAM-camera pose agreement and graph consistency.
Missing head-clock VRS files are extracted from authorized manifests with
bounded HTTP ranges and ZIP CRC checks, using four prefetch workers while
scoring proceeds. No full archive fallback. New metrics for the four approved
visual cases match the review's metrics and coverage exactly.

Completed all71windows/6887frames without skips or failed alignment checks.
GT primary frame flags:2111/6887(30.65%); affected windows50/71(70.42%);
body-part/frame flags4.03%; foot/nonfoot frame flags15.28/26.88%. The latter
can overlap. Pooled flagged-voxel capsule intrusion mean/p95 is3.65/6.56cm.
At the2cm margin, radius scales0.8/1.0/1.2 give17.77/30.65/42.40% frame flags.
These do not validate physical collisions; capsule anatomy, ordinary contacts,
sparse surfaces and dynamic-map content remain relevant to interpretation.

Maximum SLAM-camera translation mismatch is0.0153mm, rotation mismatch
2.13e-6degrees, nearest-time mismatch0.5255ms, and world/window roundtrip
error0.000807mm. Head-to-RGB-camera distances span12.10–16.73cm (median13.83cm).
The checked source/normalization/window protocol is preserved in the attached
quantitative ZIP. The four approved visual cases reproduce all metric and
coverage fields exactly.

`metrics_legacy_test71/COMPLETE.json` and
`final_publication.json=complete_verified` establish completion. The Notion
quantitative table was read back, and the uploaded PNG and quantitative ZIP
were downloaded and SHA256-verified. `results.json`, `per_case.csv`,
`per_frame.csv`, `summary.json` and `REPORT.md` retain the full evidence.

## Thigh/shin/foot exclusions: current GT results

`rescore_gt_scene_collision.py --source <audited-full-body-run> --out <fresh-run>`
reuses the complete approved-cloud voxel geometry and saved alignment checks.
It decodes the exact GT bundles and reproduces the full-body control for every
window before recomputing the changed capsule union and connectivity. Each
reduced-policy intrusion-point set must be a subset of the full-body result.
This recomputation is required; subtracting old leg flags would leave the wrong
cluster support and depth values. All71control/parity and subset checks passed.

GT71 /6887frames:640flagged frames (9.29%),25affected windows (35.21%), and
1.89% active body-part/frames (16capsules). Pooled flagged-voxel capsule intrusion
mean/p95 is4.47/8.59cm. The conditional depth can rise when shallow leg overlaps
are excluded; it is calculated over the remaining flagged observations.
The historical full-body frame rate was30.65%. All thresholds, cloud points,
normalization, GT windows and alignment are unchanged.

Current artifacts are in `metrics_legacy_test71_exclude_thigh_shin_foot` under
the same operation. The new Notion table was read back and its chart and ZIP
were remotely hash-verified. New scene overlays in
`visual_review_legacy_exclude_thigh_shin_foot` use the same exclusion policy;
the legs remain visible but cannot produce red collision markers.

## Full-test modality comparison, completed 2026-10-31

[PHASE2_SCENE_EVALUATION_20261030.md](PHASE2_SCENE_EVALUATION_20261030.md)
records the completed matched11892caption/window cases,11657physical windows
and71recordings for GT, text only, image only, image+text, and the new
image+text→video→motion pipeline. This larger cohort has GT30.929% intrusion
frames; it must not be confused with the fixed71-window GT9.29% above.
All use the approved legacy cloud and16active capsules excluding thighs,
shins and feet. Full-cohort GT frame/body/depth evidence matches between
evaluations exactly. Two Notion tables, four quantitative uploads and eight
side-by-side videos were verified remotely. The monitor continues every
30minutes after completion; inference checkpoints and training are unchanged.
