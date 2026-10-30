# Phase2 modalities, scene intrusion and generated-video motion

Completed 2026-10-31; operation started 2026-10-30. These are inference-only experiments: no
training, loss, checkpoint or frozen reasoner computation is modified.

## Cohort and reference

The immutable input list comes from Scratch2 Cosmos
`operations/phase2_generation_modalities_20261028`:11892caption/window cases,
11657physical windows and71recordings. Its configuration supplies the exact
Self→cross/final-hidden/Knorm/caption-only B256200k checkpoint, hashes,
generator statistics, evaluator checkpoint and separate evaluator statistics.
Use the saved paths rather than reconstructing checkpoint names.

Direct variants are actual text-only, image-only with empty caption, and
image+text. Native empty-caption/no-image T2M is the common CFG reference.
Retain original batches of4, paired per-case CPU motion noise, neutral shapes,
T97/20fps, UniPC35/shift3/CFG2. First regenerated batch raw features match the
original saved outputs bitwise; `first_batch_generation_parity.json` is proof.
GT embeddings reproduce the existing CPU-decoded reference. Retrieval uses
32candidates and3grouping seeds. Always include measured GT in reports.

Full direct measurements completed all11892cases, with no skipped cases:

| Input | Collision frames % | Mean/P95 intrusion cm | R@3 % | FID |
|---|---:|---:|---:|---:|
| GT | 30.929 | 4.211 / 8.258 | 91.695 | 0 |
| Text only | 37.684 | 4.399 / 8.621 | 96.367 | 0.019463 |
| Image only | 34.291 | 4.295 / 8.440 | 52.597 | 0.029787 |
| Image + text | 36.751 | 4.351 / 8.559 | 97.077 | 0.018421 |
| Image + text → video → motion | 36.404 | 4.327 / 8.497 | 87.562 | 0.082044 |

Image+text minus text-only is -0.933percentage points; recording-bootstrap
95% interval[-1.953,-0.054]. Image+text minus image-only is+2.459points,
interval[0.401,4.594]. Image-only's lower overlap rate comes with much poorer
caption retrieval. These observations do not validate physical collisions or
establish better motion quality from overlap alone. One retained source case
has10head/camera separation outlier frames. Final remote table readback and
PNG/quantitative ZIP SHA256 are verified; the direct operation's
`COMPLETE.json` and `publication.json=complete_verified` record completion.
The generated-video pipeline also completed all11892cases with zero skips.
Its collision rate is 0.346 percentage points below direct image+text, with
recording-bootstrap 95% interval [-1.684, 1.193]. This does not establish a
collision improvement. Retrieval and FID are worse. Its original foot-skating
metric is lower (0.069674 versus 0.119762), but feet-above-5cm frames are more
frequent (20.910% versus 15.458%); lower skating alone does not show better
grounding. GT feet-above-5cm is 16.267% on this same cohort.

## Scene metric

Use the approved head-map SLAM graph and finite raw points satisfying SDK
confidence thresholds: inverse-distance std<0.0004 and distance std<0.02m.
Round confidence fields to float32, promote for comparison, and do not cap
the display point count or remove GT-overlap points. Display uses every
retained point and a depth buffer.

Metric points are2cm voxel centroids. Flag a frame when at least3connected
voxels lie more than2cm inside the capsule union. Exclude both thighs, shins
and feet before computing any union, clusters or body-part statistics;
16capsules remain, including hips. Preserve radius/margin sensitivities.
This is a capsule/cloud intrusion proxy, not validated physical penetration.

Use audited Aria→Kimodo basis, cumulative source-window yaw/origin and
calibrated floor. Canonicalize GT's first delta only. Preserve every generated
feature without prediction fitting, recentering or grounding. An acceleration
crop must cover GT and all compared generated trajectories with ample margin;
it cannot be limited to the GT path. Final pipeline aggregation verifies its
GT frame flags against the direct full cohort exactly.

A stored head/camera inconsistency was found near the end of a Sherri Scott
window. Camera/motion timestamps agree and independent full-source versus
window world reconstruction differs by only6.56e-8m. Anatomical separation
is therefore a source-quality warning, not a reason to drop the benchmark
case. Flag these frames, retain all poses, and require independent world
roundtrip agreement within0.1mm for flagged cases. The direct operation saves
the diagnostic, original gate/source and correction record.

The earlier GT71fixed-window result9.29287% uses a different cohort and must
not substitute for the full-test GT measurement.

## New two-stage condition

1. Supply only firstRGB+caption to the same frozen Phase1EMA200k video
   specialist, with bridges disabled, native UniPC30/shift10/guidance1.
   Pin the first causal Wan latent every step. No motion/camera or future
   ground-truth RGB is an input. A zero/ones motion-template test verifies
   video independence bitwise.
2. Decode to lossless uint8RGB, restore the observed first frame and re-encode
   using the training Wan pipeline. MP4 is an export, not the inference input.
3. Original Phase3baseline200k reconstructs motion with UniPC35/shift3 and
   `null + 2*(caption-null)`. Keep generated video and enabled bridges
   identical in both branches. Inject caption into both reasoners using the
   existing audited V2M caption context. No video CFG. Initial motion noise
   matches the direct variants. Preflight verifies native-null parity,
   unchanged video and70conditional/null calls.

This compares specialist Phase2 generation with a Phase1→Phase3 pipeline;
changes cannot be attributed solely to adding generated video.

## Reusing saved generated motions

All four conditions retain every generated motion: 11892 per condition,
47568 trajectories total, each97frames at20fps. A full read/finite-value
inventory is saved as `generated_motion_inventory.json` in the pipeline
operation. Regeneration is unnecessary for additional metrics.

- Direct operation `samples/batch_*.npz`: 2973 files with `raw_features`
  of shape `[3,4,97,283]`, float32. Axis0 order is **image_only, text_only,
  image_text**; `case_id` identifies the four cases in each batch. Batch-file
  starts use filtered manifest positions, not the numeric prefix of a case ID.
- Pipeline operation `case_outputs/<case_id>/motion.npz`: 11892 files,
  each containing `raw_features` and `normalized_motion`, both `[97,283]`
  float32. The matched caption/window/recording/neutral/GT references are in
  each operation's identical `cases.json`.

`raw_features` is generator-denormalized UniEgo283 in the camera/head-aligned
representation; do not apply generator mean/std a second time. Decode SOMA30
joint positions using `camera_head_recanonicalization.decode_uniego` as in
the existing scene scorer. Keep the saved window/world transform for scene
metrics and use a new evaluator's own normalization when that metric requires
it. Neither saved predictions nor their placement should be fitted to GT.

## Final acceptance and interrupted export recovery

`full_completion_audit.json` verifies all11892cases,11657physical windows and
71recordings: no skips, requested per-case guidance, finite saved motions,
exact GT measurement arrays and per-case frame/body/depth evidence against
the direct sweep, inference/checkpoint identities, and quantitative ZIP CRC
and contents. GT retrieval is unchanged; the maximum text-projection difference
from batch-size/hardware rounding is3.28e-7, below the predeclared2e-6 bound.
Existing worker GT guards and all metrics remain unchanged.

The audit found one interrupted export at filtered position11785,
`nymeria_ti2m_11831_S19__20231117_s0_randy_martin_act4_hernaf_5080`.
The expired allocation had saved the atomic video NPZ but left a zero-byte
MP4 and no video-audit JSON. Replaying the same official sampler/seed on
PRO6000 reproduced both native and saved motion-input latents bitwise and
restored the97-frame/20fps export. Hashes confirm the original video cache,
motion, motion audit, measured batch and collision evidence are unchanged.
See `interrupted_export_recovery.json` and the retained failed/then-passed
acceptance logs. The frozen inference driver was not modified.

`publication_readback_audit.json` freshly checks both Notion tables, four
quantitative PNG/ZIP uploads and all eight published comparison videos against
their local files. `post_completion_monitor_acceptance.json` verifies the
02:16KST scheduled tick after quantitative completion, including GPU probes
and Notion update. Monitoring remains active every30minutes after completion.
Retained running allocations were preserved; only the unused pending
evaluation replacement2649070 was cancelled, with a pending-state filter.
The allocation details below are historical, not a current resource listing.

After the user authorized cancelling idle GPUs, jobs2646811,2646945,2646946,
2647359 and2648261 were cancelled (4RTX4090+1RTXPRO6000). Each was verified
idle twice with no GPU processes; controller terminal states are recorded in
`idle_allocation_cleanup_20261031.json`. Shared job2608785 remains because
its second card still hosts the application. Its evaluation card is idle.
Other pending requests and the persistent30-minute monitor were preserved.

## Code, artifacts and operations

- Direct entry: `motion_expert_t2m_edge/evaluate_scene_modalities.py`.
- Two-stage entry: `motion_expert_t2m_edge/evaluate_generated_video_motion.py`.
- Shared geometry: `nymeria_baselines/phase2_collision_scene.py`.
- Aggregation/publication: `phase2_collision_report.py` and
  `generated_video_motion_report.py` in this directory.
- Direct operation: `/scratch2/whwjdqls99/cosmos/operations/phase2_scene_collision_20261030`.
- Pipeline operation: `/scratch2/whwjdqls99/cosmos/operations/phase2_generated_video_motion_20261030`.
- Each saves configuration/source identities, raw generated features,
  resumable claims, per-case/per-frame collision evidence, standard quality
  metrics and recording-bootstrap paired95% intervals (5000draws).
- Completion requires all cases, final aggregation and verified Notion table
  readback plus remote media/quantitative-archive SHA256. A launch note or
  preflight is not completion. Report page:
  https://app.notion.com/p/407d2a44f44b81a5ad53dbb685cdbd9e .
- Three direct full-cloud scene comparisons and two pipeline comparisons
  are already published with verified hashes. Pipeline panels show GT video,
  generated video, observed first image, GT motion and all four predictions.
  Green identifies observed RGB; amber identifies generated video frames.
- Three further five-way scene comparisons include the generated-video motion,
  using the same complete approved cloud, depth buffer and camera framing.
  They use fixed prior71review cases0,1,3 (full-case indices0,138,407) and are
  remotely hash-verified under the pipeline's `scene_comparisons`. All eight
  published videos fully decode to97frames/20fps with uniform timestamps;
  see `published_video_acceptance.json`.
- `handoff_output_audit.json` checks the first345completed pipeline cases:
  GT frame/body collision flags and intrusion-depth arrays match the direct
  evaluation bitwise, generated motions/metrics are finite, and guidance
  audit records preserve no video CFG and text CFG2. This is a correctness
  audit of that subset, not an interim aggregate performance claim.
- `source_qc_case_acceptance.json` verifies the known original06703/Sherri
  Scott source-head outlier at filtered position6677 completed without being
  skipped. All10outlier frames remain flagged; GT frame/body flags and depth
  arrays match the direct evaluation bitwise, with6.56e-8m world roundtrip.
  Final pipeline archives include the acceptance records, published-video
  decode checks and observed worker GPU metadata under `execution_resources`.

All model/data/scoring/rendering work runs in owned Yonsei compute allocations.
Source/log reads and publication can run on login. Do not use cs-gpu-01 or
node37. Existing worker2608785/node33 uses only UUID
`GPU-247c7e54-ee27-bb1f-6b9d-02c909b7417a`; the other allocated GPU belongs to
a separate application. A Slurm step there must expose both GPUs before
masking by the intended UUID. New single4090 requests use8CPUs,64GB,3days.
Job2647359 accepts a 2-hour minimum backfill slot while
retaining the 3-day maximum (`--time-min=02:00:00`). Its minimum was reduced
from 8 hours at 2026-10-31 00:21 KST as the evaluation nears completion.
The pipeline is resumable
per case; running workers, checkpoints and sampling settings are unchanged.
Request2646946 remains the fixed3day request. Controller readback is saved
in `backfill_request_2647359.json` in the pipeline operation.
Request2646946 started on node35 at 2026-10-31 00:30:48 KST, with its full
3-day allocation. The 00:46 scheduled check verified its actual RTX4090,
active evaluation PID and progressing case outputs. Eight workers were live
(4 RTX4090 + 4 RTXPRO6000), with 8815 cases complete and no worker errors.
Job2647359 subsequently started on node40 at 01:14:23 KST with its full
3-day allocation. The 01:16 tick verified nine live workers
(5 RTX4090 + 4 RTXPRO6000), 10004 completed cases and no worker errors.
Its worker log confirms measured case outputs after model loading.

Additional PRO6000 capacity was available with only2CPUs free on the ASUS
nodes. Single-GPU request2648261 therefore uses2CPUs/32GB; it obtained node56
for an8hour backfill slot, ending2026-10-31 04:24KST. The frozen model and
sampler source remain identical. `pro_first_case_acceptance.json` verifies a
PRO-owned completed case: finite motion, requested guidance, and bitwise GT
frame/body flags and intrusion depths against the direct sweep. Request2648286
expanded to ASUS/Gigabyte with a4hour minimum to fit a shorter backfill slot;
it obtained node55 until2026-10-31 01:25:29KST and is producing measured cases.
The3day maximum is retained when a longer slot is available. Request2648659
obtained another node55 GPU at22:32:20KST for4hours, ending2026-10-31
02:32:20KST, and is producing cases. Its2CPU/32GB resources and
4hour-minimum/3day-maximum request are unchanged. Replacement2648691
started early on node55 at21:59:55KST for4hours,
ending2026-10-31 01:59:55KST, and is producing cases. At the22:16scheduled
check all six workers were live (3RTX4090+3RTXPRO6000),3663cases were complete,
and no worker errors appeared; the status was remotely verified in Notion.
Completed cases and video/motion caches are reused. Current queued/running
state must be read from Slurm and live processes.
The22:46scheduled tick verified seven live workers (3RTX4090+4RTXPRO6000),
4634completed cases and zero errors. Backup2649070 covers ASUS/Gigabyte with
the same single-PRO resource request, queued before the short allocations end.
Node58 has rebooted since its old reported anomaly; if allocated there, verify
current CUDA health. The original4090 workers continue.
Inference now spans RTX4090 and RTXPRO6000. Actual GPU names/UUIDs are saved
in `allocations/JOB.gpu.json` by the live monitor, with separate controller
and hardware records for the PRO allocation. `runtime_throughput_snapshot.json`
records recent real completion intervals; it is not a controlled hardware
benchmark. Recovery reads the per-job `.cpus` sidecar so a2CPU allocation is
not asked to start an8CPU step.

Tmux `eval_monitor_30min_20261030` keeps checking and updates one verified
Notion status block every1800seconds, including after completion as explicitly
requested. Its script/history/status live in the direct operation. When
the direct sweep completes it hands known idle evaluation GPUs to the
two-stage pipeline. Preserve allocations, check time remaining and queue
replacements as needed. Do not disturb unrelated jobs or the other node33 GPU.

The monitor now records live GPU utilization/memory, compute PIDs and matching
job-local evaluation processes through `live_gpu_health.py`. Unknown queries
or timeouts are not evidence of a stopped worker and never cause a restart.
Twice-verified idle allocations can resume via `worker_resilient.sh`, after
a120second startup grace. That wrapper checks case locks before loading a
model, rescans unfinished cases released by an expired peer, preserves cached
generations/measurements, and serializes recovery per allocation and final
publication globally. It does not alter any sampling settings. Real
cross-process flock checks passed on2646811; `recovery_acceptance.json` is
the evidence. Currently healthy evaluation PIDs are not interrupted.
