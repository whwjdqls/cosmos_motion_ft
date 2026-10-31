# Shared Agent Context

## Central Phase2 A0–A3 implementation (2026-10-31)

- A2 follow-up: full Kimodo369 stats completed and hashes/counts verified:
  120929windows/11888119frames,642recordings, non-limited train split. Shared
  path `/scratch2/whwjdqls99/cosmos/stats/kimodo369_camhead_v1` is readable by
  groupys_ms (including minsu); export PHASE2_KIMODO_STATS_DIR to that path.
  Actual A2 training-source Nymeria+BONES rows,369-D collation and physical-loss
  backward pass with the completed stats passed on cnode01/job2653025.
  Evidence `phase2_heads_fix_20261031/a2_runtime_ready.json`. Ready for GPU
  smoke/capacity acceptance; no full frozen-Edge GPU test/production start is
  claimed by whw. The first diagnostic incorrectly used train=False, selecting
  BONES validation; the corrected check uses actual training sources. This
  does not certify completeness of every held-out BONES file.
- Historical velocity clarification: commit890eec6 (2026-07-02, "update bs_
  scripts for v prediction") implements direct v=eps−x0 feature MSE plus decoded
  clean-pose/smooth losses in bs_train.py. Its loop uses gradient clipping and
  nonfinite checks, with no max(50,25*EMA) batch guard; no planted-foot penalty
  appears in that commit. This supports the user's recollection that velocity
  could learn. The exact historical learned checkpoint has not been identified.
  Do not generalize the limited new A1 initialization probes into a claim that
  velocity prediction itself cannot learn. The proposed EMA-bootstrap change
  remains unapplied; no A0 process or allocation was changed.

- Follow-up fix: A3's minsu GPU preflight failed before training because the
  Edge-local `motion_heads.MotionHeads` did not accept the new `motion_dim`
  keyword. Fixed that actual wrapper, preserving283-D initialization/state/
  outputs bit-for-bit. Added real local-head constructors,369-D forward/
  backward/reload and production model-constructor coverage to the CPU check.
  All15 targeted checks pass on cnode01/job2653025. This is CPU constructor
  acceptance, not a successful full frozen-Edge GPU preflight. A0 was not
  restarted and no allocation was changed. Evidence:
  `/scratch2/whwjdqls99/cosmos/operations/phase2_heads_fix_20261031`.
  Each smoke/capacity retry now uses a fresh timestamped output, preserving
  failed evidence and avoiding stale-source/finished-checkpoint reuse. The
  production resume source-fingerprint guard remains strict.

- Read `motion_expert_t2m_edge/ABLATIONS_PRO6000_20261031.md` for the current
  implementation, minsu thin wrapper, exact conditioning partition and acceptance.
  Central entrypoint: `motion_expert_t2m_edge/run_ablation.sh`; no production
  trainer was launched. Existing allocations/requests are retained.
- User explicitly chose historical BONES double dropout (19% effective) for
  all four runs. A3 is T2M/TI2M50/50, BONES75% within T2M, Nymeria T2M dropout10%,
  TI2M dropout30%. Expected totals: text41.625%, image+text35%, image-only15%,
  text/image-null8.375%. Null still receives shape and valid length.
- A1 is direct flow velocity, physical losses on reconstructed clean motion.
  Default x0/UniEgo checkpoint contracts are unchanged; new output objective/
  codec/stat identity is explicit and strictly checked. Do not treat A1's raw
  output as x0. High-noise zero-head A1 CPU examples trigger the unchanged Nano
  loss guard; full-batch GPU acceptance must verify actual nonzero updates.
  Follow-up CPU batch256 initialization probe (three seeds; two real windows
  repeated at the reference source/task proportions) measured A1 losses
  90.3778/93.3365/92.6937, all above the initial guard50. A0 was0.64435.
  These are not representative training batches, but the guard would zero all
  three probes, with foot velocity dominating the A1 losses. A1 is not cleared
  for production; do not equate optimizer-step counters with nonzero updates.
  Evidence: `phase2_heads_fix_20261031/a1_guard_probe.json` in scratch2 operations.
- A2 uses the existing SOMA30 Kimodo369 layout with original contact labels,
  the same calibrated windows/shapes and source UniEgo guard. Its own train-only
  stats use the original120929-window/11888119-frame Nymeria population. Do not
  feed369 features into283-only evaluation drivers without the proper adapter.
- CPU job2653025/cnode01: new objective/codec/dropout plus historical regression
  checks pass (30 unique tests); real BONES/Nymeria positional roundtrip errors
  below3e-8m and exact contacts. No GPU/account acceptance under minsu is claimed.
  Limited statistics smoke:172windows/17134frames; not production-valid.
- Full A2 statistics preparation completed on the four-CPU allocation retained
  by tmux `p2_ablation_cpu_20261031`, under
  `/scratch2/whwjdqls99/cosmos/stats/kimodo369_camhead_v1`. Its complete manifest,
  counts and hashes were verified; see the completion note above. Evidence:
  `/scratch2/whwjdqls99/cosmos/operations/phase2_ablation_impl_20261031`.
  Notion report: https://app.notion.com/p/409d2a44f44b81ff9d2cd1efaedd666e .

## Shared source across Yonsei accounts (2026-10-31)

- Execution for the new Phase2 ablations is assigned to minsu1206. The user's
  explicit follow-up says **do not cancel the existing allocations**. Preserve
  whw-owned requests2650571–2650574 and other allocations; no cancellation
  occurred when this destination decision was recorded.
- User wants the implementation kept in `/home/whwjdqls99/cosmos_motion_ft`.
  Under `minsu1206`, keep only thin launch scripts invoking this checkout;
  do not fork/copy the training implementation into that account.
- `minsu1206` exists and shares group `ys_ms`. The central checkout, framework,
  environment and inspected base-model/stat metadata have readable/traversable
  permissions. This is a permissions inspection, not a minsu-account GPU test.
- Launch jobs under their real owning account. Checkpoints, logs, W&B state,
  mutable caches and build caches should use that account's writable storage.
  Some current Yonsei launchers force whwjdqls99 paths; configure/adjust those
  centrally before reuse rather than assuming a copied wrapper is sufficient.
  Preserve data/model/stat identities and record the source revision per run.
- `/home/minsu1206` is private (0700); install its thin launch scripts from
  that account. Do not broaden home permissions or copy credentials.

## Phase2 one-GPU training benchmark (2026-10-31)

- `motion_expert_t2m_edge/TRAINING_BENCHMARK_20261031.md` records the real
  Self→cross/Knorm/caption-only recipe on the idle RTX4090 of2608785/node33.
  Forty isolated updates, zero skipped updates, no production checkpoint changes.
  Microbatch128: accum1 mean1.745s/median0.959s; accum2 global256
  mean3.338s/median2.461s. Loader waits average0.783/1.460s respectively.
- nvidia-smi peak22.484GiB; PyTorch peak allocated21.275GiB/reserved22.012GiB.
  This short run does not establish exhaustive token-length capacity or DDP
  throughput. The process exited0 and the card is idle again; the other card's
  application remains untouched. Scratch2 Cosmos operation
  `phase2_training_benchmark_20261031` has per-update timings and provenance.
  Report: https://app.notion.com/p/409d2a44f44b81e5b4c7d2266be3e9c3 .

## Full-test modality collision and generated-video motion evaluation (2026-10-30)

- `nymeria_baselines/PHASE2_SCENE_EVALUATION_20261030.md` records two completed
  inference-only evaluations. Both reuse the immutable11892caption/window
  cases (11657physical windows,71recordings), frozen Self→cross/Knorm/
  caption-only B256200k specialist, evaluator-specific statistics and measured
  GT from `phase2_generation_modalities_20261028`. Exact checkpoint paths and
  hashes come from saved configurations, not launcher defaults.
- Direct GT/text-only/image-only/image+text regeneration preserves the original
  four-case batches, noise and UniPC35/shift3/CFG2; its first saved batch matches
  the original raw predictions bitwise. Retain raw features and per-frame
  collision evidence. Original71fixed-window GT metrics are a different cohort.
  Direct evaluation completed all11892cases with zero skips; Notion table
  readback and quantitative ZIP/PNG SHA256 are verified (`COMPLETE.json`).
  Collision-proxy frame rates:GT30.929%,text37.684%,image34.291%,both36.751%.
  Both minus text:-0.933points, paired recording-bootstrap95%[-1.953,-0.054].
  Image-only retrieval remains much poorer; overlap alone is not motion quality.
- New condition: first image+caption generates video with the original frozen
  Phase1EMA200k, bridges disabled, native UniPC30/shift10/guidance1. Decode to
  lossless RGB, restore observed frame0 and Wan re-encode before original
  Phase3baseline200k reconstructs motion with UniPC35/shift3/textCFG2. Both
  CFG branches retain exactly the same generated video and enabled bridges;
  there is no videoCFG, GTmotion or futureGTRGB input. Preflight passed.
- Both evaluations completed all 11892 cases without skips on 2026-10-31.
  The generated-video pipeline has collision-proxy frames 36.404%, R@3 87.562%,
  FID 0.082044 and feet-above-5cm frames 20.910%. Direct image+text is
  36.751%, 97.077%, 0.018421 and 15.458%, respectively. The collision
  difference is -0.346 percentage points, recording-bootstrap 95% interval
  [-1.684, 1.193]; there is no clear collision improvement. These compare
  a Phase1→Phase3 pipeline against a Phase2 specialist, not just video input.
  `full_completion_audit.json` verifies all cases' guidance, finite motions,
  exact GT measurements/body flags/depths, checkpoint/source hashes and ZIP.
  `publication_readback_audit.json` independently verifies both remote tables,
  four quantitative uploads and all eight previously fully decoded videos.
  One time-limit-interrupted MP4/audit at position11785 was repaired by replaying
  the same official sampler/seed on PRO6000. Native and motion-input latents
  reproduce bitwise; cached inference inputs, motions and scores are unchanged
  (`interrupted_export_recovery.json`). Do not infer complete display exports
  merely from an atomic generated-video NPZ. The inference source is frozen.
  The 02:16 scheduled monitor tick verified completion and continued GPU/Notion
  checks afterward (`post_completion_monitor_acceptance.json`). Retained active
  allocations were preserved. Only unused pending replacement2649070 was
  cancelled after the full acceptance passed; other requests were untouched.
- Follow-up user authorization to cancel idle GPUs (2026-10-31, about03:00KST):
  cancelled fully idle jobs2646811,2646945,2646946,2647359 and2648261,
  releasing4RTX4090s and1RTXPRO6000. Slurm controller confirms CANCELLED
  for all five (`idle_allocation_cleanup_20261031.json`). Job2608785 remains
  because its other card, UUID8c3f840c, still serves the application; the
  evaluation card UUID247c7e54 is idle. Pending PRO requests were untouched.
  This user-authorized cleanup supersedes the earlier retention preference
  for these five completed allocations. The30-minute monitor remains active.
- Use approved SDK legacy clouds, fixed source-window transforms,16active
  capsules excluding thighs/shins/feet,2cm metric voxels and connected support.
  Shared scene support covers every compared trajectory. Never fit, recenter
  or ground predictions. Source head/camera anatomical outliers are reported
  without dropping cases or modifying poses; independent full-source/window
  world reconstruction must still agree within0.1mm. The operation retains
  the failing gate, diagnostic and corrected source snapshot.
- Scratch2 Cosmos operations: `phase2_scene_collision_20261030` and
  `phase2_generated_video_motion_20261030`. Trust each `COMPLETE.json` plus
  verified Notion publication receipt for completion. Three direct scene
  comparisons and two two-stage examples are already remotely hash-verified
  on https://app.notion.com/p/407d2a44f44b81a5ad53dbb685cdbd9e .
  Three further scene comparisons add generated-video motion as a fifth
  method on the fixed prior71review cases0,1,3 (full-case indices0,138,407).
  All eight published videos fully decode to97frames/20fps with uniform
  timestamps; `published_video_acceptance.json` records this verification.
  `handoff_output_audit.json` verifies the first345completed pipeline cases'
  GT frame/body flags and intrusion depths against the direct sweep bitwise,
  with finite outputs and the requested no-videoCFG/textCFG2 settings.
  `source_qc_case_acceptance.json` also verifies filtered position6677 (the
  original06703/Sherri Scott source-head outlier): completed, not skipped,
  all10frames flagged and GT flags/body hits/depths bitwise equal to direct.
  World roundtrip remains6.56e-8m. The final pipeline quantitative archive
  includes these acceptance records and observed worker GPU metadata.
- GPU workers during the completed evaluation (historical; query current Slurm):
  2608785/node33/evaluation UUID247c7e54 only,
  2646811/node40, 2646945/node39 and 2646946/node35. The latter started at
  2026-10-31 00:30:48 KST and runs through November 3 with the same immutable
  evaluation. Job2647359 started on node40 at 2026-10-31 01:14:23 KST with
  one RTX4090, 8 CPUs, 64 GB and a full 3-day allocation. Both additional
  requests exclude cs-gpu-01,node37,node31,node33.
  Request2647359 permits a 2-hour minimum for earlier backfill, reduced from
  8 hours at 2026-10-31 00:21 KST as the evaluation nears completion;
  its maximum remains 3 days. Request2646946 retains a fixed 3-day request.
  Controller readback and unchanged model settings are recorded in
  `backfill_request_2647359.json` under the pipeline operation.
  Follow-up:2648261 obtained one RTXPRO6000 on node56 with2CPUs/32GB,
  fitting the otherwise idle GPU/2CPU capacity. It received an8hour backfill
  allocation ending2026-10-31 04:24KST. The same frozen inference code is
  advancing; `pro_first_case_acceptance.json` verifies finite motion, both
  guidance branches, source identity and bitwise GT collision flag/depth
  parity on an observed PRO-owned case. The second single-PRO request2648286
  broadened to ASUS/Gigabyte and a4hour minimum; it obtained node55 for4hours,
  ending2026-10-31 01:25:29KST, and is producing finite measured cases.
  Single-PRO request2648659 obtained another node55 GPU at22:32:20KST for
  4hours, ending2026-10-31 02:32:20KST, and is producing measured cases.
  Its2CPU/32GB resources and4hour-minimum/3day-maximum request are unchanged.
  Replacement2648691 started early on
  node55 at21:59:55KST for4hours (ends2026-10-31 01:59:55KST) and is advancing
  the same case queue. The22:16tick verified all six workers live (3RTX4090,
  3RTXPRO6000),3663cases complete, zero errors, and Notion status readback.
  Later status must come from current Slurm/process evidence.
  The22:46tick verified seven live workers (3RTX4090+4RTXPRO6000),4634cases
  complete and no errors. Backup2649070 was queued across ASUS/Gigabyte with
  the same single-PRO resources, ready for a short allocation's expiry.
  The 2026-10-31 00:46 scheduled tick verified eight live workers
  (4 RTX4090 + 4 RTXPRO6000), 8815 completed cases, no worker errors and a
  verified Notion status update. Current state still requires live evidence.
  The 01:16 tick verified nine live workers (5 RTX4090 + 4 RTXPRO6000),
  10004 completed cases and no worker errors. New worker2647359 subsequently
  wrote measured cases; both requested extra RTX4090s are now in use.
  Never restart a live worker from a stale
  observation; case claims and saved video/motion outputs remain authoritative.
  Inference used RTX4090 and RTXPRO6000. Node58 rebooted2026-10-26 after
  the old anomaly; verify current CUDA if that requested allocation lands there.
  Recent real outputs show~10.8s/PRO case and~12.6–12.8s/4090 case; these
  are operational observations on different cases, not a hardware benchmark.
  Recovery reads optional `allocations/JOB.cpus`, preserving the2CPU request.
  Node33's other GPU belongs to the Habitat/application task; leave it alone.
  For2608785 expose both allocated GPUs in the Slurm step, then select the
  evaluation UUID explicitly: `--gres=gpu:1` selects the other GPU there.
- User explicitly requested tracking every30minutes even after completion.
  Tmux `eval_monitor_30min_20261030` persists that monitor and verifies a Notion
  status update each tick. `live_gpu_health.py` checks actual owned GPU UUIDs,
  compute PIDs and matching job-local evaluation processes. Unknown/timeout
  observations never trigger a restart. A twice-verified idle allocation can
  resume through `worker_resilient.sh`; healthy live workers remain untouched.
  Its real cross-process flock acceptance verifies held cases are not stolen
  and released unfinished cases become claimable, avoiding holes after expiry.
  The wrapper also uses a per-allocation recovery lock and shared summary lock.
  Newly started allocations have a120second startup grace. Completed direct
  GPUs have transferred to the pipeline; check remaining allocation time.
  Keep the explicit monitoring goal active until the user ends it.

## Collision checks exclude thighs, shins and feet (2026-10-30)

- User requested ignoring both thighs, shins and feet when checking collisions.
  This is now the default `collision_body=exclude_thigh_shin_foot` in the scorer
  and scene-review CLI. Keep hips, torso, head and arms:16active capsules.
  Exclude the six leg capsules before union depths, connected-cluster support,
  frame flags and body-part statistics. Keep the full skeleton for visual context.
  Historical22-capsule reproduction requires `--collision-body full_body`.
- `nymeria_baselines/rescore_gt_scene_collision.py` reused audited approved-cloud
  geometry and recomputed all71windows/6887frames. All71full-body controls
  reproduce the previous numeric metrics/hits exactly; reduced point IDs are
  subsets and all excluded-capsule flags are false. No downloads or scene changes.
- Current GT result:640/6887frames (9.29287%),25/71windows,1.89215% active
  body-part/frames; pooled capsule intrusion mean/p95 4.46633/8.59478cm.
  Prior full-body result30.65195% is historical. New artifacts:
  Scratch2 Cosmos `operations/gt_scene_collision_test71_20261030/`
  `metrics_legacy_test71_exclude_thigh_shin_foot`. Table readback and quantitative
  ZIP/PNG SHA256 are verified on the existing Notion report. The proxies remain
  distinct from validated physical collision labels.
- Matching four-view overlays are regenerated under
  `visual_review_legacy_exclude_thigh_shin_foot`; use their own `COMPLETE.json`
  and publication receipts to establish visual completion.

## Approved Nymeria scene visualization default (2026-10-30)

- User confirmed the point cloud in the final legacy-filter review is correct
  and explicitly requested using it from now on. Use
  `nymeria_baselines/review_scene_viewer.py` and `scene_gl_renderer.py` as the
  standard for future scene/body visualizations on Yonsei and Grasp.
- Use head-map points in the matching SLAM graph with SDK confidence filtering:
  inverse-distance std <0.0004 and distance std <0.02m. Round confidence fields
  to SDK float32 storage, then compare as Python floats. Keep every finite
  retained raw point; no count cap, spatial/ceiling cut or display voxelization.
  Do not remove points based on overlap with GT.
- Keep depth-buffered rendering, gray translucent points of radius0.008m,
  GT RGB, two perspective views and a top view, frame-zero head axes and the
  camera/head-derived/torso direction arrows. Preserve audited coordinates.
- Reference artifacts: Scratch2 Cosmos
  `operations/gt_scene_collision_test71_20261030/visual_review_legacy_allpoints_sdk`.
  All four97-frame/20fps videos and PNGs are remotely hash-verified in Notion:
  https://app.notion.com/p/409d2a44f44b81839675f2890d0245c6 .
  See `nymeria_baselines/SCENE_COLLISION.md` for the protocol. Historical generic
  scorer defaults and collision results are not the approved visualization
  source. Follow-up: the user requested GT collision metrics with the approved
  cloud. A fresh rerun is in the operation's `metrics_legacy_test71`, with
  `cloud_filter=nymeria_legacy_sdk`, no mixing of historical cached scores,
  original 2cm voxel/capsule definitions and independent clock/SLAM checks.
  Completed71/71 windows and6887frames, no skips. All source/canonicalization,
  independent clock/SLAM and graph checks passed. Fifty-nine missing head-clock
  members were range-extracted with ZIP CRC checks; original12were reused.
  GT:2111/6887frames flagged (30.65195%),50/71windows,4.03461% body-part/frames.
  Foot/nonfoot frame flags15.27516/26.87672%. Pooled flagged-voxel capsule
  intrusion mean/p95:3.64536/6.56335cm. With2cm margin, radius0.8/1.0/1.2 gives
  17.77/30.65/42.40% frame flags. These remain capsule/cloud intrusion proxies,
  not signed physical scene depth or validated true-collision labels.
  `COMPLETE.json`, `final_publication.json=complete_verified`, both exit codes0,
  remote table readback and PNG/quantitative ZIP SHA256 establish completion.
  All per-window/per-frame data, protocol/source snapshot and four approved
  RGB/scene videos are available on the same verified Notion report.

## Cross-server continuation handoff (2026-10-30)

- `migration/RESUME_YONSEI_20261030.md` documents four unfinished runs:
  control107793, hiddenFiLM92879, trajectoryFiLM63864, EgoControl91345.
  Drive `gggdrive:cosmos_data/yonsei_resume_20261030` has verified documentation,
  source and metadata; the separate checkpoint backup may still be in progress.
  Trust remote `READY.json`, not this launch note, for final handoff readiness.
- All three Phase3 contracts require the historical shared `data.py` hash
  c21a7bb57e2276f668ea7fed5bfba35ea46756f3e9443003c6efce82ac985262.
  A verified overlay from087ecbe2 is supplied only for a dedicated destination
  checkout. Current Yonsei code is untouched. Start with Drive
  `resume_addendum/RESUME.md`, which supersedes the initial bundled guide.
- Operation: Scratch2 `cosmos/operations/server_resume_20261030`. Checkpoint
  worker runs on retained2608785; tmux `server_resume_upload_20261030` monitors
  it and updates Notion page409d2a44f44b817d8e81d32fbe18e99a after verification.
  Destination paths/hardware have not been supplied or GPU-tested.

## Two-task Phase1 final200k guided comparison (2026-10-29)

- Follow-up16:10KST: user authorized releasing idle PRO6000s. Verified both
  GPUs on node55 had no GPU processes/memory use and cancelled2608761 after
  completed training/evaluation. All six other trainers were preserved.
  `operations/status_idle_release_20261029` holds the queue snapshot and verified
  Notion status report. Do not reuse the now-cancelled allocation.
- `native_phase_training/TWO_TASK_GUIDANCE2_20261029.md` records the requested
  final two-task/prefix1/no-text200k versus original200k comparison. Same71
  windows, official EMA UniPC30/shift10. Forward uses cameraCFG2/textCFG1;
  inverse keeps native guidance1. Both no-caption and original-caption
  references are labeled separately; measured GT is included throughout.
- Scratch2 operation `phase1_twotask200k_guidance2_20261029` reuses20 audited
  guided original videos and71 inverse predictions, generates missing results,
  and publishes tables plus fixed-index three-panel videos to Notion page
  `408d2a44f44b80e882cfcc8d8f4b7869`. Completed all71 cases with no skips;
  eight videos, two camera figures and quantitative ZIP remotely hash-verified.
  `COMPLETE.json`, `publication.json=complete_verified` and
  `publication_resume.exit_code=0` establish completion. Publication alone moved
  to the faster login-network connection after ML work finished on node55;
  the original coordinator's exit143 records that handoff, not inference failure.
- Matched no-caption cameraCFG2: original/two-task PSNR19.284/19.751,
  SSIM0.5979/0.6184, LPIPS0.2903/0.2789. Inverse rotation0.3606/0.3142deg,
  scale-matched translation5.754/5.146mm, Sim3ATE3.623/3.164cm. Paired95%
  intervals favor two-task; task mix and training caption policy both differ.

## Joint-trained Phase3 matched V2M evaluation (2026-10-29)

- `nymeria_baselines/JOINT_PHASE3_MATCHED71_20261029.md` documents evaluation of
  the existing joint3/equal200k checkpoint against original200k on the Notion
  Video2motion page. Same71 fixed windows, four video/caption × scale1/2 settings,
  supplied neutral shapes, UniPC35/shift3, unchanged Body23 and Nymeria TMR.
- The guidance reference disables whole bridges and retains captions; it is not
  trained missing-video CFG. Joint training changes task exposure at equal200k.
  This measures V2M performance, not joint-generation sample quality.
- Scratch2 operation `phase3_joint_v2m_matched_20261029` contains configuration,
  parity gates, per-case evidence and Notion receipts. Trust `COMPLETE.json` and
  `notion_publication.json` (`published_verified`) for completion; do not infer it
  from this launch note. No existing trainer or checkpoint was changed.
- Completed08:03KST: all284 predictions, verified Notion tables and quantitative
  ZIP. Caption/scale2 joint versus original: MPJPE13.98/14.13cm,
  FID0.132200/0.140510, R@3 65.10/69.79% (GT70.83%). Native V2M error/FID
  worsen; results are mixed, and paired MPJPE intervals cross zero.

## Foundary Phase1 LoRA plus video projections (2026-10-28)

- User assigned sep-2 GPUs2/3 to a separate two-task Phase1 run with one
  initial RGB frame, LoRA, and trainable video input/output projections.
  Use `native_phase_training/train_phase1_edge_video_io_foundary.sh` and
  adaptation mode `global_lora_video_io`. The usual action modules and
  global Q/K/V/O LoRA16/alpha32 train, plus the full weights/biases of
  `vae2llm` and `llm2vae`. The VAE and remaining backbone stay frozen.
- Video projections retain pretrained initialization and use the LoRA rate:
  effective2e-5 initially, while action modules retain8e-5. Preserve the
  prefix1/no-caption/exclude5 forward:inverse40:25 recipe, T97/20fps/256,
  native loss/noise, FP32 EMA, global32=2×16, accumulation1, and200k updates.
  Initialize from official Edge base with a fresh optimizer and distinct W&B
  run. Never initialize production from the disposable acceptance checkpoint.
- The mode is recorded in the existing run contract and rejects resume as
  standard LoRA. The Foundary launcher accepts explicit prefix1/17 and mode
  overrides while preserving its former prefix17/global_lora defaults.
- 57 CPU tests passed (six Edge trainability/optimizer/contract/preview checks
  and51 existing Nano contract regressions). Config comparison against the
  one-frame baseline has identical model/data/noise/loss/scheduler settings;
  only adaptation mode and optimizer selection add the video projections.
  Two-GPU forward/inverse audits passed with finite gradients and actual
  updates to all four new tensors. They add788672 parameters, for15669440
  total trainable parameters. Full checkpoint2 restored successfully; update3
  changed both video projections and produced four97-frame/20fps video
  comparisons plus four camera plots on the fixed held-out cases.
- Operational plan, source snapshot, tests, probe/resume and production logs
  are in `cosmos_storage/operations/phase1_video_io_p1_20261028` on sep-2.
  Live logs and acceptance records establish preview/production status.
  Existing camera GPUs0/5 and naeive adaptation GPUs1/7 remain separate.
- Production launched with W&B `jungbinc-upenn/cosmos3_camera/51xcizf1`,
  tmux `cosmos_p1_video_io_p1_gpu2_3`. The model/optimizer/scheduler match
  the accepted test; production restores normal40:25 task sampling, uses
  fresh official base loading and the full200k horizon. The operations
  pointer is `cosmos_storage/operations/phase1_video_io_current_ops.txt`.
- Production passed both task audits on both ranks, with nonzero gradients
  and optimizer updates to all four video-projection tensors. All four initial
  video comparisons and four camera plots were verified uploaded to W&B.
  Training continued beyond step33 at roughly1.15s/update. See
  `running_verified.json` for the evidence snapshot; live logs supersede it.

## Foundary two-GPU resumes (2026-10-28 afternoon)

- User requested two GPUs per experiment and authorized any unoccupied GPUs.
  Resume existing camera Phase1 on sep-2 GPUs0/5 and revised x0 naeive
  adaptation on GPUs1/7. Each uses global32=2×16, accumulation1, unchanged
  objectives/optimizer/LR schedule and its original200k horizon and W&B ID.
- Verified complete model/EMA, optimizer, scheduler and trainer DCP components
  at camera checkpoint20048 and motion checkpoint6237. The motion log reports
  SIGKILL on2026-10-28 at08:57UTC; the cause is not established. Camera logs
  end at08:54UTC. Neither previous trainer nor tmux session was still running.
- Operations are `cosmos_storage/operations/resume_two_gpus_each_20261028`,
  with `camera/` and `motion/` plans, scripts, saved prior configurations and
  logs. `phase1_current_ops.txt` and `naeive_adaptation_current_ops.txt` select
  the current respective operations directories. Live logs establish status.
- Motion launcher now accepts two GPU indices. Global batch accounts for
  replica/shard degrees, preserving the saved contract at16 clips per rank.
  Only rank0 writes the contract and preview configuration/media; task audit
  files are separate per rank. Both ranks participate in EMA sampling.
  Nine CPU regression checks and the two-GPU config/contract check passed.
- tmux sessions: `cosmos_p1_p17_gpu0_5` and
  `cosmos_naeive_adaptation_x0_gpu1_7`. W&B IDs remain `17u3iurd` and `m8owvunc`.
  The native loader restores existing RNG entries; the new motion rank has no
  saved rank1 RNG and retains initialized RNG. Do not claim bit-identical replay.
- Both two-GPU trainers recovered their complete checkpoints and continued
  updates after resumed EMA previews on both ranks. Both motion ranks passed
  all four task gradient audits. `running_verified.json` in each operations
  subdirectory records the evidence. W&B previews are held until the resumed
  step catches up to committed history (camera20288, motion6605 before stop);
  this prevents overwriting history when replaying unsaved updates.

## Foundary naeive adaptation and Phase2 staging (2026-10-28)

- User assigned sep-2 GPU7 to the separate **naeive adaptation** ablation in
  `native_motion_phase1/`. It replaces native camera actions with the existing
  normalized camera/head-aligned283-D Nymeria motion and trains global LoRA16/
  alpha32 plus fresh action heads. No Phase2 expert or Phase3 bridge is present.
- User selected equal25% text→motion, text+image→motion, video→motion and
  motion→video, with one reference RGB frame for motion→video. Batch32 on GPU7,
  accumulation1, T97/20fps/256 and200k updates. Use the same112503 validated
  Phase3 cache windows. See `native_motion_phase1/README.md` for the neutral
  shape prefix, fixed zero visual placeholder for T2M, and loss-masked padding
  from283 to288 channels required by native BF16 grouped GEMM.
- The original velocity acceptance included five native packing/loss tests, four-task batch32 GPU backward,
  complete checkpoint save/resume, and four native EMA qualitative previews
  passed. Production starts fresh from official Edge base. Exact plan, launcher,
  acceptance evidence and live logs are under node-local
  `cosmos_storage/operations/native_motion_phase1`. Live logs establish status.
  Existing camera Phase1 on GPUs0/5 must stay untouched.
- The original native-velocity run, W&B `jungbinc-upenn/cosmos3_camera/h4r7mobn`,
  is stopped. Its complete step600 model/EMA/optimizer/scheduler/trainer
  checkpoint and original source snapshot are preserved. The user requested
  keeping shared native motion/video noising while changing the motion target
  and loss to match Phase2/3. Use the separate revised objective
  `MOTIONP1_OBJECTIVE=x0_reconstruction_shared_waver_v1`; the compatibility
  default `native_velocity` still reproduces the original run.
- Revised motion predicts direct clean x0 with the exact Phase2/3 loss:
  feature1, centered decoded joints1, temporal displacement5, contact0.05,
  planted-foot velocity1 and foot height10, with no additional multiplier.
  Video remains native velocity with scale10. Native WAVER shift3 is unchanged;
  `independent_action_schedule=False` keeps the configured action logit-normal
  distribution inactive. Sampling converts only motion x0 to velocity using
  actual UniPC integration sigma, preserving observed shape/padding masks.
- Revised run operations are in `cosmos_storage/operations/naeive_adaptation_x0_20261028`.
  Nine CPU tests passed, including exact Phase2/3 loss/gradient parity, native
  video loss/gradient parity and actual-sigma sampler conversion. All four
  GPU7 batch32 backward tasks passed. Full checkpoint4 restored and update5
  passed; live logs and acceptance records establish preview/production status.
  The production recipe starts fresh from official Edge base, with a separate
  W&B run and contract that rejects resuming the old velocity objective.
- Revised x0 production launched on sep-2 GPU7 in tmux
  `cosmos_naeive_adaptation_x0_gpu7`, W&B `jungbinc-upenn/cosmos3_camera/m8owvunc`,
  displayed as `naeive adaptation | x0 + motion losses | Edge | batch32`.
  Full save/resume and all four97-frame/20fps EMA previews passed before launch;
  production model/optimizer/scheduler and source hashes match the accepted test.
  All four initial production videos and individual motion-loss metrics were
  verified in W&B, with continuing updates and the four-task gradient audit passed.
  Use the new operations `plan.json` and `training.log` for current progress.
- Nymeria RGB staging is complete on sep-2:1479files/728MP4s/297149670906bytes,
  source-stream and local SHA256 match for every file. Destination is
  `/mnt/local/jinhyung.kim/cosmos_storage/weka/nymeriaplus_kimodo_proportional/video`.
  All713 indexed recordings are local;12 real Phase2 T2M/TI2M samples across
  train/test passed without skipped-window fallback. Receipts/readiness are in
  `cosmos_storage/operations/stage_phase2_nymeria`. Phase2 itself is not launched.

## Foundary Edge Phase1, prefix17 / global32 on GPUs0 and5 (2026-10-27)

- User assigned sep-2 GPUs0 and5 to Cosmos3-Edge two-task Phase1 with17
  reference RGB frames, global batch32 and no accumulation. Use
  `native_phase_training/train_phase1_edge_foundary.sh`; it selects exact GPU
  UUIDs and checks node-local paths without fabricating a Slurm allocation.
- No captions, exclude5, forward:inverse40:25, T97/20fps/256, global LoRA16/alpha32
  plus action heads; retain the existing losses, LR schedule/EMA and200k horizon.
  Set `FOUNDRY_GPU_INDICES=0,5`, `NATIVEP1_CLIPS_PER_GPU=16`, replicate degree2,
  shard degree1 and `grad_accum_iter=1`; do not add accumulation or more GPUs.
  User requested W&B qualitative previews like
  EgoControl, inside the training process and the same training W&B run.
- `validation.py` supports fixed prefix1 or17. The17-frame path conditions five
  Wan latents, checks their preservation and scores generated RGB frames17–96.
  Four fixed held-out test cases produce video comparisons and camera plots
  after launch/resume and every5k, with RNG and training modes restored.
- First two updates, full native checkpoint and four videos/four camera plots
  passed; training resumes the same run's complete state toward200k. W&B:
  `jungbinc-upenn/cosmos3_camera/17u3iurd`. Exact local plan/logs/verification:
  `/mnt/local/jinhyung.kim/cosmos_storage/operations/phase1_current_ops.txt`.
  Live logs establish current state. Only GPUs0 and5 are assigned to this task.
- The batch32 continuation restores the same run's full step390 checkpoint.
  Its native model/inference contract is unchanged; LR and200k update horizon
  are retained. Record the batch change in the same W&B run because its first390
  updates used64 clips. The historical output directory still contains `bs64`.
- The user then assigned GPUs0 and5, explicitly selected global32 (16 clips per
  GPU, accumulation1), and confirmed coordinated sharing of GPU5 with the GR00T
  server PID3036385. Leave that server untouched. The launcher supports
  `FOUNDRY_GPU_INDICES=0,5`, `NATIVEP1_CLIPS_PER_GPU=16`, and the explicit
  `FOUNDRY_SHARED_GPU_PIDS=3036385` exception. It still rejects unexpected PIDs.
  Two-GPU operations are `phase1_p17_bs32_2gpu_20261027T222100Z`, selected by the
  active operations pointer. Full native checkpoint777 was verified and loaded;
  update778 onward uses two replicas. Model/inference contract, optimizer and
  scheduler are unchanged. Single-GPU batch32 averaged2.14s/update; both ranks
  passed training and the resume preview. Live logs establish current speed.

## Standing experiment reporting preference (2026-10-28)

- Always publish experiment/status reports and quantitative conclusions to the
  user's Notion Experiments workspace, with GT metrics, exact checkpoints,
  protocol/coverage, artifact paths, and selected qualitative comparisons.
  Parent: https://app.notion.com/p/binne/Experiments-396d2a44f44b8053b736ceb63350f26b
- Keep repository documentation and local machine-readable evidence as well.
  Reuse an existing report page for progress/results when possible. Verify the
  remote write and include its link in the user-facing report. If access fails,
  preserve the prepared report locally and explicitly say it is not published;
  never claim an upload succeeded without checking.

## Phase-1 latest no-text comparison and camera CFG (2026-10-29)

- `native_phase_training/CAMERA_CFG_20261029.md` records the frozen194233
  no-text/prefix1 versus original200k comparison on71 fixed test windows.
  The original is evaluated both with and without captions; all video metrics
  exclude observed frame0 and include measured GT references.
- `native_phase_training/camera_cfg_inference_config.py` adds inference-only
  camera CFG using a camera-token-absent branch with the same image/text.
  Six fixed cases sweep1/1.5/2/3/5; both inference caches are disabled throughout
  this sweep. Native/scale1 latents match bitwise in the same loaded model on
  all6 cases; separate-process pixels differ, including independent native
  repeats. The original failed pixel audit is retained; do not claim full
  cross-process bitwise reproducibility.
- Scratch2 operation `phase1_latest_vs_baseline_20261029` contains immutable
  settings, metrics, media and separate main/early/camera_cfg publication states.
  Completed2026-10-29 04:37KST: main71 metrics, six comparison videos, two
  camera figures and six CFG videos published/hash-verified (plus two early
  previews). Main exit0; camera_cfg/recovery.json records final success after
  the preserved initial control failure. No trainer or checkpoint changes.
- Report: https://app.notion.com/p/407d2a44f44b81608660d1fe4a51036e .

## Phase-2 image-conditioning diagnostic (2026-10-28)

- `motion_expert_t2m_edge/IMAGE_CONDITIONING_DIAGNOSTIC.md` records the inference-only
  Self→cross/Knorm/caption-only B256 200k test on32 distinct held-out recordings,
  three seeds, two layout-matched wrong-image controls and native no-image T2M.
- Correct images improve pure-noise feature MSE and sampled frame-0 MPJPE;
  whole-clip/PA gains are mixed. The image path is functional, with limited
  useful conditioning; this does not establish a foot-contact/floating fix.
- `nano_caption_only` changes T2M packing only. TI2M uses Edge's multimodal chat
  template during both training and inference, retaining the image under text CFG.
- Yonsei evidence/visuals: Scratch2 Cosmos `operations/phase2_image_conditioning_20261028`.
  Existing training/checkpoints are unchanged; allocation2608785 is retained.
- Full-test extension: `evaluate_image_conditioning.py` samples all11892 valid
  Nymeria TI2M cases (11657 physical windows,71 recordings) from noise under
  correct image/two wrong-image permutations/native T2M, sharing captions,
  shapes,lengths,seeds. UniPC35/shift3/CFG2, checkpoint200k unchanged. Use the
  Nymeria evaluator's own statistics and32-candidate retrieval with3 grouping
  seeds. GT floor offsets and every batch's measured reference are verified.
  Scratch2 operation `phase2_image_generative_20261028` holds immutable inputs,
  progress, resumable batches, measurements and final Notion publication.
  First batches on both2608785 workers passed; live logs determine completion.
- Report: https://app.notion.com/p/407d2a44f44b81a5ad53dbb685cdbd9e . The32-case
  diagnostic and two remotely hash-verified playable videos are published;
  the coordinator adds full-test metrics only after successful aggregation.
- Resume correction: GT embeddings must use NymeriaShapeTMRDataset's CPU
  decoding path, matching the saved reference. A2.38e-7m CPU/GPU joint
  difference amplified to8.88e-4 in one GT embedding; the original CPU path
  matched exactly at batch64. Do not widen the guard or drop that case.
  Generated sampling/scoring is unchanged; final GT uses the common measured
  reference. The formerly failing batch passed after this correction.
- `diagnose_joint_cfg.py` tests the user's joint text/image CFG request on the
  same32 diagnostic cases/seeds: current text-CFG2 versus joint CFG1/2/3/5.
  Joint null is native T2M empty text with no image, so prompt layout changes
  too. No training modifications. Scratch2 operation `phase2_joint_cfg_20261028`
  saves per-case measurements and6-panel previews; publication appends to the
  same Notion report. This small cohort is separate from the full11892-case
  benchmark. The retained allocation's resume coordinator runs the preview on
  GPU1 while full-test shard0 continues, then resumes full-test shard1.
- The joint-CFG32-case sweep completed and published its metrics/two videos.
  `diagnose_joint_cfg.py --target image` adds the image-only follow-up with
  the same caption in conditional TI2M and reference T2M branches:
  `x0_text + s*(x0_text_image - x0_text)`, s=1/2/3/5. Check caption token IDs
  and absent reference image tokens. Native prompt-layout differences remain.
  Scratch2 operation `phase2_image_only_cfg_20261028` shares retained2608785
  GPU1 with a40% process allocator cap (observed total16.5GB), leaving both
  full-test shards advancing. Results/previews publish to the same Notion page.
  Frozen training and full-test inference code are unchanged by this option.
- Image-only CFG completed32/32 cases ×3 seeds, with metrics and two verified
  videos in the report. CFG2 slightly lowers MPJPE40.01→39.17cm (paired interval
  crosses zero), but worsens PA/initial-pose errors; stronger scales worsen
  pose errors further. No clear overall benefit. Full96-pair control audit
  passed; unguided outputs match joint CFG1 bitwise, recomputed text-CFG2
  baselines differ slightly across batch packing. Use each sweep's baseline.
- User clarification: generation quality, not MPJPE, is the objective. The
  earlier image-only CFG experiment retained text; it is NOT image-only input.
  `evaluate_generation_modalities.py` now compares actual image-only (empty
  caption), text-only (no image), and image+text, all using the same native
  unconditional T2M reference atCFG2/UniPC35/shift3. No GT motion conditioning.
  Scratch2 operation `phase2_generation_modalities_20261028` preserves the
  exact11892-case manifest and checkpoint/evaluator hashes; full-test FID,
  R32, matching distance, diversity and contact metrics include measured GT.
  Interpret image-only R32 as association with withheld captions, not prompt
  compliance. Native T2M/TI2M layouts differ. Six fixed original diagnostic
  cases provide GT/image/text/both previews with seeds0/1/2; these previews
  are separate from full-test quantitative results. The retained2608785
  coordinator publishes videos as ready and full metrics after completion
  to the same Notion report. Live operation logs establish actual status.
- Generation-input preview recovery: a cached legacy flat decoder lacks
  `decode_transforms`; `render_motion.head_rotation_from_features` now prefers
  its package-qualified sibling. On saved GT and all three generated modes,
  old/sibling joint decodes matched bitwise and head rotations matched the
  sibling transforms. Sampling/metrics are unchanged. `renderer_fix.json`
  records the check. `recover_compute.sh` / `recovery.exit_code` are the active
  preview/shard1/final-publication coordinator; original shard0 continues.
  The original coordinator may record `0 1` for its failed preview branch;
  use `final_summary_ready.json` and recovery status for completion.
- At2026-10-28 14:23KST, all six generation-input preview cases completed with
  three seeds and all three modalities. The first two four-panel seed0 MP4s
  were uploaded and remotely hash-verified in Notion, section
  `407d2a44-f44b-8196-9ad5-e630a5e6c7cf`. Full-test shard0 was advancing and
  recovered shard1 was loading; full metrics were NOT complete at this snapshot.
- Follow-up: user requested more informative examples and text CFG fixed at2
  with varying image guidance. `evaluate_text2_image_cfg.py` uses
  `u + 2*(t-u) + w*(both-t)`, w=0/0.5/1/2/3/5. Weight0 is exactly text-only
  CFG2; weight2 is exactly joint CFG2. The image increment includes conditional
  text/image interaction; native T2M/TI2M layouts differ. Guidance is applied
  once; no training changes and no MPJPE ranking.
  Scratch2 operation `phase2_text2_image_cfg_20261028` freezes eight original
  diagnostic slots6,16,14,27,30,24,7,19 by action variety/GT motion before the
  sweep, seeds0/1/2. Four actions (6,16,14,30) get both modality-comparison and
  six-weight comparison videos in a new Notion page:
  https://app.notion.com/p/407d2a44f44b817ba15ad37cef229360 . These are selected
  illustrations, not aggregate evidence. A short preview shares PRO6000 job
  2608784's spare VRAM with a20% process cap. `full_dispatch.sh` waits for both
  ongoing full-test evaluations to finish, then runs all11892 cases × six
  weights on retained2608785 and publishes GT/FID/R32/diversity/contact metrics.
  `preview.exit_code`, `full.exit_code`, and `publication.json` establish status.
- The action-diverse eight-case preview finished successfully at14:47KST;
  all eight selected videos (four actions × modality/sweep views) were remotely
  hash-verified. The full11892-case weight sweep remains independently queued.
- User then requested long-root-travel and visible-object examples.
  `operations/phase2_root_object_cfg_20261028` scans GT across all11892 cases
  and adds four long-travel windows (source indices2706,10864,11747,4161;
  paths5.44–6.82m, net5.12–6.67m over4.8s) and four object windows from
  original slots26/9/12/31 (laptop, sliding door, vanity drawer, bag/chair).
  Supplemental slots100–103/200–203 are interleaved and are distinct from
  the earlier slot numbering. Selection uses GT travel/visible targets, not
  generated quality or MPJPE. All use text2/image0,0.5,1,2,3,5 plus image-only,
  seeds0/1/2. `visualize_conditioning_examples.py` delegates sampling unchanged
  and adds raw world-XZ root plots and separate display-only target outlines.
  Original RGB remains untouched. Outputs are added as toggles on the same
  Notion follow-up page. This does not create a second full-test sweep.
  The short supplemental preview shares retained PRO6000 job2608784; live
  `preview/progress.json`, `exit_code`, and `publication.json` determine completion.
- The long-root/object supplement completed successfully at15:15KST on
  2026-10-28, all eight cases × three seeds. Its eight Notion toggles contain
  sixteen remotely hash-verified MP4s and twelve verified figures (eight root
  plots plus four input-object highlights). Original input hashes remained
  unchanged; all guidance endpoint checks passed. `final_audit.json` records
  the verification. These qualitative runs are complete; do not conflate
  them with the still-pending full-test six-weight quantitative sweep.

## Foundary action/video projection ablation (2026-10-28)

- User requested an additional model training video input/output projections
  alongside the action projections, matching the running Phase1 recipe on
  idle GPUs. GPUs3,5 were rechecked empty and admitted; the existing action-only
  run on6,7 remains active. New run initializes fresh from the same official
  Edge base, with global32=2x16/accum1, prefix1, two tasks40:25, no captions,
  exclude5,200k steps, LambdaLinear, EMA,15min/5k checkpoints and W&B previews.
- New Edge-only mode `action_video_projections` trains exactly9,246,912
  parameters: `action2llm`, `llm2action`, `action_modality_embed`, `vae2llm`
  and `llm2vae`, including projection biases. Backbone/reasoner/VAE/timestep
  parameters remain frozen; no LoRA. Existing LR policy is preserved: initial
  effective action LR8e-5 and video LR2e-5,500-step plateau then linear decay.
- Foundary launcher accepts `--adaptation action_video_projections`; the
  separate shell wrapper enforces distinct naming. Train/inference contracts
  persist the new mode and reject LoRA/Nano combinations. The four-update
  distributed gate confirms finite nonzero gradients in all four projections
  and no frozen-parameter gradients. Peak allocation34.30GiB per GPU.
- Operations: `/mnt/local/jinhyung.kim/cosmos/operations/foundary_phase1_20261028T162949Z`.
  W&B: `jungbinc-upenn/cosmos3_camera/q3qthues`. Production passed step50 at
  roughly1.05s/update, and first validation completed in63.5s. All four videos
  and four camera plots were verified remotely. Logs supersede this snapshot.
  `recipe_verified.json` confirms the dataloader, scheduler and model config
  match the action-only run; only adaptation/optimizer selection differs.
  Twelve focused CPU tests passed, plus the real two-rank gradient/memory gate.

## Foundary sep-1 Phase-2/3 data staging (2026-10-28)

- User authorized copying the missing Nymeria assets from shared storage to
  sep-1 NVMe. Completed9 snapshot archives plus1,479 video/source-sidecar files
  and canonical evaluation-window JSONs; approximately359GB of staged payload.
  Existing Phase1 latents, base models and authoritative video manifest were
  reused. Shared sources were read serially with rsync capped at256MiB/s.
- Dataset root: `/mnt/local/jinhyung.kim/cosmos/weka/nymeriaplus_kimodo_proportional`.
  Packed motion, native-camera and floor18D caches live under
  `/mnt/local/jinhyung.kim/cosmos/caches`. Archived frozen Edge Phase1/Phase2
  specialists and fixed validation inputs are under local `weka/cosmos_motion_ft_runs`.
  This is the camera/head-v1 Nymeria path; BONES and raw VRS are not staged.
- Reproducible worker, receipts/inventories, environment helper and CPU report:
  `/mnt/local/jinhyung.kim/cosmos/operations/stage_phase23_20261028`.
  After activating the local cosmos environment, source its `environment.sh`
  for the matching statistics, cache paths and frozen specialist paths. Pass
  `--cache "$EDGE_PHASE3_CACHE"` and the appropriate camera/trajectory cache
  flags to Phase3; these explicit helper variables do not override CLI defaults.
- Verified archive SHA256, extracted sizes/nanosecond mtimes, unchanged cache
  indices, all108760 cached latent identities,642 motion/camera sources,
  112503 Phase3 rows and12 sampled windows. Phase2 has120929 T2M/112937 TI2M
  windows; three samples per route passed CPU loading, including image decode.
  All713 current manifest videos exist locally. No Phase2/3 GPU job was launched.
- Phase3 identity/latent reads now resolve recognized archived paths through
  `runtime_paths`, preserving fingerprints. Custom Phase2 restores explicitly
  set `EDGE_ARTIFACT_SOURCE_SITE=yonsei` to select existing pinned metadata
  hashes; strict weight/statistics/population checks remain. Correct-profile
  acceptance and wrong-profile/changed-source rejection were checked.

## Foundary sep-1 Phase-1 action-only launch (2026-10-27)

- On2026-10-28 the user requested GPUs6,7 and explicitly retained global32:
  two ranks at16 clips each, accumulation1. Both devices were rechecked empty
  after stopping only our old trainer at committed checkpoint11672. The
  four-update distributed gate passed on both ranks (~1.03s/update), and
  production restored model/EMA, optimizer, scheduler and trainer at11672.
  Current operations: `/mnt/local/jinhyung.kim/cosmos/operations/foundary_phase1_20261028T050711Z`.
  Current W&B: `jungbinc-upenn/cosmos3_camera/hq0tnf0s`.
  Resumed validation11673 uploaded all four videos/four camera plots. Production
  averages about1.07s/update (roughly1.9x the prior single-GPU run); first new
  rolling checkpoint12403 completed and training continued. Keep global32.
  The single-GPU runs below are retained as provenance and are stopped.
- User subsequently requested batch32. The batch64 run was stopped only after
  checkpoint421 committed; GPU7 then passed the four-update batch32 gate
  (40.81GiB peak allocated, finite gradients in both action projections).
  Production resumed model/EMA, optimizer, scheduler and trainer at421 with
  global32=1x32/accum1. The loader has no saved state and restarts its sampling.
  The first resumed preview at422 completed in74.3s; all four videos and four
  PNGs were verified remotely. Initial steady updates take about2s, versus4s
  at batch64, with similar clips/second and half the samples per update.
  Previous single-GPU W&B: `jungbinc-upenn/cosmos3_camera/8fds1lkr`.
  Continuation evidence and source snapshot:
  `/mnt/local/jinhyung.kim/cosmos/operations/foundary_phase1_20261027T222226Z`.
  The original run below is retained as provenance, not an active trainer.
- User authorized process-free GPUs:2x32 clips or1x64, global64, prefix1,
  forward/inverse dynamics only, and W&B. They accepted the Yonsei action-only
  selection including the2048-parameter modality embedding alongside both
  action projections (8,458,240 trainable parameters total).
- `native_phase_training/train_phase1_foundary.py` selects and rechecks empty
  GPUs, records a live-parent admission, and runs a four-update memory/gradient
  gate before fresh production. The Slurm launchers retain their2x16 defaults;
  no fabricated Slurm ID is used for Foundary preview callbacks.
- On sep-1 only GPU7 was empty. Its1x64 gate passed finite losses/gradients,
  both action projections received gradients, and peak allocated memory was
  53.84GiB. Production passed its first fixed four-case preview and continued
  at roughly4s/update. W&B run:
  `jungbinc-upenn/cosmos3_camera/rar9yvk0`; all four videos/four camera PNGs were
  verified remotely. Live logs supersede this initial launch observation.
- Operations and exact source snapshot:
  `/mnt/local/jinhyung.kim/cosmos/operations/foundary_phase1_20261027T215005Z`.
  The archived Yonsei run contract matches exactly; global batch64 differs
  from Yonsei's32. LR remains LambdaLinear: action LR8e-5 for500 updates, then
  linear decay to zero at200k. Data, model assets, outputs and checkpoints use
  node-local NVMe. Rolling checkpoint cadence is15min, plus fixed saves every5k.

## Phase-1 action-only two-task ablation (2026-10-27)

- `native_phase_training/train_phase1_edge_two_tasks_action_only.sh` reuses the
  prefix1/no-caption two-task baseline with the existing `action_only` mode.
  Only action input/output projections and the action modality embedding train
  (8,458,240 parameters); no LoRA. Default existing launchers retain global LoRA.
- Preserve exclude5, T97/20fps/256, forward:inverse40:25, global32=2x16/accum1,
  FP32 EMA, losses and200k LR schedule. Initialize from the official Edge base,
  with a fresh optimizer and a distinct run; explicit same-run resume is supported.
- The A6000 probe's resolved model, optimizer, scheduler and data configurations
  match the current no-caption baseline except adaptation mode, LoRA enablement
  and optimizer parameter selection. The final gradient audit combines ranks;
  the older sharded4090 probe's rank-local assertion was not a valid full pass.
- Yonsei retained job2616216/node25 uses two48GB A6000s. Operation evidence is
  Scratch2 Cosmos `operations/phase1_action_only_a6000_20261027`; its successful
  probe and recipe gate precede production. Logs establish actual progress.
  W&B uses `jungbinc-upenn/cosmos3_camera`; existing trainers stay untouched.
- The24-update replicated A6000 gate passed:34.40GiB peak allocated and35.84GiB
  reported device memory per GPU, finite losses/gradients and both action heads
  receiving gradients. Production starts fresh from base, not from probe weights.
- Production launched on2026-10-27 and advanced through its first six updates
  at about7.8s/update, with35.8GiB device memory and both GPUs busy. W&B run
  `jungbinc-upenn/cosmos3_camera/6p03fz3d`; the run name is
  `edge_phase1_T97_20fps_2gpu_bs16_action_only_exclude5_fd_invdyn_no_text_p1_200k_yonsei_20261027`.
  `operations/phase1_action_only_a6000_20261027/STATUS.json` records the launch
  and resolved production/probe recipe match; live logs supersede this snapshot.
- User requested EgoControl-style **in-process** W&B previews. The action-only
  launcher now enables `NATIVEP1_WANDB_PREVIEWS=1`; four fixed held-out test71
  cases are sampled with native EMA UniPC30 after the first resumed update and
  every5k. The first full pass at step117 completed in73s (four97-frame videos
  plus inverse-camera plots), then training returned to about7.8s/update.
  Outputs are under `<run>/validation/`; `production_previews.log` is the live
  operation log. Model/data/optimizer/scheduler configs remain unchanged.
  The node25 resume worker sets `NCCL_P2P_DISABLE=1` after an initial NCCL stall;
  a64-cycle two-rank all-reduce/barrier test passed with this workaround. This
  is an operation-specific setting, not a change to other launchers or Grasp.
- W&B API verification confirmed all four comparison MP4s and four camera PNGs
  uploaded to `6p03fz3d`, with model, VAE-GT and GT-self metrics at validation
  step117. `wandb_previews_verified.json` records remote file sizes. Training
  continued through step163 after the preview pass (15:04KST snapshot).
- The regular step5000 preview also completed on2026-10-28, followed by
  continuing training. W&B API verification confirmed all eight new media files
  and model/VAE-GT/GT-self metrics at validation step5000. The memory-probe
  launcher explicitly disables these previews to isolate training memory.

## Single-repository source restore (2026-10-27)

- `migration/SINGLE_REPO.md` documents the current code transfer. Dependencies
  are locked in `migration/dependencies.lock.json`; bootstrap restores separate
  Git checkouts under ignored `.deps/`, preserving native revision guards.
- Four UniEgoMotion and six EgoAllo source changes formerly only in Scratch2
  are preserved as checked patches. Cosmos Edge/Nano and Kimodo were clean
  at the recorded revisions. Keep third-party weights/data out of Git.
- `COSMOS_SITE=custom` adds explicit new-server roots without changing Yonsei
  or Grasp selection. Shell/Python share `migration/custom_paths.py`. Six
  focused path tests and all five dependency source restores passed.
- Installed-version snapshots cover Cosmos/cu128, UEM/cu118 and EgoAllo/cu126;
  they do not prove a fresh environment or GPU run works on an unknown server.
  Preserve strict checkpoint/cache fingerprints during artifact migration.
- The 2026-10-27 consolidation includes the recent experiment packages and
  migration setup in Git. The code-bundle exporter can additionally capture
  subsequent unpublished working-tree changes for an explicit source snapshot.

## Phase-3 runtime profile and mask cache (2026-10-27)

- Measured on Yonsei RTX4090, current global32 / two ranks / microbatch16.
  Steady loader wait was below1ms, model forward about1.2s and backward with
  recomputation about2.6s. Adding loader workers does not address this profile.
  Evidence: Scratch2 Cosmos `operations/phase3_speed_20261027`.
- `scripts/train_phase3_performance.py --training-module MODULE ...` is an
  opt-in wrapper around the unchanged trainer. It reuses per-forward FP32
  bridge masks across layers/recomputation, preserving the attention edges,
  original dtype/TF32 settings, activation checkpointing, optimizer and recipe.
  A fixed-batch test gave identical predictions/loss and only about1.2% lower
  runtime; do not describe this as a substantial real-run speedup. Repeated
  original backward passes themselves have non-bitwise-identical gradients.
- Original source/data/recipe fingerprint checks remain enforced. The separate
  execution manifest lives in `performance_runtime/` and subsequent saves
  include `runtime_performance` with implementation hashes. Old samplers and
  checkpoint readers remain compatible. The wrapper is site independent.
- Fused FP32 SDPA and disabling motion-block checkpointing gave no useful
  speedup. TF32 saved roughly6-7% but changed predictions/gradients enough to
  reject for this continuation; it was not enabled in production.
- Control was checkpointed at45370 and resumed on retained2608758 using the
  mask cache. The other active trainers were not interrupted. Yonsei's existing
  resume worker applies the cache to the three Phase3 variants on their next
  launch via `resume_20261026/performance_policy.json`; live logs, not this
  dated note, establish current progress.

## EgoControl adapted to Edge, M2V baseline (2026-10-26)

- Isolated implementation and comparison contract: `egocontrol_edge/README.md`.
  User requested EgoControl conditioning on Cosmos3-Edge, the exact Phase3
  Nymeria/SOMA training windows, first observed RGB frame, M2V only, and the
  current Phase1 optimizer/schedule/loss/global32 recipe. Warm-start the same
  Phase1 EMA200k generator as original Phase3; use a fresh optimizer.
- Uses the immutable112503-window Phase3 cache and its saved camera/head
  normalization. Converts SOMA Body23 to EgoControl's96x138 controls; both
  global pose modulation and per-block pose cross-attention are implemented.
  Follow-up: default conditions only layers3,7,11,15,19,23,27 (matching Phase3);
  no extra final-output modulation. Earlier28-block weights are legacy history,
  not initialization for the fresh seven-block run. See `conditioning_layout`
  in resolved config and the comparison contract.
  Fixed held-out monitoring samples and model/GT/VAE_GT video metrics go to
  UPenn W&B after step1/every5k; see `egocontrol_edge/README.md` for protocol.
  This is an adapted-method benchmark, not a parameter-matched ablation or
  exact reproduction of the original Cosmos-Predict2 checkpoint.
- Match caption-free first-frame+motion inference, held-out windows, native
  sampler/seeds and generated-frame metrics. Disclose our Phase2 pretraining,
  historical Phase3 caption training, Body23 versus30, parameter counts and
  M2V exposure. Retain100k and200k for equal-exposure/equal-update reporting.
- Owned GPU acceptance checks precede replacing the prefix17 trainer on
  retained job2608784. Keep the allocation and unrelated prefix1 training.
  Operation logs in Scratch2 `cosmos/operations/egocontrol_edge_20261026`
  establish actual acceptance/launch status; this entry is not a claim that
  production has started. Run artifacts go to Scratch2
  `cosmos/cosmos_motion_ft_runs/cosmos3_camera/egocontrol_edge`.
- Native dataset now exposes `_build_cached_index` as a protected hook; its
  default remains the original builder. This baseline supplies its already
  validated Phase3 index, avoiding a redundant full filesystem scan.

## Phase-1 no-caption PRO6000 resumes (2026-10-22)

- User requested resume, not fresh initialization: prefix1 on retained two-GPU
  job2490075; prefix17 automatically on pending two-GPU job2490076. Their latest
  observed checkpoints were87120 and95353 respectively; select the latest
  complete checkpoint from each original run's pointer at actual launch.
- Coordinator and plan: Scratch2 Cosmos `operations/phase1_resume_pro6000_20261022`.
  `dispatch.sh PREFIX JOB` polls the named allocation every30seconds, then uses
  an owned compute step and the existing no-text launcher with strict training
  state resume. It never initializes fresh, changes the recipe, or cancels jobs.
- Preserve T97/20fps/256, prefix1 versus17, no captions, exclude5, forward:inverse
  40:25, global32=2x16/accum1, original losses/LR/EMA and200k horizon.
- Both resumed trainers use `jungbinc-upenn/cosmos3_camera` and saved CLI login,
  not stale allocation API keys. This note records the launch configuration;
  operation logs establish actual allocation, checkpoint loading and progress.

## Phase-3 live W&B metric sidecars (2026-10-22)

- `scripts/stream_phase3_wandb.py` backfills and follows existing Phase-3
  `metrics.jsonl` files without modifying/restarting trainers or checkpoints.
  Run on an owned compute node, never login/cs-gpu-01/node37; no GPU is used.
- W&B destination: `jungbinc-upenn/cosmos-motion-ft`, comparison group
  `edge-phase3-fourtask-camfilm-20261021`. The four-task control and context
  camera-FiLM run have separate deterministic IDs derived from output paths.
- The user selected their current UPenn login. The launcher unsets inherited
  `WANDB_API_KEY` and uses the saved login; older allocations carried a different
  account's key. Earlier `scene_motion` copies are historical, not live targets.
- Charts use `train/step` (optimizer updates). W&B history steps are source
  record indices, preserving appended resume histories. Restart resumes from
  remote history; partial final JSONL records wait for their terminating newline.
- Only existing scalar metrics and the saved training contract are uploaded.
  This does not add validation, videos, or per-task losses that were not logged.
  System telemetry is disabled to avoid attributing the logger's host to training.
- Yonsei launcher/state/logs: `operations/phase3_wandb_20261022` under the
  Scratch2 Cosmos root; `launch.sh control` or `launch.sh camfilm` first flushes
  historical metrics, then follows new records. A sidecar must be relaunched
  on a retained allocation after its host allocation expires; existing JSONL
  records allow backfilling any gap. Runtime logs establish actual sync status.

## Edge phase-3 bridge implementation (2026-09-24)

- New isolated package: `motion_expert_phase3_edge/README.md`. Frozen Edge
  phase1 EMA200k plus self-cross-final-hidden/K-norm/caption-only B256 motion
  phase2 200k, seven zero-gated bridges at layers3,7,11,15,19,23,27.
- Nymeria only, all valid train recordings, T97/20fps/256, camera/head-aligned
  statistics. Preserve original losses: video10; motion1/1/5 and contact0.05/1/10.
  Native video reduction includes masked conditioning elements in its denominator.
- Cache preparation and model checks must run on allocated compute nodes, never
  login/cs-gpu-01/node37. Yonsei outputs/cache default to Scratch2; Grasp paths
  are explicit. Existing trainers and source checkpoints are untouched.
- Production is gated by GPU/DDP parity, gradient, overfit and resume checks.
  Only bridges train; global batch32, LR2e-4, warmup1k, cosine0, target200k.
  Keep latest plus100k bridge checkpoints; do not cancel retained allocations.
- This entry records implementation, not successful validation or observed
  training. Report actual stage/optimizer logs from the operations directory.

## Projection A K/V norm capacity ablations (2026-09-20)

- See motion_expert_t2m_edge/KV_CAPACITY_ABLATIONS_20260920.md and the
  site-portable train_kv_capacity_ablation.sh launcher. Three separate mixed
  runs: lr10 B128/2e-3/200k, long300k B128/2e-4/300k, and self_joint
  B256/2e-4/200k on two GPUs. Existing trainers and checkpoints are unchanged.
- User's follow-up changed long300k to a separate continuation from the finished
  B128 200k checkpoint, including optimizer/RNG. Extending the cosine horizon
  restarts LR around5e-5 at200k and decays tozero at300k. The other two start
  fresh. All preserve the baseline data, statistics,
  prompt, losses, warmup, seed, visualization cohorts and 10k checkpoint policy.
- EDGE_MOTION_SELF_ATTENTION_BEFORE_JOINT=1 adds an attention-only motion
  residual before every joint block: SA -> JA -> ONE FFN, not legacy gap
  Transformer blocks. Reasoner K/V projection plus learned K/V norms remain;
  the frozen reasoner never sees motion. Default false and checkpoint contract
  resolution preserve existing architectures on Yonsei and Grasp.
- Yonsei new outputs stay on Scratch2. User approved canceling idle2291683
  for replacement two-4090 request2294892. Active2291900 is not canceled;
  its existing successor2293349 resumes its own run after allocation.
- No standalone tests were requested. Actual startup/optimizer progress must
  be established from runtime logs, not inferred from this implementation note.

## Nymeria evaluator adaptation and GT32 gate (2026-09-19)

- Baseline GT-only scoring with 32 candidates and seeds 0,1,2: BONES R@3
  99.90 +/-0.02%; Nymeria 69.80 +/-0.25%. These are paired-only scores,
  not the existing full-pool similar-caption-positive scores. User target
  is at least85% Nymeria GT R@3 with this unchanged three-seed protocol.
- Baseline training loss fell while full-pool paired validation R@3
  plateaued near2.6% after5k. Text embedding dispersion stayed nonzero.
  Frozen text geometry and stale queue negatives are hypotheses, not
  established causes. Do not claim the test threshold has been achieved.
- New pipeline: shape_aware_TMR/train_nymeria_adapt32.py, invoked through
  nymeria_run.sh on an owned GPU, with explicit --baseline-run and --out.
  It initializes from the old validation-selected5k weights, trains the
  small C45 text encoder at0.2x motion LR (LLM2Vec remains frozen/cached),
  uses batch256, queue0, and40k new steps. Same data, statistics, shape
  model and loss weights; no changes to existing Phase2 generators.
- Nymeria evaluation additionally logs paired R32 R@1/2/3 for seeds0,1,2.
  New run selection uses validation/R32_R03_mean; full-pool selection
  remains the default for old recipes. Test is evaluated only after
  selecting best by validation, not used to pick a checkpoint.
- Baseline curves are under Yonsei operations/nymeria_evaluator_retrain_20260919.
  New output is shape_tmr/nymeria_c45_camhead_adapt32_jointtext_b256_q0_20260919.
  Job2281723/node35 is the planned retained allocation. Launcher/status
  and live logs, not this entry, determine actual training progress.

## Final-hidden cross-attention K/V norm, no added key RoPE (2026-09-19)

- New opt-in mixed batch128 experiment: run
  `bash motion_expert_t2m_edge/train_norm_ablation.sh self_cross_final_hidden_knorm_vnorm_nokrope mixed`
  with `COSMOS_SITE=yonsei` or `grasp` on an owned GPU allocation.
- Preserve final post-norm frozen reasoner memory, seven independent learned
  K/V projections, motion self-attention, cross-query RoPE and the FFN. Apply
  independent trainable per-128-D-head RMSNorm to projected K and V. Disable
  only added cross-key RoPE; V was already unrotated. The frozen reasoner's
  native positional computation remains intact, so its memory is not
  position-free. Cross norm gains follow motion norm storage dtype (BF16).
- New settings `EDGE_CROSS_ATTENTION_V_NORM=rmsnorm` and
  `EDGE_CROSS_ATTENTION_REASONER_ROPE=0` default to none and 1 respectively.
  Checkpoint contracts and inference resolution retain both; older checkpoints
  retain their original V/no-norm and cross-key/native-RoPE behavior.
- The launcher also retains cross-K RMSNorm. Same mixed data, camera/head
  statistics, parity losses, LR schedule and visualization recipe, batch128,
  seed0, 200k target, positive 10k checkpoint multiples only. Yonsei output
  stays on Scratch2; Grasp uses RUN_ROOT. Existing trainers are not restarted.
- Initial launch allocation: retained idle job2263514/node34, about 6h40m
  remaining when selected. This entry documents configuration, not observed
  training progress or test success. Resume later only this distinct output.

## Edge Phase-2 separate self/cross-attention experiments (2026-09-16)

Two fresh mixed-data runs replace each of the seven joint-attention blocks
with pre-norm motion SA, pre-norm reasoner CA, then one SwiGLU FFN, each with
its own residual. They keep width2048, GQA16/8 x128 and FFN3072. Motion SA
includes only shape plus valid frames; CA includes only the same sample's
reasoner tokens. The frozen reasoner never reads motion.

- EDGE_MOTION_ATTENTION_MODE=self_cross_final_hidden: all seven blocks use
  the final reasoner token sequence after backbone final norm. Each owns new
  2048-to-1024 K/V projections, with no extra K/V norm. New CA queries and
  projected keys receive native mRoPE once. TI2M includes image tokens.
- EDGE_MOTION_ATTENTION_MODE=self_cross_layer_kv: blocks use native K/V from
  reasoner layers[3,7,11,15,19,23,27]. Native K already has its own frozen
  normalization and RoPE; V is unchanged. No interface K norm, K/V adapter,
  or repeated K rotation. Motion CA Q uses its own norm/projection/RoPE.
- Defaults remain joint. Contracts record mode, memory and position policy;
  old contracts imply joint. Inference resolves architecture from checkpoints
  before model creation and rejects conflicting explicit environment settings.
- Launch bash motion_expert_t2m_edge/train_norm_ablation.sh MODE mixed,
  selecting COSMOS_SITE=yonsei or grasp. Same loss/data policy as the
  reasoner_k_fp32 mixed baseline: batch128, accum1, workers8, seed0, 200k,
  LR2e-4/warmup1k/cosine floor0, centered1/1/5 plus contact0.05/1/10,
  75%T2M/25%TI2M, BONES50% within T2M, camhead statistics, variable lengths
  capped at200, TI2M97, same caption policy and fixed visualizations every5k.
  Save only positive10k checkpoint multiples. Outputs stay on Scratch2 at
  Yonsei; Grasp uses the existing RUN_ROOT mapping.
- These are not parameter-matched experiments. Extra SA/CA projections add
  capacity, and removal of the learned interface norm is an additional
  difference from the named reference. Do not claim a unique causal result.
- Requested GPU acceptance checks are in validate_self_cross.py. Operations
  are under /scratch2/whwjdqls99/cosmos/operations/phase2_self_cross_yonsei_20260916.
  The staged launcher checks actual batch128 before production. Initial
  allocations2231899(node38) and2263514(node34) were idle before reuse.
  This entry describes implementation and intent, not observed test success
  or verified production progress. Never cancel retained allocations on exit.

This repository is a cluster-oriented research checkout for adapting NVIDIA Cosmos-3 Nano into human-motion and egocentric world-model variants. It is not a standalone Python package. Real runs require a site-specific Cosmos framework checkout, compatible conda environments, Slurm GPU nodes, and persistent datasets/checkpoints. The physical path contract for the Yonsei datacenter and Grasp lab is `migration/SITE_STORAGE.md`; `/home/jungbin_cho`, `/weka/jungbin`, and `/mnt/shared/jungbin_cho` references are legacy H200/A100 provenance unless a section explicitly says otherwise.

Primary current work areas: `motion_expert_joint_attention/` and the isolated native-Cosmos Phase 1 path in `native_phase_training/`.

This file is the shared source of context for both Claude and Codex. Agent-specific bootstrap files such as `AGENTS.md` and `CLAUDE.md` intentionally point here and should stay small.

## Motion renderer: floor layers and initial Head orientation (2026-09-17)

- User-approved default style on Yonsei and Grasp: XZ view half-extent 1.6 m,
  skeleton bone linewidth 1.35 and joint marker area 8.0. Keep the initial
  Head-axis arrows and color legend, omit the top numeric orientation overlay,
  and retain correct above/below-floor layering. These are shared renderer
  defaults for new visualization processes, not per-run display overrides.
- Shared `motion_expert_joint_attention/render_motion.py` splits every bone at
  world Y=0 and disables mplot3d's collection-level automatic z-order.
  Above-floor fragments stay opaque; below-floor fragments are rendered behind
  the translucent floor. Negative heights remain visible, with no grounding,
  recanonicalization, or changes to stored joints/features.
- Optional `gen_head_rotation0` / `gt_head_rotation0` show a fixed frame-0
  RGB local-axis triad plus world +Z reference, without a numeric plot overlay.
  Offline manifests retain yaw/pitch/roll and initial Head XZ for diagnostics.
  The rotation is decoded from raw 283-D features
  including the first canonical delta; it is never inferred from positions.
  Canonicalization is yaw-only: nonzero pitch/roll are not inherently errors.
- Edge training visualization, Edge sampling and Nano joint-training
  visualization supply the real head rotations. Already-running Python
  trainers retain previously imported code until restarted; do not restart
  them just to update rendering.
- `motion_expert_t2m_edge/rerender_saved.py` re-renders immutable saved Edge
  visualization NPZs, emits first-frame PNGs and a provenance manifest, and
  uses explicitly supplied generator mean/std for older GT-normalized arrays.
  Run rendering on a compute node, never a login node. Paths are CLI inputs
  so the workflow works on both Yonsei and Grasp.

## Documentation Policy

### Scheduled five-hour Phase-2 evaluation follow-up (2026-09-18 evening)

- YONSEI_PHASE2_EVAL_20260918_2045.md records the completed sweep started
  at20:45:20 KST on retained job2281723/node35. Five new checkpoints were
  evaluated, sixteen exact prior results reused, and no results were missing.
  The snapshot was pinned at20:46:48 and completed21:16:33, exit0.
- New BONES R@3: Projection A mixed256/150k61.50; final-hidden128/180k57.29;
  projected-K/V-norm128/110k45.49; final-hidden cross-Knorm256/70k44.99;
  layer-K/V128/200k38.04. GT93.93 and Nano200k82.31 remain references.
  Full FID/skating/penetration/airborne metrics and GT are in the report.
- Layer-K/V training finished200k, completed final visualization, returned
  to the retained shell, and its GPU was observed idle. Do not resume it.
- Separate Nymeria evaluator training has NOT started: preparation stopped
  on identical duplicate source windows; fixing that new-code assumption
  is awaiting explicit user approval. No Nymeria test result exists yet.
- These reports are BONES/C45 metrics, not proof of Nymeria evaluator
  progress or resolution of floating. Existing unfinished trainers were
  not stopped, resumed, or modified by this follow-up.

### Nymeria-specific C45 evaluator implementation (2026-09-18)

- Separate Nymeria evaluator pipeline: shape_aware_TMR/NYMERIA_EVALUATOR.md.
  Use train_nymeria_c45.sh on an owned GPU node, COSMOS_SITE=yonsei or grasp.
  Yonsei caches and outputs default to Scratch2; BONES C45 is not overwritten.
- Keep the official C45 model/loss/initialization: batch128, LR5e-5,
  warmup1k, 10k steps, frozen text encoder, official30fps TMR186 statistics.
  Generator UniEgo283 statistics are never evaluator statistics.
- Preserve recording-level test holdout; select on separate validation,
  not the test set. Default camera/head Nymeria native windows are cached;
  known quality drops and all additional exclusions are recorded.
- Nymeria snapshots bind data/statistics/text/recipe fingerprints and save
  optimizer, contrastive queue and RNG state. Pretrained initialization no
  longer overwrites checkpoint-restored weights in st_train.py.
- Report GT, case counts, duplicate-aware and plain/tie-aware retrieval.
  Physical metrics are independent; no claim that evaluator training fixes
  floating. See the detailed document for explicit dataset adaptations.
- This entry records implementation, not validated behavior, observed
  training progress, or achieved retrieval. Current jobs/results belong
  in timestamped operational reports; no existing trainer was restarted.

### Current Yonsei Phase 2 handover and evaluations (2026-09-18)

- Read YONSEI_PHASE2_STATUS_20260918.md for the repository-local run map,
  launcher recipes, complete evaluated-result table, GT/Nano references,
  normalization contract, allocation history, and Yonsei/Grasp scope.
- Snapshot at 2026-09-18 15:19 KST: final-hidden128 job2275411/node36,
  layer-KV128 job2263514/node34, Projection A mixed256 job2270088/node36,
  final-hidden+cross-K-norm256 job2279743/node34, and projected-K/V-norm
  mixed128 job2278800/node40 are training. Both original Projection A
  batch128 runs finished200k and were evaluated. Live Slurm/logs supersede
  this dated snapshot; none of these tables is a live dashboard.
- The requested projected-V-norm variant HAS started. It resumed from40000
  on replacement2278800 after2248640 expired. Do not launch a duplicate.
- The final-hidden DDP run moved from node37/job2270089 to node34/job2279743
  at the user's request, resuming10000. Its new20000 checkpoint and all12
  visualizations completed, and later training advanced past50000 without
  an observed timeout recurrence. Keep node37 excluded for this replacement.
  The600-second timeout remains unchanged; no timeout fix was applied.
- Full evaluation details remain on Yonsei Scratch2. The repository-local
  summary makes results readable from Grasp without implying shared storage.
  The two generators' mean/std and the C45 evaluator's mean/std are distinct.
- The operations registry is historical, and runtime_state.json is timestamped.
  Do not use the unused run_eval.sh wrapper with its stale followup.py target;
  invoke evaluate_idle.py with the established evaluation environment instead.
- The September17 preparation section below is historical and is superseded
  for current job IDs and launch/completion status by this September18 handover.


### Active 30-minute operations goal and projected-V norm variant (2026-09-17)

- User authorized monitoring current runs every30min, obtaining replacement
  allocations after confirmed termination, and resuming the same unfinished
  runs. A completed trainer's idle GPU first evaluates missing latest Edge
  checkpoints across both run roots, using the established BONES protocol and
  documenting Nano/GT/model retrieval and physical metrics. Preserve retained
  allocations; an expired observation timeout is not allocation termination.
- Operation registry and prepared replacement/evaluation launchers:
  /scratch2/whwjdqls99/cosmos/operations/phase2_autopilot_20260917
  The active Codex goal performs the 30-minute checks and decisions; these
  helper files alone are not a running automatic scheduler.
- Initially tracked: final-hidden128 job2231899, layer-KV128 job2263514,
  Projection A mixed128 job2248639, Projection A BONES128 job2248640,
  Projection A mixed256 job2270088, final-hidden+cross-K-norm256 job2270089.
  Do not resurrect intentionally canceled native/as-is job2250537. Do not
  automatically resume stopped Phase1 runs from their pending legacy requests.
- Existing Projection A has projected K -> learned norm -> native RoPE and
  projected V without a norm. The requested NEW variant adds a separate
  fresh trainable native RMSNorm to each projected 128-D V head, with FP32
  affine gains; seven blocks add896 parameters. K's existing learned norm
  is unchanged, V is not rotated, and frozen reasoner K/V are untouched.
- Opt in with EDGE_REASONER_V_NORM_TRAINABLE=1 together with Projection A.
  Default0 leaves all existing models unchanged. Checkpoint contracts record
  reasoner_v_norm_trainable, old checkpoints imply false, and inference
  resolves the flag before construction. Strict loading rejects mismatches.
- Prepared command: bash motion_expert_t2m_edge/train_norm_ablation.sh
  reasoner_kv_project_vnorm mixed. This is a fresh mixed BONES50-within-T2M,
  batch128, 200k, camera/head-stat, parity-loss/LR experiment dated20260917,
  with the original A recipe and10k checkpoint interval. It runs only AFTER
  a completed-run GPU has finished the missing-latest evaluation sweep.
  No GPU is currently reassigned merely to start this additional variant.
- No standalone tests were requested. This records preparation, not a claim
  that the pending V-norm run has started or completed validation.


### Final-hidden self/cross with cross-key RMSNorm, DDP256 (2026-09-17)

- User requested starting a fresh final-hidden self/cross mixed run on the
  pending two-4090 job2270089 as soon as it allocates, with global batch256
  and the discussed projected-cross-key RMSNorm. Do not change the existing
  batch128 final-hidden run or the Projection A DDP run on job2270088.
- Opt in with EDGE_MOTION_ATTENTION_MODE=self_cross_final_hidden and
  EDGE_CROSS_ATTENTION_K_NORM=rmsnorm. The default remains none, preserving
  earlier models. Each of seven motion blocks applies its own trainable
  native-class RMSNorm over each 128-D projected cross-key head, before
  native RoPE exactly once. Cross V, motion self-attention, and the frozen
  reasoner's entire computation remain unchanged.
- This is not the old joint-attention reasoner-interface K norm. Final-hidden
  memory is still the frozen post-final-norm token sequence, followed by
  independent learned K/V projections per motion block. Norm gains follow
  the configured motion-norm storage dtype, BF16 for this run; the native
  RMSNorm reduction is FP32. The seven norms add 896 affine parameters.
- The existing cross_attention_key_normalization contract field records
  rmsnorm versus none. Inference resolves this before model construction;
  strict checkpoint contracts reject mixing normalized and unnormalized
  architectures. Old joint/layer-KV/final-hidden defaults remain unchanged.
- Launcher: bash motion_expert_t2m_edge/train_self_cross_knorm_ddp.sh inside
  an owned two-GPU allocation, COSMOS_SITE=yonsei or grasp. Use batch128/rank,
  accumulation1, workers8/rank, 200k target, LR2e-4, warmup1000, cosine floor0,
  centered losses1/1/5 plus contact0.05/1/10, camera/head stats, BONES50 within
  T2M and existing T2M/TI2M task weights. Save only positive 10k checkpoints.
- Separate output: edge_7layer_self_cross_final_hidden_knorm_..._gbs256_...
  dated20260917. Yonsei outputs use Scratch2; Grasp uses RUN_ROOT.
  SELF_CROSS_OUTPUT_ROOT and FIXED_VIZ_SOURCE are optional path overrides.
  Copy the existing final-hidden Nymeria fixed cases; append two BONES T2M
  examples, visualize at start and every5k using the updated shared renderer.
- Durable watcher tmux: phase2_finalhidden_knorm_ddp256_watch_20260917.
  Operation directory:
  /scratch2/whwjdqls99/cosmos/operations/phase2_finalhidden_knorm_ddp256_20260917_job2270089
  Its watch_and_run.sh polls Slurm every30s, launches an owned overlapping
  compute step after job2270089 is RUNNING, and preserves the allocation
  when training exits. Logs and trainer.exit stay in that operation directory.
  Only scheduler polling runs on the login host, never model/data work.
- No tests or smoke runs were requested. This records configuration and
  automatic-launch intent, not observed training progress.


### Projection A mixed-data DDP batch256 launch (2026-09-17)

- User requested canceling native/as-is K/V mixed allocation2250537 and a
  separate Projection A mixed experiment on retained two-4090 allocation2270088
  (node36, 16 CPUs, 128G RAM). The existing Projection A batch128 run is untouched.
- Launch with `bash motion_expert_t2m_edge/train_kv_projection_ddp.sh` on an
  owned two-GPU compute allocation with `COSMOS_SITE=yonsei` or `grasp`.
  Torchrun uses two ranks, batch128/rank, accumulation1, global batch256.
- Preserve the batch128 Projection A recipe: seven joint blocks, identity-
  initialized K/V projections plus learned K norm, mixed T2M/TI2M, BONES50
  within T2M, camera/head statistics, seed0, 200k target, LR2e-4, warmup1000,
  cosine floor0, per-task centered losses1/1/5 and contact0.05/1/10.
  Workers8/rank; save only positive 10k multiples, no rolling checkpoints.
- A distinct `edge_7layer_reasoner_kv_project_..._nanoparity_gbs256_...`
  output dated20260917 starts fresh; `--resume auto` subsequently resumes only
  this new output, never the batch128 checkpoint. Yonsei writes to Scratch2;
  Grasp uses RUN_ROOT. `PROJECTION_A_OUTPUT_ROOT` can override the output root.
- Reuse the existing Projection A Nymeria fixed suite. `FIXED_VIZ_SOURCE` can
  override its location. Visualize every5k and initially, using five Nymeria
  samples per task plus two fixed BONES T2M examples and the updated renderer.
- Retained tmux session: `phase2_ddp256_2x4090_a_20260917`. Do not cancel this
  allocation when its trainer exits. This entry records launch intent, not
  observed training progress or a completed acceptance test.


### Fixed BONES examples in Edge training visualizations (2026-09-17)

- When training visualization is enabled with Nymeria as its primary source,
  mixed runs with BONES T2M training mass append two held-out BONES-SEED T2M
  examples by default. `--viz-bones-samples` controls this count; zero disables
  the extra suite. BONES-only visualization already uses BONES and does not
  append duplicates. There are no BONES TI2M examples.
- The additional frozen inputs live in
  `<run>/visualizations/fixed_samples_bones/manifest.json`; the original
  `fixed_samples` manifest, Nymeria cases, seeds and sampling batches remain
  unchanged. BONES-suite setup restores training RNG state afterward.
- Each visualization step saves generated-versus-GT BONES videos and arrays
  with a `bones_` prefix for mixed suites. W&B uses
  `visualizations/bones/text2motion/sample_XX`, and step manifests retain both
  input-manifest paths and source labels. The shared default renderer applies.
- This is visualization only: do not change training data mixture, losses or
  checkpoints. It is site-independent and takes effect in new or resumed
  Python trainers; existing live processes are not restarted for this change.


`AGENTS_ALL.md` is the only canonical high-level context document for this repo. Keep detailed project context here, then verify behavior against code before editing.

### Nano-style caption-only Projection A + V norm ablation (2026-09-19)

- New opt-in `EDGE_TEXT_PACKING=nano_caption_only` removes only the T2M
  BOS/EOS/start-of-generation wrapper. Empty CFG text uses one Edge EOS,
  matching Nano's historical raw-caption motion path with Edge's tokenizer.
- TI2M retains its original multimodal chat template, as Nano does. Caption
  rewriting, data/statistics, losses, architecture, and optimizer are unchanged.
- Portable launcher: `bash motion_expert_t2m_edge/train_norm_ablation.sh
  reasoner_kv_project_vnorm_captiononly mixed`, with `COSMOS_SITE=yonsei`
  or `grasp` inside an owned GPU allocation. Fresh B128, 200k, 10k saves.
- Planned Yonsei launch uses idle job2281723/node35, tmux
  `phase2_projA_vnorm_captiononly_b128_20260919`, with a separate scratch2
  output. Runtime logs, not this launch note, determine current status.
- The checkpoint `text_packing` contract preserves inference/resume behavior;
  default native-prefix runs remain unchanged. Details and exact run name:
  `motion_expert_t2m_edge/CAPTION_ONLY_ABLATION.md`.

Root `README.md`, `AGENTS.md`, `CLAUDE.md`, `motion_expert_joint_attention/README.md`, and `motion_expert_joint_attention/DESIGN_7TASK.md` are now short pointers/stubs. Do not re-expand them into parallel sources of truth.

Stale large docs were removed on 2026-07-06: root `COSMOS.md`, `DESIGN.md`, `SAMPLING_NOTES.md`, and `PLAN_ego_video_camera_world_model.md`. Their still-relevant content was consolidated here. `nymeria_world/` remains on disk but is older work and not the source of truth for current joint-attention training.

## Yonsei fresh unfiltered Phase 1 (2026-09-14)

The user requested a fresh Cosmos3-Edge Phase-1 run to 300k, not a continuation
of the filtered 200k model. Launch configuration lives in
`/scratch2/whwjdqls99/cosmos/operations/phase1_edge_unfiltered300k_yonsei_20260914`.
The initial source is the pretrained Edge DCP; later allocations may resume
only this new run's complete model/optim/scheduler/trainer checkpoints.

- Run name: `edge_phase1_T97_20fps_2gpu_bs16_camera_wearer_lora_unfiltered_300k_yonsei_20260914`.
- Outputs: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/`
  followed by the run name. Checkpoints, training logs, TensorBoard and W&B
  staging stay on Scratch2; existing datasets and pretrained models stay on Lustre.
- Native Phase-1 dataset selection now accepts comma-separated
  `NYMERIA_EXCLUDE_ORIGINAL_VIDEO_NAMES`. An explicit empty value removes the
  five-person name exclusion; an unset variable preserves the historical five
  exclusions on both Yonsei and Grasp. Test splits and standard usable-caption,
  camera, train-split and available-latent checks are unchanged.
- This launcher explicitly empties the name and optional quality filters.
  It preserves T97, 256 resolution, 20FPS, four tasks, camera-wearer wording,
  global LoRA, action loss10 and 2 GPUs x16 clips = global batch32.
- LambdaLinear keeps base LoRA LR5e-5 and action multiplier4, with factors
  f_start=f_max=.4, warmup500, cycle300000, f_min=0. Initial effective learning
  rates are2e-5 and8e-5. Save every5k. This is a fresh 300k schedule.
- Both automatic checkpoint-evaluation callbacks and trainer validation are
  disabled; no visualizations are submitted. W&B scalar logging remains online.
- Initial retained allocation2215401 has two PRO6000s. A dependent two-GPU
  replacement may resume this new run; keep four-GPU requests2194983/2194977 held.
  Never cancel the retained allocation merely because its trainer exits.
- Scratch2 is temporary storage; preserve selected checkpoints before the site's
  purge deadlines. This entry records configuration, not observed training progress.

## High-Level Thesis

The project extends Cosmos-3 Nano into an omnimodal world model centered on human motion and egocentric camera action. The recurring pattern is: keep Cosmos's reasoner/understanding pathway frozen, treat text as conditioning rather than a target, and adapt the generation pathway or add motion-specific generation pathways for continuous per-frame outputs.

Text is always a condition, not a captioning target. Do not train `video -> text`, `motion -> text`, or VLM captioning tasks unless explicitly requested. For Cosmos-based motion runs, use raw text strings through the Cosmos processor/reasoner, not precomputed LLM2Vec embeddings, except in the separate BONES-only `motion_expert/bs_*` POC that intentionally uses cached LLM2Vec.

Motion can be both generated output and a control signal. Pure `motion -> video` without scene/image context is considered ill-posed; use `motion + image/text -> video`.

## Experiments

There are four experiment families historically. Current active work is `motion_expert_joint_attention/`; the others are legacy/reference unless the user explicitly asks for them. Do not mix their data contracts, trainers, or representations.

### Root Text-to-Motion Finetune

Files: `train_motion_ft.py`, `sample_motion.py`, `motion_decode.py`.

Goal: text to 369-d Kimodo/BONES-SEED motion. Cosmos reasoner is frozen. The model trains motion heads plus LoRA or full `_moe_gen`. The root run uses raw text through Cosmos's own processor, not LLM2Vec.

Motion representation: 369-d Kimodo/BONES-SEED at 20 fps:

`[0:3] smooth_root_pos`, `[3:5] heading(cos,sin)`, `[5:95] local joint positions`, `[95:275] global rot6d`, `[275:365] velocities`, `[365:369] foot contacts`.

Stats come from `/weka/jungbin/seed/stats/soma_uniform_motions_20fps/`. Decode and losses live in `motion_decode.py`, a pure-torch bit-exact port of Kimodo decode/FK. Known gaps: heading augmentation is off; fps fixed at 20.

Export uses the `kimodo` env; training/sampling uses the `cosmos` env. Full BONES export lives at `/weka/jungbin/seed/cosmos_text_motion_full`.

Sampling convention: training path is `x_t = (1-t)x0 + t*noise`, target `v = noise - x0`, where `t=1` is noise and `t=0` is clean. Euler sampling integrates from `t=1` to `t=0` with `x <- x - dt * v_hat`. Sample-time packing must pack the current `x_t` and current `t`; do not re-noise inside the sampler. CFG uses an empty caption as approximate unconditional.

Healthy root checkpoint overlay in `sample_motion.load_model` previously loaded 406 trainable tensors: 397 `_moe_gen`, 5 motion heads, and 4 `time_embedder`.

### `motion_expert/` POC

Files: `motion_expert/README.md`, `motion_expert/BONES_SEED_POC.md`, and `motion_expert/*.py`.

This is an older standalone MotionExpert text-to-motion POC. It does not instantiate the Cosmos generator.

Main Nymeria POC: text goes through a frozen Cosmos reasoner once, hidden states `H_R` are cached, and a small trainable MotionExpert transformer one-way cross-attends to `H_R`. Information flow is `text -> frozen reasoner -> cached H_R`, plus `neutral_joints -> shape token`, plus noised motion tokens. MotionExpert predicts x0. There is no generator token or generator attention edge.

Representation: 283-d UniEgo/SOMA-30:

`[0:270]` local SE(3) per joint as 30 x `[rot6d + trans]`, `[270:279]` canonical delta, `[279:283]` foot contacts. Decode uses `decode_uniego_torch.py`, bit-exact to Kimodo, no FK. World convention is Y-up, +Z-forward; visualization remaps to matplotlib Z-up and negates X.

Important POC lessons: v1 velocity + feature MSE crumpled/spun; v2 fixed quality with x0 prediction, cached `H_R`, decoded losses, and AdaLN-zero; v3 added floor grounding and viz coordinate fix. Loss is feature MSE + centroid-relative decoded-joint pose + decoded joint-velocity smoothness. Absolute joint loss was unstable and drift-dominated.

BONES-only `bs_*` POC: uses BONES-SEED proportional 283-d UniEgo, cached LLM2Vec pooled text embeddings, an in-context text token plus shape token, no Cosmos and no reasoner. Runs in `kimodo`, not `cosmos`. It uses `SOMABonesSeedDataset` indexing by subclassing and overriding motion I/O. Text is dropped to the cached `""` embedding for CFG; shape is never dropped. This POC is intentionally different from the Cosmos raw-text runs.

**Machine boundary (confirmed 2026-07-14): `motion_expert/bs_*` is A100-machine-only.** Do not run, debug, or validate these BONES scripts on the H200 `a3ultra` machines. They work in the intended A100 environment, whose data and run roots use `/mnt/shared/...`. On the H200 machine, the `kimodo` environment does not provide `cosmos_framework` or `diffusers`, and the local Cosmos checkout exposes UniPC under `cosmos_framework.model.vfm` while the pulled BONES UniPC code expects the older `cosmos_framework.model.generator` namespace. Consequently the BONES official-UniPC paths fail there. This restriction does not apply to `motion_expert_joint_attention/`, native Phase 1, or H200 Phase-3 job `2870`.

Native-schedule BONES Phase-2 POC: `motion_expert/bs_native_flow.py`, `bs_train.py`, and `bs_sample.py` keep the BONES model/data/x0/loss recipe fixed while replacing the unshifted schedule with Cosmos's action-like shifted logit-normal training sigma and shifted 1000-step inference ladder. Default motion shift is 3. Training-time sigma and diagnostic Euler/Heun integration are local formulas pinned to NVIDIA Cosmos Framework commit `3d9c0878fd0dde76eac98161aed0493d85a036fd`; the later official-UniPC path imports the audited Cosmos scheduler supplied by the A100 `kimodo` environment. The BONES model itself still has no Cosmos reasoner or generator. Launch through `motion_expert/sbatch_bs_native_phase2.sh` on the A100 machine, never on an H200 or login CPU.

Native-schedule POC launch status on 2026-07-11: Slurm smoke job `10386` passed five real-data optimizer steps on one A100 with finite losses/gradients, shifted sigma mean about `0.70-0.74`, and 10.3 GB peak memory. Production job `10387` runs `bs_native_x0_logitnormal_shift3_w1_1_5_200k` for 200k steps, batch 128, loss weights `1/1/5`, reusing `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_incontext_v1/bs_train_index.json`. Its run directory is `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_w1_1_5_200k` and Slurm log is `/home/jungbin_cho/cosmos_motion_ft/slurm-bsnatp2-10387.out`.

`motion_expert/bs_tmr_eval.py` is the in-memory generation evaluator for comparing native- and
legacy-schedule BONES checkpoints with the shape-aware C45 TMR. It follows Kimodo's benchmark
metric contract without writing generated motions or embeddings: generate normalized proportional
283-D UniEgo at 20 FPS, unnormalize and decode to SOMA-30 joints, resample joints to 30 FPS and
featurize with C45's official-stat `TMRMotionRep`, and pass the same centered proportional
`neutral_joints` to both the generator and C45 shape encoder. It reports protocol and plain
R-precision, TMR FID, benchmark foot skate/contact metrics on native 20-FPS joints, and bone-length
MAE versus the conditioned skeleton. Generated four-channel foot contacts are thresholded at 0.5.
Generation uses the benchmark LLM2Vec cache because the training cache intentionally omits almost
all held-out benchmark captions; both caches store outputs of the same frozen text representation.
The full content/overview comparison launcher is `motion_expert/sbatch_bs_tmr_eval.sh`; its default
evaluator is C45 step 5000, selected as C45's best full-overview checkpoint.

Full in-memory C45 comparison completed as Slurm job `10570` on 2026-07-13. It discovered 917
content/overview cases and scored 911; the six exclusions were proportional GT UniEgo windows with
non-finite values. Both step-200k generators used 100 steps, CFG 2, identical case-seeded noise,
and C45 step 5000. Native versus legacy: protocol R@3 `69.15` vs `54.34`, plain R@3 `58.84` vs
`42.81`, and FID gen-GT `0.05255` vs `0.08028`. Physical quality favored legacy: predicted-contact
skate `13.85` vs `10.23` cm/s, contact consistency `0.818` vs `0.879`, skate ratio `0.260` vs
`0.212`, and conditioned-skeleton bone MAE `0.360` vs `0.269` cm. GT references were `1.90` cm/s,
`1.000`, `0.102`, and `0.181` cm respectively. Aggregate result:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_vs_legacy.json`.

Native Heun sampler follow-up: `bs_native_flow.sample_x0_heun` implements explicit trapezoidal
RK2 on the same shifted native ladder, with an Euler final interval because its endpoint is sigma
zero. `bs_sample.py --native_solver heun` exposes it for normal sampling, and `bs_tmr_eval.py
--native-solver heun` exposes it for in-memory evaluation. A 50-step run has 99 denoiser
evaluations, or 198 model forwards with two-branch CFG, versus Euler-100's 100/200. Five focused
native-flow tests pass and GPU smoke job `10606` passed. Full job `10607` scored all 911 cases:
Heun-50 versus Euler-100 protocol R@3 `69.59` vs `69.15`, plain R@3 `57.96` vs `58.84`, FID
gen-GT `0.05193` vs `0.05255`, predicted-contact skate `13.73` vs `13.85` cm/s, contact consistency
`0.820` vs `0.818`, skate ratio `0.258` vs `0.260`, and bone MAE `0.352` vs `0.360` cm. This is
effectively a tie: Heun gives marginal FID/physical improvements but does not fix foot skating and
slightly lowers plain R@3. Result:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_heun50.json`.

Official Cosmos-3 UniPC follow-up: `bs_native_flow.sample_x0_unipc` does not reimplement UniPC. It
imports NVIDIA `cosmos-framework==1.2.2`'s `FlowUniPCMultistepScheduler` from the `kimodo` env and
calls its `step` method directly; installed scheduler SHA-256 is
`03aef1959f273b704ca4954f69b2a34df0fdd412f6acc8ab91625eeed78cf4fe`, byte-identical to audited
Cosmos Framework commit `3d9c0878fd0dde76eac98161aed0493d85a036fd`. The only local adapter converts
guided x0 to the scheduler's required flow velocity as `(x - x0_cfg) / sigma`. Preserve the real
wrapper's defaults: 35 steps, order 2, `bh2`, `flow_prediction`, `predict_x0=True`, lower-order
final, shift 3, and no dynamic shifting. The official code shifts its constructor sigma range and
then shifts the inference ladder again in `set_timesteps`; do not replace this with the earlier
single-shift Euler/Heun diagnostic ladder.

UniPC GPU smoke job `10608` passed. Full job `10609` evaluated all 911 cases with C45 step 5000,
CFG 2, and 35 denoiser evaluations/70 CFG model calls: protocol R@3 `69.81`, plain R@3 `59.06`,
FID gen-GT `0.05143`, predicted-contact skate `13.54` cm/s, consistency `0.821`, skate ratio
`0.257`, and bone MAE `0.352` cm. Relative to Euler-100, this is +0.66 protocol R@3, +0.22 plain
R@3, 2.13% lower FID, and slightly better physical metrics with 65% fewer denoiser evaluations.
Result: `/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_content_overview_native_unipc35.json`.

Foot-skating loss follow-up launched 2026-07-13. `motion_expert/bs_losses.py` contains tested,
configurable contact-aware losses; `bs_train.py` records/logs them, and
`sbatch_bs_native_phase2.sh` exposes all weights through `BS_NATIVE_W_*`. The SOMA-30 contact order
is `[LeftFoot, LeftToeBase, RightFoot, RightToeBase]`, indices `[24,25,28,29]`. Contact BCE is
class-balanced and centered on raw contact threshold 0.5; horizontal physical foot speed and raw
foot-height reconstruction are masked with GT contacts so the model cannot avoid them by predicting
no contact. Raw Y channels are used for height because they exactly equal decoded Y under UniEgo's
yaw-only canonical frame and avoid unstable cumulative-decoder gradients. Four focused loss tests
pass; final contact-aware GPU smoke `10616` passed with gradient norms `2.7-5.5`.

Two 200k controlled runs train from scratch. Job `10622` is
`bs_native_x0_logitnormal_shift3_w1_10_100_inline10k_200k`, changing existing
feature/joint/smooth weights to `1/10/100`. Job `10623` is
`bs_native_x0_logitnormal_shift3_contactaware_c0p05_v1_h10_s2_inline10k_200k`, retaining `1/1/5`
and adding contact/foot-velocity/foot-height weights `0.05/1/10` with contact logit scale 2.
Separate dependent evaluations `10619/10620` were canceled. `bs_train.py` now initializes C45 and
all 911 GT references once, then evaluates the live model in-process every 10k checkpoint and at
the final step. Results are `<run>/inline_eval/step_XXXXXX.json` plus `history.json`; no motions or
embeddings are saved. CPU/CUDA RNG states and training mode are restored after each callback.
Integration smoke `10621` passed the train/save/live-eval/write path. Native checkpoint
visualizations also use official UniPC-35. Full rationale, paths, smoke ablations, and evaluation
outputs are in `motion_expert/BONES_SEED_POC.md` section 18.

Early 2026-07-14 results motivated two paired hybrid follow-ups. Job `10644` keeps the stronger
`1/10/100` reconstruction weights and adds foot velocity/height weights `0.5/5` without contact
BCE. Job `10645` is identical but adds contact BCE weight `0.025` at logit scale 2. Their run
directories are respectively `bs_native_x0_logitnormal_shift3_w1_10_100_foot_v0p5_h5_inline10k_200k`
and `bs_native_x0_logitnormal_shift3_w1_10_100_softcontact_c0p025_v0p5_h5_s2_inline10k_200k` under
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/`; both use the same inline 10k C45 evaluation.

Shape-awareness evaluation was strengthened on 2026-07-14. `bs_tmr_eval.py` now supplements
conditioned-skeleton bone MAE with actor-centered bone correlation/slope/variance and an optional
paired counterfactual pass. The counterfactual uses identical text, duration, and initial noise but
conditions on the most different natural held-out skeleton, then measures generated/requested bone
delta alignment, target advantage, and retrieval retention. New inline evaluators default to
`farthest`; processes already running before this edit retain the old imported callback. Initial
smoke `10650` failed before Python because Slurm used `/bin/sh` for a `pipefail` wrapper, and blocked
job `10651` was canceled. Corrected explicit-Bash smoke `10679` gates final-200k backfill `10680`,
which also waits for training jobs `10644/10645`; output is
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_shape_counterfactual_final200_ablation_runs.json`.
Smoke `10679` completed successfully with all five metric tests and an eight-case paired
UniPC/C45 integration pass. Foot-only hybrid job `10644`, soft-contact job `10645`, and final shape
backfill job `10680` all completed. The backfill report is
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/c45_step5k_shape_counterfactual_final200_ablation_runs.json`.

Full-contact continuation/all-benchmark follow-up launched 2026-07-15. `bs_train.py` supports a
strict model-only warm start with global checkpoint numbering and a restart-local LR schedule. The
source step-200k checkpoint has no optimizer or RNG state, so job `10747` is explicitly a fresh
AdamW warm start rather than an exact resume: unchanged full-contact recipe, steps `200k->500k`,
LR `5e-5`, 1k local warmup, cosine decay, and seed `200000`. It keeps 10k in-process 911-case C45
overview evaluation. Run:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_contactaware_c0p05_v1_h10_s2_continue200to500k_lr5e-5_seed200000`.
`sbatch_bs_all_t2m_eval.sh` evaluates all six applicable content/repetition overview/single/multi
text-to-motion suites using UniPC-35, physical metrics, and paired shape intervention, then
`bs_all_benchmark_summary.py` writes one structured report. Step-200k job `10746` writes under
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_tmr_eval/full_contact_200k_all_text2motion_unipc35_shape_cf`;
dependent final-500k job `10748` writes the corresponding `full_contact_500k...` directory.
Constraint-conditioned suites are marked not applicable because this generator has no constraint
input. Continuation/evaluation/unit smokes `10742/10743/10744` all passed. Initial job `10746`
revealed that benchmark readers looked up raw text in a cache keyed by Kimodo-sanitized prompts;
this was fixed in `bs_tmr_eval.py`, `st_inline_eval.py`, and `official_tmr_eval.py`. Audit `10751`
recovered all 36 valid timeline prompts, and correction job `10752` rebuilt the report with
9,124/9,162 discovered cases; only 32 non-finite GT motions and six sub-10-frame requests remain.
Per-suite protocol/plain R@3 is `70.58/59.60`, `63.93/54.28`, `70.06/62.55`, `77.89/65.33`,
`64.74/52.59`, and `77.30/68.89` in the order above. The case-weighted suite means are
protocol/plain R@3 `71.56/60.80`, FID `0.02195`, contact skate `3.44 cm/s`, bone MAE `0.382 cm`,
shape correlation `0.969`, and counterfactual response slope `0.906`; these are means of
suite-level metrics, not one merged retrieval computation.

Generator-normalization ablation launched 2026-07-15 as job `10860`. It reproduces the
full-contact 200k recipe, including the historical global DataLoader RNG and pre-increment periodic
checkpoint indexing, but replaces the proportional-data stats with
`motion_expert/stats/uniego283_{mean,std}.npy` (tag `nymeria_grounded_uniego283`, SHA prefixes
`bd1d6bdc`/`ee069e3a`). Run:
`/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_native_x0_logitnormal_shift3_contactaware_c0p05_v1_h10_s2_nymeria_grounded_stats_inline10k_200k`.
Control smoke `10859` with the original proportional stats reproduced the archived baseline's first
update exactly to printed precision; alternate-stat smoke `10857` passed five finite updates.
`bs_normalization.py` now pins paths, hashes, shape, dtype, and a tag in configs/checkpoints.
`bs_sample.py` and `bs_tmr_eval.py` resolve stats from checkpoint metadata by default and reject
silent mismatches; the evaluator resolves each generator independently, allowing valid mixed-stat
comparisons. Full rationale and job audit are in `motion_expert/BONES_SEED_POC.md` section 21.

### `nymeria_world/` Camera World Model

Files: `nymeria_world/*`.

Goal: egocentric video + text + ego-camera world model. This is older native-camera work and is not the current source of truth for joint-attention training. Human body motion and audio are out of scope there. This experiment uses Cosmos's native training/inference stack (`cosmos_framework.scripts.train` and `cosmos_framework.scripts.inference`), not `train_motion_ft.py`.

Camera action is Cosmos native `camera_pose`, domain id 2, raw dim 9 `[pos(3), rot6d(6)]`, zero-padded to 64 only for Cosmos action heads. Do not z-score camera actions. The representation is relative SE(3): `Delta T_t = T_{t-1}^{-1} T_t`, implemented through `pose_abs_to_rel(rotation_format="rot6d", pose_convention="backward_framewise")`.

Coordinate convention: source camera npz is raw Aria device frame, but the chosen training/eval convention is RGB optical plus `Rz(-90deg)`, matching the upright video. Finetuning can learn any consistent frame, but OpenCV/upright-video frame aligns best with Cosmos zero-shot predictions.

Zero-shot inverse dynamics showed a 7-17x translation scale gap. The diagnosis is action temporal-step/training-distribution mismatch, not coordinate frame, fps, de-normalization, or metric-scale error. The model emits action magnitudes similar to about 7-frame GT displacement. Finetuning pretrained camera action heads should adapt the convention.

Native modes: `forward_dynamics`, `inverse_dynamics`, and `policy`. `image2video` was tried in a 4-task mixture but is dropped for multi-GPU training because it has no action, so action-head parameters can get no gradients on all-i2v steps and distributed collectives desync.

Training modes:

- LoRA default: reasoner frozen; LoRA on generator attention plus pretrained camera action heads kept trainable. Action heads are not re-initialized in this camera experiment.
- Full-gen: reasoner frozen; full generation pathway trainable. Requires FSDP sharding (`dp_shard=8`) because replicated optimizer state OOMs.

Checkpoints are LoRA/action-head deltas. Native inference cannot overlay LoRA directly, so evaluation is merge-then-sample: `export_merge_lora.py` -> `prep_test_eval.py` -> `run_infer_merged.sh` or `sbatch_infer_3tasks.sh` -> `viz_eval_samples.py`.

Inference must run from `cosmos-framework`, use `--no-guardrails`, use `.jsonl` inputs, set local Wan VAE override when needed, and set `lora_enabled=false` for merged checkpoints.

### `native_phase_training/` Official-Compatible Phase 1

Read `native_phase_training/README.md` before editing this path. That README is the detailed runbook for the current official-compatible Phase 1 camera/video generator run.

Cosmos3-Edge Phase-1 continuation status on 2026-09-03: production job `528782`
survived three batch preemptions, reached permanent checkpoint step 20,000, and
then failed after step 22,336 because the W&B device-monitor callback exhausted
the quota of its default home-backed artifact staging directory. This was not a
GPU, CPU, host-memory, or model failure; its peak host RSS was about 280.5 GiB
against 512 GiB. `sbatch_phase1_edge_4gpu_bs8_batch.sh` now redirects both W&B
data/staging and cache to Weka under `.wandb_runtime`. Replacement job `529842`
loaded complete rolling checkpoint step 22,241, reused W&B ID `6kwex4fb`,
passed the formerly failing W&B callback at step 22,400, and completed all
100,000 updates with exit code zero on 2026-09-07. Its final DCP is the
production run's `checkpoints/iter_000100000`; W&B run `6kwex4fb` finished
normally. Final-eval jobs `537114` and `539785` exposed two regressions in a
later explicit-site change: an unset `COSMOS_SITE`, then a Grasp artifact root
missing its `/weka` suffix. Both mappings and the evaluator launcher are fixed,
and final-checkpoint evaluation was resubmitted as job `541395`. Known
ECC-faulted `ll-l40-1` remains excluded. Verify dynamic Slurm state before
reporting evaluation progress.

Purpose: train a new Nymeria camera/video generator LoRA that keeps the training and sampling contract close to NVIDIA's native Cosmos-3 Nano action/video setup. This path exists because the older joint-attention Phase 1 and custom sampler showed poorer visual quality than base Cosmos with the official sampler. The new path uses saved Wan-VAE latents for training speed, but it preserves native Cosmos sequence construction, rectified-flow noising, loss masks, losses, action heads, checkpoint format, and official inference compatibility. Batch cardinality is an optimization choice and does not affect sampler compatibility.

This directory is intentionally isolated from `motion_expert_joint_attention/`. It has no motion expert, no 3-way joint attention, and no modality bridge. It should produce a frozen video/camera specialist that future bridge or motion experiments can reuse.

Main files:

- `native_phase_training/latent_omni_model.py`: `LatentOmniMoTModel(OmniMoTModel)`. If `video_latents` is present in the training batch, it uses those cached clean VAE latents instead of encoding pixels. Without `video_latents`, it falls back to native `OmniMoTModel`, which is why official inference still works.
- `native_phase_training/latent_nymeria_dataset.py`: cached-latent Nymeria camera dataset. It emits native action-SFT fields plus dummy video metadata and real `video_latents`. Forward/policy prompts use official action JSON, inverse text is exactly empty, and image-to-video uses the official generic duration/resolution prose. New caches can carry an immutable `latent_cache_contract.json`; production high-tier training requires and validates it before workers start.
- `native_phase_training/latent_nymeria_dataset.py` also defines `CyclingDataLoader`, which is required for long runs. Native `IterativeJointDataLoader` assumes child streams are infinite; finite map-style `DataLoader` streams can otherwise exhaust and make the trainer silently spin without advancing `global_id`.
- The same file defines `LatentAwareIterativeJointDataLoader`. The stock counter sees only the `[3,97,1,1]` dummy video; the local override adds the real `25*8*8=1600` cached-latent patch tokens. Production defaults to exactly four clips per GPU (`NATIVEP1_CLIPS_PER_GPU=4`); setting that environment variable to `0` restores the audited 45,056-token-budget packing mode.
- `native_phase_training/experiment.py`: registers Hydra experiment `world_camera_nymeria_latent_nano`, resolves local tokenizer/VAE paths, defaults to `resolution=256`, supports a run-pinned resolution/shift override, and builds the four native streams.
- `native_phase_training/AUDIT.md`: records the finite-loader, prompt, cached-latent packing, and evaluation-contract audits and fixes.
- `native_phase_training/prep_test_eval.py`: held-out official-inference input builder for forward/inverse/policy/image-to-video, pinned to T97/action96, 20 FPS, 256, and shift 3.0. Use this instead of the historical 480/shift-10 helper.
- `native_phase_training/visualize_checkpoint.py`: validates four mode-specific official outputs, creates GT/generated videos for forward/policy/image-to-video, camera plots for inverse/policy, and a manifest.
- `native_phase_training/evaluate_forward_dreamsim.py`: optional canonical-full71 forward evaluator using official DreamSim 0.2.1 over all 96 generated suffix frames, with early/middle/late summaries.
- `native_phase_training/evaluate_forward_cdfvd.py`: optional canonical-full71 set evaluator using CVPR 2024 content-debiased FVD with VideoMAE-v2-SSv2. This is deliberately not legacy TensorFlow/I3D FVD; it uses all 96 generated suffix frames for the full score and all 32 frames per horizon. See `native_phase_training/FORWARD_VIDEO_METRICS.md` for the pinned revision and exact contract.
- `native_phase_training/checkpoint_eval_callback.py` plus `sbatch_checkpoint_eval.sh`: production rank 0 submits an isolated official four-mode EMA/UniPC evaluation after every successful checkpoint save. The stock in-training generation callback stays disabled.
- `native_phase_training/run_contract.py`: persists architecture-critical Phase-1 settings in each run and resolves/validates them before evaluation. Schema 2 includes model resolution, T, and training shift in addition to adapter/task settings. Manual evaluation must never rely on default environment values.
- `native_phase_training/latent_cache_contract.py`, `validate_latent_cache.py`: immutable cache geometry/provenance plus exact expected-file-set validation.
- `native_phase_training/prepare_phase1_eval_tier.py`, `validate_eval_inputs.py`: convert canonical held-out records to a released model tier and reject resolution/shift/T conflicts before inference.
- `native_phase_training/sbatch_precompute_latents_720tier.sh`, `sbatch_phase1_native_camera_720tier.sh`: isolated full-cache and controlled high-tier Phase-1 launchers.
- `native_phase_training/world_camera_nymeria_latent.toml`: production TOML.
- `native_phase_training/run_latent_train.py`: training entrypoint. It assigns TensorBoard to `$TB_LOG_DIR` if set, otherwise `${job.path_local}/tensorboard`, avoiding the old shared fallback directory.
- `native_phase_training/inference_config.py`: official inference shim; use it with `--config-file native_phase_training/inference_config.py`.
- `native_phase_training/sbatch_phase1_native_camera.sh`: production Slurm launcher.

Default data contract:

- latent root: `/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97`;
- `NYMERIA_NUM_FRAMES=97`, `NYMERIA_RESOLUTION=256`, fps 20;
- latent `.npz` keys: `latents [48,25,16,16]`, `camera_action [96,9]`, `image_size`;
- camera action is raw Cosmos `camera_pose`, domain id 2, raw dim 9, padded to 64 only for the native action projection. Do not z-score it.

High-tier training contract added 2026-07-26:

- model config tier is `NYMERIA_RESOLUTION=720`;
- the Nymeria spatial transform/cache key is `--resolution 480`, which yields raw/transformed `640x640` frames in this pipeline;
- cached latents are fp16 `[48,25,40,40]`; camera action remains `[96,9]`;
- released Cosmos Nano config maps tier 720 to shift 10, and this experiment follows that released checkpoint/config contract;
- full unfiltered train input has 119,632 caption rows but 115,583 unique physical `(uuid,start)` cache files, estimated near 414 GiB;
- full cache root is `/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97_720tier_640`;
- run name is `native_phase1_camera_json_720tier640_bs4_lora5e5_action4x_ema_100k`;
- the controlled comparison keeps all four tasks, prefix 1, global Q/K/V/O LoRA, action weight 10, LR `5e-5`, action LR 4x, four clips/GPU, EMA, 100k steps, and 5k saves. Only spatial tier/cache geometry and corresponding released shift change from Original.

The real 8-GPU batch-four smoke used 640-pixel cached inputs, reached finite losses on all ranks, stayed near 117-119 GiB per H200, and saved a complete DCP at iteration 1. Exact resume restored all state and saved iteration 2. Official EMA/UniPC inference from that checkpoint completed all four modes at 640x640/T97/20 FPS under `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_720_bs4_v2/official_inference_4mode_iter2_T97_v2`; action-bearing modes are finite `[96,9]`, and GT/generated provenance is explicit in `viz/manifest.json`. Production submitted 2026-07-26 from pushed commit `a1e49f4`: full-cache array `3109`, then Phase-1 job `3110` under `afterok:3109`. Current-state details and logs are recorded in `native_phase_training/README.md`.

Task mix:

- `forward_dynamics`: 40%;
- `inverse_dynamics`: 25%;
- `policy`: 20%;
- `image2video`: 15% video regularizer.

Prompt contract:

- forward dynamics and policy: `ActionPromptJsonFormatter` JSON, matching official action inference;
- inverse dynamics: exactly empty prompt;
- image-to-video: plain caption plus official generic duration/FPS and resolution prose, without action-viewpoint prose;
- 10% CFG dropout is applied after formatting and drops the whole prompt to empty-string conditioning.

Native RF/loss settings are intentionally inherited from Cosmos Nano:

- video train-time distribution: `waver`;
- image train-time distribution: `logitnormal`;
- `independent_action_schedule=false`, so action reuses the sample's vision/video sigma; configured `train_time_action_distribution=logitnormal` is dormant;
- train-time loss weight: `uniform`;
- action loss weight: `10.0`;
- shift config: `{256:3, 480:5, 720:10}`.

Trainable default:

- generator LoRA on `q/k/v/o_proj_moe_gen`, rank 16, alpha 32;
- action heads `action2llm`, `llm2action`, and `action_modality_embed`;
- frozen reasoner and frozen base generator weights.
- native PowerEMA remains enabled. The optimizer updates regular LoRA/action parameters; after each step Cosmos updates an FP32 EMA model, and official checkpoint evaluation uses `--use-ema-weights`. Because the generic implementation duplicates and traverses the full frozen model, it is memory-inefficient for LoRA-only training, but it is retained for this official-compatible baseline.

Vision-head recommendation as of 2026-07-10: keep `vae2llm` and `llm2vae` frozen for the first native baseline. The goal is to preserve base Cosmos visual quality while adapting camera control through generator LoRA and action heads. A later ablation can unfreeze `llm2vae`/`vae2llm` and possibly `time_embedder` at lower LR if the baseline underfits, but that increases the risk of perturbing the pretrained visual distribution.

Schedule/LR recommendation as of 2026-07-12: do not use the old 200k-step Phase-1 schedule or uniform `2e-4` LoRA/action-head LR. The fixed-four baseline runs 100k max steps with `NATIVEP1_LORA_LR=5e-5`, `NATIVEP1_ACTION_LR_MULT=4.0`, and checkpoints every 5k. With `f_start=f_max=0.4`, the first 500 steps are a flat plateau at `2e-5` effective LR for generator LoRA and `8e-5` for the action modules, followed by linear decay. Keep these rates for the first fixed-four baseline: they are already 10x/2.5x below the old custom run's effective LoRA/action rates, and changing batch size plus LR simultaneously would obscure the comparison. Evaluate early checkpoints and lower the LoRA rate only if visual quality drifts.

Phase-1 visual-quality audit as of 2026-07-23: read
`native_phase_training/PHASE1_VISUAL_QUALITY_AUDIT.md` before attributing the
historical run's prettier videos to the model. The inspected historical output
is 640x640 and at least one old evaluation loaded
`/weka/jungbin/tmp_merge/m97f_7000/model`; current A output is 256x256 and has an
architecture recovered from its saved run config before direct DCP load. A
direct A/100k EMA prefix-1 test at 256 with UniPC shift 10 was effectively tied
with shift 3, so shift is not the visual-quality fix.
FP16 cache precision is also a very low-probability cause: Wan encode uses bf16
autocast, and 9.83M values from 32 cached T97 files were all exact after a bf16
round trip. If a fully matched old/new sample still favors the old model, the
leading hypothesis is global-LoRA optimization: old training mixed
forward/inverse/policy across ranks on every up-to-128-clip update and used
action weight 10; A uses one homogeneous task per 32-clip update, adds pure I2V,
and uses action weight 2. This may let Nymeria video reconstruction dominate the
shared LoRA while camera heads still achieve strong inverse metrics. It remains
a training-side hypothesis. The completed 2026-07-24 five-sample
same-checkpoint 2x2 test identifies spatial inference tier as a separate,
dominant factor: at fixed shift 3, 720-tier sampling lowers adjacent RGB change,
temporal second difference, and flow-compensated residual by
`17.7%/18.8%/14.0%` versus 256; shift 3 to 10 at fixed 256 changes those by only
`-0.7%/-0.4%/-1.9%`. The old run's dataset config is itself 256, so this is not
high-resolution Nymeria training; the larger pretrained spatial-token inference
path is producing the smoother result. See the audit for exact paths and
caveats.

Official inference syntax gotcha: the config must be passed as the relative Python file path:

```bash
--config-file native_phase_training/inference_config.py
--experiment world_camera_nymeria_latent_nano
```

Do not pass an absolute `.py` path, and do not pass module form without `.py`.

Required local env:

```bash
cd /home/jungbin_cho/cosmos-framework
export PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework:${PYTHONPATH:-}
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export BASE_CHECKPOINT_PATH=/weka/jungbin/cosmos3_nano_dcp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

Smoke status on 2026-07-10:

- One-step post-patch train smoke passed on `a3ultravis-a3ultranodeset-1`, GPU 1.
- Smoke checkpoint: `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/cosmos3_camera/camera_world/native_phase1_smoke_train_postpatch/checkpoints/iter_000000001`.
- Official inference loaded that exact checkpoint and successfully sampled `forward_dynamics`, `inverse_dynamics`, and `policy`.
- Each smoke output had `sample_outputs.json`, `vision.mp4`, action content shaped `96 x 9`, and 97-frame 256x256 MP4 at 20 fps.
- Official inference output dir: `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/official_inference_postpatch`.

Production OOM incident:

- Slurm job `2792` (`nativep1`) failed on 2026-07-10 after reaching the first training forward pass.
- It was CUDA OOM on rank 2, not a data/checkpoint bug.
- Evidence in `/home/jungbin_cho/cosmos_motion_ft/slurm-nativep1-2792.out` lines 1552 and 1700: GPU 2 had only 623 MiB free; our process used 123.29 GiB; external PID `2951793` used 15.88 GiB.
- PID `2951793` was owned by `jmleeluck` and was running `scripts/train_latent_grpo_libero.py ... --device cuda:2` on bus `00000000:96:00.0`, which maps to GPU 2.
- The production config also had `model.config.compile.enabled=true`, while successful smokes had compile disabled. TorchInductor compile increased first-step memory pressure.

Patch after job `2792`:

- `sbatch_phase1_native_camera.sh` now requests `#SBATCH --exclusive`;
- it prints `nvidia-smi` state before launching;
- it runs a memory preflight and fails before torchrun if any GPU has less than `NATIVEP1_MIN_FREE_MIB` free, default 132000 MiB;
- it sets `model.config.compile.enabled=false` in the production override.

Replacement job after OOM:

- Slurm job `2799` (`nativep1`) was submitted on 2026-07-10.
- It was pending on resources after submission. Launcher state at submission: 100k max steps, default LoRA LR `2e-4`, compile disabled.
- This was superseded by the lower-LR split recipe: base LoRA LR `5e-5`, 4x LR multiplier for `action2llm`, `llm2action`, and `action_modality_embed`, run name `native_phase1_camera_latent_lora5e5_action4x_100k`.
- If job `2799` still exists, cancel/resubmit before treating it as the active baseline because Slurm copied the old launcher at submission time.
- Job `2800` was submitted with the lower-LR split recipe on 2026-07-10. At submission it was pending on resources.
- Job `2800` was cancelled before start after the finite-dataloader audit. Resubmit only after verifying the `CyclingDataLoader` wrapper remains active in `native_phase_training/experiment.py`.
- After the cycling fix, a forced-exhaustion smoke (`NYMERIA_MAX_SAMPLES=8`, 20 iterations) passed and saved `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/cosmos3_camera/camera_world/native_phase1_cycling_exhaust_smoke/checkpoints/iter_000000020`. Official inference loaded it and produced successful forward/inverse/policy samples under `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1/official_inference_cycling_exhaust_smoke`.
- Job `2801` was submitted on 2026-07-10 with the lower-LR split recipe plus `CyclingDataLoader`. At submission it was pending on priority.
- The 2026-07-11 audit found job `2801` used legacy prose action prompts and fixed 32-sample packing based on dummy pixels. It was superseded by the from-base token-budget run `native_phase1_camera_json_tokpack_lora5e5_action4x_100k`; do not resume job `2801` into either the token-budget or fixed-four recipe.
- Corrected smoke checkpoint `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_contract/cosmos3_camera/camera_world/native_phase1_json_tokpack_smoke_20260711/checkpoints/iter_000000004` completed train/save/load. Official inference output `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_contract/official_inference_4mode_v2_20260711` passed all four modes with separate directories, UniPC shift 3, 97-frame 256x256/20-FPS MP4s, and `[96,9]` actions for action modes. Its `viz/manifest.json` covers all four modes.
- Job `2838` started on 2026-07-12 with the token-budget configuration and was intentionally cancelled at step 4,399 on 2026-07-12 before its first 5k checkpoint; it produced no training checkpoint. Replacement job `2852` was submitted from the fixed-four launcher under `native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k`, with exactly four clips per GPU and PowerEMA retained. At submission it was pending on scheduler priority, with node 3 as the estimated backfill target.
- Keep the stock `EveryNDrawSample` callback disabled for cached-latent training; it sees 1x1 dummy GT, covers only the selected stream, lacks action visualization, and uses mismatched generic sampler defaults. Production sets `NATIVEP1_AUTO_EVAL=1`: the rank-0 post-save callback submits a separate one-GPU Slurm job that runs official EMA/UniPC inference and `visualize_checkpoint.py` for all four modes. Set it to 0 for smoke/debug runs.

Final fixed-four Phase-1 status (2026-07-14): job `2852` completed 100k steps at `/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k`. The final **100k EMA** DCP at `checkpoints/iter_000100000`, not a historical joint `.pt`, is the Phase-3 generator initialization. Its full official-inference evaluation used all 71 held-out forward windows and all 71 held-out inverse windows on node 3 GPU 2; outputs are under `eval_full71_inverse_forward/iter_000100000`. All 142 samples passed, and analysis wrote 71 inverse camera plots plus 71 forward comparison MP4s. Inverse means: rotation `0.213036 deg`, direction cosine `0.838294`, length/scale ratio `1.007017`, normalized translation error `0.003177 m`, and Sim(3) ATE `0.023574 m`. Forward means, excluding the conditioned frame 0: PSNR `19.4988 dB`, SSIM `0.612583`, and LPIPS-Alex `0.285336`. Early/middle/late horizon means are PSNR `23.2029/18.5262/16.7672 dB`, SSIM `0.724805/0.589087/0.523856`, and LPIPS `0.159399/0.306662/0.389947`. The 100k checkpoint is the best of the 12 fully evaluated checkpoints (5k through 100k) on rotation, direction cosine, translation error, ATE, PSNR, SSIM, and LPIPS. Relative to 90k, it improves PSNR by `0.1223 dB`, SSIM by `0.00405`, LPIPS by `0.00386` lower, and ATE by about 3.1%; mean scale ratio moves slightly farther from ideal 1.0 (`1.00515` to `1.00702`), although its median improves to `0.99860`.

Phase-1 video-quality ablation evaluation correction (2026-07-22): the automatic `checkpoint_evals` suite contains only five fixed held-out sources expanded over prefixes `[1,9,17,33,49]`; its `n=5` aggregates are diagnostics, not the original benchmark. The five underlying GT clips/actions/poses are byte-identical to the first five historical full-71 instances, although the compact ablation captions intentionally replace standalone `C` with `A person`. A-E now also configure a distinct 10k-cadence `eval_full71_inverse_forward` callback using the exact historical 71 prefix-1 forward and 71 inverse records, EMA, official shift-3 UniPC, 30 steps, and guidance 1 on an exclusive eight-GPU node. The full-71 launcher and legacy all-checkpoint driver resolve the saved architecture contract before model import. Only `eval_full71_inverse_forward` should be compared to the original Phase-1 metrics above; retain `checkpoint_evals` for variable-prefix, policy, I2V, and qualitative inspection.

A/B were already running when the full-71 callback was added, so their in-memory callback sets remain the old compact-only version. Explicit 70k full-71 backfills are Slurm jobs `3052` (A) and `3053` (B); compact 70k jobs are `3050/3051`. D/E were pending and will load the dual-callback configuration when they start. Future A/B 10k checkpoints need explicit full-71 submission unless those training jobs are restarted; do not mistake the continued compact callback submissions for canonical evaluation.

Final Phase-1 video-quality comparison as of 2026-07-24: canonical full-71 prefix-1 forward PSNR/SSIM/LPIPS, DreamSim, and FP32 VideoMAE-v2 CD-FVD are Original-100k `19.4988/.612583/.285336/.170482/281.614`, A-100k `19.5343/.614062/.284598/.169134/295.561`, B-100k `18.6945/.586307/.315056/.177504/304.385`, D-100k `18.3851/.576988/.328101/.189021/319.056`, and historical-7k under the current contract `18.4828/.574474/.322919/.175386/282.414`. Corresponding inverse rotation/translation/ATE are Original `.213 deg/3.18 mm/2.36 cm`, A `.272/3.83/2.41`, B `.287/4.04/2.52`, D `.315/4.39/3.03`, and old 7k `.268/4.68/3.99`. C stopped at 65k and has no full-71 report; its compact five-source forward macro is `15.203/.430/.447` and inverse is `.601 deg/12.99 mm/9.63 cm`. The compact five-prefix forward macros are Original `16.955/.497/.368`, A `17.203/.505/.359`, B `17.507/.519/.356`, C as above, and D `16.756/.489/.379`. Original is the best balanced model; A ties it on aligned frame metrics but loses camera/CD-FVD, B benefits from long prefixes but loses prefix-1 quality, and C/D underperform global LoRA.

The historical step-7000 comparison is not a strict EMA match: its old trainable-delta DCP was reconstructed with `nymeria_world/export_merge_lora.py` and sampled from merged regular weights, while Original/A/B/D use EMA. The exact five-sample 2x2 matrix holds the historical prompt, condition image, `96x9` camera actions, seed, guidance, UniPC-30 solver, model, and inference command fixed. Both 720-tier cells are smoother than both 256-tier cells; shift has only a small fixed-tier effect. Artifacts are under `iter_000007000/sampler_matrix_fdpolicy5_exact_20260724`, including `temporal_diagnostics.json` and five labeled comparison MP4s. Detailed tables, caveats, and interpretation are in `native_phase_training/PHASE1_VISUAL_QUALITY_AUDIT.md`.

The 2026-07-24 high-tier follow-up sampled the Original/A/B/D 100k EMA checkpoints on all 71 canonical prefix-1 forward records at the official 720 tier (raw 640x640), shift 10, UniPC-30, guidance 1, and seed 0. Compared with each model's canonical 256/shift-3 result, paired second-difference improves `19-22%`, flow residual improves `15-23%`, and DreamSim improves for every model, while PSNR/SSIM/LPIPS worsen. CD-FVD changes Original `281.614->308.912`, A `295.561->301.229`, B `304.385->303.792`, and D `319.056->295.649`; D has the best high-tier CD-FVD and flow residual, but weaker aligned fidelity than Original/A. Treat the token-aware-LoRA explanation as an inference, not proof. Artifacts are under `phase1_ema100k_resolution_matrix_5_20260724/full71_720`: 284 raw samples, merged analysis, advanced metrics, 20 per-model and 5 cross-model labeled MP4s. Do not switch the 256-trained Phase-3 bridge directly to this token geometry without bridge validation/retraining.

Verification bar for this directory is higher than a config import. After changes, run pycompile, TOML dryrun, one-step train smoke, final DCP save, official inference load, and official samples for forward/inverse/policy.

### `motion_expert_joint_attention/` Current Main Work

Read order: this file first, then `motion_expert_joint_attention/task_plan.py`, `joint_motion_model.py`, `mot_joint_attention.py`, `mot_joint_layer.py`, and the specific trainer/sampler/dataset file being edited.

This is a self-contained model wrapper around frozen Cosmos-3 Nano reasoner+generator. It adds a trainable motion expert as a true third MoT pathway (`_moe_motion`) with shared joint attention. Historical/default runs use 3-way joint attention. New bridge experiments are flag-gated with `--coupling bridge_local` and keep the native generator path separate from the motion expert, then apply a trainable local gen-motion bridge.

Default/historical packing is:

`[reasoner text | generator image/video/camera | motion shape+frames]`

Reasoner rows use causal attention over reasoner only. Generator and motion rows share the full attention block and attend densely over `{reasoner, generator, motion}`. The reasoner never reads generator or motion rows. This asymmetric attention is load-bearing.

`mot_joint_attention.py` owns only the two-call attention mask:

1. causal call over reasoner rows only;
2. full call with queries from generator+motion rows and keys/values from all rows;
3. scatter outputs back.

`mot_joint_layer.py` owns weight routing by role:

- reasoner rows use plain frozen reasoner weights;
- generator rows use `_moe_gen` frozen/LoRA/full weights;
- motion rows use trainable `_moe_motion` weights.

The mask mode and the weight role are separate concepts. Motion shares generator full-attention behavior but uses its own weights.

Sparse-depth motion is default. `MOTION_LAYER_STRIDE=3` gives motion layers `{2,5,...,35}` for 12 motion blocks. `stride=6` gives 6 blocks. Plain layers own no `_moe_motion` params, run only reasoner+generator, and pass motion rows through unchanged.

Bridge-local mode:

- CLI: `--coupling bridge_local`.
- At former motion layers, reasoner+generator runs without motion K/V, and reasoner+motion runs without generator K/V.
- `modality_bridge.py` then applies a zero-initialized gated bridge between generator and motion rows.
- Directional masking prevents clean condition tokens from reading noisy target tokens. Cross-modal residual updates are role-driven: only noisy target query rows may read source rows in another modality. For `video2motion`, noised motion may attend local clean video. For `motimg2video`, noised video may attend local clean motion. Experimental joint-target tasks permit bidirectional updates because both generator and motion rows are targets. The causal Wan 4x mapping for 97 source frames is latent groups `{0}`, `{1..4}`, `{5..8}`, ..., `{93..96}`; source frame `m` maps to latent frame `(m+3)//4`. Camera action row `i` covers source interval `[i,i+1]`, including direct local camera-motion edges.
- The old 3-way path remains available as `--coupling joint` and is the default for old checkpoints.

Motion weights are freshly initialized, never copied from `_moe_gen`. `_reset_motion_params` uses small normal init; qk norms/layer norms are fresh; `llm2motion` is zero initialized. Motion FFN width defaults to `MOTION_INTERMEDIATE_SIZE=3072`. Shared attention fixes hidden width 4096 and Q/K/V head geometry.

`FrozenCosmos` must build with `action_gen=True`, because camera tasks require `action2llm`, `llm2action`, and `action_modality_embed`.

## Base Seven-Task Contract and Experimental Tasks

Task names come from `task_plan.py` and must be used verbatim.

The single conditioning mechanism is `condition_mask`: `True` means clean/condition/no loss; `False` means noised/supervised. Text with instruction gets 10% CFG drop during training. `inverse_dynamics` and `video2motion` use empty text. Empty text still tokenizes to a structural EOS token, so the reasoner segment remains non-empty.

Tasks:

- `inverse_dynamics`: video -> camera. Clean: all video. Target: all camera action channels `[:9]`. Empty text.
- `forward_dynamics`: camera + text + image -> video. Clean: frame-0 image and all camera. Target: future video latent frames.
- `policy`: text + image -> camera + video. Clean: frame-0 image. Targets: all camera and future video.
- `text2motion`: text -> motion. Clean: text and shape token. Target: valid motion frames.
- `textimg2motion`: text + image -> motion. Active/correct path is `--textimg_condition reasoner`, which sends frame 0 through the Qwen-VL reasoner image path and packs no generator rows for this task. Historical `--textimg_condition generator` packs the image as one clean generator latent frame and is deprecated; keep it only for old checkpoint/run compatibility. Phase-2 bridge-era pretraining should use `text2motion + textimg2motion` with `--textimg_condition reasoner` before freezing the motion expert.
- `motimg2video`: motion + text + image -> video. Clean: motion, text, frame-0 image. Target: future video latent frames.
- `video2motion`: video -> motion. Clean: all video and shape token. Target: valid motion frames. Empty text.

Phase-3-only opt-in tasks have zero default sampling weight and therefore do not alter historical runs:

- `video2camera_motion`: video -> camera + motion. Clean: all video and shape token. Targets: camera action channels `[:9]` and valid motion frames. Empty text. Each target branch has relative loss weight 0.5 so the two-target sample retains a total branch budget of 1.
- `camimg2video_motion`: camera + image -> video + motion. Clean: frame-0 image, all camera actions, and shape token. Targets: future video latent frames and valid motion frames. Empty text. Each target branch has relative loss weight 0.5.

For generator-conditioned tasks, image is not a separate modality; it is video-latent frame 0. A task with image but no video packs exactly one clean latent frame. The exception is `textimg2motion`, which should use `--textimg_condition reasoner` for new runs: raw frame-0 pixels are preprocessed by the Qwen-VL reasoner processor and appear in reasoner rows instead of generator rows. Do not start new TI2M runs with `--textimg_condition generator` unless explicitly reproducing a historical checkpoint.

Reasoner-side image conditioning implementation fact as of 2026-07-11: the local `nvidia/Cosmos3-Nano` snapshot materializes `net.language_model` as `Qwen3VLTextForCausalLM`, so it does not expose `language_model.visual` by default. The snapshot does include a standalone visual tower under `vision_encoder/` (`model.safetensors`, `config.json`) plus a Qwen3VL processor. `motion_expert_joint_attention/cosmos_loader.py` lazily loads the 576.4M-param `Qwen3VLVisionModel`, freezes it, attaches it as `net.language_model.visual`, and then uses NVIDIA's `prepare_multimodal_reasoner_inputs` helper. Normal text-only and generator-latent paths do not load this tower.

The released processor accepts 256x256 images. Its configured minimum area is 65,536 pixels. A direct processor check on node 3 measured: source 224 -> processor grid `[1,16,16]` -> 64 merged visual tokens; source 256 -> the same 64 tokens; source 640 -> grid `[1,40,40]` -> 400 tokens. New Phase-2 TI2M uses `--reasoner_image_size 256` at both dataset and encoder boundaries, reducing the visual sequence by 6.25x versus raw 640 frames. This size is saved in checkpoints and replayed by `sample.py`; old reasoner-image checkpoints without the field retain historical 640 behavior.

Motion tokens include one clean shape token from `neutral_joints` plus valid motion frames. Camera tokens are Cosmos camera pseudo-actions with domain id 2 and loss only on channels `[:9]`; zero-padded channels are ignored. Camera loss is weighted by `ACTION_LOSS_WEIGHT=10`.

## Joint-Attention Data

`nymeria_joint_dataset.py` builds two NymeriaPlus indices:

- `_index`: aligned windows where video, camera, and motion share the same `(uuid, start)`. Normally its aligned length equals output/padding `T`. The corrected Phase-2 T2M+TI2M run is the one guarded exception: output `T=200`, but `_index` uses 97 valid aligned frames for reasoner-image TI2M and pads/masks its remaining 103 batch rows. This exception is rejected if generator/video/camera tasks are active.
- `_t2m_index`: native captioned motion windows for Nymeria `text2motion`, independent of T and padded/masked to T. Used only when mode is `text2motion` and no video/camera are needed.

Alignment invariant:

- Motion-output tasks with no video need no video-frame alignment. `text2motion` should use native motion windows.
- `textimg2motion` uses the same 97-frame NymeriaPlus window as the source video and sends frame 0 to the reasoner. It does not use generator rows or a full 200-frame video window.
- Tasks that pack video with camera or motion must use `_index`.

BONES-SEED is motion-only and is routed only to `text2motion`; it cannot supply image/video/camera tasks. BONES text pools are overview/natural, single-timeline, and multi-timeline. Active pair files use only `content_natural_desc_4` for overview/natural rows; single-timeline and multi-timeline rows are kept as-is. `--bones_frac` controls the fraction of text2motion mass drawn from BONES. A 2026-07-11 ordered-source audit checked every emitted natural row against `seed_metadata_v004.csv`: train has 120,074/120,074 exact desc-4 matches and val has 6,541/6,541; there were zero other overview captions. Train has 585,312 total rows and val has 18,459 total rows. New training fails fast if BONES was requested but could not load; it no longer silently changes the source mixture.

Both Nymeria and BONES motion use shared 283-d UniEgo stats (`uniego283_mean.npy`, `uniego283_std.npy`) in the joint-attention run. This is a documented approximation for BONES. Only motion is z-scored. Camera stays raw. Video stays in VAE latent space.

The dataset humanizes Nymeria captions by rewriting standalone `C` to `a person`.

Training-time caption dropout is controlled by `--cfg_dropout` and defaults to 0.10. It is applied by `nymeria_joint_dataset.py` to all instruction-caption tasks during train only, and is forwarded into the BONES `UniegoPairsDataset` for BONES text2motion rows. It does not apply to validation/test data. For `textimg2motion --textimg_condition reasoner`, the current implementation drops text but keeps the frame-0 image condition; there is no image-condition dropout flag. Sampling CFG for this task uses real text + same image for the conditional branch and empty text + same image for the null branch.

Feature guard skips extreme motion windows with `|z|max > 20`.

Precomputed Wan-VAE latents are preferred for video. `precompute_latents.py` writes `.npz` files under a latent root such as `joint_latents` or `joint_latents_T97`, with latents, camera action, and metadata. For non-default T, latent roots are suffixed like `joint_latents_T97`. Set `WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth`.

## Floor Calibration

NymeriaPlus UniEgo motion has per-sequence vertical bias after GT grounding. BONES convention leaves contacted feet about +2-3 cm above y=0, while Nymeria initially had median foot penetration around -5.4 cm.

`precompute_floor_calibration.py` computes per-sequence deltas:

`delta_seq = d_minc(seq) - c0`

where `d_minc` is the median contacted-foot min-y after GT grounding and `c0 = +2.42 cm` is the BONES target convention. Contact channels `[279:283]` map to joints `[24,25,28,29]`. Heights use Y channel `j*9+7`.

At dataset index-build time:

`entry["off"] = ground_offset_y + delta_seq`

This is the calibrated total vertical shift that `ground_features` subtracts. Original values are recoverable as `off_gt` and `delta`. Future camera-motion absolute alignment must use the calibrated total `off`.

Window drops from `floor_calibration.json` are skipped in both `_index` and `_t2m_index`:

- `wrong_floor`: enough contacts and the contact median is more than 0.20 m from expected floor.
- `residual_penetration`: calibrated min-foot-y below -0.20 m.
- `extreme_y`: a later defense-in-depth scan found post-calibration joint-Y magnitudes above 2.5 m that survived the first two criteria.

The completed 713-sequence calibration file is noisier than the early ~2% estimate: among usable captioned windows, train drops 7,173/128,102 (5.60%: 4,640 wrong-floor, 2,181 residual-penetration, 352 extreme-Y) and test drops 714/13,487 (5.29%: 488/198/28). The active Phase-2 and native Phase-3 logs confirm these exact train drops at dataset construction, leaving 120,929 native T2M windows and 112,937 aligned T97 windows. Every train/test sequence has its own measured delta; no sequence uses the global fallback. The full71 camera-oriented list was not motion-quality filtered and happens to contain five dropped windows (7.04%): four wrong-floor and one residual-penetration. Direct diagnostics show these are gross, not borderline: wrong-floor contact-height errors are 0.58, 2.64, 0.66, and 1.24 m, and the residual case reaches 0.415 m below floor. Use the floor-valid result or select replacement windows for motion evaluation.

Filtering does not prove that every retained SOMA fit is clean. The calibration summary over retained windows has median min-foot-y +0.34 cm and no windows below -20 cm, but 15.22% have at least one min-foot observation below -5 cm; that minimum-over-window statistic can include transient fit/contact artifacts and is not itself a wrong-floor label. `_load_motion` therefore also rejects normalized `|z|max > 20` at runtime before the sample reaches the loss. In the completed 200k Phase-2 run this guard rejected about 247,961 attempts out of approximately 51.2M sampled items (~0.48%), concentrated in 574 unique starts. This adds retry/log overhead but prevents those extreme samples from training the model. Moderate residual motion-fitting noise remains a real data-quality risk and should not be described as fully solved.

Normalization provenance matters for existing checkpoints. The active `motion_expert/stats/uniego283_{mean,std}.npy` files were generated on 2026-06-24 from the older raw-`ground_offset_y` pair set; the per-sequence calibration/drop file was generated on 2026-07-02. `audit_motion_stats.py` recomputes read-only, versioned candidate stats after current calibration/filtering and the runtime `|z|max > 20` guard, and refuses to overwrite the active stats. The 2026-07-15 train audit starts from 120,929 floor-filtered windows, removes 616 runtime-guard failures, and computes candidate stats over 120,313 windows/11,828,053 frames under `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/motion_quality_audit_20260715`: clean joint-Y mean shifts only +3.24 cm, but median joint-Y std changes from 0.628 m to 0.228 m (clean/old ratio 0.364); foot-Y std changes from about 0.585 m to 0.066-0.069 m. All 30 Y channels have clean/old std below 0.8, while the non-Y median ratio is 0.973. A separate guarded 71-sequence test audit gives joint-Y std 0.227 m, so this is not a train/test shift. Phase 2's physical decoded losses mitigate relative pose and velocity errors, but its centroid-relative pose loss removes global translation, so the inflated old Y scale still underweights global/floor-height feature error. Existing Phase-2/Phase-3 checkpoints remain internally consistent with the old stats and must continue using them. A clean-stats experiment needs versioned mean/std paths saved in checkpoint metadata and a Phase-2 retrain from scratch; never silently repoint the global config because that would reinterpret old checkpoint outputs.

`precompute_floor_calibration.py` now includes `extreme_y` directly and deduplicates blacklist output by physical `(uuid,start,end)` while retaining annotation-occurrence statistics. A full low-priority reproduction on 2026-07-15 matched the active c0, all 713 deltas, and all 7,611 effective drop windows exactly; there are 7,887 dropped annotation-row occurrences because the source manifest contains duplicate interval annotations.

If calibration JSON is missing, the dataset warns and proceeds uncalibrated. BONES samples are untouched.

## Objectives and Timestep Convention

In `motion_expert_joint_attention/train.py`, motion objective is selected by `--objective`; the current argparse default is `x0`, while vision and camera are always velocity objective to match Cosmos rectified flow.

Velocity objective: `x_t = (1-t)x0 + t*noise`, target `v = noise - x0`, sample by Euler from `t=1` to `t=0`.

X0 objective: model predicts clean x0. Historical `--motion_schedule legacy` uses unshifted logit-normal sigma plus a linear DDIM-in-sigma ladder. New `--motion_schedule native` uses Cosmos's shifted logit-normal training sigma and shifted 1000-step inference ladder while retaining the empirically better x0 target.

For each target modality, the same per-sample time used to create that modality's noisy state must be passed to `model.forward`. Ordinary one-target tasks can use the compatibility `t_or_sigma` argument. Joint-target tasks must pass independent `motion_t_or_sigma` and `gen_t_or_sigma`; sharing one sampled training time would change both frozen specialists' pretrained marginals.

Critical Cosmos timestep scale: Cosmos heads multiply by `TIMESTEP_SCALE=1e-3` internally. If raw `t in [0,1]` is passed directly, the embedder sees nearly constant `t*1e-3`, loss can drop while samples remain noise. `JointMotionModel.forward` divides `t_or_sigma` by `TIMESTEP_SCALE` exactly once before passing to heads, so the internal embedder sees true flow time. Train and sample must both go through this pre-scale. Do not additionally scale in callers.

Motion losses in joint-attention training: masked 283-d feature loss, decoded-joint loss, and smoothness loss, with documented weights `1/10/50`. Vision loss is flow MSE on latent channels. Camera loss is flow MSE on action channels `[:9]` times 10.

For current Phase 2, base seven-task, and experimental joint-target runs, all motion-supervised tasks train x0 prediction by default. `text2motion`, `textimg2motion`, `video2motion`, `video2camera_motion`, and `camimg2video_motion` train `motion_pred` directly against clean normalized motion `x0` via `flow.add_noise_x0_masked` unless `--objective velocity` is explicitly passed. `motimg2video` packs motion as a clean condition only and has no motion target/loss. The checkpoint records `args["objective"]`, and `sample.py` dispatches motion sampling to `sample_x0` or `sample_velocity` from that saved value.

Native motion schedule implementation as of 2026-07-11:

- `flow.sample_sigma_native_logitnormal` matches current Cosmos action-style training: CPU standard-normal draw, sigmoid, `sigma=1-t_raw`, rational shift `s*sigma/(1+(s-1)*sigma)`, default shift 3, then `x_sigma=(1-sigma)x0+sigma*eps` with clean x0 as target.
- `flow.native_inference_schedule` is bit-identical to `FlowUniPCMultistepScheduler.set_timesteps`: float32 `(N-1)/N` endpoint, linear base ladder, rational shift, integer `sigma*N` model timesteps, final zero sigma.
- Historical `--motion_native_solver euler` is the BONES POC-proven first-order x0/straight-path update on that exact native ladder. It remains available for reproducing old checkpoints and evaluations, but it is schedule-identical rather than solver-identical to generator UniPC.
- New native-schedule motion runs default to `--motion_native_solver unipc`. It converts `x0_hat` to `v_hat=(x_sigma-x0_hat)/sigma` and drives NVIDIA's official `FlowUniPCMultistepScheduler`, making integration solver-identical to native generator sampling while retaining the motion expert's x0 prediction target. The adapter is covered by official-ladder, fixed-noise, perfect-x0, and dispatch tests.
- Training and sampling both feed the quantized/shifted normalized sigma through the same `JointMotionModel.forward` timestep pre-scale. Vision/camera objectives and schedules are unchanged.

The BONES POC uses `shift(sigmoid(z))`, while current Cosmos code computes `shift(1-sigmoid(z))`; these are the same distribution by logistic-normal symmetry. The real joint-attention implementation follows the current framework ordering exactly.

Checkpoint metadata now stores `args`, `task_weights`, and an explicit `data_policy` block. The latter records native T2M versus aligned TI2M temporal policy, `ti2m_frames`, caption dropout, no image dropout, reasoner-image size, BONES overview caption policy, motion schedule/shift/timestep count, and native solver. Resume warns on drift in schedule, mRoPE, coupling, image size, and task-specific TI2M length.

## Trainability and Checkpoints

For joint attention, motion is always fully trained unless `--freeze_motion` is used for Phase 1. Reasoner and generator each choose one of frozen, LoRA, or full.

Important toggles:

- `--gen_lora`: LoRA on generator q/k/v/o projections.
- `--reasoner_lora`: LoRA on reasoner q/k/v/o projections.
- `--gen_full`: train all `_moe_gen` plus gen I/O heads (`vae2llm`, `llm2vae`, `action2llm`, `llm2action`, `action_modality_embed`).
- `--freeze_gen`: keep generator LoRA/full/action-head params frozen even when `--gen_lora` or `--gen_full` is used to instantiate/load them. This is for bridge-only Phase 3 runs that warm-start Phase-1 generator LoRA but train only bridges.
- `--freeze_motion`: build motion expert but exclude it from optimization.
- `--init_gen <ckpt>` and `--init_motion <ckpt>`: warm-start disjoint subsets.
- `--motion_intermediate`, `--motion_layer_stride`, `--tasks`, `--task_weights`, `--bones_frac`, `--objective`.

Generator/reasoner LoRA and `gen_full` params live under `cosmos.net`, so use `model.named_all_parameters()` and `trainable_parameters()` instead of plain `model.parameters()`.

Checkpoints store trainable deltas only (`trainable_state_dict`). Pure DDP manually all-reduces every trainable param, materializing zero grads for unused task-specific params to avoid multi-task NCCL desync. Rank-0-only visualization is wrapped in pre/post barriers so other ranks cannot enter the next backward while rank 0 samples. Non-finite optimizer-step decisions are reduced across ranks so every replica/shard steps or skips together; the non-finite loss guard uses a graph-connected finite zero rather than `NaN*0`. Bridge-only checkpoints with `--freeze_gen --freeze_motion` are not self-contained: their `model` state contains only bridge tensors. `sample.py` / eval must first replay recorded `args["init_gen"]` and `args["init_motion"]`, then overlay the bridge checkpoint delta; otherwise V2M samples can be exactly zero because the frozen motion expert is missing.

Known discrepancy: `config.TASK_WEIGHTS` and `task_plan.TASK_WEIGHTS` differ. `train.py` uses `config.TASK_WEIGHTS`; dataset normalizes weights when sampling. `task_plan` weights are only for its self-check.

Known stale flag: old docs/scripts may mention `--data_mix`; current `train.py` uses `--tasks`, `--task_weights`, and `--bones_frac`.

Base-weight key remapping was fixed on 2026-07-02. Runs before that may have random reasoner attention and random action heads despite docs claiming pretrained reuse. The fixed loader remaps reasoner attention and action I/O heads correctly.

Camera mRoPE packing in `gen_heads.build_gen_segment` was fixed on 2026-07-02 to pack camera tokens at the vision segment start temporal offset, matching native Cosmos, instead of sequentially after video.

Motion mRoPE has an explicit `--motion_mrope {legacy,cosmos3d}` flag in train/sample/eval model loading:

- `legacy`: old behavior and default for backward compatibility. Motion rows use sequential positions after the generator segment, expanded across all three mRoPE axes.
- `cosmos3d`: official-style 3D mRoPE for motion. Motion frame tokens are a `T x 1 x 1` temporal grid with temporal compression factor 1, so motion frame `k` shares the same temporal coordinate family as video/camera frame `k` in video+motion tasks. The shape token is non-temporal conditioning and is placed on the last text-time plane with distinct spatial axes to avoid collisions with text, video patches, and motion frame 0.

## Planned Joint-Attention Curriculum

The same model/mask supports a 3-phase curriculum:

1. Phase 1: camera tasks only, `--tasks inverse_dynamics forward_dynamics policy --gen_lora --freeze_motion`. Motion expert built but not stepped; no motion tokens are packed.
2. Phase 2: text2motion, historically `--tasks text2motion`, optional BONES via `--bones_frac`. Motion pathway trains; generator frozen. Bridge-era corrected Phase 2 should train `text2motion + textimg2motion` with `--textimg_condition reasoner` so the motion expert has seen reasoner visual tokens before it is frozen.
3. Phase 3 has two distinct experiment families. Historical joint runs train all seven tasks with 3-way attention. The active bridge experiment trains only `video2motion + motimg2video`, recreates and freezes the Phase-1 generator LoRA and Phase-2 motion expert, and optimizes only `LocalModalityBridge` modules.

The new Phase-2 3D-mRoPE variant run is `ja_t2m_x0_T200_mrope3d`, launched by `motion_expert_joint_attention/sbatch_t2m_both_mrope3d.sh`. It is the same recipe as `ja_t2m_x0_T200` except `--motion_mrope cosmos3d`; it trains `--tasks text2motion --bones_frac 0.5 --T 200 --objective x0 --batch_size 32 --steps 200000` on 8 GPUs. Slurm job `2765` was submitted on 2026-07-06 and started on `a3ultravis-a3ultranodeset-1`. Its run dir is `/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_x0_T200_mrope3d`.

The native-schedule corrected Phase-2 launcher is `motion_expert_joint_attention/sbatch_t2m_ti2m_native_mrope3d.sh`. Its run is `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d` with:

- T2M + TI2M weights 0.75/0.25 and `bones_frac=0.5`, giving effective mass 37.5% Nymeria T2M, 37.5% BONES T2M, and 25% Nymeria TI2M;
- output/padding `T=200` for full/native T2M capacity, but exactly 97 aligned valid Nymeria frames for TI2M;
- reasoner-side TI2M image at 256x256, no image dropout, text CFG dropout 0.10;
- x0 target, native shifted logit-normal schedule, shift 3, 1000 timesteps, and the historical POC-proven native-ladder Euler sampler;
- `motion_mrope=cosmos3d`, joint coupling, stride 3 / 12 motion blocks, batch 32 per rank, 8 GPUs, 200k steps;
- 96-hour Slurm wall limit. The previous T2M+TI2M run measured about 1.26 s/step, so 200k steps alone need about 70 hours; a 48-hour request predictably stops near 137k before visualization/checkpoint overhead;
- in-training visualization every 2k steps with two T2M and two TI2M samples. TI2M MP4s show conditioning image | GT | generated; rendering uses stride 2 at 10 FPS while raw arrays keep all frames.

Pre-submit verification on node 3 passed: exact native schedule/UniPC ladder checks; real 97-valid-frame TI2M padded to 200; full 993.73M-trainable-parameter T2M and TI2M forward/backward with frozen-gradient assertions; checkpoint optimizer save/reload with zero skipped tensors; post-load 200-frame T2M and 97-frame TI2M sampling; condition/GT/generated MP4 creation; and a two-GPU run with zero parameter divergence.

Production submission: Slurm job `2844` (`t2mtinat`) was submitted on 2026-07-11 with no dependency. It requested one exclusive `a3ultra` node, 8 GPUs, 64 CPUs, and 96 hours. Slurm output is `/home/jungbin_cho/cosmos_motion_ft/slurm-t2mtinat-2844.out`; run outputs go to `/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d`. It completed successfully with exit code 0 on 2026-07-14 after 2d 2h 28m. The final named checkpoint is `ckpt_step200000.pt` (about 6.19 GB); bridge launchers name this file explicitly so they cannot silently consume a moving `latest.pt`.

The contact-aware native Phase-2 variant launcher is `motion_expert_joint_attention/sbatch_t2m_ti2m_native_mrope3d_contact.sh`. It keeps job `2844`'s architecture, native x0 training schedule, Cosmos3D motion mRoPE, T2M/TI2M mixture, aligned 97-frame TI2M policy, 256x256 reasoner image, batch size, LR, cosine schedule, and 200k steps. Its training-objective change is the proven BONES contact objective: `1*L_feature + 1*L_joint + 5*L_smooth + 0.05*L_contact_BCE + 1*L_contact_horizontal_foot_velocity + 10*L_contact_foot_height`, with contact-logit scale 2 and motion FPS 20. It also selects official NVIDIA UniPC-35 for in-training sampling/visualization and records that solver in checkpoints; this changes sampling, not the training noising or x0 loss. Contact BCE and physical terms use ground-truth contact masks; the four contacts map to SOMA-30 joints `[24,25,28,29]`, and foot height uses raw local-pose Y channels to avoid cumulative-decoder gradient spikes. The new terms are opt-in (`train.py` defaults them to zero), logged to TensorBoard, checkpoint metadata, and `data_policy`.

This variant deliberately uses the original pre-floor-filtering NymeriaPlus shared stats for both Nymeria and BONES: `/home/jungbin_cho/cosmos_motion_ft/motion_expert/stats/uniego283_mean.npy` and `uniego283_std.npy`. The launcher verifies their SHA-256 hashes (`bd1d6bdc...b400d3` and `ee069e3a...65f28c`) before starting. It does not use the 2026-07-15 cleaned candidate statistics. In-training visualization remains enabled every 2k steps and includes T2M plus reasoner-image TI2M; the launcher requires visualization setup/rendering to succeed.

UniPC conversion verification on 2026-07-15 passed the native training-sigma, official scheduler-ladder, perfect-x0 endpoint, fixed-initial-noise, and default-dispatch contracts on node 2. A real one-update smoke at `/weka/jungbin/cosmos_motion_ft_runs/_smoke_phase2_contact_native_unipc35_115_20260715` then trained with all requested losses, saved a 149-tensor checkpoint plus optimizer state and TensorBoard event, and completed one T2M-200 and one reasoner-image TI2M-97 visualization with official UniPC-35. Both generated arrays were finite, both MP4s passed `ffprobe`, the TI2M condition image was 256x256, and checkpoint/config/manifest metadata all recorded `motion_native_solver=unipc`. Pending Euler-configured job `2995` was canceled before start. Replacement production job `2997` was submitted with the UniPC-35 launcher; at submission it was pending for Slurm priority. Its run directory is `/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35`, and its log is `/home/jungbin_cho/cosmos_motion_ft/slurm-t2mticnt-2997.out`.

Job `2997` later failed on 2026-07-17 at logged step `102280`, not from OOM or a model/loss failure. Losses remained finite, peak allocation stayed at `67.1 GB/GPU`, and the valid periodic checkpoint/latest pair was completely written at step 100k. A DataLoader worker then reported `could not unlink the shared memory file /torch_*: No such file or directory`; an eight-rank 16M-element gradient all-reduce stopped completing and hit the 600-second NCCL watchdog. Evaluator smoke processes had also been launched by SSH on the same Slurm-exclusive node/GPU during exactly this interval. Slurm exclusivity does not prevent SSH processes from bypassing Slurm, so never run evaluation or smoke workloads on a node occupied by this eight-GPU training job even when `nvidia-smi` appears to show spare memory.

Recovery hardening on 2026-07-18 keeps the scientific recipe unchanged but reduces DataLoader IPC pressure from `8 workers x prefetch 2` to `4 workers x prefetch 1` per rank, adds a 300-second DataLoader timeout, saves every 5k steps, and gives torchrun two bounded worker-group restarts. A timeout now becomes a process failure that torchrun can restart instead of waiting indefinitely for NCCL; `--resume auto` reloads the newest complete weights, optimizer, and step after either a Slurm resubmission or an in-job worker restart. Parser, four-worker DataLoader, shell-syntax, compile, and diff checks passed. Resume job `3000` was submitted with the same run name and 200k target; it will load `latest.pt` at step 100k and continue from step 100001. At submission it was pending for priority because all four nodes were allocated. Its Slurm log is `/home/jungbin_cho/cosmos_motion_ft/slurm-t2mticnt-3000.out`.

Phase-2 motion training does not maintain a model-weight EMA. `motion_expert_joint_attention/train.py` optimizes the regular motion-expert parameters and saves those directly; its `_loss_ema` is only a scalar used by the explosive-batch guard. The BONES native-schedule POC likewise has no model EMA. Adding motion-weight EMA is a separate future experiment, not part of job `2844`.

The final Phase-2 checkpoint has a dedicated shape-aware C45 evaluation path in `prepare_shape_tmr_eval.py`, `precompute_nymeria_tmr_text.py`, `eval_phase2_shape_tmr.py`, and `sbatch_phase2_shape_tmr_eval.sh`. It evaluates T2M on all six official BONES-SEED text-to-motion suites and on every floor-filtered, direct-guard-valid Nymeria test annotation window from all 71 held-out sequences. Different captions for the same physical Nymeria start remain distinct evaluation cases. TI2M is evaluated only on Nymeria's aligned 97-frame windows. The active floor calibration removes known wrong-floor/residual/extreme-Y rows first; the old-stat training guard then rejects `|z|max > 20` directly, without dataset retry/substitution.

Sampling uses the checkpoint's native x0 schedule (shift 3, 1000 training timesteps) with official Cosmos UniPC for 35 steps and deterministic per-case initial noise. T2M uses text CFG 2 and a same-text/same-noise farthest-natural-skeleton intervention. TI2M is sampled twice with identical image and noise: `ti2m_cfg2` keeps the image in both conditional/null branches and drops only text in the null branch, while `ti2m_no_cfg` uses guidance 1 and performs only the conditional image+text branch. Image dropout is not part of either TI2M evaluation. TI2M does not use the skeleton counterfactual because keeping the original actor image while replacing body proportions would create contradictory conditions.

The C45 bundle is `/weka/jungbin/shape_aware_motion_eval_c45_20260715`; evaluator checkpoint `artifacts/evaluator/c45_step_00005000.pt` and `artifacts/evaluator/stats/motion` are paired. Those C45 stats are used only inside the frozen evaluator. Phase-2 outputs are first unnormalized with `motion_expert_joint_attention/stats/uniego283_{mean,std}.npy`, decoded to SOMA-30 joints at 20 FPS, then resampled to C45's 30 FPS representation. The benchmark LLM2Vec cache is retained byte-for-byte and extended with every valid BONES/Nymeria evaluation caption absent from the base cache, only after a local encoder parity check against known cached rows. Reported outputs include protocol/plain retrieval, TMR FID, foot/contact metrics, bone-shape metrics, counterfactual response for T2M, per-cohort JSON, a case-weighted six-suite BONES summary, and representative MP4s. The production Slurm script gates the full 8-GPU sweep on a one-GPU end-to-end smoke covering all three cohort types, both TI2M CFG modes, C45 metrics, counterfactual sampling, checkpoint loading, and rendering.

Contact-aware Phase-2 step-100k BONES evaluation completed on 2026-07-17 at `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35/eval_bones_shape_tmr_c45_unipc35_step100000`. It uses the same six manifests, cached benchmark text, C45 checkpoint/stats, native shift-3 x0 schedule, UniPC-35, CFG 2, and deterministic per-case noise as the earlier final-checkpoint evaluation. All six suites and 9,124 cases completed. The case-weighted suite means are: bone MAE `0.47281 cm`, contact consistency `0.85979`, predicted-contact skate `5.0693 cm/s`, height skate `19.2935 cm/s`, maximum contact velocity `26.2800 cm/s`, skate ratio `0.10715`, FID gen-GT `0.03350`, plain R@3 `58.0776`, protocol R@1/2/3/5/10 `46.779/61.584/69.028/76.304/83.659`, shape centered correlation `0.93496`, shape response slope `0.83560`, shape variance ratio `0.89367`, counterfactual delta cosine `0.97583`, counterfactual response slope `0.85504`, and counterfactual target advantage `2.3752 cm`. All 42 JSON and 42 NPZ rank shards, 12 finite visualization arrays, and six MP4s were validated.

Against the original non-contact Phase-2 final checkpoint at step 200k, the contact-aware 100k checkpoint reduces predicted-contact skate by `55.66%`, height skate by `27.74%`, maximum contact velocity by `51.62%`, and skate ratio by `49.38%`; the skating reduction occurs in every suite (`53.7-56.7%`). It is worse on FID (`+52.90%`), protocol R@3 (`-5.48` points), plain R@3 (`-6.46` points), bone MAE (`+6.26%`), and contact consistency (`-0.0144`). Shape tracking is mostly stable: centered correlation changes `-0.005`, response slope `+0.003`, and variance ratio `+0.008`. This comparison is useful but not controlled because it changes both the loss and training duration (`100k` contact-aware versus `200k` non-contact); do not attribute all regressions or gains solely to contact losses without a same-step comparison.

## Bridge Hypotheses and Representation Risks

These notes are experiment hypotheses, not established results.

Modality-bridge hypothesis: full generator-motion joint self-attention is expensive and may perturb the pretrained generator distribution by adding motion tokens directly into generator attention. For frame-aligned physical translation tasks (`video2motion`, `motimg2video`), a trainable gen-motion bridge with temporal locality may be a better inductive bias. The active Phase-3 bridge experiment freezes the Phase-1 generator/video/camera model and the Phase-2 motion expert, then trains only `modality_bridge.py` so generator and motion hidden states can exchange local information without changing either expert's weights. Directional masking is task-specific: for `video2motion`, noised motion reads local clean video while video cannot read noisy motion; for `motimg2video`, noised video reads local clean motion/shape while motion cannot read noisy video. Same-modality bridge attention remains enabled. Twelve bridges are inserted after the native reasoner+generator update and the reasoner+motion update at sparse motion layers 2,5,...,35. Their generator/motion residual gates are zero-initialized, making the bridge an exact no-op before training. This is still a hypothesis; successful optimization and useful alignment are not established by the smoke tests.

The head-camera Phase-3 variant adds an explicit, flag-gated relative-geometry contract without changing either frozen specialist or adding a trainable adapter. `head_camera_alignment.py` decodes UniEgo's SOMA `Head` joint (index 6) and maps its frame-to-frame transform into the upright-RGB camera-action frame using a robust global train-split calibration. For `T_world_camera = T_world_head X`, it computes `R_camera = R_X^T R_head R_X` and `t_camera = R_X^T(t_head + (R_head-I)r_X)`. Absolute translations are intentionally unused. `estimate_head_camera_calibration.py` fitted `head_camera_calibration_train.json` from all 642 train sequences: 61,632 relative-action samples, train median/mean translation error 1.54/3.09 mm per frame, and train median/mean rotation error 1.44/1.58 degrees. On all 71 test sequences, 11,892 T97 windows that pass the same floor and `|z|` guards have median/mean translation error 2.61/3.57 mm and rotation error 1.28/1.38 degrees; the lever arm improves held-out median translation from 4.80 mm. This validates the relative approximation but does not establish reliable absolute head-camera alignment.

With `--head_camera_alignment`, data/model flow is deliberately asymmetric and leakage-guarded. For M2V, the model derives a clean 96x9 camera-action condition only from its clean motion input and packs it through the frozen Phase-1 camera pathway; synchronized GT camera action is used only for a no-grad calibration diagnostic. For V2M, the model still receives only clean video. Synchronized camera action is carried in the separate `camera_alignment_action` batch field and supervises relative camera action decoded from predicted motion x0 with robust translation/rotation losses; it is never packed as a V2M condition. `task_plan.resolve_sample` rejects derived camera conditioning for every task except M2V. The intended production weights are `w_head_camera_trans=0.05`, `w_head_camera_rot=0.05`, with 2 cm and 5 degree normalization scales. This remains an experiment: the geometry is verified, but improvement in V2M/M2V metrics must be measured rather than assumed.

`eval_all.py` automatically writes `motion_recon/video2motion/head_camera_alignment_metrics.json` for head-camera checkpoints. It reports mean/median/p90 V2M translation and rotation action errors plus the GT-motion calibration floor on the identical windows; floor-valid output is split when needed. `--eval_head_camera_alignment` enables the exact same train-calibrated evaluation-only metric for a historical baseline checkpoint without changing that model's inputs, weights, or sampling. This is required for a paired baseline/headcam comparison. The metric measures relative action consistency, not absolute camera pose. M2V keeps the existing PSNR/SSIM/LPIPS video metrics because camera motion cannot be read directly from generated pixels without a separate inverse-dynamics estimator. `merge_phase3_clean71.py` also merges the 66 floor-valid rows plus five deterministic replacements for this metric.

Phase-1 camera-coordinate reminder: all native Phase-1 action targets are from the preprocessed **upright RGB optical-camera** trajectory, not the raw Aria device trajectory. `nymeria_camera_rgb_dataset.py` rewrites the manifest's `/camera/` path to `/camera_rgb/`, loads `cam_world_pos_upright` and `cam_world_rot_upright`, and computes the 9D metric relative action with Cosmos `pose_abs_to_rel(..., pose_convention="backward_framewise")`. Preprocessing first forms `T_world_rgb = T_world_device @ T_device_rgb` from the recording's VRS device-to-RGB extrinsic and then applies the optical-axis rotation matching the upright video. Inverse dynamics, forward dynamics, and policy therefore all use relative upright-RGB-camera translation `(3)` plus rotation-6D `(6)`. The head mapping and all metrics below target this same frame.

The 2026-07-20 **test-actor oracle** is a separate, explicitly leaky diagnostic requested to answer how well GT motion can reconstruct camera after calibrating each held-out actor from GT. `estimate_test_actor_head_camera_calibration.py` fits one fixed `R_X,r_X` for each of the 17 actors represented by the motion-clean71 benchmark. It uses the exact 71 held-out windows (66 original floor-valid windows plus five deterministic replacements), synchronized GT UniEgo motion, and synchronized GT upright-RGB camera actions. Each six-parameter actor transform starts from the train-global calibration and minimizes the same normalized Phase-3 geometry objective: SmoothL1 translation at a 2 cm scale plus rotation-matrix chord at a 5 degree scale, with a bounded rotation correction and lever arm. The fit and reported clean71 windows are identical. Consequently this is an **in-sample oracle floor**, not a leakage-free evaluation, deployable calibration, training target, or model-selection metric. The loader requires `kind=oracle_test_actor_head_camera_calibration`, `split=test`, and explicit `uses_test_gt_motion`, `uses_test_gt_camera`, and `diagnostic_only` flags before accepting the artifact.

The metric components are:

- `translation_m` / `rotation_deg`: V2M-predicted motion mapped with the production train-global rigid calibration, compared with synchronized GT upright-RGB camera action.
- `gt_calibration_translation_m` / `gt_calibration_rotation_deg`: GT motion mapped with that same train-global calibration. This is the leakage-free calibration/representation/data floor paired to the model rows.
- `oracle_actor_translation_m` / `oracle_actor_rotation_deg`: V2M-predicted motion mapped with the actor's test-GT-fitted transform. This diagnoses whether an oracle actor transform transfers from GT motion to model output; it is not a fair model score.
- `gt_oracle_actor_translation_m` / `gt_oracle_actor_rotation_deg`: GT motion mapped with its own test-GT-fitted actor transform. This is the requested in-sample actor-calibrated floor.

For every 97-frame window, translation is the mean Euclidean error over 96 relative translations in metres, and rotation is the mean SO(3) geodesic angle over 96 relative rotations in degrees. JSON aggregation then reports mean/median/p90 across windows. `per_actor_aggregate` first groups those window rows by actor; the overall aggregate remains window-weighted, not actor-balanced. Absolute pose, absolute origin, and accumulated trajectory error are not part of these metrics.

Vanilla Phase-3 step-200k motion-clean71 results (`head_camera_alignment_metrics_motion_clean71.json`):

| Motion source | Rigid calibration | Translation mean | Rotation mean |
|---|---|---:|---:|
| V2M prediction | train-global | 6.394 mm | 1.281 deg |
| GT motion | train-global | 4.445 mm | 1.235 deg |
| V2M prediction | test-actor oracle | 6.709 mm | 1.259 deg |
| GT motion | test-actor oracle | 4.056 mm | 1.205 deg |

Per-actor GT floors are:

| Actor | Windows | Train-global GT | Test-actor-oracle GT |
|---|---:|---:|---:|
| S01 | 7 | 5.490 mm / 1.040 deg | 5.164 mm / 1.017 deg |
| S02 | 9 | 3.558 mm / 1.180 deg | 3.789 mm / 1.147 deg |
| S03 | 3 | 2.624 mm / 0.886 deg | 2.087 mm / 0.882 deg |
| S04 | 1 | 2.795 mm / 0.875 deg | 1.423 mm / 0.864 deg |
| S05 | 5 | 5.319 mm / 1.193 deg | 4.865 mm / 1.179 deg |
| S06 | 3 | 7.923 mm / 1.715 deg | 6.719 mm / 1.723 deg |
| S07 | 6 | 5.297 mm / 1.168 deg | 5.191 mm / 1.175 deg |
| S08 | 4 | 3.638 mm / 1.289 deg | 3.379 mm / 1.250 deg |
| S09 | 1 | 6.123 mm / 1.261 deg | 2.815 mm / 1.236 deg |
| S10 | 5 | 5.374 mm / 1.417 deg | 4.994 mm / 1.430 deg |
| S11 | 1 | 6.442 mm / 1.373 deg | 2.252 mm / 1.335 deg |
| S12 | 7 | 3.698 mm / 1.175 deg | 3.606 mm / 1.141 deg |
| S13 | 2 | 3.476 mm / 0.937 deg | 2.335 mm / 0.937 deg |
| S14 | 3 | 4.752 mm / 1.461 deg | 3.589 mm / 1.232 deg |
| S16 | 7 | 4.095 mm / 1.296 deg | 3.942 mm / 1.228 deg |
| S17 | 2 | 2.414 mm / 0.658 deg | 1.636 mm / 0.672 deg |
| S19 | 5 | 3.548 mm / 1.686 deg | 3.857 mm / 1.655 deg |

The oracle reduces the aggregate GT floor by 8.76% in translation and 2.46% in rotation, so actor-specific rigid geometry explains a small but measurable part of the mismatch. It does not explain the main V2M translation problem: applying the GT-fitted actor transforms to predicted motion worsens translation by 4.93% (`6.394 -> 6.709 mm`) while improving rotation by only 1.70%. Some actors also worsen even on GT, because the optimizer minimizes the combined normalized robust objective rather than either reported mean independently and the SOMA Head-to-worn-camera relation is not perfectly rigid. Single-window actors S04/S09/S11 are especially in-sample and must not be used to claim generalization. Do not subtract the GT floor as though errors were independent; use paired rows and report the calibration contract.

The same step-200k motion-clean71 V2M output confirms that its dominant decoded error is global/root trajectory: MPJPE is `0.187697 m`, root error is `0.163340 m`, root-centered MPJPE is `0.078462 m` (custom diagnostic, no rotation/scale fit), and PA-MPJPE is `0.089861 m`. Per-window MPJPE and root error correlate at `0.9738`. Mean root error grows from `0.0776 m` at frame 0 to `0.1033/0.1447/0.2170/0.3112 m` at frames 24/48/72/96 and is mostly horizontal. UniEgo hybrid decodes further localize the problem: predicted local features with GT global delta give `0.098156 m` MPJPE; GT local features with the predicted full global delta give `0.141005 m`; keeping all GT features except predicted delta translation gives `0.136229 m`; replacing only predicted delta rotation with GT barely changes full prediction (`0.185798 m`). Therefore global delta translation is a major contributor, but local body error also remains; this is not a pure rigid head-calibration problem.

For a direct camera-prior comparison, recomputation on the identical floor-valid66 windows used raw relative and accumulated camera actions rather than mixing the Phase-1 scale-normalized headline with raw Phase-3 errors. Phase-1 inverse dynamics gives `3.317 mm` raw per-step translation, `0.209 deg` rotation, raw trajectory mean/RMSE/endpoint `0.0728/0.0857/0.1492 m`, and Sim(3)-aligned ATE `0.02335 m`. Phase-3 V2M motion mapped through the train-global head calibration gives `6.146 mm`, `1.261 deg`, `0.1418/0.1727/0.3100 m`, and `0.03740 m`. GT motion through that global calibration already gives `4.519 mm`, `1.207 deg`, `0.1322/0.1618/0.2839 m`, but only `0.00328 m` Sim(3) ATE; the large raw-versus-Sim(3) gap reflects calibration/frame bias rather than trajectory-shape failure alone. The Phase-1 camera model is clearly the stronger metric-camera prior, but vanilla `video2motion` does not run or expose its frozen camera-action prediction path, so that prior cannot automatically correct UniEgo root motion. A head-camera auxiliary loss or an explicit joint `video -> camera + motion` objective is required to couple them.

The 2026-07-20 **relative-mapping** GT visualization makes that raw-versus-Sim(3) gap concrete. `visualize_gt_head_camera_alignment.py` uses the exact motion-clean71 windows and no model output, but it is not a direct overlay of the two stored world trajectories. For each window it decodes the floor-calibrated/canonicalized GT SOMA skeleton, forms the global-calibration camera pose `H_t X`, and applies one constant left world transform that makes the stored Phase-1 upright-RGB camera pose exactly equal at frame 0. All later measured camera poses remain unchanged. Its divergence therefore measures the production global Head-frame mapping after frame-0 pose alignment. Static plots and 97-frame/20-FPS MP4s show the skeleton, both trajectories, and both upright optical-camera `+Z` directions. Outputs are under `eval_full71_step200000_unipc30/gt_camera_vs_gt_motion_viz`; `summary.json` records all 71 rows and transform/action reproduction checks, `all71_summary.png` is the aggregate diagnostic, and `cases/` contains seven ranked PNGs plus worst-endpoint, worst-local-error, and median MP4s. All three MP4s are valid H.264 `1320x880`, 97-frame clips.

This visualization shows that the large accumulated GT floor is a coherent frame-direction effect amplified by locomotion, not 96 independent millimetre errors or a motion-scale failure. Across clean71, mean/median per-step translation error is `4.445/3.639 mm`, while mean/median frame-96 position gap is `0.276/0.176 m` (p90 `0.686 m`, max `1.687 m`). The observed sequence-window SOMA-Head-to-upright-camera rotation differs from the train-global `R_X` by mean/median `14.73/12.79 deg`; measured net displacement is mean/median `1.321/1.141 m`. The simple rotation-chord term `2*d*sin(theta/2)` correlates `0.875` with endpoint gap, compared with `0.820` for per-step translation, `0.654` for displacement alone, and `0.408` for frame rotation alone. The worst window moves `3.40 m`, has `26.2 deg` median frame error, predicts a `1.542 m` chord gap, and observes `1.687 m`; the second moves `2.67 m`, has `32.5 deg`, predicts `1.494 m`, and observes `1.519 m`. Yet clean71 Sim(3)-aligned trajectory RMSE is only `3.24 mm` mean (`2.92 mm` median), with fitted scale `1.0028` mean/`0.9998` median. Thus GT motion usually preserves trajectory shape and metric scale very closely, but a global head-camera frame is not accurate enough per sequence to express translation in Phase-1's local camera coordinates. A per-sequence rigid alignment can visually remove most of the raw path error, but it is unavailable at inference and must not be treated as a leakage-free solution.

While implementing the visualization, a separate absolute-orientation diagnostic bug was found and fixed in `calibration_sample_from_arrays`: directly decoding a nonzero-start UniEgo slice incorrectly treated its first relative `canon_delta` as a sequence-absolute transform. The utility now decodes the sequence prefix before selecting absolute head orientations, and `_verify_head_camera_alignment.py` has a synthetic nonzero-start regression. Relative actions, model training, existing head-camera losses, and the visualization trajectories were never affected because they canonicalize the window and use frames 1+ deltas. The saved train-global calibration was also effectively unaffected: 637/642 fit windows start at zero, and a full corrected recomputation changes its rotation by only `0.0117 deg` and lever arm by `0.052 mm`; relative fit metrics are unchanged. Keep the active calibration JSON unchanged for checkpoint compatibility. The test-actor oracle's final transforms were optimized from relative actions, so only its auxiliary absolute-frame-deviation diagnostic was exposed to this bug.

The 2026-07-21 source-level audit corrected an important interpretation. `audit_nymeria_camera_motion.py` independently decodes only the UniEgo Head and compares it directly with `camera_rgb` after the fixed Aria-Z-up to Kimodo-Y-up basis change, with no fitted calibration or frame-0 alignment. Across all 728 sequences that have camera sidecars, proportional motion/UniEgo/raw camera/upright camera timestamps match exactly, every sequence's step-lag search selects zero frames, median MPS sampling error is about `0.246 ms`, and the per-sequence median camera-origin-to-Head-joint distance is tightly centered at `0.139 m`. The direct position streams therefore normally do share a metric world frame. In the William clean71 worst mapping case, the measured camera, decoded Head trajectory, and `Head + global lever` trajectory overlap directly; the apparent `1.687 m` endpoint divergence appears only after the noisy global Head-to-camera orientation is forced to equal the measured camera pose at frame 0. Corrected plots are under `/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/plots`.

The corrected direct visualization set in that `plots/` directory includes `heldout_motion_clean71_direct_head_camera.png` for all 71 selected windows and two H.264 `1320x880`/20-FPS animations: `direct_head_camera_william_start0.mp4` (97 frames) and `direct_head_camera_kevin_source_jump_1450_1600.mp4` (150 frames). Every view uses the stored upright RGB camera pose and independently decoded UniEgo SOMA Head pose after only the fixed world-basis change; there is no fitted transform and no frame-0 pose alignment. Blue/orange paths are camera origin/Head joint, arrows are each frame's own `+Z`, and the lower panels show direct separation and translation-step continuity. William stays at `0.133 m` median/`0.140 m` maximum separation with maximum camera/Head steps `0.070/0.060 m`; Kevin reaches `1.003 m` separation and `1.644/0.956 m` steps. `direct_head_camera_manifest.json` records the contract, all 71 per-window metrics, and animation paths.

The same audit independently reproduces the camera preprocessor, ruling out a hidden device-to-RGB/action transform bug: over all 728 valid sequences, worst raw-to-upright-RGB position/rotation differences are `1.53e-5 m`/`6.34e-8`, and worst stored relative-action translation/rotation reconstruction errors are `3.23e-5 m`/`0.03 deg` (float storage precision). Two tooling issues were fixed in `/home/jungbin_cho/nymeria_kimodo_pipeline/camera` without rewriting active data. `preprocess_camera_rgb.py` had the diagnostic `K_rgb_upright` principal-point mapping for the opposite image-rotation direction; the checked old sidecar differs from the correct Pillow-clockwise K by up to `16.3 px`. Intrinsics are never loaded by current training/evaluation and camera poses/actions were already correct. New sidecars carry `preprocess_version=2`, corrected K, continuity arrays, and version-aware freshness checks; existing sidecars remain legacy until an explicit rebuild. `extract_camera_all.py --motion-root` previously changed discovery but workers still used the default hard-coded root; root/output paths now propagate into every worker. Alternate-root, current-sidecar, VRS/K, and temporary full `camera_rgb` preprocessing smokes pass on node 2.

The broad non-rigidity is rotational and is already present in the source SMPL fit. Over 728 sequences, the per-sequence fixed SOMA-Head-to-camera relation has mean/median full-rotation residual `15.06/14.57 deg` and horizontal-heading residual mean/median `13.05/12.27 deg`. Even within non-overlapping T97 windows the sequence-level mean residual is `8.80 deg` full rotation and `7.81 deg` heading. On all 71 test sequences, the original SMPL Head has `14.53 deg` mean residual to camera, the converted SOMA Head has `14.42 deg`, and SMPL-to-SOMA Head differs by only `0.923 deg` mean. Thus the converter is not creating the main orientation problem; the fitted source Head orientation is not a rigid proxy for the worn RGB camera. Across 642 train sequences, sequence relations differ from a train-global relation by `10.58 deg` mean; grouping by named actor reduces sequence-to-actor-mean disagreement only to `6.57 deg`, so actor calibration cannot remove the remaining session/time/pose variation. The fitted translation lever is much more stable: train mean Head-frame offset is approximately `[-0.0133, 0.0629, 0.1236] m`, with a typical sequence-mean residual of only `4-5 mm` after excluding source discontinuities.

The 2026-08-31 `camhead_v1` absolute-origin follow-up keeps its camera-aligned Head rotation and unchanged Head position, then fits only a constant lever on the exact qfilterv1 clean population. The selected train-only robust lever is `[-0.02214354, 0.05470125, 0.12229882] m`, 1.358 cm from the historical relative-action lever. On all 71 held-out sequences / 1,156,017 clean union frames, corrected absolute optical-center error improves `3.7000 -> 3.4082 cm` (7.89%) while relative translation changes `2.4304 -> 2.4531 mm` (+0.93%) and rotation stays `0.002043 deg`. The same-population original representation is `1.4437 cm / 3.5762 mm / 20.1067 deg`; there is no single row that wins all three quantities. Train-actor calibration reaches 3.3072 cm, while an explicitly leaky per-test-sequence oracle reaches only 2.5710 cm, proving that one constant corrected-Head-frame lever cannot recover most of the absolute-position residual. Keep `head_camera_calibration_train.json` unchanged for existing relative-action checkpoints; the new calibration is experimental/opt-in only. Full leakage contract, 713 per-sequence rows, hashes, aggregate plots, and a 12-case trajectory overlay are under `/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/absolute_lever_refit` and documented in `motion_expert_joint_attention/CAMERA_HEAD_RECANONICALIZATION.md`. Because this is calibration-only, it cannot change motion, contacts, skating, floating, or feet.

The 2026-08-31 standard-SOMA visualization under the sibling `qualitative_soma_mesh`
directory renders the same twelve diverse/diagnostic windows as synchronized original
and `camhead_v1` 1920x1080 H.264 MP4s, plus peak-impact posters and aggregate plots.
It uses Kimodo's restored 18,056-vertex/36,108-face standard SOMA-77 skin, compacted
for rendering to 6,706 vertices/13,576 faces; it is a common visualization body, not
subject-specific SOMA-X identity geometry. Across 1,217 displayed source frames,
frame-weighted camera SO(3) error is `32.726843 -> 0.002231 deg`. The common skin makes
the Head correction visibly large (case-mean affected-surface p95 `73.86 mm`), while
the selected windows retain maximum decoded position/non-Head-rotation changes of only
`0.003328 mm`/`2.70e-6 deg`; maximum foot-height and absolute contact-skate changes are
`5.44e-10 mm` and `1.71e-5 cm/s`. Surface displacement is a derived LBS response to the
new Head rotation, not a changed UniEgo position channel. The known source jump and
existing floor defects remain visible in both panels. Exact per-case metrics, asset
hash, and interpretation contract are in `qualitative_soma_mesh/gallery_manifest.json`
and `motion_expert_joint_attention/CAMERA_HEAD_RECANONICALIZATION.md`.

There is also a sparse but real source-data discontinuity problem that must not be conflated with the broad orientation issue. Forty-two of 728 camera-bearing sequences contain at least one interval where direct Head-camera separation exceeds `0.5 m`; 47 trip a conservative camera-step gate of at least `0.25 m` or `30 deg` in one 20-FPS frame. Split by component, only 12 sequences have >=`0.25 m` camera translation steps while 39 have >=`30 deg` rotations; similarly, 51 sequences have >=`0.25 m` Head translations but 227 trip the angular gate. The angular count can include genuine fast turns and should be treated as a conservative exclusion criterion, not proof that all 227 recordings have coordinate resets. The metre-scale translations are unambiguous: four train sequences contain `1.54-3.34 m` camera jumps with rotations up to `168.8 deg`; held-out `S17/20230918_s0_kevin_shaw_act2_5g4k0z` has a `1.644 m` camera jump and a simultaneous `0.935 m` raw-SMPL Head jump at transition `1519->1520`, despite MPS `quality_score=1.0` and sub-millisecond sampling error. This proves that at least the severe events are upstream source/registration behavior, not UniEgo decoding.

Exact T97 impact depends on the consumer. The unfiltered native Phase-1 index has 722/119,632 train windows (`0.6035%`) in the conservative camera-or-motion union, but only 95 (`0.0794%`) actually trip the camera gate relevant to its camera targets: 40 translation and 74 rotation windows, with overlap. The motion Phase-2/3 dataset first removes the existing floor-calibration drop list; its remaining aligned population has 539/112,937 affected train windows (`0.4773%`): 89 camera-gate windows (38 translation, 70 rotation), 457 Head-gate windows (70 translation, 452 rotation), 84 direct cross-modal step disagreements, and 93 >0.5 m separation windows. Only 120/112,937 (`0.1063%`) hit the stricter union of translation jumps, cross-modal translation disagreement, or >0.5 m separation; the other 419 are rotation-gate-only. Its corresponding conservative test count is 53/11,938 (`0.4440%`), of which nine are strict translation/separation cases. None of the selected motion-clean71 or replacement-five evaluation windows intersects these masks, so their reported root divergence remains an orientation/representation result rather than a catastrophic-jump artifact. Canonical artifacts are `summary_all.json`, `details_all.jsonl`, `training_window_impact_T97.json`, and complete unfiltered/Phase-2/3-filtered affected-window JSONLs under `/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/`. Existing floor filtering does not remove every source jump; future training should additionally exclude the listed rows through a versioned manifest/cache. Do not mutate old checkpoints or cached data in place.

The 2026-07-21 ranked clean-window visualization adds a stricter warning: even floor filtering plus the 539-window conservative source mask does **not** make every position trajectory reliable. `visualize_ranked_head_camera_windows.py` reconstructed all 112,937 Phase-2/3 train T97 rows, removed the conservative union, and ranked the remaining 112,398 rows. Within-window fixed Head-to-camera SO(3) residual has mean/median/p90/p95/p99/max `8.70/8.14/15.41/17.94/23.52/48.26 deg`. A separate no-fit translation score computes `RMSE((p_camera[t]-p_camera[0])-(p_head[t]-p_head[0]))` directly in the shared world frame: mean/median/p90/p95/p99/max is `7.01/5.73/14.79/17.39/21.28/45.59 cm`. This raw score is not expected to be zero even for rigid data because the physical camera lever arm rotates around the Head, but its worst cases expose smooth source drift that the single-step and 0.5 m separation gates miss. The worst is `S09/20230620_s0_marie_vasquez_act4_dhbf58@7759`: raw RMSE/endpoint/max `45.6/56.5/57.0 cm`, mean step disagreement `10.6 mm`, and fitted-lever residual mean `93.6 mm`, despite direct separation remaining below the existing 0.5 m gate. The worst rotational case is `S04/20230711_s0_frederick_young_act1_2imlal@12779`, with `48.26 deg` within-window residual. Best visual cohorts require at least 0.5 m raw camera path so static clips cannot win trivially. Outputs are under `/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final/ranked_clean_train_T97`: 80 unique-window detailed PNGs, four 20-case contact sheets, four H.264 2000x1400/20-FPS/97-frame montages, and `summary.json`. Blue camera and orange Head paths are always raw; the green fitted-lever path appears only as a separate detailed diagnostic. The active dataset still applies only its historical floor drop list, not this conservative source mask or a new smooth-drift mask.

Filtered native Phase-1 follow-up (2026-07-21): `native_phase_training/build_camera_motion_quality_filter.py` reconstructs the exact usable+captioned+cached T97 Phase-1 population and writes a strict physical-window exclusion artifact. The pinned file is `/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/camera_motion_quality_filter_v1_T97.json`, SHA-256 `1fd6465890cbf175068db839beb8bb220f6964090ff2c583cbf50d5001989848`. It excludes hard camera/Head/cross-modal jumps, >0.5 m direct separation, >25 degree mean residual after a best fixed per-window Head-to-camera rotation, or >5 cm RMS residual after a best fixed Head-frame lever. The 25-degree rotation cutoff is just above train p99 `24.68`; the 5 cm lever cutoff is over four times train p99 `11.49 mm`. The separate 30-degree one-frame angular gates are conservative and can include genuine fast turns; they are not proof of source resets. Train drops `1,583/119,632` rows (`1.323%`, `1,524` unique) and retains `118,049`; test drops `113/12,613` (`0.896%`) and retains `12,500`. Duplicate captions for one physical window are all removed. Global-extrinsic disagreement, raw relative-path RMSE, and the motion floor blacklist are explicitly not filter criteria. Full thresholds, overlap counts, nearby-cutoff sensitivity, per-subject rates, source hashes, and exclusions are embedded in the artifact and documented in `native_phase_training/AUDIT.md`.

`latent_nymeria_dataset.py` now validates filter kind/version/T/split/duplicates/summary counts and logs exact per-reason row removal. `sbatch_phase1_native_camera.sh` requires a pinned SHA for any filtered launch. The two controlled launchers are `sbatch_phase1_native_camera_qfilter.sh` (original `40/25/20/15` four-task mix) and `sbatch_phase1_native_camera_qfilter_no_i2v.sh` (drops only I2V, leaving raw `40/25/20`, effectively `47.06/29.41/23.53%`). Both remain 100k fixed-four, global batch 32, LoRA LR `5e-5`, action-head LR 4x, action loss 10, native RF, PowerEMA, save every 5k, and official shift-3 UniPC inference. Auto-eval is disabled for these launches to avoid forty uncontrolled one-GPU submissions while multi-node jobs are pending; evaluate selected checkpoints manually. Acceptance smokes under `/weka/jungbin/cosmos_motion_ft_runs/smoke_native_phase1_qfilter_20260721` exercised every active training stream, saved and reloaded both EMA DCPs, sampled all four inference modes for both with NVIDIA's official entrypoint, and validated eight `256x256`/97-frame/20-FPS outputs, `96x9` action arrays, and two complete four-mode visualization manifests.

Filtered Phase-1 production launched 2026-07-21 as job `3017` (`np1qf4`, node 1, four tasks) and job `3018` (`np1qf3`, node 0, no I2V). Run directories end in `native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_qfilterv1` and `_qfilterv1_noi2v`; logs are `slurm-np1qf4-3017.out` and `slurm-np1qf3-3018.out`. Launch evidence confirms eight ranks, the exact filter SHA/counts, intended stream ratios, and finite updates for both. Old jobs `3011`, `3010`, and `3003` were canceled one node at a time around the start of other-user job `3016`; `3003` was kept until `3017` completed all-rank model/filter initialization, preserving the requested queue fairness for pending two-node job `3014`.

The four-task filtered LR ablation is `sbatch_phase1_native_camera_qfilter_lr1e5.sh`, job `3021`, run name `native_phase1_camera_json_bs4_lora1e5_action4x_ema_100k_qfilterv1`. Its only scientific change from job `3017` is optimizer base/LoRA LR `1e-5` instead of `5e-5`; action-head multiplier remains 4x (`4e-5` effective), and filter/tasks/batching/EMA/native RF/scheduler/100k-save contract stay fixed. It was submitted with `afterany:3016` because 21 foreign SSH processes occupied every GPU on the sole Slurm-idle node; do not clear that dependency until all eight GPUs pass the existing 132-GB-free preflight.

Native Phase-1 video-quality ablations A-D were implemented on 2026-07-21. They all use the pinned qfilterv1 population, whole-token `C -> A person`, global batch 32, 100k horizon, save every 5k, base/LoRA LR `5e-5`, action modules at 4x, PowerEMA, action loss weight 2, and per-sample active-suffix normalization. Exact cached Wan causal boundaries are RGB `[1,9,17,33,49]` -> latent `[1,3,5,9,13]`; do not silently change these to the originally requested `[1,8,16,32,48]`, because those cut through causal four-frame latent groups and require either target-frame leakage or dropped clean frames. Visual RF noising and MSE indexes cover only the suffix; camera sequence offsets remain unchanged.

A is prefix-1 global generation Q/K/V/O LoRA with all four tasks. B is variable-prefix global LoRA with all four. C is variable-prefix action-interface-only. D is variable-prefix camera-token-only generation K/V LoRA; optional E is its prefix-1 factorization. C/D/E explicitly drop I2V training: native dummy action loss prevents a backward/FSDP crash, but I2V video loss has no trainable path in C and D's camera mask is empty, so those would be zero-update optimizer/scheduler/EMA steps. Their active `40/25/20` streams normalize to `47.06/29.41/23.53%`; I2V remains a mandatory frozen-prior evaluation at every prefix. Config import fails if action-only/camera-KV mode leaves I2V active.

`native_phase_training/camera_token_lora.py` preserves NVIDIA LoRA/DCP key names and masks the K/V residual to finalized packed action-token rows; text/video rows use exactly frozen base projections and zero-initialized B preserves the base output. `prefix_inference.py` is a narrow official-inference shim that validates contiguous explicit latent prefixes for forward/policy/I2V. `evaluate_prefix_suite.py` reports suffix-only PSNR/SSIM/LPIPS at full and relative early/middle/late horizons, inverse camera metrics, and policy full/suffix-reanchored camera metrics. Each 10k checkpoint evaluation uses official EMA UniPC (action 30/guidance 1/shift 3; I2V 35/guidance 6/shift 3), five held-out sources x five prefixes, all pair videos and GT/five-prefix grids, and writes `COMPLETE.json` only after metrics and visualization succeed.

Native Phase-1 pair/grid videos encode frame provenance directly. Green borders and `GT REFERENCE`/`GT CONDITION` headers identify reference or clean conditioning frames; red borders and `GENERATED` identify sampled suffix frames. The switch occurs at RGB frame `prefix_length` (prefix 9 means GT frames 0-8, generated frames 9-96). `viz/manifest.json` stores the same zero-based inclusive ranges under `video_frame_provenance`; do not remove this marking when changing visualization code.

Prefix-evaluation JSONLs retain `source_name`, `rgb_prefix_length`, and `latent_prefix_length` for local grouping, but NVIDIA's Pydantic inference schema forbids those extra fields. `sanitize_prefix_inference_inputs.py` validates and strips only those three fields into each output's `inference_inputs/` directory before the official CLI runs; `condition_frame_indexes_vision` remains intact. Visualization and metrics must continue to read the original enriched JSONLs. Passing the enriched files directly to `cosmos_framework.scripts.inference` fails before sampling.

Launchers are `sbatch_phase1_video_quality_{A,B,C,D,E}.sh`. Production jobs are A `3025`, B `3026`, C `3027`, and D `3028`; E was not submitted. A showed finite updates at about 1.0-1.1 s/step after launch. B-D perform a two-step eight-GPU DCP train/save/restart preflight on first allocation, then exact-resume into 100k. Status is dynamic; inspect `squeue`, the Slurm logs, and run-local completion markers before reporting it.

On 2026-07-22, A/B/C 10k and 20k evaluations were recovered directly on spare node-3 GPUs because Slurm copies `3029`-`3034` remained pending. Every one of the six run-local outputs has 80 successful official EMA-UniPC samples, 95 visualization records, all fixed-prefix video/camera metrics, and `COMPLETE.json` under `checkpoint_evals/iter_000010000` or `iter_000020000`. The redundant queued jobs were canceled only after those artifacts were validated; jobs `3025`-`3028` were not canceled. This real run exposed and verified the inference-schema sanitizer described above.

A/30k and C/30k were completed the same way on 2026-07-22, after which pending copies `3035/3036` were canceled. Both `iter_000030000` evaluation directories have 80 successful samples, 95 marked visualization records, all five-prefix metrics, and `COMPLETE.json`; jobs `3025`-`3028` were left unchanged. A/30k inverse means are `0.4047 deg`, direction cosine `0.8935`, translation error `6.764 mm`, and ATE `4.102 cm`; C/30k is `0.6466 deg`, `0.7533`, `14.628 mm`, and `11.543 cm`. These are fixed five-source diagnostics. B had no completed 30k DCP at that observation time, so no B/30k result should be inferred.

The final artifact is `eval_full71_step200000_unipc30/head_camera_calibration_test_actor_motion_clean71_joint_oracle.json` under the vanilla Phase-3 run. `backfill_oracle_actor_head_metrics.py` reproduced the pre-existing global metrics before writing (maximum discrepancy `0.00000004 m / 0.000156 deg`), appended actor IDs and four oracle keys to saved rows without resampling, and updated all71, floor-valid66, replacement5, motion-clean71, and summary JSON. Provenance records overlap as 66/71 for the original all71 set, 66/66 for floor-valid, 5/5 for replacements, and 71/71 for motion-clean71. `eval_all.py --oracle_test_actor_calibration PATH` supports the same fields in future evaluations without changing sampling or model inputs. The discarded attempts to fit actor orientation from unaligned full-sequence frames or from absolute orientation averages were rejected because they produced 14-degree or larger inconsistencies; only aligned windows and the joint relative-action objective are retained in code/results.

The isolated launcher is `sbatch_phase3_bridge_native_v2m_m2v_headcam.sh`. It keeps the exact baseline Phase-1 100k EMA and Phase-2 200k specialists, 50/50 V2M/M2V mixture, T97, global batch 32, 12 local bridges, LR/schedule, save cadence, and required two-per-direction visualization. The only learning-path changes are the derived M2V camera condition and V2M auxiliary losses; both motion and video checkpoint sampling use their native shift-3 UniPC solvers.

Verification on 2026-07-18 used free H200 node 0. Synthetic geometry, gradient, task-plan leakage, collate separation, native bridge locality, Python compile, launcher syntax, and diff checks passed. An exact-checkpoint one-GPU smoke loaded all 293 Phase-1 generator tensors and 149 Phase-2 motion tensors with zero missing/shape mismatches. V2M produced finite loss 0.1521, nonzero bridge gradients, zero frozen-specialist gradient leakage, and initial head-camera error 1.4 cm/2.48 degrees; the weighted auxiliary contribution was about 0.014, so it did not dominate. M2V produced finite loss 0.1404, nonzero bridge gradients, zero leakage, and its clean motion-derived camera condition was 1 mm/0.36 degrees from synchronized GT for that sample. The full-path smoke at `/weka/jungbin/cosmos_motion_ft_runs/smoke_phase3_headcam_full_20260718` performed an optimizer step, wrote a 96-tensor checkpoint plus optimizer/TensorBoard state, and produced valid 97-frame V2M and M2V media. A separate reload smoke at `smoke_phase3_headcam_reload_20260718` restored all 96 tensors and optimizer state with zero mismatches, then reran both UniPC samplers, VAE decode, motion/video rendering, and manifests successfully.

Production job `3003` (`p3brheadcam`) was submitted on 2026-07-18 and started on `a3ultravis-a3ultranodeset-0`. Its 8-rank startup loaded all specialist tensors cleanly, built the required two-V2M/two-M2V held-out visualization set, and advanced through step 220 with finite losses, no skipped optimizer updates, nonzero bridge gates, about 1.94 seconds/iteration after startup, and 122.6 GB peak allocated memory (well below the 143 GB H200 limit). Its run-local TensorBoard event and config were also created with the expected alignment and solver fields. Run directory: `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam`. Slurm log: `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brheadcam-3003.out`. This status is dynamic; inspect `squeue -j 3003`, `sacct -j 3003`, and the log before claiming current progress or success.

Step-10k paired evaluation completed on 2026-07-19. Both checkpoints use the exact same T97, seed-0, CFG-1, native shift-3 UniPC-30, full-71 plus deterministic replacement-5 contract. Outputs are `eval_full71_step010000_unipc30` and `eval_motion_clean_replacement5_step010000_unipc30` under each run. All 304 MP4s per run pass `ffprobe`; each run also has 152 finite `[97,283]` motion arrays and 76 finite `[48,25,16,16]` generated-latent files. The paired report is `compare_step010000_vs_baseline_step010000.json` in the headcam run.

On motion-clean71, headcam versus baseline changes V2M MPJPE `0.24662 -> 0.23522 m` (4.62% better; 38/71 wins), PA-MPJPE `0.13585 -> 0.12090 m` (11.00%; 55/71), feature MSE `0.48969 -> 0.43709` (10.74%; 49/71), acceleration error `0.006723 -> 0.006813` (1.35% worse; 34/71), and root error `0.21677 -> 0.20701 m` (4.50%; 35/71). M2V changes PSNR `15.3880 -> 16.4445 dB`, SSIM `0.48695 -> 0.51738`, and LPIPS `0.45467 -> 0.40057`; headcam wins 62/71, 62/71, and 60/71 rows. Relative head-camera translation improves `8.740 -> 7.286 mm` (16.64%; 60/71), and rotation improves `1.515 -> 1.014 degrees` (33.04%; 71/71). All-71 and floor-valid66 show the same direction on every metric except acceleration error. This supports the alignment intervention at 10k, especially for rotation and M2V, while showing that decoded root/MPJPE gains are modest and not universal.

Operational provenance: the first SSH full-71 attempt was terminated when another user's Slurm allocation populated node 2. Exclusive Slurm job `3004` safely reran both full sets and completed sampling/visualization, but exited during postprocessing because the launcher called system Python without NumPy. A second merger bug placed `_video_payload`'s return after another function's return. Both launcher and control-flow bugs are fixed and helper-tested; the completed outputs were merged successfully without resampling. Do not run SSH evaluators on occupied training nodes.

The latest comparison was pinned at request time to headcam 30k. Exclusive six-GPU job `3008` (`p3hdcmp`) runs the same full-71/replacement-5 protocol through `sbatch_compare_phase3_headcam.sh` for three checkpoints: headcam 30k, baseline 200k (literal latest-to-latest), and baseline 30k (matched training budget). It was pending for resources on 2026-07-19. It will write `compare_headcam_step030000_vs_baseline_step200000.json` and `compare_step030000_vs_baseline_step030000.json` in the headcam run. Superseded pending job `3007` was canceled before start. Treat this status as dynamic and verify Slurm/output markers before reporting 30k results.

The corrected causal 4x locality for a 97-frame clip and 25 Wan-VAE latent frames is `{0}`, `{1..4}`, `{5..8}`, ..., `{93..96}`. In code, motion frame `m` maps to latent frame `ceil(m/4)`, implemented as `(m+3)//4`. The old rule `[4*g, 4*g+3]` was wrong: it assigned source frames 1..3 to latent 0 and omitted source frames 94..96. `_verify_native_bridge_contracts.py` asserts every one of the 97 motion frames has exactly one bridge-local latent.

Bridge risk: if the frozen motion expert only learned text/text-image-to-motion and never learned to interpret generator/video features, the bridge must translate generator hidden states into useful motion-expert activations through residual edits alone. This may be enough, but should be evaluated as a capacity/alignment question rather than assumed. Monitor bridge gate values, bridge output norms, `video2motion` reconstruction, and `motimg2video` qualitative control. Because generator weights stay frozen in this hypothesis, Phase-1 camera/video metrics should be checked for non-regression.

Sampler compatibility for a native Phase-1 generator plus frozen motion expert:

- Native Phase 1 and native-schedule Phase 2 share a normalized shifted RF noise coordinate and 1000-step scale, but intentionally do not share the training-time base distribution or prediction target. Generator targets use Waver clean-time, convert to noise-time, apply resolution shift 3, and predict velocity. Motion targets use logit-normal noise-time, apply shift 3, and predict clean x0. New motion runs convert x0 to velocity at each step and use the same official UniPC scheduler class as the generator; historical job `2844` and its derived Phase-3 run recorded Euler and remain reproducible. Historical Phase-2 checkpoints without `motion_schedule=native` still use the legacy unshifted contract.
- The bridge should not force these into one global diffusion process. Treat each bridge task as a target-modality task. For `video2motion`, video is clean conditioning and motion is the noised target, so use the motion expert's motion `x0` noiser/sampler. For `motimg2video`, motion is clean conditioning and video is the noised target, so use the native Cosmos generator RF noiser/sampler.
- This is acceptable because the bridge aligns hidden states, not raw sampler states. It is wrong to train bridge video targets with the old custom `motion_expert_joint_attention` video sampler and then evaluate with official Cosmos inference. For the experimental joint-target objectives, sample motion and generator training times independently from their native marginals and pass both into the model; do not force one sampled scalar on both targets.
- `train.py`/`sample.py` now checkpoint independent `motion_schedule` and `gen_schedule` contracts. `gen_schedule=native` uses the exact CPU float32 Waver transform plus shift during training and `FlowUniPCMultistepScheduler` during video sampling; `legacy` preserves uniform-time Euler behavior for old checkpoints. For standalone Phase-1 visual-quality evaluation, the authoritative path remains `cosmos_framework.scripts.inference` through `native_phase_training/inference_config.py`; the joint sampler is required once bridge layers participate.

The 2026-07-11 Phase-2 run implements the earlier proposed inference-coordinate unification: native action-style logit-normal sigma with x0 prediction. Joint-target bridge training still uses target-modality training distributions rather than forcing simultaneously noised video and motion into one sampled scalar process. At inference, both native specialists can share one shift-3/1000 UniPC state because their inference noise coordinate is the same. The coupled sampler makes one model call per UniPC evaluation, passes the common quantized model timestep to both specialists, and converts motion x0 to velocity with the scheduler's actual sigma (which need not equal the quantized model timestep divided by 1000). Video-style `waver` for motion remains an unimplemented ablation.

The old bridge run/launcher (`ja_phase3_bridge_v2m_m2v_from_t2m3d_p1cam`, `sbatch_phase3_bridge_v2m_m2v.sh`, historical job `2774`) is retained only for provenance. It used historical custom specialists/schedules and the incorrect local frame map; do not resume it into the native experiment.

Active native bridge launcher: `motion_expert_joint_attention/sbatch_phase3_bridge_native_v2m_m2v.sh`. It uses 50/50 V2M/M2V, `T=97`, batch 4/GPU on 8 GPUs, 200k steps, LR `2e-4`, 1k warmup, cosine decay, and saves every 5k. The Slurm request is 120 hours: the measured historical bridge rate of about 1.738 s/step requires about 96.6 hours for optimizer steps alone, so the former 96-hour request could not cover checkpoint I/O and required visualization. It loads the final Phase-1 100k **EMA** from native DCP through `checkpoint_utils.py`, recreating LoRA rank 16 / alpha 32; using the old alpha-16 default would halve the trained adapter residual despite a shape-clean load. It loads the explicit Phase-2 200k `.pt`, freezes both specialists, and trains 805.40M bridge parameters. Generator-facing samples use native `[BOS,text,EOS,SOG]` reasoner packing, FPS-modulated float 3D-mRoPE at source FPS 20, and temporal modality margin 15000; this is required in addition to matching Waver/shift-3/UniPC. Mixed-task loss is normalized by local batch size: motion aggregate is scaled by the number of motion-target samples over `B`, while each vision/camera sample contributes its loss divided by `B`. The previous implementation summed per-sample vision/camera losses but used one aggregate motion mean, so effective task weight changed with batch composition.

Contact-Phase-2 vanilla bridge ablation (submitted 2026-07-20): `motion_expert_joint_attention/sbatch_phase3_bridge_native_v2m_m2v_contactp2.sh` changes only the frozen Phase-2 specialist and its matched V2M loss recipe relative to the naive native bridge. It initializes motion from the completed contact-aware checkpoint `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35/ckpt_step200000.pt`. The historical non-contact Phase 2 and its vanilla Phase 3 both used motion feature/joint/smooth weights `1/10/50` with no contact terms; this new pair consistently uses `1/1/5`, contact BCE `0.05`, GT-contact-masked foot velocity `1`, foot height `10`, contact-logit scale `2`, and 20 FPS. These losses apply only to V2M because M2V carries clean condition motion and has no motion target. Everything else remains the naive baseline: Phase-1 100k EMA, 50/50 V2M/M2V, no head-camera supervision, no joint-target tasks, frozen specialists, 12 local bridges, T97, global batch 32, LR `2e-4`, 1k warmup, cosine decay, 200k steps, and required two-per-direction visualization every 5k. Motion and video visualization use native shift-3 UniPC-30. Focused native-bridge, native-motion-flow/UniPC, and four contact-loss contracts passed before submission. Production job `3011` (`p3brp2ct`) requests one exclusive 8-GPU node for five days; its run directory is `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k` and log is `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brp2ct-3011.out`. At submission it was pending for resources; verify dynamic state with `squeue -j 3011` and `sacct -j 3011`.

Checkpoint visualization is required, not best-effort, for this launcher. Every 5k checkpoint and the final 200k state sample two fixed held-out V2M and two fixed held-out M2V windows with 30 native solver steps and guidance 1.0. V2M writes the clean source video, GT|predicted motion, and a combined comparison MP4. M2V writes GT/generated latent arrays, separate decoded MP4s, and a GT|generated comparison. `--require_viz` synchronizes setup/runtime failures across ranks and terminates the run coherently rather than silently training without results. `--viz_only` exists as a no-training/no-checkpoint verification seam.

Pre-submit verification on node 3 used actual Phase-1 90k EMA and Phase-2 190k weights. All 293 generator tensors and 149 motion tensors loaded with zero missing/shape mismatches. One real V2M and one real M2V forward/backward passed finite-loss and frozen-gradient checks. A metadata-checkpoint reload then sampled one held-out V2M case with native x0/Euler and one M2V case with native velocity/UniPC and wrote motion arrays/metrics, generated video latents, and `summary.json`. A second built-in smoke on node 2 repeated the two real forward/backward paths while Phase 2 was running: V2M loss `0.1500`, M2V loss `0.1581`, nonzero bridge gradients, and zero frozen-specialist gradient leakage. Final-pair verification then used Phase-1 100k EMA plus Phase-2 200k: 293/293 generator and 149/149 motion tensors loaded cleanly; V2M loss was `0.3095`, M2V loss was `0.1846`, bridge gradients were nonzero, and frozen-gradient checks passed. A separate `--viz_only` run sampled one held-out example per direction and produced valid V2M source/GT-prediction comparison media plus M2V GT/generated/comparison media under `/weka/jungbin/cosmos_motion_ft_runs/smoke_phase3_bridge_viz_exact_p1ema100k_p2native200k/viz_step000000`. These checks prove wiring, loading, noising, gradient routing, both sampler executions, VAE decoding, and media writing; they do **not** prove bridge quality. Residual risks include the Phase-2 expert's changed reasoner context inside Phase 3, prompt-format differences between native Phase 1 and joint Phase 3, UniEgo's incomplete camera/world placement information for M2V, and the large bridge capacity/zero-gate optimization dynamics.

Phase-3 joint-target multitask ablation (implemented 2026-07-19): `motion_expert_joint_attention/sbatch_phase3_bridge_native_multitask.sh` is the third directly comparable bridge run after (1) vanilla V2M/M2V and (2) the head-camera-supervision variant. It uses the same Phase-1 100k EMA, Phase-2 200k native checkpoint, frozen-specialist topology, 12 local bridges, T97, batch 4/GPU, global batch 32, LR `2e-4`, 1k warmup, cosine decay, 200k steps, and checkpoint/required-visualization interval 5k. It deliberately does **not** enable head-camera losses, so its intervention is the two added objectives rather than a combination of the second and third variants. The production mixture is 35% `video2motion`, 35% `motimg2video`, 15% `video2camera_motion`, and 15% `camimg2video_motion`. Each joint task's two target branches have relative weight 0.5. The resulting expected branch mass is motion 0.50, video 0.425, camera 0.075, with total loss budget 1/sample; this preserves the vanilla motion exposure while retaining 70% direct corner-task sampling.

The two joint objectives are `video -> camera + motion` and `camera + frame-0 image -> future video + motion`, both without text. During training, motion uses its shifted logit-normal x0 marginal and generator targets use their shifted Waver velocity marginal, sampled independently. During joint inference, `sample.py` uses one coupled shift-3 UniPC state and one forward call per solver evaluation. Bridge masking is derived from clean/noisy token roles rather than hard-coded task names: only target rows receive cross-modal residuals; both target modalities can update in joint tasks; the original V2M/M2V one-sided masks remain unchanged. Joint inverse visualization writes source video, GT/pred motion, GT/pred camera arrays, and a camera trajectory plot. Joint forward visualization writes GT/generated video plus GT/pred motion. With `--viz_n 8`, each checkpoint gets two held-out records for each of the four active modes.

Pre-submit verification for this ablation used the exact final specialists. CPU contract tests `_verify_phase3_multitask_contracts.py` and `_verify_native_bridge_contracts.py` passed. Real T33 and T97 one-GPU backward smokes exercised all four tasks with finite losses, nonzero bridge gate/core gradients after gate opening, and zero frozen-specialist gradient leakage. A two-step train/save/reload run restored all 96 bridge tensors plus optimizer state exactly and executed all four native samplers, VAE decoding, motion rendering, camera plotting, and manifest writing; all arrays were finite and all MP4 frame counts were valid. A two-rank ordinary-training smoke on node 1 GPUs 2/7 completed four mixed-task steps at 51.0 GB peak per added process with no skipped updates or collective hangs; `COSMOS_DDP_PARAM_CHECK=1` reported bit-identical replicated parameters after every optimizer step. Smoke output is `/weka/jungbin/cosmos_motion_ft_runs/smoke_phase3_multitask_ddp2_t33_20260719`. These checks establish execution and serialization contracts, not that the added objectives improve held-out quality.

Production job `3010` (`p3brmulti`) was submitted on 2026-07-19 with no dependency. It requests one exclusive 8-GPU `a3ultra` node, 64 CPUs, and five days. Its run directory is `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask`, and its Slurm log is `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brmulti-3010.out`. At submission it was pending for `Priority`; the log is not created until allocation. The launcher fails before model construction if any allocated GPU already uses more than 2 GiB, guarding against SSH processes that bypass Slurm exclusivity. Treat job state as dynamic and inspect `squeue -j 3010`, `sacct -j 3010`, and the log before reporting progress.

Interpret the ablation cautiously. Joint-target success would demonstrate that one self-attention bridge can express both directional and simultaneous cross-modal updates, strengthening the flexibility argument. It would not by itself prove self-attention is the best bridge architecture; that claim requires a matched-capacity bidirectional cross-attention baseline. `video -> camera + motion` may also admit two mostly independent predictions from the clean video, so improvement on that task alone is weak evidence that the generated camera and motion communicate with each other. The independent off-diagonal training times and the joint forward objective apply stronger pressure to use cross-target context, but whether that pressure improves ordinary V2M/M2V remains an empirical question. Evaluate all three variants with identical held-out windows, native UniPC-30 overrides, seeds, and matched training steps.

Native Phase-3 production job `2865` (`p3brnative`) ran normally through step 5,000 on 2026-07-14, then failed during the required checkpoint visualization. Training itself was healthy: finite losses, no skipped updates, about 1.811 s/step, and peak allocated GPU memory about 117.5 GB. It successfully wrote both `ckpt_step005000.pt` and `latest.pt` under `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k`; a metadata/mmap load confirmed step 5,000, 96 model tensors, optimizer state, and the expected V2M/M2V task list. The partial `viz_step005000` directory is **not** a complete evaluation: it contains the first V2M motion artifacts only and must not be treated as a successful four-sample visualization.

The job-2865 failure was a distributed collective mismatch, not OOM, bad training data, timeout, or checkpoint corruption. Only rank 0 runs `do_viz`, but `precompute_latents.load_vae` constructed `Wan2pt2VAEInterface`, whose `_video_vae` unconditionally called the framework's world-group `sync_model_states(model)`. Ranks 1-7 were already waiting in `run_checkpoint_viz`'s scalar status broadcast. The incompatible collective ordering made a nonzero/garbled status reach peer ranks; rank 2 raised the misleading `required checkpoint visualization failed ... on rank 0`, then torchrun terminated rank 0 before it could log its own traceback. `load_vae(..., rank_local=True)` now suppresses only that constructor synchronization for rank-0-only visualization and restores the framework function in `finally`; all ordinary VAE training/evaluation/precompute callers keep synchronized construction. The temporary visualization VAE is released and the CUDA cache cleared before rank 0 broadcasts status, so it is not retained during later optimizer steps. Visualization setup/barrier/status now use a separate CPU/Gloo group with a two-hour timeout because four 30-step samples may legitimately exceed the default 10-minute NCCL timeout; gradient synchronization remains on the normal NCCL group. A local contract test verified VAE-sync suppression/restoration, and Python syntax/diff checks passed.

The exact 8-rank fix gate is Slurm job `2869` (`p3brvizsmk`), launched by `sbatch_phase3_bridge_native_viz_ddp_smoke.sh`. It reloads the saved step-5,000 bridge plus the exact final specialists and must produce one held-out V2M and one held-out M2V visualization through `--viz_only`. Production resume job `2870` (`p3brnative`) has dependency `afterok:2869` and uses the regular launcher with `--resume auto`, so it will continue at step 5,001 only if the distributed visualization gate succeeds. At documentation time both are pending: `2869` for cluster priority/resources and `2870` for its intentional dependency. Logs are `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brvizsmk-2869.out` and `/home/jungbin_cho/cosmos_motion_ft/slurm-p3brnative-2870.out`. Treat all later status as dynamic and check `squeue`, `sacct`, and the logs before making current-progress claims.

Representation risk: camera action is raw metric relative SE(3) delta (`pos(3)+rot6d(6)`, loss on `[:9]`) and is naturally action-like. Current motion is normalized 283-d UniEgo, with body-centric/canonicalized pose features and foot contacts; it is not the same kind of raw metric state as camera action. This representation is probably reasonable for `video2motion` because the target is UniEgo features. It may be more ambiguous for `motimg2video`, because a normalized/canonicalized body motion condition may not fully specify absolute camera-frame body placement or the body-camera relation. If `motimg2video` is weak, representation alignment should be considered before blaming only the bridge architecture. Possible future auxiliary conditions include camera/world-frame root or pelvis deltas, root velocity, or camera-relative body trajectory, but these are not implemented.

Main launch wrappers include `run.sh`, `sbatch_t2m_both.sh`, `sbatch_t2m_both_mrope3d.sh`, `sbatch_t2m_ti2m_native_mrope3d.sh`, `sbatch_phase3_7task.sh`, `sbatch_phase3_bridge_native_v2m_m2v_contactp2.sh`, `sbatch_phase3_bridge_native_multitask.sh`, `sbatch_precompute.sh`, `sbatch_precompute_T97.sh`, `run_eval.sh`, and `run_camera_eval.sh`.

## Visualization and Evaluation

Joint-attention motion renderer: `render_motion.py`, runs in `cosmos`, no Kimodo import. It renders GT left/blue and generated right/red, with root-tracking XZ viewport, floor grid, trajectory trail, SOMA-30 parents, skipped fingertip-end joints, Y-up/+Z-forward input, negated X to match Kimodo display, and mplot3d Z-up remap.

`render_viz.py` is a Kimodo-env alternative. `_autorender.sh` watches viz dirs.

Phase-1 camera eval/viz note: the existing `/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/eval_all/camera_eval` contains inverse-dynamics outputs and metrics. On 2026-07-06 an additional FD/policy qualitative eval for `/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/ckpt_step200000.pt` was run on the T2M node into `/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/eval_all/fd_policy_camera_eval` with 8 test windows. It produced raw generated videos under `fd_out/` and `policy_out/`, and 16 GT|generated side-by-side mp4s under `viz/` (`*_fd.mp4`, `*_policy.mp4`). This was intentionally kept separate from the existing inverse-dynamics `camera_eval` directory to avoid overwriting old metadata.

Phase-3 held-out evaluation must pass `/weka/jungbin/cosmos_motion_ft_runs/joint_attention/full71_windows.json` to `eval_all.py --windows_json`. As of 2026-07-15, that argument constrains camera, motion, and video tasks in the listed order and requires all requested windows to resolve before evaluation starts. Older `eval_all.py` revisions forwarded the list only to `eval_camera`; V2M/M2V silently used the ordinary dataset index instead (an inspected historical output contained 71 windows from one S01 sequence). Do not label old V2M/M2V outputs from that path as the canonical 71-sequence benchmark. The corrected M2V evaluation excludes conditioned frame 0 and reports PSNR, SSIM, and LPIPS-Alex over frames 1-96 using the same GT resize/pad and metric implementation as native Phase-1 forward evaluation. It writes GT/generated comparison MP4s for every evaluated window. Corrected V2M evaluation writes GT/prediction motion MP4s. Both tasks report the canonical all-71 aggregate plus a 66-window aggregate excluding the five windows flagged by floor calibration; both are retained so benchmark coverage and motion-GT/conditioning-quality sensitivity remain explicit.

Phase-3 step-35k canonical evaluation completed on 2026-07-15 for `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/ckpt_step035000.pt`, using T97, 30 native steps, guidance 1, exact `full71_windows.json`, motion native shift-3 Euler/x0, and video native shift-3 UniPC/velocity. Output is `eval_full71_step035000_native30`. All 71 V2M and 71 M2V cases completed; motion/video arrays are finite and have expected shapes, and all 284 MP4s (142 primary comparisons plus M2V GT/generated component clips) pass `ffprobe`. All-71 V2M is MPJPE `0.26389 m`, PA-MPJPE `0.09835 m`, feature MSE `0.35335`, acceleration error `0.00622`, and root error `0.23721 m`. The floor-valid 66 subset is MPJPE `0.20526 m`, PA-MPJPE `0.09491 m`, feature MSE `0.32637`, acceleration error `0.00609`, and root error `0.17788 m`. All-71 M2V is PSNR `16.1292 dB`, SSIM `0.51260`, and LPIPS-Alex `0.42438`; the floor-valid 66 values are `16.1430 dB`, `0.51585`, and `0.42295`.

For a sequence-balanced motion-clean comparison, the five flagged rows were replaced deterministically by same-sequence held-out T97 windows that have cached video latents, are absent from the floor blacklist, satisfy old-stats `|z|max <= 10`, have at least ten contact frames, contact median within 5 cm of the calibrated floor, and minimum foot height at least -10 cm. The replacement-only evaluation is `eval_motion_clean_replacement5_step035000_native30`; it completed all five V2M/M2V cases and wrote 20 valid MP4s. Combined with the retained 66, `eval_full71_step035000_native30/motion_clean71_provenance.json` records the exact 71-row composition. Motion-clean71 V2M is MPJPE `0.21364 m`, PA-MPJPE `0.10098 m`, feature MSE `0.34501`, acceleration error `0.00637`, and root error `0.18647 m`; M2V is PSNR `16.1189 dB`, SSIM `0.51157`, and LPIPS-Alex `0.42318`. Preserve all three reports: all-71 exposes sensitivity to the original camera-oriented list, floor-valid66 avoids invalid motion GT without replacement, and motion-clean71 restores one valid row per held-out sequence. In V2M, text is disabled in both training and sampling; manifest captions shown by older evaluation renders were metadata labels only. New renders explicitly label them as metadata-only to avoid implying text conditioning.

Phase-3 step-85k canonical evaluation completed on 2026-07-16 with the identical step-35k protocol and deterministic windows/noise: T97, 30 steps, guidance 1, motion native shift-3 Euler/x0, and video native shift-3 UniPC/velocity. Outputs are `eval_full71_step085000_native30` plus `eval_motion_clean_replacement5_step085000_native30`; the merged clean71 metrics and provenance are stored in the full-eval directory. All 71 full rows and five replacements completed, all motion/video arrays are finite with expected shapes, and all 304 MP4s pass `ffprobe`. Compared with step 35k, motion-clean71 V2M changes from MPJPE `0.21364` to `0.20938 m`, PA-MPJPE `0.10098` to `0.10079 m`, feature MSE `0.34501` to `0.36916`, acceleration error `0.00637` to `0.00644`, and root error `0.18647` to `0.18304 m`: decoded pose/root errors improve modestly, while normalized feature and temporal errors worsen. Motion-clean71 M2V changes from PSNR `16.1189` to `16.3416 dB`, SSIM `0.51157` to `0.51311`, and LPIPS-Alex `0.42318` to `0.40071`, a clearer perceptual improvement. The same direction holds for all-71 and floor-valid66. Treat step 85k as better for M2V but mixed, not categorically better, for V2M.

Motion-UniPC correction on 2026-07-16: `eval_all.py` now exposes an inference-only `--motion_native_solver {euler,unipc}` override, forwards it through `sample.load_joint_model`, and records the resolved motion/generator schedules and solvers in `summary.json`. It also hard-errors when an explicit `--ckpt` path is missing; BASE smoke is permitted only when `--ckpt` is omitted. This fixes a dangerous prior behavior where a deleted checkpoint silently evaluated an untrained BASE model. The attempted step-35k UniPC directories `eval_full71_step035000_motion_unipc30` and `eval_motion_clean_replacement5_step035000_motion_unipc30` are invalid for exactly this reason: checkpoint retention had removed `ckpt_step035000.pt`, their logs say `no ckpt ... BASE model`, the static predictions have near-zero acceleration, and both directories are marked with `INVALID_EVALUATION.json`. Do not use those metrics or videos. The historical step-35k Euler evaluation remains valid because it ran while the checkpoint existed, but a genuine step-35k UniPC rerun requires restoring that checkpoint externally.

The valid step-85k V2M UniPC-30 rerun is `eval_full71_step085000_motion_unipc30`, with five clean replacements in `eval_motion_clean_replacement5_step085000_motion_unipc30`. It uses the same windows, initial noise, steps, and guidance as the Euler evaluation; all 71 predictions are finite `[97,283]` arrays and all 71 comparison MP4s pass `ffprobe`. All-71 UniPC metrics are MPJPE `0.26657 m`, PA-MPJPE `0.09955 m`, feature MSE `0.38859`, acceleration error `0.00642`, and root error `0.24030 m`. Floor-valid66 values are `0.20575`, `0.09658`, `0.36255`, `0.00629`, and `0.17918`; motion-clean71 values are `0.21382`, `0.10325`, `0.38616`, `0.00655`, and `0.18668`. Compared with step-85k Euler on motion-clean71, UniPC is respectively `2.12%`, `2.43%`, `4.61%`, `1.69%`, and `1.99%` worse; only 9/71 cases improve MPJPE and 2/71 improve feature MSE. The UniPC motions are dynamic (`accel_pred` mean `0.00536`, versus `0.00524` for Euler and about `2.9e-7` for the invalid BASE fallback), so this is a small solver-quality regression rather than static collapse. M2V does not invoke the motion sampler because motion is clean conditioning, so its previously reported metrics are unchanged.

Phase-3 step-110k canonical evaluation completed on 2026-07-17 in `eval_full71_step110000_unipc30`, with the five deterministic clean replacements in `eval_motion_clean_replacement5_step110000_unipc30`. Step 110k was the latest complete checkpoint when this evaluation was launched; step 115k was saved while it ran. The contract is T97, exact `full71_windows.json`, 30 steps, CFG 1, seed 0, native shift-3 x0/UniPC for V2M, and native shift-3 velocity/UniPC for M2V. All 142 motion arrays are finite `[97,283]`, all 71 video NPZ outputs are finite, and all 304 full-plus-replacement MP4s pass `ffprobe`.

Step-110k all-71 V2M is MPJPE `0.24743 m`, PA-MPJPE `0.09890 m`, feature MSE `0.38056`, acceleration error `0.00635`, and root error `0.22310 m`; floor-valid66 is `0.18831`, `0.09629`, `0.35434`, `0.00622`, and `0.16328`; motion-clean71 is `0.19644`, `0.10271`, `0.37741`, `0.00648`, and `0.17098`. Against the valid step-85k UniPC motion-clean71 result, these improve by `8.13%`, `0.52%`, `2.26%`, `1.13%`, and `8.41%`, respectively; per-case wins are 49/71 MPJPE, 39/71 PA-MPJPE, 49/71 feature MSE, 44/71 acceleration error, and 47/71 root error. Predicted acceleration remains dynamic (`0.00522` mean), so this is not the static BASE-fallback failure.

Step-110k all-71 M2V is PSNR `16.4682 dB`, SSIM `0.51868`, and LPIPS-Alex `0.39873`; floor-valid66 is `16.4969`, `0.52278`, and `0.39571`; motion-clean71 is `16.4863`, `0.51861`, and `0.39553`. Against step 85k on motion-clean71, PSNR improves `+0.1448 dB`, SSIM `+0.00550`, and LPIPS `-0.00518`, with per-case wins of 46/71, 48/71, and 49/71. Step 110k is therefore consistently better than step 85k on the aggregate V2M and M2V metrics under the corrected sampler, though PA-MPJPE improves only marginally.

Phase-3 steps 140k, 150k, and 160k were evaluated on 2026-07-18 with the exact step-110k contract: T97, exact `full71_windows.json`, seed 0, CFG 1, 30 steps, native shift-3 x0/UniPC for V2M, and native shift-3 velocity/UniPC for M2V. Their full outputs are `eval_full71_step{140000,150000,160000}_unipc30`; deterministic replacement outputs are `eval_motion_clean_replacement5_step{140000,150000,160000}_unipc30`. For every checkpoint, all 71 original rows plus five replacements completed, all 152 full-plus-replacement motion arrays are finite `[97,283]`, all 76 generated video-latent files are finite `[48,25,16,16]`, and all 304 MP4s are 97-frame files that pass `ffprobe`. Each summary records both resolved solvers as UniPC, all specialist loads had zero missing/shape-mismatched tensors, all 96 bridge tensors overlaid, and no BASE fallback occurred. `run_phase3_recent_eval.sh` is the repeatable serial launcher; `merge_phase3_clean71.py` combines floor-valid66 plus replacement5, updates provenance/summary files, and was regression-checked to reproduce the existing step-110k aggregates exactly.

Recent V2M results (lower is better):

| Step | Set | MPJPE m | PA-MPJPE m | feature MSE | accel error | root error m |
|---|---|---:|---:|---:|---:|---:|
| 140k | all71 | 0.24468 | 0.08915 | 0.35504 | 0.00616 | 0.21883 |
| 140k | floor-valid66 | 0.18381 | 0.08619 | 0.32646 | 0.00604 | 0.15741 |
| 140k | motion-clean71 | 0.19191 | 0.09227 | 0.35029 | 0.00629 | 0.16511 |
| 150k | all71 | 0.23895 | 0.09036 | 0.34215 | 0.00624 | 0.21461 |
| 150k | floor-valid66 | 0.17822 | 0.08744 | 0.31428 | 0.00612 | 0.15341 |
| 150k | motion-clean71 | 0.18458 | 0.09342 | 0.34000 | 0.00637 | 0.15906 |
| 160k | all71 | 0.25007 | 0.08786 | 0.34407 | 0.00621 | 0.22465 |
| 160k | floor-valid66 | 0.18879 | 0.08446 | 0.31592 | 0.00608 | 0.16298 |
| 160k | motion-clean71 | 0.19551 | 0.09032 | 0.34056 | 0.00632 | 0.16924 |

Recent M2V results (PSNR/SSIM higher is better; LPIPS lower is better):

| Step | Set | PSNR dB | SSIM | LPIPS-Alex |
|---|---|---:|---:|---:|
| 140k | all71 | 16.5244 | 0.51963 | 0.39651 |
| 140k | floor-valid66 | 16.5524 | 0.52368 | 0.39340 |
| 140k | motion-clean71 | 16.5482 | 0.51987 | 0.39287 |
| 150k | all71 | 16.4897 | 0.51880 | 0.39275 |
| 150k | floor-valid66 | 16.5106 | 0.52293 | 0.38954 |
| 150k | motion-clean71 | 16.4985 | 0.51890 | 0.38953 |
| 160k | all71 | 16.6157 | 0.52177 | 0.38835 |
| 160k | floor-valid66 | 16.6436 | 0.52602 | 0.38472 |
| 160k | motion-clean71 | 16.6188 | 0.52169 | 0.38460 |

There is no single best recent checkpoint. On the sequence-balanced motion-clean71 report, 150k is best for MPJPE (`0.18458 m`), feature MSE (`0.34000`), and root error (`0.15906 m`); 160k is best for PA-MPJPE (`0.09032 m`) and every M2V metric; 140k has the lowest acceleration error (`0.00629`). Against 110k, 150k lowers clean71 MPJPE by `6.04%`, feature MSE by `9.91%`, and root error by `6.98%`; 160k lowers PA-MPJPE by `12.06%` and improves M2V by `+0.1325 dB` PSNR, `+0.00308` SSIM, and `-0.01094` LPIPS. Mean predicted accelerations remain dynamic at approximately `0.00509`, `0.00521`, and `0.00515` for 140k/150k/160k, so none exhibits the invalid static BASE-fallback behavior.

Phase-3 high-tier M2V resolution audit (2026-07-24): baseline `ckpt_step200000.pt` was sampled on the exact sequence-balanced motion-clean71 benchmark, not only the original five replacement windows. The set is the canonical 66 floor-valid held-out rows plus five deterministic same-sequence replacements. The temporal contract is unchanged: 97 RGB frames, 97 aligned clean motion frames, and 25 Wan latent frames. The earlier 33-frame observation came from accidentally probing the default T33 cache and was never this checkpoint's training contract. The high-tier path uses `--resolution 480`, which means the official Cosmos **720 tier** but produces square 640x640 decoded videos and `[48,25,40,40]` latents in this pipeline. It overrides only the native generator inference shift from checkpointed 3 to 10 and otherwise retains native velocity prediction, UniPC-30, CFG 1, and seed 0. M2V motion is clean conditioning, so no motion solver is invoked.

Matched motion-clean71 suffix metrics:

| Baseline step 200k inference | PSNR dB | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| 256 tier / shift 3 | 16.64946 | 0.52182 | 0.38470 |
| 720 tier / shift 10 | 16.74922 | 0.60141 | 0.44467 |

High tier changes PSNR by `+0.09976 dB` and SSIM by `+0.07959`, but worsens LPIPS by `+0.05997`. This is mixed aligned-fidelity evidence, not proof that high-tier inference is categorically better. The full high-tier result is `eval_m2v_motion_clean71_step200000_720tier_s10_unipc30`; all 71 rows completed and its merged `summary.json`, metrics, latent outputs, and videos agree on the exact set. The dedicated three-way gallery `compare_m2v_motion_clean71_step200000_256s3_vs_720s10` contains 71 validated `GT | 256/S3 | 720/S10` MP4s.

Resolution is a real Phase-3 bridge extrapolation. At training resolution, each layer presents `25*16*16 = 6,400` video tokens to the bridge; high tier presents `25*40*40 = 40,000`, or 6.25 times as many. `LocalModalityBridge` keeps full generator-to-generator and motion-to-motion attention and uses temporal locality only for cross-modal generator-motion edges. Thus the frame mapping remains correct, but cross-modal softmax competition, per-motion-query key count, and same-modality attention statistics change. Scalar gates do not explicitly compensate for this token-count shift. More resolution-robust follow-ups would normalize cross-attention separately from same-modality attention, pool video tokens per frame before cross-modal exchange and broadcast afterward, or train the bridge on mixed spatial resolutions. Treat these as design hypotheses, not validated fixes.

The earlier five-replacement diagnostic remains useful for the head-camera checkpoint. Its high-tier suffix metrics are baseline step 200k `16.0745/0.49518/0.45498` and headcam step 115k `16.3287/0.49746/0.45837` for PSNR/SSIM/LPIPS. The five four-way videos are in `compare_m2v_720_base200k_head115k`; the aborted node-2 directory ending `_aborted_node2_20260724` is noncanonical.

The exact 71-window cross-phase gallery is:

`/weka/jungbin/cosmos_motion_ft_runs/phase1_phase3_motion_clean71_resolution_compare_20260724/compare_gt_p1_256_720_p3_256_720`

Each validated 97-frame, 20-FPS, 1280x256 MP4 is ordered `GT | Phase-1 256/S3 | Phase-1 720/S10 | Phase-3 256/S3 | Phase-3 720/S10`. Green denotes the GT tile or clean frame-0 condition; red denotes generated frames 1-96. Phase-1 uses the latest 100k EMA checkpoint and official Cosmos UniPC-30 sampler. Exact `(sequence UUID, start frame)` and condition hashes established that 66 canonical Phase-1 outputs were directly reusable; only the five replacement windows required new 256/S3 and 720/S10 inference. The initial 256 smoke correctly rejected local bookkeeping fields that are not part of the official inference schema; routing records through the existing sanitizer fixed this before production. No model output was silently paired by task index.

Important comparison caveat: `Phase-1` in that gallery means the historical **Original** action-weight-10 checkpoint, which also initialized vanilla Phase 3. It does not mean ablation A. A file named `phase1_ema100k_resolution_matrix_5_20260724/viz/by_model/A_*.mp4` uses the separately trained action-weight-2, quality-filtered, caption-replaced A checkpoint and contains `GT | A 256/S3 | A 720/S3 | A 720/S10`; compare the cross-phase Phase-1 tiles with `by_model/original_*.mp4` instead. Corresponding saved records use the same prompt, clean first frame, numerical camera actions, EMA, UniPC-30, CFG 1, seed 0, and shifts 3/10. The saved videos are still not bit-identical reruns: canonical Original 256 used 8-GPU FSDP with model-level resolution 256, while the compact matrix used one GPU and model-level resolution 720 even for explicit 256 rows; full-71 high tier used single-GPU shards. For inspected S07, reused Original versus matrix Original output-to-output PSNR is `43.36 dB` at 256 and `37.11 dB` at high tier, compared with only `30.77/22.87 dB` against matrix A. Thus the main visible discrepancy is model identity, with smaller topology/config-context drift. For a strict resolution ablation, resample both tiers from one checkpoint using one matched process per tier and matching model-level resolution. Detailed evidence is in `native_phase_training/PHASE1_VISUAL_QUALITY_AUDIT.md`.

Canonical artifacts and tools:

- Phase-3 high-tier clean71 result: `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/eval_m2v_motion_clean71_step200000_720tier_s10_unipc30`.
- Phase-3 exact high-tier cache: `/weka/jungbin/cosmos_motion_ft_runs/phase3_m2v_720tier_conditions_T97_motion_clean71_20260724`.
- Phase-1/Phase-3 exact inputs, reuse manifest, missing-five outputs, and five-way gallery: `/weka/jungbin/cosmos_motion_ft_runs/phase1_phase3_motion_clean71_resolution_compare_20260724`.
- Headcam five-window high tier: `/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam/eval_m2v_replacement5_step115000_720tier_s10_unipc30`.
- Phase-3 tools: `shard_windows_json.py`, `run_phase3_m2v_720_shard.sh`, `merge_phase3_m2v_720.py`, and `compare_phase3_m2v_720.py`. `eval_all.py --gen_shift_override` records checkpoint/effective shifts and `--expected_m2v_latent_hw` fails early on a wrong cache.
- Phase-1/cross-phase tools: `prep_test_eval.py --windows-json`, `prepare_phase1_motion_clean_reuse.py`, `run_phase1_forward_resolution_shard.sh`, and `compare_phase1_phase3_resolutions.py`.

`sample.py` provides standalone per-task samplers. `eval_all.py`, `eval_camera.py`, and `eval_motion_recon.py` are main evaluation helpers.

In-train Phase-2 visualization balances T2M and TI2M. T2M also balances Nymeria/BONES where quota permits. It saves full-frame generated and GT `.npy` arrays plus MP4s; TI2M additionally saves the 256x256 conditioning PNG and renders conditioning image | GT motion | generated motion. Rank 0 samples while all other DDP ranks wait at barriers. Rendering is best-effort and must not break training.

## Cosmos Reference Constraints

The project-relevant Cosmos 3 constraints are consolidated here:

- Cosmos has AR/reasoner and DM/generator subsequences.
- AR tokens use causal attention over AR only.
- DM tokens use full attention with keys/values from AR+DM.
- AR/reasoner is not updated from DM/generator tokens.
- Non-language modalities use projection heads plus modality embeddings.
- Vision generation uses Wan2.2 VAE latents; VAE is frozen.
- Action uses domain-aware projections, relative pose pseudo-actions for ego/effector pose, 6D rotations, and domain ids.
- Cosmos3-Nano has 36 layers, hidden 4096, 32 attention heads, 8 KV heads, head dim 128, FFN 12288.
- Generator training uses rectified-flow velocity objective and masks clean conditioning tokens out of loss.
- Generator pre/mid-training freezes the reasoner and updates generation-specific params.
- Cosmos mid-training action mixture includes forward dynamics, inverse dynamics, and policy.
- Mid-training uses action loss scale 10, FusedAdamW lr `1e-4`, wd 0.05, grad clip 1.0, LambdaLinear start factor 0.4/cycle 100k, and shift 3/5/10 for 256p/480p/720p.
- Robot policy post-training is a structural analog for adding a continuous action-like modality: fresh heads, 5x LR on new action params, lr `2e-4`, shifted schedule around 5 for action sampling. However, root 369-d motion is absolute kinematic motion, not Cosmos delta action.
- For action generation, default Cosmos sampling uses steps 50, guidance 1 for forward/inverse dynamics, shift 5; policy uses fewer steps and guidance 3. Audio/visual generation uses different CFG/shift.

Project-specific assessment: the old root 369-d motion finetune structurally matches Cosmos but diverges in LR, head LR multiplier, timestep/noise schedule, LoRA usage, constant LR, and no CFG text dropout.

## Environment and Launch Rules

Most real commands must run on cluster nodes. This checkout alone cannot import Cosmos unless the external framework/env are available.

The fixed `/home/jungbin_cho` and `/weka/jungbin` commands below describe the
historical H200 environment. On Yonsei or Grasp, use the explicit site roots in
`migration/SITE_STORAGE.md`; do not silently fall back to another machine's
paths.

Invariant `cosmos` preamble:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd /home/jungbin_cho/cosmos-framework
```

The working directory must be `/home/jungbin_cho/cosmos-framework` for Cosmos config paths such as `QWEN_JSON`. Invoke scripts in this repo by path or through wrappers.

Use `motion_expert_joint_attention/run.sh` for Python entrypoints in that directory. `run.sh` sets env, cwd, and `PYTHONPATH`.

Single-GPU `srun` needs `--ntasks=1`; otherwise `srun` may fan out tasks per CPU and OOM. Do not let `srun` and `torchrun` both fan out ranks.

Inference often needs `--no-guardrails` because nodes lack `uvx` and guardrail downloads fail.

Python stdout can be block-buffered in logs. Monitor real training via TensorBoard.

Runs/checkpoints usually live under `/weka/jungbin/cosmos_motion_ft_runs/<run_name>/`. Joint-attention runs usually write TensorBoard events directly under their run directory. Native Phase 1 now writes events under `${job.path_local}/tensorboard` unless `TB_LOG_DIR` is set. Older native job `2801` wrote to the shared fallback `/weka/jungbin/cosmos_motion_ft_runs/tensorboard`.

Generated logs, Slurm outputs, cached latents, `.npy/.npz` stats, mp4s, and checkpoints are artifacts. Do not edit them unless explicitly requested.

## Environments

- `cosmos`: Py 3.13 / torch cu128-ish environment for Cosmos-3 Nano, training, sampling, inference, and `cosmos_framework`.
- `kimodo`: Py 3.10 / torch 2.4 environment for Kimodo datasets, BONES export, and some decode/render workflows.
- `nymeria_plus`: projectaria/VRS preprocessing for camera extraction.

The `kimodo` and `cosmos` environments cannot share one process. Typical root pipeline is export in `kimodo`, train/sample in `cosmos`, decode/render in `kimodo` or with pure-torch ports.

## Verification

Use the smallest relevant verification after edits.

Joint-attention CPU checks:

```bash
python motion_expert_joint_attention/mot_joint_attention.py
python motion_expert_joint_attention/mot_joint_layer.py
```

Joint-attention GPU smoke:

```bash
bash motion_expert_joint_attention/run.sh motion_expert_joint_attention/smoke_7task.py --gen_lora --steps 3
```

Other diagnostics in `motion_expert_joint_attention/` include `_verify_*.py`, `_diag_*.py`, `compare_uniego_stats.py`, and objective/floor/load checks.

Root checks include `verify_export.py`, `verify_full_export.py`, and `verify_motion_decode.py`.

Native Phase 1 checks:

```bash
python -m unittest native_phase_training.test_contracts
python -m py_compile native_phase_training/*.py
bash -n native_phase_training/sbatch_phase1_native_camera.sh
```

TOML/config dryrun:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth \
BASE_CHECKPOINT_PATH=/weka/jungbin/cosmos3_nano_dcp \
NYMERIA_RESOLUTION=256 \
bash native_phase_training/run_latent_train.sh --dryrun \
  trainer.max_iter=1 \
  checkpoint.save_iter=1000 \
  job.name=native_phase1_config_check \
  model.config.parallelism.data_parallel_replicate_degree=8 \
  model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.parallelism.context_parallel_shard_degree=1 \
  model.config.parallelism.cfg_parallel_shard_degree=1
```

Full native Phase 1 verification after code changes means more than import success: run a GPU train smoke on a node, let the trainer save a DCP checkpoint, then load that exact checkpoint through `cosmos_framework.scripts.inference` using `--config-file native_phase_training/inference_config.py --sampler unipc --use-ema-weights` and sample `forward_dynamics`, `inverse_dynamics`, `policy`, and `image2video`. Verify separate output names, `status=success`, action shape `96 x 9` for action modes, and a 97-frame 256x256 MP4 at 20 fps for every mode; run `visualize_checkpoint.py` and verify its four-mode manifest.

Do not assume imports or training work on the login/head machine outside the cluster env.

## File Map

Root:

- `AGENTS_ALL.md`: canonical context for agents and the only high-level source of truth.
- `AGENTS.md`: Codex bootstrap pointer to `AGENTS_ALL.md`.
- `CLAUDE.md`: Claude bootstrap pointer to `AGENTS_ALL.md`.
- `README.md`: short pointer to `AGENTS_ALL.md`.
- `train_motion_ft.py`: root text-to-motion trainer.
- `sample_motion.py`: root sampler.
- `motion_decode.py`: 369-d decode/loss.
- `verify_visual_tower/`: local checks for the Cosmos Nano standalone Qwen3-VL visual tower and reasoner-image smoke path.

`native_phase_training/`:

- `README.md`: official-compatible Phase 1 runbook, smoke evidence, OOM postmortem, and launch/eval commands.
- `latent_omni_model.py`: native `OmniMoTModel` subclass with optional cached-video-latent training input.
- `latent_nymeria_dataset.py`: Nymeria cached-latent native action/video dataset.
- `build_camera_motion_quality_filter.py`: exact T97 camera/Head discontinuity and rigid-consistency filter builder.
- `prep_test_eval.py`: native held-out T97/256/shift-3 official-inference input builder.
- `visualize_checkpoint.py`: four-mode official-output visualizer and manifest writer.
- `checkpoint_eval_callback.py`, `sbatch_checkpoint_eval.sh`: post-save isolated four-mode evaluation submission and worker job.
- `test_contracts.py`: prompt, packing, and evaluation-contract CPU tests.
- `experiment.py`: Hydra experiment registration for `world_camera_nymeria_latent_nano`.
- `world_camera_nymeria_latent.toml`: production TOML for the native camera/video LoRA run.
- `run_latent_train.py`, `run_latent_train.sh`: training entrypoints. `run_latent_train.py` sets a run-local TensorBoard log dir.
- `inference_config.py`: official inference config shim.
- `sbatch_phase1_native_camera.sh`: production Slurm launcher with exclusive-node request, memory preflight, and compile disabled.
- `sbatch_phase1_native_camera_qfilter.sh`, `sbatch_phase1_native_camera_qfilter_no_i2v.sh`: pinned filtered four-task and no-I2V launchers.
- `sbatch_phase1_native_camera_qfilter_lr1e5.sh`: exact filtered four-task LR ablation with base/LoRA LR `1e-5` and unchanged 4x action-head multiplier.

`motion_expert_joint_attention/`:

- `README.md`: short pointer to `AGENTS_ALL.md` and active code read order.
- `DESIGN_7TASK.md`: redirect stub kept for code docstring compatibility; task truth lives in `task_plan.py` and this file.
- `mot_joint_attention.py`: two-call mask.
- `mot_joint_layer.py`: role weight routing and sparse-depth layer.
- `joint_motion_model.py`: packed assembly, sparse 3-way/2-way interleave, timestep pre-scale, freeze/toggle logic, sampling.
- `cosmos_loader.py`: builds/materializes/freeze-loads Cosmos, tokenization, layer handles. Only file that should touch framework net directly.
- `motion_heads.py`: motion I/O heads and local timestep embedder.
- `gen_heads.py`: video/image/camera encode/decode via frozen Cosmos net.
- `task_plan.py`: task specs, condition masks, loss specs.
- `flow.py`: noising, losses, samplers.
- `nymeria_joint_dataset.py`: 5-modality data loader and collate.
- `dataset.py`: 283-d BONES/Nymeria motion pair loader.
- `build_bones_pairs.py`: offline BONES pair export.
- `precompute_latents.py`: offline Wan-VAE latent precompute.
- `precompute_floor_calibration.py`: per-seq floor deltas and drop list.
- `decode_uniego_torch.py`, `uniego_layout.py`: 283-d decode/layout.
- `head_camera_alignment.py`, `estimate_head_camera_calibration.py`, `head_camera_calibration_train.json`: relative SOMA-Head to upright-camera mapping, reusable rigid fitter, train-only estimator, and production constants.
- `camera_head_recanonicalization.py`, `build_camera_head_aligned_uniego.py`, `compare_camera_head_aligned_uniego.py`, `visualize_camera_head_aligned_uniego.py`, `visualize_camera_head_aligned_soma_mesh.py`: versioned `camhead_v1` rotation correction, corpus builder, exhaustive preservation comparator, skeleton gallery, and controlled standard-SOMA surface gallery. `estimate_camera_head_absolute_lever.py` and its focused test evaluate the separate experimental absolute-origin lever without changing the corpus or production calibration; `test_camera_head_soma_mesh.py` covers the standard-skin expansion/LBS invariants. See `CAMERA_HEAD_RECANONICALIZATION.md`.
- `audit_nymeria_camera_motion.py`, `summarize_nymeria_alignment_quality.py`, `visualize_nymeria_alignment_audit.py`: source-level timing/world-frame/continuity audit, exact Phase-1 versus floor-filtered Phase-2/3 window impact, and corrected direct-world visualizations.
- `estimate_test_actor_head_camera_calibration.py`, `backfill_oracle_actor_head_metrics.py`: explicitly test-leaky per-actor GT oracle fitting and offline augmentation of already-sampled V2M metrics; never use the oracle artifact for training/model selection.
- `merge_phase3_clean71.py`, `compare_phase3_evals.py`: corrected clean-71 aggregation and paired baseline/candidate comparison.
- `config.py`: dims, paths, task weights, defaults.
- `train.py`, `sample.py`, `eval_all.py`, `eval_camera.py`, `eval_motion_recon.py`.
- `prepare_shape_tmr_eval.py`, `precompute_nymeria_tmr_text.py`, `eval_phase2_shape_tmr.py`: final Phase-2 BONES/Nymeria C45 evaluator.
- `render_motion.py`, `render_viz.py`.
- `sbatch_phase3_bridge_native_v2m_m2v_headcam.sh`: isolated head-camera Phase-3 production launcher.
- `sbatch_phase3_bridge_native_multitask.sh`: isolated joint-target Phase-3 ablation launcher (35/35/15/15 V2M/M2V/joint-inverse/joint-forward).
- `_verify_phase3_multitask_contracts.py`: CPU contracts for joint task plans, role/locality masks, independent training sigmas, coupled state splitting, and scheduler compatibility guards.
- `sbatch_compare_phase3_step10k_headcam.sh`: exclusive paired step-10k full-71/replacement-5 evaluation launcher.
- `sbatch_compare_phase3_headcam.sh`: parameterized exclusive paired evaluation launcher (`PHASE3_COMPARE_STEP`).
- `run.sh`, other `sbatch_*.sh`, `_verify_*.py`, `_diag_*.py`.

`motion_expert_t2m_edge/`:

- Isolated Cosmos-3 Edge Phase-2 T2M+TI2M path. TI2M image+caption tokens come
  from the frozen Edge reasoner vision path; there are no video-generator
  tokens, no three-way reasoner/generator/motion attention, and no Phase-1
  adapter initialization.
- `model.py`, `layer.py`, `attention.py`: frozen Edge Nemotron reasoner plus the
  original Nano-style trainable motion pathway. Shared-attention interfaces use
  Edge hidden/head geometry (2048 residual, 16 Q heads, 8 KV heads, head dim
  128). Capacity is seven evenly spaced motion blocks at
  `[3,7,11,15,19,23,27]`, each with the original fresh width-3072 three-linear
  SwiGLU rather than the Edge reasoner's width-9216 ReLU-squared FFN.
- `edge_loader.py`: loads the frozen Edge reasoner and its required
  `k_norm_und_for_gen` tensors from the pinned Edge DCP plus the pinned frozen
  SigLIP2/projector bundle for TI2M, then discards the unused video-generation
  pathway.
- `data.py`: uses native-caption `_t2m_index` for Nymeria T2M, aligned `_index`
  for 97-frame reasoner-image TI2M, `camera_head_recanonicalization_v1`, its
  matching mean/std, and floor calibration. Defaults are Nymeria-only:
  0.75/0.25 T2M/TI2M and `bones_frac=0`. BONES is retained only as an explicit
  T2M ablation. Standalone uppercase `C` becomes sentence-aware `The camera
  wearer` / `the camera wearer`, matching retrained Phase 1.
- `train.py`, `train_visualization.py`, `sample.py`, `checkpoint.py`: x0
  shifted-native Phase-2 training, strict motion-only checkpoint contracts,
  UniPC sampling, online W&B scalar/media logging, and deterministic held-out
  GT-vs-generated rendering. Production freezes five Nymeria test samples per
  task plus their exact inference seeds, logs loss/gradient norm every 20
  steps, and renders/uploads T2M `GT | generated` plus TI2M
  `conditioning image | GT | generated` at step 0 and every 5k steps.
- `run.sh`: selects the Edge framework and v1 representation paths. Launch CPU
  checks with `bash motion_expert_t2m_edge/run.sh -m unittest discover -s
  /home/jungbinc/cosmos_motion_ft/motion_expert_t2m_edge/tests -v`; use
  `sbatch_l40_smoke.sh`, then the canonical one-GPU `sbatch_train_1gpu.sh`.
  `sbatch_train_1gpu_batch.sh` is a shared-`batch` partition wrapper around the
  same canonical training command.
- `SMOKE_RESULTS.md`: the one-L40S schema-v3 Nymeria-only T=16 and
  production-shape T=200/TI2M=97 gates pass finite forward/backward, nonzero
  motion gradients, zero frozen gradients, strict reload, and TI2M sampling at
  7.50 GiB peak allocation. Online W&B/media job `528071` additionally passed
  scalar loss/grad logging, fixed T2M/TI2M sampling, local H.264 rendering,
  media upload, and W&B API verification. Schema-v1 prototype and schema-v2
  BONES-mixture evidence is retained but superseded. Multi-GPU DDP is optional,
  not an execution gate. Smoke runs prove wiring/memory only, not learned quality;
  production still requires
  C45, foot-skating/floating, trajectory, and diverse SOMA comparison.
- First instrumented one-L40S production job `528072` ran under
  `edge_7layer_nymeria_t2m_ti2m_v1_wandb_viz`; W&B run `bupzjaj5` has ten
  fixed production-shape step-0 videos, five TI2M condition images, the
  full-prompt table, and verified loss/gradient rows through step 20. Slurm
  preempted it twice before the old 5k checkpoint interval. The superseding
  launcher atomically overwrites `recovery_latest.pt` every 250 steps, stores
  model/optimizer/RNG/data-epoch state, uses `--resume auto`, and requests
  Slurm requeue plus USR1 warning. Replacement `liu-compute` job `529512`
  reached model/data initialization but failed before step 0 when W&B 0.27.2
  timed out after 30 seconds waiting for its local service port; it produced no
  checkpoint. The repaired launcher moves mutable W&B data/cache from the full
  shared `/home` filesystem to Weka, waits up to 300 seconds per attempt, and
  retries three times with the same persisted ID; all 14 contract tests pass.
  Repaired `liu-compute` job `529849` remained pending and was cancelled before
  allocation once shared-`batch` job `529851` began on a healthy L40 at
  `dj-l40-0`, avoiding duplicate writers. Job `529851` successfully reused W&B
  run `isptsmkc`, uploaded the fixed five T2M plus five TI2M step-0 comparisons,
  and reached step 20 with finite loss `1.089077`, gradient norm `49.5103`, and
  16.31 GiB peak allocation. A manual USR1 test wrote the 1.41 GB atomic Weka
  recovery checkpoint at step 75, and the job continued through logged step 80;
  a memory-mapped load confirmed schema 3 plus optimizer/data-epoch/full RNG
  state. Job `529851` later advanced its rolling recovery checkpoint to step
  92,500; its final automatic restart failed at zero runtime because the new
  explicit site selector was not exported by the submitted Grasp launcher.
  The repaired launchers export `COSMOS_SITE=grasp`, retain the Grasp `/weka`
  artifact root, and explicitly select the Edge rather than Nano DCP.
  Replacement batch job `541394` loaded step 92,500 plus
  optimizer/data-epoch/RNG state, reused W&B run `isptsmkc`, passed RNG
  restoration, and completed finite step 92,520 (`loss=0.115096`, gradient
  norm `246.0626`, 16.27 GiB peak allocation).
  It is the active Phase-2 run under
  `edge_7layer_nymeria_t2m_ti2m_v1_wandb_viz_preemptsafe`; known ECC-faulted
  node `ll-l40-1` is excluded. First `liu-compute` job `528415` had a zero-runtime
  launch-level `FAILED (0:53)` event on `ll-l40-0`; no user log, output, W&B
  run, or checkpoint was created. At the same second, five incumbent jobs on
  that node failed, four other new tasks also died with zero-runtime `0:53`,
  and unrelated jobs then started successfully. Treat this as a transient
  scheduler/node resource-handoff incident, not Phase-2 execution; exact
  prolog/cgroup/GRES/controller attribution requires administrator logs.
  Superseded pending `batch` job `528385` was cancelled before it started.

`motion_expert/`:

- `README.md`: frozen-reasoner cross-attention POC.
- `BONES_SEED_POC.md`: BONES-only LLM2Vec in-context POC.
- `reasoner.py`, `precompute_hr.py`, `hr_cache.py`: reasoner hidden-state cache.
- `motion_expert.py`, `flow.py`, `train.py`, `sample.py`, `viz.py`.
- `bs_*`: BONES-only in-context LLM2Vec POC; `bs_normalization.py` owns checkpoint-pinned
  generator mean/std provenance and mismatch checks.

`nymeria_world/`:

- `README.md`: native camera-action world-model docs.
- `camera_to_action.py`, `nymeria_camera_dataset.py`, `nymeria_camera_rgb_dataset.py`.
- `extract_camera_opencv.py`, `prep_*`, `viz_*`, `run_vggt.py`.
- `launch_camera_phase2.sh`, `sbatch_camera_phase2.sh`.
- `export_merge_lora.py`, `prep_test_eval.py`, `run_infer_merged.sh`, `sbatch_infer_3tasks.sh`.

## Editing Guidance

Prefer existing local patterns and wrappers. Do not modify `/home/jungbin_cho/cosmos-framework` unless the user explicitly asks and the change is necessary.

Keep generated artifacts, Slurm logs, checkpoints, cached latents, stats files, mp4s, and `.npy/.npz` outputs untouched unless asked.

When changing joint-attention code, preserve these invariants:

- reasoner never attends to gen/motion;
- generator and motion attend densely over reasoner+generator+motion;
- mask ownership stays in `mot_joint_attention.py`;
- weight routing stays in `mot_joint_layer.py`;
- plain layers do not create `_moe_motion` params and pass motion rows through;
- motion init stays fresh unless explicitly changing the experiment;
- timestep pre-scale happens exactly once in `JointMotionModel.forward`;
- camera action stays raw 9-d with loss on `[:9]`;
- BONES remains text2motion-only.

When changing `motion_expert_t2m_edge/`, preserve these separate invariants:

- this Phase-2 path remains two-role T2M/TI2M (causal frozen reasoner,
  including visual tokens for TI2M, plus full motion queries), never the
  Phase-3 three-role joint-attention model;
- shared attention requires Edge's residual and Q/K head geometry, while the
  private motion FFN stays the smaller Nano-style SwiGLU;
- there are exactly seven motion blocks at `[3,7,11,15,19,23,27]`;
- the reasoner never reads motion, and its detached/frozen branch receives no
  gradient;
- checkpoints contain only motion-owned tensors and reject architecture,
  representation, stats, DCP, or framework drift;
- data stays `camera_head_recanonicalization_v1`, with native Nymeria T2M
  spans and 97-frame aligned Nymeria TI2M. BONES defaults off; if explicitly
  enabled for ablation it is routed only to T2M, remains labeled legacy
  motion-only, and is never treated as camera/head-equivalent.

When changing data code, be explicit about which index is used (`_index` vs `_t2m_index`) and why. When touching floor grounding, keep `entry["off"]` semantics as calibrated total vertical shift.

When changing training, use `named_all_parameters()` / `trainable_parameters()` for optimizer, grad clip, all-reduce, and checkpoint logic.

When changing samplers, keep train/sampling noising conventions pinned to the relevant trainer and do not re-noise already-current sample states.

When changing `native_phase_training/`, preserve these invariants:

- cached `video_latents` are a training input optimization only;
- official inference must still work with ordinary `vision_path` image/video inputs and no `video_latents`;
- camera actions stay raw 9-d `camera_pose` deltas and are padded only inside the native action path;
- native Cosmos RF distributions and action loss weight stay unchanged unless explicitly running an ablation;
- action prompts must match official `ActionPromptJsonFormatter`, inverse text must remain exactly empty, and image-to-video must stay on the official generic prompt template;
- cached-latent batches must use the active fixed-four contract (`max_samples_per_batch=4`, `max_sequence_length=None`); any optional 45,056-token run must use `LatentAwareIterativeJointDataLoader`, never stock counting of the 1x1 dummy pixels;
- native Phase 1 evaluations must resolve the saved run contract before model import and use its exact resolution/shift/T; historical 256 runs are 256/shift 3/T97/action96/20 FPS, while the controlled high-tier run is 720/shift 10/T97/action96/20 FPS and decodes at 640x640;
- production checkpoint visualization defaults to a post-save official-inference Slurm job; smoke runs must disable auto-eval. Explicit user exception (2026-10-27): the two-task prefix1 **action-only** run uses `native_phase_training/validation.py` inside the trainer, like EgoControl. Its launcher enables `NATIVEP1_WANDB_PREVIEWS=1`: four fixed test71 cases after the first update following launch/resume and every5k, native EMA sampling on all ranks, rank0 media/GT-reference metrics in the same UPenn W&B run, with training RNG/modes restored. No separate preview worker. See the native README for paths and sampling details;
- do not reintroduce Torch compile into the production Slurm launcher unless a clean-node multi-GPU smoke proves it fits;
- do not start production training on a node with leftover GPU memory consumers; use the launcher preflight or inspect `nvidia-smi`.

Yonsei node exclusion: **do not use `cs-gpu-01` for GPU jobs.** Include
`--exclude=cs-gpu-01` in all new Yonsei GPU `srun` and `sbatch` allocation
requests, or append it to any existing exclusion list. On 2026-09-09, an
allocated RTX 4090 was visible to `nvidia-smi`, but both PyTorch CUDA startup
and a direct driver `cuInit(0)` failed with `CUDA_ERROR_UNKNOWN` (999).
Opening the NVIDIA UVM devices returned I/O errors, including from fresh
steps within the allocation. Treat NVML visibility as insufficient evidence
that CUDA works. Keep this exclusion until the user explicitly authorizes
retesting after an administrator repair. This rule is Yonsei-only; do not
replace or change the Grasp cluster's existing node exclusions.

Cluster operational rule: before every SSH compute action, first run `squeue -w <node>` from the login node. Only after confirming Slurm ownership may `nvidia-smi` be checked over SSH. Never launch through SSH on a node allocated to another user, even if individual GPUs look idle; Slurm exclusivity does not protect jobs from SSH-launched processes.

## Optional Raw Motion Array Cache (Yonsei and Grasp)

`motion_expert_joint_attention/motion_array_cache.py` provides an opt-in I/O
optimization shared by the Nymeria aligned/native loader and the BONES pairs
loader, including Edge Phase 2. It is disabled unless
`COSMOS_MOTION_CACHE_ROOT` is set. Existing launchers do not enable it, and
running trainers must not be restarted merely to adopt this optimization.

The offline builder extracts `features.npy` and `neutral_joints.npy` from each
source NPZ without changing their bytes, dtype, layout, or precision. Training
maps these arrays read-only, slices the same source window, and returns the
same writable float32 copies as before. Grounding, calibrated `off`, frame-zero
canonicalization, z-score stats, padding, random crops, caption policies,
bad-window retries, and task/source sampling are unchanged. Native Nymeria T2M
still uses `_t2m_index`; aligned TI2M still uses `_index`. BONES remains T2M-only.
The cache does not change image decoding, reasoner inputs, losses, or samplers.

Cache keys include the logical source path relative to the selected site's
`WEKA_ROOT`, source size and nanosecond mtime, and cache schema version. There
are no hardcoded Yonsei or Grasp paths in the cache implementation. Logical
paths are not symlink-resolved, so a WEKA-to-scratch symlink retains its source
identity. `COSMOS_MOTION_CACHE_SOURCE_ROOT` can explicitly override the logical
source prefix; use the same override when building and training. Copied data
whose timestamps differ requires rebuilding its entries on the destination
site. Source datasets must remain immutable: size/mtime guards are not
cryptographic content verification and cannot detect a same-size edit with a
deliberately restored timestamp.

Published entries are immutable and become visible through an atomic directory
rename. Training never creates or repairs entries. Missing, stale, unreadable,
or header-invalid entries fall back to the original NPZ path with a bounded
per-process notice, rather than dropping samples or changing the mixture.
`COSMOS_MOTION_CACHE_MAX_OPEN` defaults to 64 mapped sequences per process;
workers lazily maintain separate bounded handle caches. Do not overwrite or
delete mapped cache files while a trainer is using them.

Build on a compute allocation after checking the destination's quota and free
space. The builder is CPU-only and resumable, but uncompressed arrays may need
substantially more disk space than their NPZ sources. Its filesystem free-space
guard does not account for per-user Lustre soft quotas. A scratch or node-local
cache is disposable; preserve the original source datasets and follow the
site's scratch-retention policy. Memory mapping on Lustre still incurs remote
I/O, so throughput gains require measurement, not assumption.

Example, from the site checkout with the matching `COSMOS_SITE` selected:

```bash
source ./restored_env.sh
# Set this explicitly to a cache destination with sufficient space on this site.
export COSMOS_MOTION_CACHE_ROOT=/absolute/site/cache/uniego_arrays
"${COSMOS_PYTHON}" -m motion_expert_joint_attention.motion_array_cache \
  --tree "${WEKA_ROOT}/nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1" \
  --pairs-jsonl "${RUN_ROOT}/joint_attention/bones_pairs_train.jsonl" \
  --pairs-jsonl "${RUN_ROOT}/joint_attention/bones_pairs_val.jsonl"
```

The same commands apply to Yonsei (`COSMOS_SITE=yonsei`) and Grasp
(`COSMOS_SITE=grasp`); their existing site selector supplies different physical
roots. Build the representation and manifests used by the intended run, not a
legacy UniEgo corpus. Enable `COSMOS_MOTION_CACHE_ROOT` only in a future training
launch after the cache is built. Leave worker count and all training settings
unchanged for the initial cache comparison. Unset the variable to restore the
original NPZ path. Implementation does not imply a populated cache or a tested
speedup; neither a cache build nor a benchmark was run when adding this code.

## Server Migration

The physical storage map and readiness ledger for the Yonsei datacenter and
Grasp lab is `migration/SITE_STORAGE.md`.

The authoritative restore runbook is `migration/SERVER_MIGRATION.md`.
Machine-readable contracts are `migration/GCS_ARTIFACT_MANIFEST.tsv` and
`migration/DRIVE_ARTIFACT_MANIFEST.tsv`. Cloud roots are
`gs://mm-jinhyung_kim/jungbin_cho` and rclone remote `data:`. Verify with:

```bash
python migration/verify_migration.py local
python migration/verify_migration.py gcs
python migration/verify_drive.py --include-trees
```

The current Phase-2/3 motion normalization is the Git-tracked
Nymeria-grounded `uniego283_{mean,std}.npy`, not BONES source stats or the C45
190-D evaluator stats. `motion_expert/pairs_{train,val}.jsonl` are ignored by
Git and restore from Drive through `migration/restore_from_drive.sh data`.
Standalone training visualizations and intermediate checkpoints are not
required; selected checkpoints, exact splits, mean/std arrays, calibrations,
evaluator models, and regenerable-cache contracts are enumerated in the
migration runbook.

## Edge Phase-2 Absolute Decoded-Joint Ablation (2026-09-11)

`motion_expert_t2m_edge/train.py --joint-loss-mode absolute` replaces the
centroid-relative decoded-joint term with valid-frame MSE between the directly
decoded prediction and target. Both are unnormalized with the selected motion
mean/std before decoding. There is no centroid, root, or Procrustes alignment
in this joint term; "absolute" refers to the representation's decoded canonical
coordinate frame, not external GPS/world coordinates. Padding remains masked.
Feature reconstruction, decoded joint-difference smoothness, and all three
contact-aware terms are unchanged. The historical default is `centered`.
The selected mode is stored in args.json, W&B config, and checkpoint args;
resuming across joint-loss modes is rejected. Older checkpoints without this
field are treated as `centered`.

The shared `sbatch_train_1gpu.sh` accepts `JOINT_LOSS_MODE=absolute` on either
Grasp or Yonsei. Site selection and cache locations remain independent of this
loss option. Do not switch a running centered-loss experiment to absolute loss
in place; use a separate output directory.

Yonsei's new launcher is
`motion_expert_t2m_edge/train_contact_absolute_yonsei.sh`. It starts a fresh
200k-step experiment in
`$RUN_ROOT/motion_expert_t2m_edge/edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_absjoint_w1_1_5_contact_c0p05_v1_h10_yonsei_20260911`.
It uses recanonicalized `uniego_rep_camhead_v1` Nymeria and the pinned
camera/head-aligned `clean_calibrated_uniego283_{mean,std}.npy` statistics, as
explicitly selected by the user. BONES remains motion-only UniEgo; using these
statistics does not make it camera/head-equivalent. BONES is 50% within T2M:
37.5% BONES T2M, 37.5% Nymeria T2M, and 25% Nymeria TI2M overall.

The loss is feature + absolute_joint + 5*smooth + 0.05*contact + foot_velocity
+ 10*foot_height. Batch size 128, eight workers, 1k-step LR warmup to 2e-4,
200k-step cosine schedule to 2e-5, subject wording, fixed visualizations every
5k, permanent checkpoints every 5k, and recovery checkpoints every 250 remain
unchanged. Raw motion-array caching and the parallel-read/background-write
image/conditioning caches stay enabled on Yonsei scratch. This is a new data,
statistics, and joint-loss experiment relative to the ongoing legacy/Nano-stats
run, not a single-factor comparison with that run.

Allocation request: `2204981`, one RTX 4090, eight CPUs, 64 GiB RAM, three days,
excluding `cs-gpu-01` and `node33`. Tmux session:
`phase2_edge_absjoint_4090_yonsei_20260911`; its trainer window waits for the
allocation and starts the launcher through srun, retaining the allocation after
exit. At implementation time this request was pending, not confirmed training.
The existing legacy/centered run in allocation `2184232` was not restarted.

## Edge Phase-2 Global Feature Weight Ablation (2026-09-12)

`train.py --w-global-rot 10 --w-global-trans 10` adds ten to the per-coordinate
feature weights for normalized canonical-delta rotation `[270:276]` and
translation `[276:279]`. Both extra sums retain the full 283-D feature-MSE
denominator and valid-frame mask. With `w_feat=1`, these nine coordinates
therefore have weight 11, while local-joint and contact feature coordinates
retain weight 1. These are representation deltas, not an additional decoded
world-position or geodesic rotation loss. The rotation and translation loss
components are logged separately with the full-feature denominator.

Both new weights default to zero, preserving previous experiments. They are
stored in args.json, W&B config, and checkpoint args. Resuming with different
global feature weights is rejected; older checkpoint args imply zero extra
weight. The shared one-GPU launcher accepts `W_GLOBAL_ROT` and
`W_GLOBAL_TRANS` on both Yonsei and Grasp.

Yonsei launcher:
`motion_expert_t2m_edge/train_contact_globalfeat10_yonsei.sh`.
Its separate output directory is
`$RUN_ROOT/motion_expert_t2m_edge/edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_globalfeat10_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912`.
It reproduces the September 8 `camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10`
run, including the centered decoded-joint loss, recanonicalized Nymeria,
pinned camera/head-aligned mean/std, BONES 50% within T2M, batch 128, eight
workers, 200k steps, 1k warmup to 2e-4 and cosine decay to 2e-5, contact weights
0.05/1/10, and the fixed five-per-task UniPC-35/CFG-2 visualization every 5k.
It is not the absolute-joint ablation. The representation-preserving motion,
image, and frozen-condition caches remain enabled on Yonsei scratch.

New retained allocation: `2212228`, `node31`, one RTX 4090, eight CPUs,
64 GiB RAM, three days, excluding `cs-gpu-01,node33`. Tmux:
`phase2_edge_globalfeat10_4090_yonsei_20260912`. Operations/logs:
`/scratch2/whwjdqls99/cosmos/operations/phase2_globalfeat10_camhead_20260912`.
The allocation is acquired; training launch is separate from this record.
Existing allocations and runs are left untouched.

## Edge Phase-2 Interleaved Motion Self-Attention Ablation (2026-09-12)

`EDGE_MOTION_SELF_ATTENTION_PER_GAP=1` adds one fresh motion-only Transformer
block after each of the first six joint-attention motion blocks. The joint
blocks remain at reasoner layers `[3,7,11,15,19,23,27]`; new blocks follow
`[3,7,11,15,19,23]`, giving seven joint-attention and six self-attention blocks.
There is no extra block before the first or after the final joint block.
The added blocks use width 2048, 16 Q / 8 KV heads, head dimension 128,
fresh Q/K/V/O and RMS norms, and a width-3072 SwiGLU FFN. Their bidirectional
attention reads only each sample's shape token and valid motion frames,
with packed per-sample offsets and the existing motion mRoPE coordinates.
They never read or update reasoner rows; the frozen reasoner and vision path
remain unchanged. Non-reentrant activation checkpointing recomputes only the
new blocks during training to preserve batch 128 on a 24-GiB RTX 4090.

The setting defaults to zero on both Yonsei and Grasp. Checkpoint contracts
record the new block count and positions; older schema-3 checkpoints default
to zero added blocks and are rejected under the expanded architecture.
Inference must select the same environment setting as training.
No changes to the Nano model or Phase 1 are involved.

Yonsei launcher: `motion_expert_t2m_edge/train_contact_selfattn_yonsei.sh`.
Fresh run: `$RUN_ROOT/motion_expert_t2m_edge/edge_7ja6sa_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912`.
It matches the September 8 camera/head-aligned baseline's data/statistics,
BONES fraction 0.5 within T2M, subject policy, 0.75/0.25 task mixture, batch
128, 200k updates, LR 2e-4 with 1k warmup/cosine to 2e-5, centered decoded
joint loss, weights 1/1/5 plus contact 0.05/1/10, and fixed visualization.
Absolute-joint loss and extra global-feature weights are disabled.
The exact baseline fixed-sample inputs are copied into the new run;
raw-array/image/frozen-condition caches use the existing Yonsei scratch paths.
The production launch targets retained allocation `2213496` on `node32`,
one RTX 4090, eight CPUs, 64 GiB, previously used for sampler visualization.
Tmux session: `phase2_edge_sampler_viz_4090_yonsei_20260912`,
window `selfattn_train`. Operations/logs: `/scratch2/whwjdqls99/cosmos/operations/phase2_7ja6sa_camhead_20260912_job2213496`.
This is launch configuration, not a claim that training has completed.

## Edge Phase-2 BONES-Only Text-to-Motion Ablation (2026-09-12)

The shared one-GPU launcher accepts `TASK_WEIGHTS` as JSON, retaining the
historical `{"text2motion":0.75,"textimg2motion":0.25}` default. Selecting
`TASK_WEIGHTS='{"text2motion":1.0}'` and `BONES_FRAC=1` routes every training
example to the existing BONES-SEED pair loader. The Edge dataset already
rejects an unavailable BONES source, and the trainer additionally rejects
any non-BONES, non-T2M, or image-bearing row under this configuration.
The shared adapter still constructs an unused Nymeria index at initialization;
this does not mean Nymeria examples enter the optimizer batches.

`train.py --viz-source bones` (shared launcher `VIZ_SOURCE=bones`) uses
fixed, caption-diverse BONES validation T2M examples, selected with no CFG
dropout and materialized with their exact motion, skeleton, caption and
sampling seed. It does not visualize an untrained image-conditioned task.
The default `VIZ_SOURCE=nymeria` preserves the previous five-per-task
Nymeria test-suite contract, identities, and sampler. All existing launchers
keep that default unless they explicitly opt in. The source and split are
recorded in the fixed manifest; changing an existing suite is rejected.

Yonsei launcher: `motion_expert_t2m_edge/train_contact_bonesonly_yonsei.sh`.
Fresh run: `$RUN_ROOT/motion_expert_t2m_edge/edge_7layer_bones100_t2m_camheadstats_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912`.
It matches the September 8 baseline's seven joint-attention blocks, 230.51M
trainable parameters, camera/head-aligned mean/std, subject wording policy,
batch 128, T200, 200k steps, 2e-4 peak LR / 1k warmup / cosine to 2e-5,
centered decoded-joint loss and weights 1/1/5 plus contact 0.05/1/10.
There are no added self-attention blocks, no absolute-joint term and no extra
global-feature weights. BONES remains its original motion-only UniEgo data;
this experiment neither recanonicalizes BONES nor computes BONES-only stats.
Checkpoint interval remains 5k, rolling recovery 250, visualization at start
and every 5k with UniPC-35 / CFG-2 / shift-3.

Allocation `2219279`: RTX 4090 on `node36`, eight CPUs, 64 GiB, three days.
Tmux `phase2_edge_bonesonly_4090_yonsei_20260912`; operations/logs
`/scratch2/whwjdqls99/cosmos/operations/phase2_bonesonly_camheadstats_20260912_job2219279`. Training launch and progress are separate from this configuration
record. Existing training jobs are not restarted.

## Edge Phase-2 Batch-256 / Zero-LR-Floor Ablation (2026-09-13)

The Nano contact-aware launcher uses eight ranks at batch 32 (effective batch
256). Its `motion_expert_joint_attention/train.py:lr_factor` decays cosine
to zero; the `min_lr_ratio=0.1` value in that module's configuration is not
used by this trainer. The Edge trainer instead has an explicit
`--min-lr-ratio` argument, historically defaulting to 0.1.

The shared Edge launcher now exposes `MIN_LR_RATIO`, retaining 0.1 unless
explicitly overridden. This remains portable between Grasp and Yonsei and
does not alter existing running processes or the defaults of other launchers.

Yonsei launcher: `motion_expert_t2m_edge/train_contact_bs256_lr0_yonsei.sh`.
Fresh run: `$RUN_ROOT/motion_expert_t2m_edge/edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_bs256_lr0_w1_1_5_contact_c0p05_v1_h10_yonsei_20260913`.
The only requested training-recipe changes from the September 8 camhead
baseline are actual per-GPU batch 256 (one GPU, accumulation 1) and a zero
cosine LR floor. Peak LR remains 2e-4, warmup 1k updates, total 200k updates,
seed 0, and all loss weights remain centered joint 1/1/5 with contact
0.05/1/10. The model is the original seven joint-attention layers, without
added self-attention or extra global-feature weights. Data remains 75% T2M
with 50% BONES within T2M, plus 25% Nymeria TI2M; camera/head-aligned
Nymeria data and its mean/std, original BONES motion-only representation,
subject policy, caches, T200/TI2M97 and 256px reasoner images are unchanged.

The exact baseline fixed visualization cohort is copied to the new run.
Visualization remains UniPC-35 / CFG-2 / shift-3 at start and every 5k;
scheduled checkpoints remain every 5k, rolling recovery every 250.
This is a fresh experiment, not a resume of the September 8 weights.

Retained tmux: `phase2_edge_bs256_lr0_4090_yonsei_20260913`.
Operations/logs: `/scratch2/whwjdqls99/cosmos/operations/phase2_bs256_lr0_camhead_20260913`.
The allocation wrapper records its actual job/node in `allocation.txt`,
starts production training once allocated, and retains a shell on trainer
exit. Request one RTX 4090, eight CPUs, 64 GiB, three days, excluding
`cs-gpu-01,node31,node33`. This section records configuration, not an
assertion that the job is allocated or optimizer updates have started.

### Two-GPU DDP Replacement for the Batch-256 Attempt (2026-09-13)

The physical batch-256 attempt in job `2224956` on `node40` completed
its initial ten visualizations but failed on its first training forward with
CUDA OOM: 23.45 GiB process usage on a 23.52 GiB RTX 4090. No optimizer
updates completed. At the user's explicit request, cancellation of that
retained one-GPU allocation was issued to free the GPU/QoS budget. Its
logs and output remain as failure provenance, not a trained checkpoint.

The replacement is a fresh run with two GPUs on one node, per-GPU batch
128, accumulation 1, effective batch 256, 200k optimizer updates, peak LR
2e-4, 1k warmup, and cosine decay to zero. It uses the same September 8
seven-layer architecture, mixed Nymeria/BONES data, camhead statistics,
centered losses 1/1/5 + contact 0.05/1/10, and exact fixed visualization
cohort. This is not the proposed batch-182/300k alternative.

The shared launcher accepts `NPROC_PER_NODE` (default 1). Values above one
invoke `torch.distributed.run --standalone --nnodes=1` before the existing
trainer, which already implements DDP and a distributed data sampler.
This option is site-independent; request matching GPUs/CPUs on a single
node and use one Slurm task, not one Slurm task per torchrun rank.
Existing one-GPU launchers retain their prior behavior.

Yonsei launcher: `motion_expert_t2m_edge/train_contact_ddp2_bs256_lr0_yonsei.sh`.
Run: `$RUN_ROOT/motion_expert_t2m_edge/edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_ddp2_bs128_gbs256_lr0_w1_1_5_contact_c0p05_v1_h10_yonsei_20260913`.
Retained tmux: `phase2_edge_ddp2_bs256_lr0_yonsei_20260913`.
Operations/logs: `/scratch2/whwjdqls99/cosmos/operations/phase2_ddp2_bs256_lr0_camhead_20260913`.
The request is two RTX 4090s, 16 CPUs (eight workers per rank), 128 GiB,
three days, excluding `cs-gpu-01,node31,node33`. The wrapper starts the
trainer when allocated, records job/node in `allocation.txt`, and retains
the allocation after exit. This is launch configuration, not confirmation
of allocation or successful optimizer steps. Other training jobs are unchanged.

### Nano-reference Phase-2 parity audit (Yonsei, 2026-09-13)

The user requested a thorough Edge/Nano comparison and fresh 200k experiments:
one GPU with batch 128, and two GPUs with batch 128/rank (global 256), both
peak LR 2e-4, 1k warmup, cosine floor zero. Keep the camera/head-aligned
Nymeria representation and its recomputed statistics, 75/25 T2M/TI2M,
50% BONES within T2M, and the September-8 fixed visualization cohort.

Evidence and provenance:

- The successful Nano contact checkpoint at step 200000 records AdamW
  betas 0.9/0.99, epsilon 1e-8, weight decay zero, fused optimizer, LR
  1.2461309761846451e-14 on its final update, batch 32/rank and DDP.
  Its launcher used eight ranks, giving global batch 256. Edge baseline90k
  and pre-audit DDP55k checkpoints instead record beta2=0.95.
- Nano has 132 BF16 model tensors and 17 FP32 tensors; optimizer moment
  dtypes match. Source inspection explains this: transformer motion blocks
  are BF16, but all motion heads plus the final norm remain FP32. Edge has
  90 BF16 and four FP32 tensors, retaining only the timestep MLP in FP32.
  Its old heads.to(BF16).time_embedder.to(FP32) also irreversibly rounds the
  non-persistent sinusoidal-frequency buffer. The new FP32 I/O path avoids
  this round trip entirely, not merely a later cast of rounded values.
- The checked-in Nano trainer reduces each objective component separately
  within each task and weights it by count(task)/local_batch, followed by
  rank-mean gradient reduction. Old Edge pools all valid frames and contact
  denominators across tasks. With different clip lengths these are not the
  same objective. DDP rank-local task means are not a single global
  task-normalized mean; the new path deliberately follows Nano's local
  reduction and then DDP averaging.
- Important historical uncertainty: git commit f81b1aa (2026-07-20) added
  per-task reduction together with contact objectives and later Phase-3
  features. Earlier committed b2c307f used pooled reconstruction losses.
  The successful contact checkpoint has contact arguments but lacks the
  later head-camera arguments from that same commit, so it came from an
  uncommitted intermediate version. No original source snapshot or code
  SHA is present in the imported run. We cannot prove its precise loss
  reduction from current code or Git. The new runs explicitly follow the
  current Nano per-task reference; do not describe this as a proven exact
  reconstruction of the historical successful training.
- Nano's loss-spike guard already exists in b2c307f: above max(50,25*EMA),
  zero that rank's contribution without dropping a distributed backward;
  EMA decay is 0.99. Non-finite optimizer updates are skipped coherently.
  Old Edge had no finite-loss spike guard and aborted on non-finite values.
- Nano does not explicitly enable TF32 in its trainer/root import. The new
  runs explicitly disable TF32 for FP32 matmuls, matching PyTorch's default;
  the historical job's environment-level overrides cannot be recovered.
- Old Edge checkpoint RNG state is rank-0-only and gets restored to every
  DDP rank. New guarded runs collect and restore each rank's own RNG state
  and synchronize signal-triggered checkpoint decisions. DataLoader worker
  state/in-epoch cursor are still not checkpointed: resumption is not a
  bitwise replay of uninterrupted sampling.

Compared and retained:

- Both use native shifted logit-normal x0 training with shift 3, noise
  x_sigma=(1-sigma)*x0+sigma*noise, no noise/loss on padded frames, and true
  normalized sigma at the timestep MLP. Nano divides by 1e-3 before its
  heads multiply by 1e-3; Edge passes normalized sigma directly.
- Motion I/O structure, zero-initialized output, shape token from centered
  neutral joints, additive time bias, fresh 0.02-normal transformer
  projections, width-3072 SwiGLU, causal frozen reasoner and full motion
  attention have the same role-level design. Edge necessarily retains its
  Nemotron geometry, norm family, cross-path K norm, seven joint blocks,
  native generation text prefix, and image-position offset convention.
  Nano has 12 blocks at width4096 versus Edge seven at width2048; these are
  NOT parameter-identical models. Nano raw-text/EOS-only tokenization and
  token-count TI2M motion origin differ from Edge's native prefix and next
  rotary position. They remain explicit backbone-interface differences,
  not demonstrated causes of floating.
- The same shared decoder/contact helper implements centered joint MSE,
  frame-difference MSE, contact BCE with logit scale2, XZ contact-foot
  velocity at20FPS, and raw local-joint foot-Y reconstruction. Weights stay
  1/1/5 plus .05/1/10. Global extra weights stay zero; no absolute-joint or
  added-self-attention ablation is enabled in the parity runs.
- Both route through NymeriaJointDataset: calibrated grounding, frame-zero
  canonicalization, z-scoring, native-caption-span T2M, 97-frame aligned
  TI2M, BONES motion-only crops up to200, padding masks, and no image
  dropout in TI2M CFG. Edge explicitly uses camhead Nymeria and applies its
  statistics to BONES too. BONES remains legacy head-defined motion and is
  not camera/head-equivalent. Subject normalization stays enabled.
- Shared BONES loading applies caption dropout twice (inner source and
  outer adapter), approximately19% at nominal10%, versus10% for Nymeria.
  This exists in the Nano and Edge paths and is deliberately not changed
  silently during this parity comparison. It is a separate possible
  source-conditioning ablation, not an Edge-only discrepancy.
- UniPC wrappers agree on35steps, CFG2, shift3, native integer model
  timesteps, full-precision integration sigma, velocity=(x-x0_hat)/sigma,
  and the same image/shape retained in the null-text branch. No re-noising,
  centroid removal, or artificial grounding is applied to saved generated
  joints. This is static inspection, not a numerical solver equivalence test.

Implementation/launch contract:

- Old defaults remain pooled loss, beta2=.95, strict non-finite handling,
  BF16 I/O and TF32 enabled. Old checkpoints default to that precision
  contract; loading them under FP32 I/O is rejected instead of silently
  reinterpreting an ablation. Resume also rejects optimizer/loss-policy drift.
- Fresh runs set EDGE_MOTION_IO_DTYPE=float32, ADAM_BETA2=.99,
  LOSS_REDUCTION=per_task, LOSS_GUARD=nano, ALLOW_TF32=0, MIN_LR_RATIO=0.
  All contact components, guarded-batch counts and skipped updates are now
  logged. These options work through the shared launcher on Grasp or Yonsei;
  no data/statistics or frozen framework files are changed.
- Yonsei launcher: motion_expert_t2m_edge/train_contact_nanoparity_yonsei.sh.
  Set NPROC_PER_NODE=1 or2 inside the appropriate retained GPU allocation.
  Output names are
  edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_nanoparity_gbs128_lr0_w1_1_5_contact_c0p05_v1_h10_yonsei_20260913
  and the same name with gbs256. Both are fresh output directories; neither
  resumes an older ablation. Subsequent restarts use their own recovery.
- Manual inference must also select EDGE_MOTION_IO_DTYPE=float32 for these
  checkpoints. Their saved architecture contract records this requirement.
- Retain per-user queue policy: PRO6000 requests2194983 and2194977 were put
  on user hold before freeing4090 capacity. Exclude cs-gpu-01,node31,node33
  from new4090 requests. Do not cancel unrelated active training.
- At implementation time, focused numerical/gradient tests were awaiting
  user approval and have not been run. Allocation/startup/progress evidence
  must be reported separately; implementation alone is not a running run,
  and none of these findings proves the cause of floating/drift.

## Shared quantitative metric reporting policy (2026-09-14)

For every quantitative evaluation report on Yonsei or Grasp, always report the
corresponding ground-truth (GT) metrics alongside each generated-model result.
Use the same eligible cases, sequence/window coverage, lengths, evaluator,
shape conditioning, frame rate, and physical-metric conventions for both.
Measure GT retrieval (including R@3), foot skating, and floor penetration;
do not assume GT retrieval is 100% or GT physical errors are zero. Report
GT-to-itself FID as 0 and explicitly label it as the reference baseline.
When a requested metric is unavailable, mark it unavailable for the affected
row rather than silently dropping GT or substituting a fabricated value.

## Edge Phase-2 normalization ablations (2026-09-15)

The user requested four fresh one-4090 runs: two independent normalization
ablations, each with the parity mixed Nymeria/BONES recipe and with BONES-only
T2M. Do not combine both normalization changes in these comparisons.

- Ablation 1: EDGE_MOTION_NORM_DTYPE=float32 keeps trainable motion-block
  input/post-attention/Q/K RMSNorm gains in FP32. Activation boundaries remain
  BF16; ordinary AdamW creates FP32 moment storage for these parameters.
- Ablation 2: EDGE_REASONER_K_NORM_TRAINABLE=1 gives each motion joint block
  its own trainable FP32 copy of the existing frozen reasoner-to-motion K norm.
  Initial gains and epsilon are copied without consuming initialization RNG.
  This replaces that branch's frozen norm, not an extra normalization. Causal
  reasoner Q/K/V, all frozen backbone weights, and ordinary motion norms remain
  unchanged. Seven 128-value copies add 896 trainable parameters.
- Both options default to legacy behavior on Yonsei and Grasp. Architecture
  contracts record them; old checkpoints imply BF16 motion norms and a frozen
  reasoner K norm. Cross-ablation resume/inference is rejected rather than
  silently loading weights under a different interpretation.
- Shared launcher: motion_expert_t2m_edge/train_norm_ablation.sh. Arguments are
  motion_fp32 or reasoner_k_fp32, followed by mixed or bones. Select COSMOS_SITE
  and run inside an owned GPU allocation; never on a login node or cs-gpu-01.
- All four use batch128, accumulation1, workers8, 200k updates, LR2e-4,
  warmup1000, cosine floor0, Adam betas0.9/0.99, FP32 I/O, per-task reduction,
  Nano guards, seed0, and centered losses1/1/5 plus contact0.05/1/10.
- Mixed runs retain 75% T2M (50% BONES within T2M) and 25% Nymeria TI2M.
  BONES-only runs use 100% BONES T2M. All keep camera/head-aligned statistics;
  BONES remains original motion-only UniEgo, not camera-recanonicalized data.
- Outputs on Yonsei: /scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/
  motion_expert_t2m_edge/edge_7layer_<variant>_<cohort>_nanoparity_gbs128_...
  Data/statistics/pretrained checkpoints stay on Lustre. Grasp defaults to its
  normal RUN_ROOT; NORM_ABLATION_OUTPUT_ROOT can override output placement.
- Save only positive multiples of 10k. --checkpoint-interval-only together
  with --recovery-save-every 0 suppresses rolling/signal/off-interval writes.
  Up to 10k progress may be lost on interruption. Existing runs keep their
  prior checkpoint policy. Visualization remains at start/every5k, with the
  existing fixed Nymeria or BONES suite and generated/GT comparisons.
- Console/W&B log actual FP32 gain movement relative to the current launch.
  This is an experiment, not a confirmed fix for floating. No standalone
  smoke/numerical tests were requested or run for this change.

New requests use suma_rtx4090, one GPU/eight CPUs/64GiB/three days each,
excluding cs-gpu-01,node31,node33. Tmux sessions are phase2_edge_n1_mix_20260915,
phase2_edge_n1_bones_20260915, phase2_edge_n2_mix_20260915, and
phase2_edge_n2_bones_20260915. Their launchers retain allocations after exit.
Existing training and pending PRO6000 holds/dependencies are not changed.
Scheduling and actual optimizer progress must be reported from live output,
not inferred from this configuration record.

### Ablation numbering correction and as-is K/V experiment #3 (2026-09-15)

The previously launched #1 is FP32 motion-norm parameter storage, NOT the
user's intended as-is reasoner-attention experiment. Its results must not
be presented as a test of bypassing the reasoner-to-motion K normalization.
The user requested retaining #1/#2 and adding the intended path as #3.

- #3 uses EDGE_REASONER_K_NORM_BYPASS=1. Motion queries consume exactly the
  native reasoner K after its existing native normalization and RoPE, and
  the unchanged reasoner V. The extra generator-specific/interface K norm
  is not called; native K is not rotated a second time. Reasoner Q and the
  entire frozen causal reasoner computation remain unchanged.
- Motion Q/K/input/post norms stay at baseline BF16 parameter storage, with
  FP32 I/O heads as in the parity baseline. No extra layers, parameters,
  projections, or trainable interface norm are introduced.
- The bypass defaults to false on Yonsei and Grasp. Checkpoint contracts
  record reasoner_k_norm_bypass; old checkpoints imply false. Combining the
  bypass with EDGE_REASONER_K_NORM_TRAINABLE=1 is rejected, and checkpoints
  cannot silently cross between baseline/#1/#2 and #3 at resume/inference.
- Shared launcher: bash motion_expert_t2m_edge/train_norm_ablation.sh
  reasoner_kv_as_is mixed (or bones for an explicitly requested BONES-only
  variant). The current request starts only the mixed run on the idle
  RTX4090 in retained job2250537, node39; #1/#2 and other jobs are untouched.
- Same parity recipe: batch128/accum1/workers8, 200k target, seed0, LR2e-4,
  warmup1000/cosine floor0, camera/head-aligned data/statistics, 75% T2M
  with 50% BONES within T2M, 25% Nymeria TI2M, and the existing centered
  1/1/5 plus contact0.05/1/10 losses. No #1/#2 weights are used to initialize.
- Yonsei output is on Scratch2 under motion_expert_t2m_edge with the
  edge_7layer_reasoner_kv_as_is_nymeria_bones50_t2m_ti2m_... run prefix.
  Save only positive 10k multiples; no rolling/signal checkpoint writes.
  Existing fixed visualizations remain at start/every5k. Tmux session:
  phase2_edge_n3_mix_20260915. Retain the allocation after exit.
- This is an architectural ablation, not a confirmed floating fix.
  No standalone tests were requested or run.

## Edge Phase-2 learned K/V projection ablation A (2026-09-16)

The user requested replacing both FP32 motion-norm runs (#1, mixed and BONES)
with fresh learned-K/V projection runs. Existing learned-K-norm (#2) and
native-as-is K/V (#3) experiments are not stopped or repurposed. In particular,
#3 means native-as-is K/V, not the learned-K-norm experiment.

- A projects the reasoner's existing pre-RoPE K projection and existing V,
  not the reasoner hidden state. Every one of the seven motion joint blocks
  owns two independent bias-free 1024x1024 linear maps over flattened 8x128
  KV heads. They are identity initialized without consuming baseline model
  initialization RNG. The projections add 14,680,064 trainable parameters.
- Projected K passes through a trainable copy of the original interface
  RMSNorm, then native RoPE exactly once. Its 896 norm gains use FP32 storage,
  as in #2; the ordinary motion-block norms remain BF16. Projected V has no
  extra normalization. These tensors are consumed only by motion attention.
  Frozen causal reasoner Q/K/V and weights are unchanged; motion still uses
  joint attention over reasoner plus shape/motion tokens, not cross-attention.
- Opt in with EDGE_REASONER_KV_PROJECTION=linear_identity_v1 and
  EDGE_REASONER_K_NORM_TRAINABLE=1. Default projection is none, retaining
  existing Yonsei and Grasp behavior. Contracts record the projection kind;
  older checkpoints imply none. Projected and unprojected checkpoints cannot
  silently cross-load. Projection plus native-K bypass is rejected.
- Launcher: bash motion_expert_t2m_edge/train_norm_ablation.sh
  reasoner_kv_project mixed (or bones), with COSMOS_SITE selected on a GPU
  compute node. New output names use edge_7layer_reasoner_kv_project_... and
  date 20260916. Yonsei outputs stay on Scratch2; Grasp keeps RUN_ROOT defaults.
- Both runs start fresh, not from the stopped FP32 models. They keep batch128,
  accumulation1, workers8, 200k target, LR2e-4, warmup1000, cosine floor0,
  seed0, camera/head statistics, existing centered 1/1/5 plus contact0.05/1/10
  losses, and the same source proportions and fixed visualization suites.
  Save only positive 10k multiples; no off-interval or rolling checkpoints.
- Reuse retained jobs2248639 (mixed) and2248640 (BONES) on node40, one4090 and
  eight CPUs each. Stop only the old trainer processes, not the allocations;
  old run files/checkpoints are preserved. New tmux sessions are
  phase2_edge_kvprojA_mix_20260916 and phase2_edge_kvprojA_bones_20260916.
- No standalone tests were requested. This records the experiment contract,
  not validated equivalence, observed optimizer progress, or a floating fix.
