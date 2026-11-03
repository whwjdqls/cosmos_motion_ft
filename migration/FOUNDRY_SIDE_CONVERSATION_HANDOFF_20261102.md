# Foundary side-conversation handoff to the main agent

This document consolidates the side conversation's code, runs, downloads and
evaluation results. It is a handoff, not an instruction to launch more work.
The documentation audit was on **sep-1,2026-11-02 22:17UTC**. Read live status
before acting; old launch receipts and older `AGENTS_ALL.md` entries describe
historical allocations, not current jobs.

The adjacent [audit snapshot](FOUNDRY_SIDE_CONVERSATION_HANDOFF_20261102.snapshot.json)
records actual processes, metrics, recipes, source hashes and checkpoint pointers.
All three current runs were alive with finite losses/gradients. Both new packages'
Python files in the main checkout matched their active pinned source snapshots.

## Publication update (2026-11-03)

The user requested committing the local work, merging it into master and
pushing it to GitHub. The camera/scratch packages and accompanying local
documentation are included in that integration. The Git status and snapshot
below record the earlier November 2 audit, before publication. Active trainers
continue using their existing pinned source exports.

## Where work lives and Git status

- Host: `lg-gwangmo-4xh200-sep-1`; checkout `/home/jinhyung.kim/cosmos_motion_ft`.
- Branch `foundary/phase2-no-shape-20261101`; HEAD
  `8fb2824d2fa75587098081e023b2fc839f6340be`, merging published master
  `d267a56225eac342733d7f76287e86cd6237167c`.
- New camera and scratch packages and their migration notes are **uncommitted**.
  They are not on GitHub or automatically present on sep-2. No new worktree was
  created. Active jobs execute pinned local source exports, not the mutable
  main checkout. This documentation task does not commit, push or restart jobs.
- Preserve pre-existing untracked `migration/FOUNDRY_FILM_SHARED_20261101.md`
  and `nymeria_baselines/VIDEO2MOTION_SUMMARY_20261101.md`; this side did not
  replace those documents or take over the main thread's Notion summarization.
- Current scientific additions are isolated in
  [camera prediction](../motion_expert_phase3_edge_image_joint_camera/README.md)
  and [scratch motion](../motion_expert_phase3_edge_image_joint_camera_scratch/README.md).
  Existing tracked scientific trainer files were not rewritten for these jobs.

Path abbreviations below expand to actual sep-1 local paths:

```text
L = /mnt/local/jinhyung.kim/cosmos
O = /mnt/local/jinhyung.kim/cosmos/operations
R = /mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge
```

These are node-local, not shared with sep-2. Use the node-local Conda Python
`/mnt/local/jinhyung.kim/cosmos/envs/cosmos/bin/python`; plain `python` may be absent.
For either new ablation, source its operation's `environment.sh` for paths/caches,
not another job's environment. The baseline records setup in its `launch.sh`;
read that file without rerunning the launcher against the active output.
Do not print `.site.local.sh`, credentials or the environment wholesale.

## Three current training runs

All target200k updates, global batch32=2x16, accumulation1. Snapshot steps
were36580/4030/2130 respectively; use live metrics for subsequent progress.

| Run | GPUs | Trainable parameters | W&B |
|---|---|---|---|
| Image-only joint baseline |0,4|Seven bridges,117469198|[p3-a60444d889694dfb5e5d](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-a60444d889694dfb5e5d)|
| Camera prediction with pretrained motion |2,3|Same seven bridges,117469198|[p3-b0c2529274be38864cbb](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-b0c2529274be38864cbb)|
| Camera prediction with scratch motion |5,6|Motion318605083 + bridges117469198|[p3-b0122564c04ef74531b6](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-b0122564c04ef74531b6)|

GPU1 has another researcher's vLLM process, despite0% utilization. GPU7 was
empty at audit time. Empty GPUs are not new authorization to launch other jobs.
Do not terminate other users' processes or either existing comparison run.

Exact run directories beneath `R`:

```text
baseline:
edge_p3_p1videoioema200k_imagejoint402020_bidir_clean_t0_T97_gbs32_sep1_20261102

camera prediction:
edge_p3_imagejoint402020_bidir_camera_prediction_nocamloss_T97_gbs32_sep1_20261102_2gpu

scratch motion camera prediction:
edge_p3_imagejoint402020_camera_prediction_scratchmotion_T97_gbs32_sep1_20261102_warmup
```

Exact operation directories beneath `O`, each with pinned `source_snapshot`:

```text
baseline:
phase3_imagejoint402020_bidir_videoio_20261102T020527Z

camera prediction:
phase3_camera_prediction_launch_20261102

scratch motion camera prediction (successful attempt):
phase3_camera_prediction_scratch_motion_20261102/retry_with_production_warmup
```

Use each run's `contract.json`, `sources.json`, `metrics.jsonl`, `status.json`,
`launch/status.json`, `launch/training.log`, `launch/wandb_state/status.json`
and `checkpoints/latest.json`. Its operation's `plan.json`, `RUNNING_VERIFIED.json`
or baseline `LAUNCHED_VERIFIED.json` establish launch provenance, not completion.
The scratch parent operation's `LATEST_ATTEMPT.json` points to the successful
attempt. Do not resume from its failed parent attempt or the earlier dead queue.

### Common recipe and exact specialists

V2M40%, M2V40%, joint20%; T97/20fps,256x256, latent48x25x16x16, motion283D.
Seven bidirectional time-local bridges at3,7,11,15,19,23,27; clean motion uses
E_M(0). V2M caption is empty; M2V uses caption/dropout0.1; joint sees only the
initial image and supplied neutral shape, with empty caption. Motion and video
noise are independent, logitnormal/Waver respectively, training shift3.
Native x0 physical motion losses plus video velocity weight10; clean image
excluded with the original full video denominator. AdamW2e-4, betas0.9/0.99,
weight decay0, global clip1,1000-step linear warmup then cosine to0 at200k.
BF16 computation/norms, FP32 motion I/O/bridges, TF32 off. Same112503 windows,
seed/global draw order, local loader paths,10k saves and latest+100k retention.

All three use this **newer Phase1 LoRA+video-projection EMA200k**, not the
original September four-task Phase1:

```text
/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_2gpu_bs16_lora_video_io_exclude5_fd_invdyn_no_text_p1_200k_sep2_20261028/checkpoints/iter_000200000
metadata SHA256 b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8
```

The first two use selected pretrained Phase2:

```text
/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_self_cross_final_hidden_knorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921/checkpoints/step_000200000.pt
file SHA256 6a8d58d746560c0af3d340f8d711dcfe23ce163bb741912afcffa8ebfdbadc3a
```

Data cache `L/caches/phase3_edge_camhead_T97_v1`, fingerprint
`cddae42b464476489c4a3465130dab8af99b63d16da5a14ce91c34c7af0e15ce`.
Camera-token cache `L/caches/phase3_edge_camera_conditioning_T97_revalidated_20261021`,
fingerprint `e83a61611fbb68679a3c02d4e7d68a614e5cc0e126783b3b0a3fe8be42103a05`.
No training reads use shared storage. No-shape conditioning is inactive here.

### Baseline source and logger details

The baseline's saved implementation hashes match published `d267a56` for
the shared Phase3/image-joint packages, bridge and performance policy. This
was checked against the saved run contract, not inferred from Git history.
Direct inspection of sep-2's live checkout was unavailable; do not claim it
was read. The reference for both new ablations is the saved local baseline
or camera contract, checked field by field before launch.

The original image-joint launcher called a missing `stream_wandb.py` and used
an unsupported `--run-id`. Its trainer was preserved and the supported tracked
`scripts/stream_phase3_wandb.py` runs separately, with deterministic ID above;
all history was backfilled. See baseline operation `LOGGER_REPAIR.json`.
The original `launch/wandb.log` error is historical. Verify the live sidecar
state/API before declaring a current logging failure. Both new launchers use
the corrected sidecar invocation and wait for W&B initialization.

Baseline12 CPU checks included a SHA-verified archived legacy bridge fixture
because historical Git object `f9043d3` was unavailable. The workaround is
inside operation `cpu_checks.py`; it did not rewrite shared scientific tests.
GPU native parity, freeze checks and fresh4/reload2 probes passed.

### Camera prediction: approved behavior and limitations

User explicitly selected **noise-corrupted GT camera** and **camera tokens in
attention**. All three tasks include96 noisy native action/camera transitions,
raw relative SE(3)9 padded64, domain2, with no clean camera tokens. Camera sigma
is motion sigma for V2M, video sigma for M2V/joint. Private per-draw camera RNG
preserves existing motion/video random draws for the same execution topology.
Camera tokens enter native generator attention. Existing bridges retain the
same1600 video and98 motion rows; there are no new direct camera-motion edges.

The native camera output head stays frozen; decoded camera output is detached.
Only motion/video targets are present: **no camera loss**, including no zero
weighted camera term. Motion/video gradients may still influence camera hidden
states through attention/bridges. Noisy GT supplies partial trajectory information;
improvements do not isolate an unused prediction head. Inference starts cameras
from pure noise, with no GT camera; camera accuracy is not guaranteed by training.

Seven CPU checks and exact two-rank/microbatch16 preflight passed, including
all tasks, native specialist parity, finite bridge gradients, frozen specialists,
NaN camera-output loss invariance, zero camera-readout gradient and joint-caption
invariance. Fresh4/reload2 probes passed; production starts fresh at0.
Checkpoint state is bridges+optimizer/RNG; both frozen specialists remain required.
See [camera run note](FOUNDRY_CAMERA_PREDICTION_ABLATION_20261102.md).

The package supports one GPU x16 with accumulation2, or two x16/accumulation1.
The sampler retains the same two16-sample loss groups, not bitwise cross-topology
noise/sums. Actual topology must pass its own preflight and is in the contract.
The initial idle watcher passed single-GPU preflight but stopped at an occupied
GPU check before production. It is dead; operation
`phase3_camera_prediction_idle_queue_20261102/MANUAL_RELAUNCH.json` points here.
The active two-GPU run supersedes that queue and its incomplete output directory.
Launchers wait at most60seconds for process release between stages and still
abort on occupied GPUs. They never terminate or share an occupied GPU.

### Scratch motion: approved extra change and checkpoint format

User explicitly chose **train the fresh motion expert and bridges**. A frozen
fresh decoder would remain identically zero under the original zero output
initialization. The model uses the original Phase2 constructor's fresh motion
weights, including heads, time/shape embeddings, norms and seven motion layers.
Phase1 Cosmos reasoner/generator and all native projections/LoRA stay frozen.

No learned Phase2 checkpoint is a training input. The5639-byte
`O/phase3_camera_prediction_scratch_motion_20261102/architecture_reference.pt`
contains only JSON-derived architecture/normalization metadata and the reference
camera contract hash. Tensor/trained-state fields are rejected. It is also
copied to the active run's `launch/architecture_reference.pt`. Registered motion
constructor source hashes and initial motion/bridge digests are recorded.

All other recipe fields, source modules, data, camera routing and losses match
the camera ablation. Optimizer and global clip1 cover motion+bridges. New
checkpoints include **`motion_state` and `bridges`**, optimizer, scheduler,
per-rank RNG and global draw cursor. Use this package's `sample.py` to restore
both components. Never load it with the pretrained-motion bridges-only loader.

Twelve CPU checks, two-rank complete-state preflight and fresh4/reload2 probes
passed. Motion changed, Cosmos stayed frozen, replicas were identical and
camera loss/readout gradients were absent. The first disposable test inherited
immediate peak LR and failed its overfit check; that failure is preserved.
The successful diagnostic follows the **same production1000-step warmup**, with
loss4.16353→4.15284. Production LR/recipe were not tuned. The output `_warmup`
suffix labels that corrected preflight attempt, not a different schedule.
See [scratch run note](FOUNDRY_SCRATCH_MOTION_CAMERA_ABLATION_20261102.md).

## Completed original200k + trajectory FiLM-only50k experiment

This is a separate completed experiment, not either native-camera-token run.
Operation `O/posttrain_original200k_trajfilm_only_20261101` owns its isolated
`trajfilm_only/` package, launcher, source/recipe contracts and runtime adapters.
It imports historical compatible code at `L/repos/phase3_resume_20261030`.
Main scientific files were not changed to launch it.

It strictly loads the original September Phase3 final200k (no FiLM), freezes
its seven learned bridges and both original specialists, and trains only56 new
trajectory FiLM tensors, **15663872 parameters**, for50000 additional updates.
Tasks are50% camera→motion and50% camera+video→motion, empty captions, supplied
shape. Camera18D goes **only to motion FiLM**; no camera action token enters
Cosmos. Camera-only never runs the generator and supplies no video tokens.
Both-step checks enforce this routing and frozen weights.

Global32=2x16; AdamW2e-4/betas0.9,0.99/no decay/clip1; original1000 warmup then
cosine to0 at50k; original112503-window cohort and native physical motion losses.
It begins a new optimizer/RNG/draw stream, not an optimizer continuation of
the original200k. FiLM initialization is exact identity. Startup parity,
camera-only video-poison invariance, freeze and optimizer recovery passed.

Completed checkpoint,188064051 bytes:

```text
/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_original200k_trajfilm_only_c2m_cv2m_T97_gbs32_50k_sep1_20261101/checkpoints/step_000050000.pt
```

W&B [58178b3c](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/58178b3c).
Operation `status.json` and run `status.json` say complete50000; its earlier
`RUNNING_VERIFIED.json`/README are launch-time evidence, not current progress.
Checkpoint stores **`film_state` only**, optimizer/schedule/RNG/cursor and base
SHA; original base/specialists remain required. Load with
`trajfilm_only.model.EdgePhase3` and `load_film_state_dict`, not an older full
hidden-state/trajectory-FiLM checkpoint loader.

Its original base is specified in operation `plan.json`, extracted beneath
`original_checkpoint/`; SHA256
`3dc642508076c2d4199014a8bae82bbec330fc45d8de23cf02eff75089c3bc1d`.
Its Phase1 is the **original four-task September20260905 EMA200k**, metadata
`0b5a35c3365d2a600f058f8200d8a8df170023daae375acb8a932b83206a4a38`,
not the newer video-projection specialist used by the three current runs.
Original selected Phase2 is the same6a8d58… full-file identity above.

## Completed FiLM-only evaluation:426 predictions and text follow-up

Operation `O/eval_trajfilm_only50k_matched71_20261102T151804Z`, with
`text_followup/`. Authoritative result: `ALL_CONDITIONS_COMPLETE.json` says
`complete_with_historical_baseline_difference`,426 verified predictions.
`comparison.md`, `all_conditions.csv` and `all_conditions_summary.json` contain
the combined report; per-case metrics, predictions, embeddings, retrieval groups,
caption overlay and source/hash receipts are retained. No worker remains active.

Protocol: same71 held-out T97/20fps cases/seeds/shapes, UniPC35/shift3, native
guidance1. Body23 physical scoring, exact TMR evaluator91f9d8, matching stats,
train-only completion63ab10; FID71, R32 groups/seeds0/1/2,64 scored/7 dropped.
Camera enters FiLM only. Caption-conditioned inference is an intervention on
caption-empty-trained Phase3/FiLM; text reaches both applicable reasoners.

| Fresh same-runtime arm | MPJPE cm | FID | R@3 % |
|---|---:|---:|---:|
| Original200k, video |21.2042|0.255292|43.2292|
| Original200k, text+video |16.3278|0.133506|68.7500|
| FiLM-only50k, camera |13.8083|0.485704|37.5000|
| FiLM-only50k, video+camera |10.1866|0.166217|55.2083|
| FiLM-only50k, text+camera |10.8880|0.132272|66.1458|
| FiLM-only50k, text+video+camera |9.8589|0.117289|68.7500|

**Unresolved historical baseline discrepancy:** published original no-text
MPJPE20.5621cm versus fresh21.2042cm. Original GT physical metrics and TMR
embeddings/retrieval match exactly. Unmodified standalone original inference
matches the wrapper bitwise on two cases. Cause is unresolved; no proof of
hardware drift. The strict initial parity failure and its `status.json=failed`
are intentionally preserved. Later combined completion/report retain the caveat;
do not mistake that old status for an unfinished426-output evaluation, erase the
failure or silently replace published baseline values. Paired intervals use the
fresh same-runtime original; camera arms observe additional measured data.

Text improves camera-only MPJPE by2.92cm, paired95% CI[-3.89,-1.97]. Text on
video+camera changes MPJPE by-0.33cm, CI[-1.06,+0.39], spanning zero; semantic
scores improve. Full paired intervals and ground/contact diagnostics are in
`comparison.md`. No new Notion content was written for this426-output report.
Reference: https://app.notion.com/p/3ffd2a44f44b806ca79ec5987d6fd1f3 .

## Downloaded original/joint200k archives and other state

Both requested Drive archives were downloaded into shared storage and remotely
readback/SHA256 verified. Receipt:
`O/download_phase3_pair_20261101T211746Z/COMPLETE.json`.

```text
/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_data/phase1_twotask_prefix1_phase23_20261031/original_phase3_200k/original_phase3_edge_200k.tar
SHA256 5ebf5bcca805799aef3233bbbe2e2e24397dbdfa618e631dbb71689f4c464f7e

/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_data/phase1_twotask_prefix1_phase23_20261031/joint_phase3_200k/joint_phase3_edge_200k.tar
SHA256 c735ea9d80aabdd3cf0ebcdf82a9d2d5342e981e54db94e24dd7a951f61da60d
```

Remote folder IDs1jRLJ-Sn5FBByIi5QJ-qBaE6WrKSeXYQB(original) and
1bXbbtDaLu_rteP6h8pn5hbXHMk3rYCLI(joint), under
`gggdrive:cosmos_data/phase1_twotask_prefix1_phase23_20261031/`.
The download receipt certifies archives, not automatic extraction. Original was
later extracted for FiLM-only; do not assume joint is extracted locally.

User authorized stopping Phase2 no-shape to free0,4 for the joint baseline.
`O/phase2_no_shape_after_ego_20261101/status.json=stopped_by_user`; latest
preserved checkpoint20k, stopped around27k. Do not restart its old monitor.
EgoControl full-test evaluation is now complete11677 windows, recorded in
`O/eval_egocontrol_fulltest_20261101/status.json` and `results/summary.json`.
These statuses are context, not new evaluation or no-shape launch requests.

Both older full hidden-state/trajectory-FiLM training runs completed200k locally;
their current run directories are in `O/resume_film_20261030/plan.json` and
their live `status.json` files say complete200000. Historical resume receipts
show their original92k/63k starting points and must not be read as current steps.

Existing shared FiLM packages are documented in
[FOUNDRY_FILM_SHARED_20261101.md](FOUNDRY_FILM_SHARED_20261101.md):

```text
/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_models/phase3/hidden_state_film_200k_20261101/checkpoints/step_000200000.pt
/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_models/phase3/trajectory_film_180k_20261101/checkpoints/step_000180000.pt
```

Their shared index is an older snapshot: its trajectory entry says still training,
but the local trainer subsequently completed200k. This side did not replace the
shared180k with200k, or back up the new camera/scratch runs to shared storage.
Check exact package contracts and copy to sep-2 NVMe before training/inference;
do not start repeated loaders from shared or substitute a different specialist.

Older FiLM qualitative publications found during this audit are separate
historical evidence: operations `qual_film_m2v_fourcase_20261101_side` and
`qual_film_m2v_motioncfg_fourcase_20261101_side` have `COMPLETE.json=published_verified`
for https://app.notion.com/p/40bd2a44f44b809aa8aee044384f3d74 . They used original200k,
hidden180k and trajectory150k,4 cases and native/motion-guidance1/1.5/2. The
motion-null reference blocks cross-modal bridge edges and is an inference
intervention, not trained text CFG. Do not conflate those videos or old
hidden150k/trajectory130k test71 tables with the new FiLM-only50k evaluation.

## Main-agent continuation checklist

1. Read this handoff and the snapshot; recheck `hostname`, live statuses,
   processes and storage. Preserve the three running jobs. No waiting auto-launch
   queue remains active; no new goal was created by this side.
2. Review the two new packages and documentation before any later user-authorized
   Git synchronization. Include untracked code/docs intentionally; never use
   `git clean` or assume `git pull` transports them. Running exports are immutable.
3. For recovery use each active run's own contract/checkpoint and exact loader:
   bridges-only camera versus complete motion+bridge scratch versus `film_state`
   FiLM-only are different formats. Do not initialize from disposable probes.
4. The parent thread's Notion summarization is separate. Its pre-existing summary
   draft is preserved. New426-case results/limitations and these new run notes
   are local; no publication or shared checkpoint transfer is implied here.
5. Unresolved work is the historical original-output discrepancy, optional future
   publication and any user-requested final/milestone shared backup. This handoff
   does not authorize extra GPU jobs, bulk copies, deletions or restarting no-shape.

All training source, metadata artifacts, environment/cache settings, validation
receipts and recovery instructions needed to inspect the side's work are retained
at the paths above. No credentials, private keys or tokens are included.
