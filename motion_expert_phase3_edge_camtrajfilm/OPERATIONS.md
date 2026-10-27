# Yonsei trajectory-FiLM launch, 2026-10-23

## Approved design and deviations from the active reference

- Four caption-empty tasks with 50% motion+first-image to video, and 1/6 each
  video to motion, camera to motion, and camera+video to motion. The last
  probability uses the exact floating-point remainder so probabilities sum to 1.
- Floor-aware 18D camera trajectory, current post-bridge motion hidden state,
  and explicit normalized motion-noise embedding predict FiLM scale/shift.
- No extra FiLM gate. The zero-initialized output head preserves initial frozen
  predictions. Existing attention bridges, masks, and gates are retained.
- Frozen Phase 1 EMA 200k and Self/cross K-normalized caption-only B256 Phase 2
  200k checkpoints match the active camera-hidden-state FiLM reference.
- Losses, motion statistics, cohort, optimizer, LR schedule and global batch 32
  are unchanged. Sampling changed from the reference's equal four-task mixture
  at the user's request. This is explicitly NOT a FiLM-only controlled ablation.
- User approved preflight and launch, then authorized implementation fixes
  within this plan without repeated approval prompts, with documentation.

## Runtime locations

- Retained job: `2599826`, node34, 2x RTX4090, 16 CPUs, 128 GiB, three-day allocation.
- Tmux: `phase3_camtrajfilm_20261022`.
- Operations: `/scratch2/whwjdqls99/cosmos/operations/phase3_camtrajfilm_20261022`.
- Output: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_fourtask_balanced_notext_floor18d_motionnoise_camtrajfilm_T97_gbs32_wv10_yonsei_20261023`.
- Trajectory cache: `/scratch2/whwjdqls99/cosmos/caches/phase3_edge_camera_trajectory18_T97_camfloor_v1_20261023`.
- W&B: `jungbinc-upenn/cosmos-motion-ft`; comparison group
  `edge-phase3-fourtask-camfilm-20261021`; variant
  `balanced5050-flooraware18d-motionnoise-camtrajfilm`.
- Microbatch 16 per GPU, two GPUs, accumulation 1. Training target 200k; monitoring
  through step 100 is a health milestone, not a request to stop training.

## Fixes and evidence

Initial cache preparation correctly refused an ambiguous recording/start key
at `S07/20231101_s0_joshua_smith_act4_3y5iik@7859`. Multiple captioned entries
share that start and have distinct floor offsets. The original cache retains
the humanized caption, so the corrected join uses recording, start, and that
caption, using the original `humanize_caption` function. An allocated-node
audit found 112,503 matches, zero missing, zero ambiguous. No arbitrary offset,
average, source modification, or dropped-window fallback was used.

The corrected cache finished for all 642 recordings: 112,503 windows and
10,912,791 frame observations. Mean/std are fitted on this exact training
cohort. Original motion, latent, floor-calibration, and native camera-action
files are unchanged.

Standalone approved GPU checks passed for floor-height preservation, the total
calibrated offset, horizontal-origin invariance, zero-initialized identity with
nonzero attention gates, frame-local camera/motion dependence, explicit noise
dependence, and shape/generator/non-camera-task isolation.

Full two-GPU preflight was launched after cache completion. Its authoritative
result will be `preflight/acceptance.json` under the operations directory.
Production is gated on that result and starts from fresh specialist weights,
not the disposable preflight weights. At this entry, production and live W&B
upload are not yet confirmed. Follow `worker.log` for subsequent launch status.

All preparation and GPU checks ran on node34. Existing reference jobs and
shared AGENTS documentation were left untouched. No git operations were used.

## Preflight passed and production started

The full two-GPU preflight passed all 15 recorded checks, including both camera
paths, frozen-specialist parity, gradients, overfit, and checkpoint reload.
Fixed-batch loss decreased from 1.0990553 to 0.7530220 over 12 disposable updates.
Peak allocated preflight memory was 15.9437 GiB. Acceptance fingerprint:
`b4acd3969689130f5bfcf0a36d00192500f66516d5e012e1457cf4bce46d3352`.
Fresh production then started at step 0, global batch 32, accumulation 1.
At step 20, recorded update times were about 4-5 seconds and peak allocated
memory was 17.4682 GiB. This is a health observation, not an evaluation result.

The W&B sidecar initially received HTTP 403: an inherited `WANDB_API_KEY`
selected an account without access to `jungbinc-upenn/cosmos-motion-ft`.
An allocated-node API check confirmed the existing `~/.netrc` credentials have
default entity `jungbinc-upenn` and can access that project. A run-local marker
now tells the logging sidecar to ignore the conflicting environment key, and
the worker unsets it for future launches. This does not restart training or
alter its contract. The shared sidecar's behavior for runs without this marker
is unchanged. No credentials were written to source, logs, or these notes.

## Step-100 monitoring completed

Production passed 100 optimizer updates and was still advancing at step 130
when the final audit ran. All 11 expected log records (step 1, then every 10
through 100) were present, finite, and reported 32 samples per optimizer update.
The run remains active toward 200k; it was not stopped at this milestone.

At step 100: total loss 1.8550143, motion loss 0.0577307, video loss 0.3652298,
LR 0.00002, update time 4.3675 seconds, and peak allocated memory 17.6785 GiB.
Median sampled update time excluding the first was 4.3863 seconds. Trainable
bridges plus FiLM contain 133,133,070 parameters. Pre-clipping gradient norm
was 124.2299; clipping to 1.0 remains active. This high warmup gradient is
recorded rather than hidden; neither LR nor loss settings were changed.
The equal-four-task reference also had increasing pre-clipping norms during
warmup, but its different video frequency prevents a controlled comparison
of total loss or gradient magnitude. No output-quality claim is made at 100.

W&B authentication was fixed without stopping training. A server-side API
query retrieved steps 1 through 110, and the step-100 total/motion/video
losses matched local metrics. W&B reported the run as running:
https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-e75dd21f3b540efbe1c8

Machine-readable evidence is saved as `monitor_100.json` in the operations
directory, including the acceptance fingerprint, exact task probabilities,
step-100 metrics, and observed remote W&B steps. Future allocation resumes
must retain the same output directory, source contract, and camera cache.
