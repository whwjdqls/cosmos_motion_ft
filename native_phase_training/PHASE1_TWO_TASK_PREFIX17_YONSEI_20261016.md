# Phase-1 two-task, 17-frame conditioning ablation

## Recipe

Fresh Edge base initialization, not a continuation of the one-frame experiment.
The baseline is
`edge_phase1_T97_20fps_2gpu_bs16_camera_wearer_lora_exclude5_fd_invdyn_200k_yonsei_20261015`.
Only forward-dynamics conditioning changes: RGB frames 0..16 are observed,
corresponding to clean Wan latent indexes 0..4. The remaining 80 RGB frames are
generated. Inverse dynamics still observes all 97 video frames and predicts
the camera trajectory. Full camera conditioning in forward dynamics is unchanged.

Keep the original 40:25 forward/inverse ratio, captions, five actor exclusions,
T97/20 FPS/256 resolution, cached latents, global batch32 (2 GPUs x16), accumulation1,
LoRA16/alpha32, action heads, EMA, seed42, losses, and 200k-step schedule. Effective
initial LoRA/action LRs are 2e-5/8e-5, with the same 500-step plateau and linear
decay to zero. Checkpoints remain every5k plus rolling15min recovery saves.
Automatic visualization/evaluation stays disabled. W&B is online in cosmos3_camera.

The native video loss still includes masked conditioning elements in its
denominator. Increasing the prefix reduces supervised video latent frames from
24 to20; no compensating loss rescaling is introduced in this conditioning-only
ablation. Raw video losses across prefixes therefore need this context.

## Launch and storage

Portable entrypoint: `native_phase_training/train_phase1_edge_two_tasks_prefix17.sh`.
Set `COSMOS_SITE=yonsei` or `grasp` inside an owned two-GPU allocation. The shared
launcher preserves its one-frame default; non-default prefixes require a distinct
run name. Resume only this experiment's checkpoints using `RESUME_CHECKPOINT`.

Yonsei run directory:
`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_2gpu_bs16_camera_wearer_lora_exclude5_fd_invdyn_p17_200k_yonsei_20261016`

Operations directory:
`/scratch2/whwjdqls99/cosmos/operations/phase1_two_tasks_prefix17_20261016`

Initial allocation: job2449143, node58, 2 RTX PRO6000 GPUs, 16 CPUs. Both GPUs
were idle before launch. About six hours remained; this is not sufficient for
200k steps and a later continuation will be necessary. Existing pending requests
and the one-frame run on job2449144/node55 are not changed or canceled.

Tmux session: `phase1_prefix17_20261016`.
Runtime log: `training_job2449143.log` in the operations directory.
This document records the launch recipe, not confirmed optimizer progress.

## Automatic allocation handoff

The user requested automatic continuation after the initial allocation expires.
The persistent tmux session runs `supervise.sh` from the operations directory.
It starts on2449143, then checks the existing pending replacement jobs2449145,
2449146,2490075,2490076 every30seconds. Only owned running two-PRO6000 allocations
with at least16 CPUs and30minutes remaining are eligible. The compute worker
requires both allocated GPUs to be idle before starting; busy GPUs are skipped.
No Slurm jobs are canceled, released, or assigned dependencies by the supervisor.

Resume uses this run's `latest_checkpoint.txt` and requires model/optimizer/
scheduler/trainer DCP metadata. It restores full training state through the
existing strict-resume launcher and retains batch32 and the same200k schedule.
It never silently restarts from the base after an expired allocation. A failure
while an allocation is still running stops supervision for diagnosis, rather
than repeatedly restarting. Checkpoint durability is still limited by native
asynchronous save completion; it is not a bit-exact uninterrupted replay.

`supervisor.log`, `status.txt`, and per-job training logs record progress. A
`STOP` file in the operations directory prevents subsequent launches but does
not stop an already running training process. Completion at a committed200k
checkpoint stops the supervisor without canceling the retained allocation.
