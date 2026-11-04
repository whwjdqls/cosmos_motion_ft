# Bidirectional camera prediction with no joint training, video weight one

User explicitly requested stopping the scratch-motion camera experiment and
replacing it with bidirectional camera prediction, no joint task, no camera
loss and video loss weight1. Use the pretrained Phase2 motion specialist,
matching the existing weight-one camera experiment; only bridges train.

## Preserved scratch-motion state

Gracefully stopped on sep-1 at step46338,2026-11-03 23:50UTC. SIGUSR1 was
sent only to the two verified user-owned training ranks; both exited normally.
The trainer wrote its full recovery state before releasing GPUs5,6.

Operation prefix:
`/mnt/local/jinhyung.kim/cosmos/operations/phase3_camera_prediction_bidir5050_video_weight1_20261103`.
`SCRATCH_STOPPED_VERIFIED.json` records the verified complete checkpoint:

```text
scratch_preserved/step_000046338_full_state.pt
SHA256 6295d34ad165ad33a3bf25e0ea2ad8a7ca385a29427f51cb5e3a9af05fdcc4f9
3383183483 bytes;143 motion tensors,56 bridge tensors
optimizer, scheduler, two rank RNG states, next_global_draw1482816
```

The original run's `checkpoints/recovery_latest.pt` remains present. Its older
40k milestone was also hardlinked before stopping. The preserved motion state
is required for scratch loading; a bridges-only loader cannot restore it.
This run is stopped at46338, not complete at200k.

## Replacement recipe and source

Fresh production and remote W&B recipe were verified at2026-11-03 23:59UTC,
step10, approximately2.03s/update on GPUs5,6. All logged fresh/reload/production
task counts match the intended per-global-draw50/50 policy and have zero
joint samples. W&B:
https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-2dc6173f736a1d929a9f .
Operation `RUNNING_VERIFIED.json` records this acceptance snapshot; inspect
live metrics for later progress.

Package: `motion_expert_phase3_edge_camera_bidir5050_wv1/`.
New local run under `weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/`:
`edge_p3_camera_bidir5050_nojoint_nocamloss_wv1_T97_gbs32_sep1_20261103`.

Reference: the joint camera/video-weight1 run on GPUs1,7, contract at
`edge_p3_imagejoint402020_bidir_camera_prediction_nocamloss_wv1_T97_gbs32_sep1_20261103/contract.json`.
Only experiment identity and task probabilities differ from that reference.

| Setting | Value |
|---|---|
| Production tasks |50% video→motion,50% motion+image→video,0% joint|
| Attention |bidirectional, clean motion E(0)|
| Camera |same96 noisy-GT native camera tokens, no clean camera tokens|
| Camera clock |motion sigma for V2M; video sigma for M2V|
| Camera loss |absent; frozen/detached camera readout|
| Video loss |native velocity loss, multiplier1, original full denominator|
| Motion loss |unchanged selected Phase2 x0/physical loss weights|
| Trainable |same117469198 bridge parameters only|
| Frozen specialists |same Phase1 LoRA+video-I/O EMA200k and selected Phase2 200k|
| Training |fresh200k, GPUs5,6, global32=2x16|
| Optimizer |AdamW2e-4, betas0.9/0.99, no decay, clip1|
| Schedule |1000-step linear warmup then cosine to zero at200k|
| Precision |BF16 model, FP32 motion I/O/bridges, TF32 off|
| Data |same112503-window local cohort/camera cache and global draw order|

Joint remains in the supported diagnostic-mode dictionary with probability0.
The training sampler never selects it. Original disposable native/camera
acceptance checks still exercise joint-mode invariants; disposable weights
never initialize production. Production fresh/reload probes and live metrics
must show `joint_samples=0` and only the two intended tasks.

The new dataset subclass reapplies the policy after initialization in spawned
workers and rejects joint rows. Two CPU tests verify recipe/checkpoint identity,
20k actual per-draw task samples and boundary draws. A real two-worker loader
checks64 draws,31V2M/33M2V, no joint, exact expected modes/captions and original
motion/video/camera representations. This prevents inherited worker setup from
silently restoring40/40/20.

The launch uses a pinned node-local source export and strictly compares all
original scientific package hashes against the weight-one control. Native
two-GPU/microbatch16 acceptance and fresh4/reload2 probes precede fresh
production. Consult operation `RUNNING_VERIFIED.json`, `plan.json`, source
manifest and the run's live `status.json`/W&B for actual production status.

Existing baseline0,4, camera/video-weight10 on2,3, and joint camera/video-weight1
on1,7 continue. The original exact50k camera evaluation/publication queue is
preserved. No shared-volume data movement or deletion was performed.
