# Yonsei restart handoff, 2026-10-23

## User request and maintenance timing

The user reported a server shutdown approximately two hours after 12:38 KST
and explicitly chose **13:50 KST on 2026-10-23** for checkpointing/stopping.
They asked to document the running experiments so they can request "resume all"
after reboot. They subsequently added the two undertrained UniEgoMotion runs
as a lower-priority group. "Resume all" now means the four first-priority runs
plus those two secondary runs, subject to available capacity. Do not
automatically resume before that request.

The durable control/state directory is:
`/home/whwjdqls99/cosmos_restart_20261023`.
The timer tmux session is `cosmos_shutdown_20261023`.
Scheduling is not evidence that saving has already completed. Consult
`shutdown.log` and each run's `*.stopped.json` after 13:50. A
`*.needs_attention.json` means saving/stopping was not confirmed.

No training source, loss, LR, architecture, task mixture, or checkpoint contract
is changed by maintenance. No allocations are cancelled, and pending jobs are
left unchanged. The timer does not force-kill any trainer or restart any run.

## First priority: the four previously active experiments

| Snapshot key | Experiment | Old allocation | Hardware | Global batch | Target |
|---|---|---|---|---|---|
| `trajectory_film` | Phase 3, direct floor-aware 18D trajectory FiLM | 2599826, node34 | 2x4090 | 32, microbatch16, accumulation1 | 200k |
| `hidden_film` | Phase 3, native camera hidden-state FiLM | 2593912, node36 | 2x4090 | 32, microbatch16, accumulation1 | 200k |
| `fourtask_control` | Phase 3, four-task control without FiLM | 2583743, node31 | 2x4090 | 32, microbatch16, accumulation1 | 200k |
| `phase1_no_text_p1` | Phase 1, no-text FD/inverse-dynamics, p1 variant | 2490075, node55 | 2xPRO6000 | 32, microbatch16, accumulation1 | 200k |

The snapshot JSON files contain the actual launcher argv, working directory,
selected non-secret runtime environment, worker identities, allocation resource
request, capture-time training status, and checkpoint pointer. These are the
authoritative exact launch settings. Old PIDs/job IDs are maintenance identities,
not IDs to reuse after reboot.

The Phase 1 p17 variant was not an active trainer in this inspection. Do not
silently add it, completed experiments, or unrelated evaluations to "resume all".

## Output directories and resume sources

### Direct trajectory FiLM

`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_fourtask_balanced_notext_floor18d_motionnoise_camtrajfilm_T97_gbs32_wv10_yonsei_20261023`

- Package: `motion_expert_phase3_edge_camtrajfilm`.
- Operations: `/scratch2/whwjdqls99/cosmos/operations/phase3_camtrajfilm_20261022`.
- Existing tmux: `phase3_camtrajfilm_20261022`.
- Task probabilities: M2V50%, V2M1/6, camera-to-motion1/6, camera+video-to-motion1/6.
- All task captions are empty. This differs from the equal-four-task reference.
- Direct floor-aware camera18 + current motion hidden feature + motion diffusion
  noise embedding produce FiLM scale/shift; no additional learned FiLM gate.
- Trajectory cache: `/scratch2/whwjdqls99/cosmos/caches/phase3_edge_camera_trajectory18_T97_camfloor_v1_20261023`.
- W&B: https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-e75dd21f3b540efbe1c8
- Detailed implementation/preflight notes: `motion_expert_phase3_edge_camtrajfilm/OPERATIONS.md`.
- The original operations worker is hard-coded to job2599826/node34 and performs
  fresh preflight/launch. **Do not blindly rerun that worker after reboot.** Use
  its recorded production arguments with `--resume auto` and a new allocation.

### Hidden-state FiLM

`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_fourtask_equal_notext_contextcamfilm_T97_gbs32_wv10_yonsei_20261021`

- Package: `motion_expert_phase3_edge_camfilm`.
- Operations: `/scratch2/whwjdqls99/cosmos/operations/phase3_camfilm_20261021`.
- Existing tmux: `phase3_camfilm_20261021`.
- Four tasks are sampled equally, with empty captions.
- Camera conditioning uses native frame-to-frame camera actions/hidden states,
  not the new floor-aware trajectory features.
- Separate W&B streamer state: `/scratch2/whwjdqls99/cosmos/operations/phase3_wandb_20261022/camfilm`.

### Four-task control

`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_fourtask_equal_notext_T97_gbs32_wv10_yonsei_20261020`

- Package: `motion_expert_phase3_edge_fourtask`.
- Operations: `/scratch2/whwjdqls99/cosmos/operations/phase3_fourtask_equal_20261020`.
- Existing tmux: `phase3_fourtask_equal_20261020`.
- Four tasks are sampled equally, with empty captions and no FiLM addition.
- Separate W&B streamer state: `/scratch2/whwjdqls99/cosmos/operations/phase3_wandb_20261022/control`.

### Phase 1 no-text p1

`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_2gpu_bs16_lora_exclude5_fd_invdyn_no_text_p1_200k_yonsei_20261017`

- Script: `native_phase_training/run_latent_train.py`.
- TOML: `native_phase_training/world_camera_nymeria_latent_edge.toml`.
- Actual native training cwd: `/home/whwjdqls99/cosmos-framework-edge`.
- The snapshot records all command-line overrides. Preserve the p1 variant,
  exclusion settings, task ratios, scheduler, EMA, and saved config exactly.
- Before maintenance, a complete native checkpoint was observed at112827.
  This is **not** the final shutdown checkpoint; use the final marker instead.
- Full wall-clock checkpoints are enabled every15minutes, besides 5k milestones.
- This process has no `SLURM_LOG_DIR` sentinel configured. Do not send it SIGUSR1.

## Checkpoint-and-stop procedure

At13:50 the timer sends SIGUSR1 only to the captured Phase3 training ranks.
Each trainer finishes its update, saves the existing recovery checkpoint
(including optimizer, scheduler, RNG and sampler cursor), then exits normally.
The final checkpoint is selected by each output's `checkpoints/latest.json`.
Do not assume it is a 10k-numbered checkpoint: it may be `recovery_latest.pt`.

For Phase1, the timer waits for the next **completed** native DCP checkpoint
after13:50, normally within15minutes. DCP async mode is disabled in this run;
the latest pointer is written after the component saves return. The controller
also requires model/optim/scheduler/trainer metadata before interrupting only
the training launcher, leaving the Slurm allocation intact. If no new complete
checkpoint arrives within25minutes, it records `needs_attention` rather than
killing a writer. Small post-save progress may be discarded during shutdown.

The final Phase1 resume source is its `checkpoints/latest_checkpoint.txt`.
Full-state resume is required, not model-only initialization.

The control script records completion markers and process exit, not a checkpoint
reload test. Standalone Phase3 W&B streamers receive a graceful flush request;
trajectory-FiLM's own supervisor handles its streamer. W&B remote delivery is
not guaranteed merely by a flush request.

## Restart instructions for the next assistant

1. Read this handoff, the four snapshot JSON files, and the final stopped/error
   records under `/home/whwjdqls99/cosmos_restart_20261023`.
2. On the user's restart request, inspect live allocations. Reacquire appropriate
   srun allocations in tmux if old ones did not survive. No work on login nodes;
   exclude `cs-gpu-01` and `node37`. Do not duplicate an already running trainer.
3. For the first-priority runs, restore each recorded runtime environment and
   its package launcher. Preserve
   two GPUs, microbatch16, accumulation1, global batch32 and target200k. Never
   replay old Slurm/rank variables, `CUDA_VISIBLE_DEVICES`, PIDs or deadlines.
4. Phase3: use recorded production argv through the package's `run.sh`, retain
   the same output/cache/acceptance paths, and set `--resume auto` exactly once.
   Use the same source contract, not a fresh-model worker. Set a fresh
   `PHASE3_STOP_EPOCH` from the new allocation end minus15minutes.
5. For trajectory-FiLM, set `PHASE3_RUN_DIR` to its existing output and unset the
   conflicting `WANDB_API_KEY`; existing netrc credentials and its run-local
   W&B marker are the established working authentication route.
6. Phase1: reuse the recorded torchrun arguments and environment, replacing the
   old `checkpoint.load_path` with the latest completed checkpoint. Preserve
   `checkpoint.load_training_state=true` and strict full-state resume.
7. Restore the two standalone W&B streamers using their recorded argv/state
   directories. Keep W&B identities and existing output directories unchanged.
8. Do not alter losses, normalizations, specialist sources, mixture probabilities,
   or LR schedules while resuming. Report each resumed step and allocation.
9. After the primary runs have resources, resume the two UniEgoMotion runs below
   on spare single4090 allocations. Do not let secondary allocations consume
   GPU/QOS capacity needed by a pending primary resume. This is operational
   scheduling priority, not an instruction to change the cluster QOS.

## Shared Phase 3 dependencies

- Frozen Phase1 EMA200k: `/lustre/whwjdqls99/cosmos/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_4gpu_bs8_camera_wearer_lora_exclude5_yonsei_20260905/checkpoints/iter_000200000`.
- Frozen Phase2 Self/cross K-normalized caption-only B256200k: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_self_cross_final_hidden_knorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921/checkpoints/step_000200000.pt`.
- Motion/latent cache: `/scratch2/whwjdqls99/cosmos/caches/phase3_edge_camhead_T97_v1`.
- Native camera cache: `/scratch2/whwjdqls99/cosmos/caches/phase3_edge_camera_conditioning_T97_revalidated_20261021`.
- Environment: `/lustre/whwjdqls99/cosmos/envs/cosmos`.
- Repo: `/home/whwjdqls99/cosmos_motion_ft`; Edge framework: `/home/whwjdqls99/cosmos-framework-edge`.
- Phase3 loss: video10; motion feature1, decoded centered joints1, smoothness5;
  contact0.05, foot velocity1, foot height10. LR2e-4, warmup1k, cosine to0 at200k.

## Other allocations observed before shutdown

- 2593148, node35, 1x4090: no Python workload found; not a training resume target.
- 2490076, 2560675, 2560674: pending PRO6000 requests, not started trainers.
- 2603462: pending `cfg2-full71` evaluation request. This is separate from the
  four training runs. Check its own queue/operations state before relaunching it.
- Existing evaluation files and galleries are untouched by this maintenance.

This document and control metadata live on `/home`; model/data outputs remain
at their existing scratch2/lustre paths. This operation is not a new checkpoint
backup to lustre. Do not infer a backup from the existence of this handoff.

## Second priority: finish the two undertrained UniEgoMotion variants

Added at the user's request on2026-10-23. These were not active trainers in the
maintenance snapshot, so they are not additional targets for the13:50 stop
timer. They are additional **post-reboot resume targets**.

| Display name | Internal variant | Latest reported evaluated step | Original training target | Resources |
|---|---|---|---|---|
| RGB-only | `RGB_recon` | approximately341.4k | 350epochs, approximately555.8k steps | 1x4090,8CPUs,64GiB |
| RGB + trajectory, reconstruction | `recon` | approximately339.8k | 350epochs, approximately555.8k steps | 1x4090,8CPUs,64GiB |

The step labels above are from the user's evaluation summary, not a new
checkpoint deserialization. `last.ckpt` was found in both run directories;
the stored epoch/global-step and optimizer state govern resumption. Continue
to the original350-epoch endpoint, not350 additional epochs. Keep both at the
same secondary priority; no preference between the two was requested.

The trajectory-only and all-task runs at555.8k are reference/completed runs,
not additional training resumes. Lower scores alone do not make them unfinished.

### Exact secondary output/checkpoint paths

RGB-only output:
`/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/runs/uniegomotion-RGB_recon_nymeria_soma_b64_350ep_20261019`

RGB + trajectory reconstruction output:
`/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/runs/uniegomotion-recon_nymeria_soma_b64_350ep_20261019`

Resume each from its own `last.ckpt`, preserving the same output directory.
Never initialize one variant from another variant's weights. Do not replace
the last checkpoint with the older333480-step numbered milestone just because
that milestone is easier to identify.

### Unchanged UniEgoMotion recipe and dependencies

- Launcher: `/scratch2/whwjdqls99/cosmos/operations/uem_variants_20261019/worker.sh`.
- Entry point: `nymeria_baselines.train_uem_variants`.
- Environment: `/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/env_uem/bin/python`.
- Upstream: `/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/upstream/UniEgoMotion`.
- Clean motion/shape cache: `/scratch2/whwjdqls99/cosmos/caches/nymeria_soma_official_baselines_T97_20fps_floorclean_20261016`.
- Frozen visual-feature cache: `/scratch2/whwjdqls99/cosmos/caches/nymeria_uem_dinov2_vitl14reg_5fps_20261019`.
- B64,97frames,20FPS, seed62, original LR3e-5 and weight decay0.01. Restore the
  saved optimizer/scheduler/LR (`USE_CKPT_LR=True`); do not restart the schedule.
- Same body230 representation, train-only normalization, predicted shape,
  native loss/diffusion/EMA, and reconstruction-only task masking. No captions
  or GT-shape conditioning. `RGB_recon` must continue to omit camera inputs.
- Existing TensorBoard logs continue in each output; the launcher disables W&B.
- Existing policy: every10epochs plus `last.ckpt`; no policy change requested.
- Each may use a three-day single4090 srun in tmux, excluding `cs-gpu-01` and
  `node37`. Respect primary-run priority and the current cluster GPU/QOS limits.

Commands to run **inside separate owned GPU allocations after reboot**:

```bash
# RGB-only: refuse an accidental fresh start if its resume file is unavailable.
test -s /scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/runs/uniegomotion-RGB_recon_nymeria_soma_b64_350ep_20261019/last.ckpt && \
  bash /scratch2/whwjdqls99/cosmos/operations/uem_variants_20261019/worker.sh RGB_recon
```

```bash
# RGB + trajectory reconstruction: same-run full-state resume.
test -s /scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/runs/uniegomotion-recon_nymeria_soma_b64_350ep_20261019/last.ckpt && \
  bash /scratch2/whwjdqls99/cosmos/operations/uem_variants_20261019/worker.sh recon
```

The entry point checks its saved experiment contract and passes `last.ckpt`
to Lightning `trainer.fit(..., ckpt_path=...)` for full-state resume. Keep this
behavior. Do not regenerate representations/features or modify the contract
just to bypass a mismatch. No training was launched by this documentation update.

Machine-readable secondary queue:
`/home/whwjdqls99/cosmos_restart_20261023/secondary_resume.json`.

## Resume authorization and additional requests, 2026-10-26

The user returned after reboot and authorized allocation requests and resuming
the four primary experiments. They additionally requested resuming the no-text
Phase1 **prefix17** run on another2xPRO6000 allocation, plus a spare2x4090 srun
for later manual use. This explicitly supersedes the earlier instruction not
to add prefix17 automatically.

The confirmed shutdown recovery steps are6024 (trajectory FiLM),31802
(hidden-state FiLM),44076 (four-task control), and114951 (Phase1 prefix1).
Prefix17's existing checkpoint pointer is `iter_000095353` in
`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_2gpu_bs16_lora_exclude5_fd_invdyn_no_text_p17_200k_yonsei_20261017/checkpoints`.

Operational scripts, tmux mapping, exact resume method and submission/runtime
logs are in `/home/whwjdqls99/cosmos_restart_20261023/resume_20261026`.
Read its README.md and current runtime records before submitting duplicates.
There are now five authorized primary training resumes and one requested spare
allocation; the two UniEgoMotion runs remain secondary and were not submitted
with these requests. No training recipe or shared training source was changed.
