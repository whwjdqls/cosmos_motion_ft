# sep-1 temporal bidirectional camera-prediction replacement (2026-11-08)

Status: COMPLETE. Fresh replacement verified before stopping old W10 at150000.

User authorizes overlapping the replacement on GPUs2/3 with only the owned old
bidirectionalW10 ranks, verifying the new production run before gracefully
checkpointing/stopping the old run. Other trainers/users are outside this handover.

Fresh seven MMDiT bridges; frozen same Phase1 video-I/O EMA200k and selected
Phase2 200k, no extra Phase3 LoRA. Temporal-local bidirectional attention,
no captions at either specialist, V2M50/M2V50/joint0, T97/20fps/256, grouped BF16.
Global32=2x16, accumulation1, video loss weight1, native physical motion losses,
AdamW peak2e-4/betas0.9,0.99/weight-decay0/clip1, warmup1000/cosine200k.
EMA0.999 begins at fresh step0. No architecture or optimization changes to
other live source snapshots.

Camera behavior matches the existing ablation:96 noisy GT native raw9 SE(3)
tokens padded64/domain2; camera clock follows target sigma; no clean camera
inputs, detached camera readout and camera loss0. Camera participates in native
generator attention; no direct camera-motion bridge edges. Noisy GT cameras
retain residual information during training. Validation instead starts cameras
from private Gaussian noise, with no test GT camera input.

Fixed71 EMA V2M/M2V validation every10k, native35steps/CFG1, sample32 perGPU,
decode4, existing pinned evaluation assets and W&B scalar sidecar.

The existing `motion_expert_phase3_edge_camera_directional_nojoint` package now
accepts `--conditioning-attention bidirectional`; directional remains default
for backward compatibility, with unchanged old directional experiment IDs.
Checkpoint sampling restores saved attention and temporal scope. Package name
is retained for existing source/checkpoint contracts. CPU coverage includes
checkpoint policy restoration, aligned reciprocal edges, forbidden far-time
edges and shape exclusion, plus existing camera noise/loss/clock tests.

Required launch gates: two-rank microbatch16 native preflight, specialist parity,
nonzero gate/core gradients, actual zero-camera-loss and detached-readout checks,
finite batched camera sampling and initial-image preservation, full fresh4 to
strict reload6 probe, fresh production finite progress, remote W&B verification.
The replacement passed50 finite production updates and remote W&B verification
before the old rank0 received SIGUSR1 at09:36:02UTC. Old150k validation deferred
that handler until completion. To honor the requested stop without waiting for
both raw/EMA validation passes, the full150k checkpoint was independently copied,
SHA-verified and checked for raw bridges/EMA/AdamW/scheduler/two-rank RNG/global
draw4800000 before terminating ONLY the two identity-checked owned old GPU workers
with SIGKILL at09:40:36UTC. No optimizer update occurred after the saved150k step.
The old150k validation is incomplete/cancelled; no partial metrics are reported as
complete. Old terminal status explicitly records this externally checkpointed
stop. The CPU graceful-stop monitor was retired before the controlled cutover.
New ranks411844/411845 remained active on2/3 throughout.

Operation: `/mnt/local/jinhyung.kim/cosmos/operations/phase3_camera_bidir_local_wv1_ema_20261108T092816Z`.

New output: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_bidirectional5050_temporallocal_no_text_camera_nocamloss_wv1_ema0999_T97_gbs32_sep1_20261108T092816Z`.

Old output: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_mmdit_bidirectional5050_no_text_wv10_lateema0999_from50k_T97_gbs32_sep1_20261106T184453Z`.

## Verified cutover

Stable comparison ID: `sep1_bidir5050_w1_camera_nocamloss_EMA0`.

W&B: https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-7016123cbd2bf81fe768 .

All15 CPU tests passed. Native preflight, fresh4→reload6 full recovery, and all428
immutable source snapshot hashes passed. Production starts fresh at1 with EMA
count1; probes do not initialize it. RUNNING_VERIFIED.json records local50 and
remote50 at cutover. COMPLETE.json records old worker exit and replacement140.
The old150k backup is an independent file, separate inode/linkcount1,3759188895
bytes, SHA256 `acdc4eab84508507b32877209b4cb3d73f41397624737e2812dbe4826b8f2f38`.
Protected file: `/mnt/local/jinhyung.kim/cosmos/operations/phase3_camera_bidir_local_wv1_ema_20261108T092816Z/preserved_old_bidirectional_w10/step_000150000.pt`.

Existing directionalW10 GPUs4/7 and directionalW1 GPUs0/1 were not signaled or
reconfigured. Other users' GPUs5/6 were untouched. No shared-storage transfer.
No auto-resume of old bidirectionalW10; use the preserved contract/source snapshot
and recovery state only if explicitly requested later.
