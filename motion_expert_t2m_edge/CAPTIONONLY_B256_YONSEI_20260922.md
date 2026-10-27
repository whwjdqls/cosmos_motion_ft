# Yonsei caption-only B256 launches, 2026-09-22

## Scope

Resume Self-cross from its existing 20k checkpoint and launch a separate fresh
Projection A + learned K/V normalization run. Do not cancel either allocation.
No training, evaluation, or data computation may run on a login node or
`cs-gpu-01`. The launchers retain their Yonsei/Grasp site selection and defaults.

Shared settings: camera/head-aligned Nymeria and statistics, legacy BONES motion,
BONES 50% within T2M, T2M/TI2M weights 0.75/0.25, caption-only text packing,
T2M T200 and TI2M T97, seed 0, 200k optimizer steps, LR 2e-4 with 1000-step
warmup and cosine floor zero. Losses remain 1/1/5 with centered decoded joints
and contact/foot-velocity/foot-height weights 0.05/1/10. Numbered checkpoints
remain every 10k. Fixed visualizations include BONES and Nymeria and run every
5k. Model architecture, normalization, losses, and checkpoint contents are not
edited by this operation.

## Self-cross recovery

- Allocation: `2295130`, node39, two RTX 4090 GPUs.
- Tmux: `phase2_selfjoint256_strict3d_after2294892`, pane `%208`.
- Launcher: `motion_expert_t2m_edge/train_self_cross_knorm_ddp.sh`.
- Existing run: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_self_cross_final_hidden_knorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921`.
- Explicit resume: that run's `checkpoints/step_000020000.pt`.
- Prior process failed with CUDA OOM after the logged step 23720.
- Memory adjustment: `SELF_CROSS_BATCH_SIZE=64`, `SELF_CROSS_GRAD_ACCUM=2`;
  64 examples x 2 ranks x 2 microbatches = global batch 256 per optimizer step.
- Set `SELF_CROSS_TEXT_PACKING=nano_caption_only` and retain the existing
  `SELF_CROSS_RUN_NAME`. The optimizer and scheduler resume rather than restart.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for this launch.
- Microbatching changes batch composition and numerical accumulation; it is not
  a bit-for-bit continuation of the previous batch-128-per-rank execution.

## Fresh Projection A + K/V norm

- Allocation: `2299623`, node38, two RTX 4090 GPUs.
- Tmux: `spare_2x4090_20260921`, pane `%215`.
- Launcher: `motion_expert_t2m_edge/train_kv_projection_ddp.sh reasoner_kv_project_vnorm`.
- Run: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_reasoner_kv_project_vnorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260922`.
- Set `PROJECTION_A_RUN_NAME` to the above basename and
  `PROJECTION_A_TEXT_PACKING=nano_caption_only`.
- Batch: 128 examples x 2 ranks x accumulation 1 = global batch 256.
- Fresh initialization, not a continuation of the B128 checkpoint.
- Preserve the B128 caption-only Projection A run's fixed visualization suite
  using `FIXED_VIZ_SOURCE`; its basename is
  `edge_7layer_reasoner_kv_project_vnorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260919`.
- Learned identity-initialized reasoner K/V projections and learned K/V norms;
  joint attention, no additional self-attention block.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for this launch.

This note records the requested launch configuration, not evidence that training
has completed. Production logs are `train_job<jobid>.log` in each run directory.
Evaluation eligibility remains at least 100k steps. The separate Nymeria
evaluation import failure is not changed by this operation.

## Later user-requested return to batch 128 per GPU

The user requested returning Self-cross to `SELF_CROSS_BATCH_SIZE=128` and
`SELF_CROSS_GRAD_ACCUM=1`, still global batch 256. The batch-64/accumulation-2
configuration above is the earlier recovery attempt, not the current requested
configuration. Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` enabled;
this may reduce fragmentation but does not guarantee that every batch fits.

Stop only the Self-cross torchrun process, retain allocation `2295130`, and
resume again from the existing `step_000020000.pt`. The last observed progress
before this change was 21300, so approximately 1300 unsaved steps are repeated.
The accumulation run reached about 1.25 optimizer steps/s; a speed benefit from
returning to batch 128 per rank is not established. The fresh Projection A
caption-only B256 run on `2299623` is not changed or stopped.
