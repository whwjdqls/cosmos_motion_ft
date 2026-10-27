# Projection A + V norm, mixed, global batch 256

Requested 2026-09-19 as a fresh run after Projection A mixed B256 finishes.
Launch inside an owned two-GPU allocation:

```bash
COSMOS_SITE=yonsei bash motion_expert_t2m_edge/train_kv_projection_ddp.sh reasoner_kv_project_vnorm
```

The no-argument launcher preserves the original Projection A experiment.
Both Yonsei and Grasp site paths remain supported.

- DDP: two GPUs, batch 128 each, accumulation 1, global batch 256.
- Same identity-initialized K/V projections and learned reasoner K norm.
- Enable EDGE_REASONER_V_NORM_TRAINABLE=1, matching the B128 V-norm variant.
- Frozen reasoner unchanged; camera/head representation and matching statistics.
- T2M/TI2M weights 0.75/0.25, BONES 50% within T2M.
- 200k steps; LR 2e-4, warmup 1000, cosine floor zero.
- Centered feature/joint/smooth losses 1/1/5; contact weights 0.05/1/10.
- Only 10k checkpoints; fixed Nymeria and BONES Phase2 visualizations.
- Separate new output directory; no initialization from either previous run.

## Yonsei handoff

Allocation 2270088, node36, two RTX4090 GPUs.
The watcher waits for the original Projection A trainer PID to exit and requires
the final 200k checkpoint and idle allocated GPUs before starting. No allocation
or unrelated process is canceled.

Watcher log:
`/scratch2/whwjdqls99/cosmos/operations/projection_A_vnorm_b256_after_A_20260919/handoff_job2270088.log`

Output:
`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_reasoner_kv_project_vnorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260919`

About 24 hours remained when scheduled. Completing 200k steps will likely need
another allocation and the launcher's resume-auto behavior. This handoff does
not request an automatic future replacement allocation.
