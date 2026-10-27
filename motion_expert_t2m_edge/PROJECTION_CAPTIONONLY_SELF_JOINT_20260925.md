# Caption-only Projection A with pre-joint self-attention

New independent experiment, not a resume of the baseline. Each of the seven
motion blocks is motion-only self-attention, Projection A joint attention,
then one SwiGLU FFN. The added SA covers the same sample's shape token and
valid motion frames, with native motion RoPE and activation recomputation.
It does not replace joint attention with cross-attention or add a second FFN.
The frozen reasoner and its own attention remain unchanged.

Baseline: caption-only Projection A with learned K/V norms, mixed data, global
batch256, Yonsei20260922. Retain its learned identity-initialized K/V adapters,
K/V norms, caption policy, camera/head-aligned Nymeria/statistics, legacy BONES,
BONES50% within T2M, T2M/TI2M75/25, T200/T97, and seed0. Losses are centered
1/1/5 with contact/foot velocity/foot height0.05/1/10. Train200k with LR2e-4,
warmup1000, cosine floor0, batch128 per rank, accumulation1, two ranks and
eight data workers per rank. Save every10k; visualize the baseline's identical
Nymeria and BONES fixed suites every5k, including startup visualization.

```bash
export COSMOS_SITE=yonsei  # or grasp
bash motion_expert_t2m_edge/train_kv_projection_ddp.sh \
  reasoner_kv_project_vnorm_captiononly_self_joint
```

Yonsei output defaults to Scratch2 under `motion_expert_t2m_edge/`, with basename
`edge_7layer_reasoner_kv_project_vnorm_captiononly_self_joint_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260925`.
On Grasp, output remains under `RUN_ROOT`; use `FIXED_VIZ_SOURCE` and
`FIXED_VIZ_BONES_SOURCE` for the local copies of the baseline suites if needed.
Existing launcher variants retain their defaults. No training/model Python
source or existing checkpoint is changed by this launch.

Initial requested allocation: retained job2310667, node39, two RTX4090 GPUs.
Runtime logs, not this note, establish successful startup and optimizer progress.
The separate Nano180k evaluator is queued on A6000 job2325525, with operations
under `/scratch2/whwjdqls99/cosmos/operations/nano_phase2_180k_eval_20260925`.
Edge phase3 on job2299622 is not interrupted.
