# Final-hidden self/cross attention: caption-only, global batch 256

## Experiment

Fresh 200k-step run using the final-hidden self-attention -> cross-attention ->
FFN architecture with cross-key RMSNorm. The sole experimental change relative
to the 20260917 K-norm B256 baseline is text packing:
`nano_caption_only` instead of `edge_native_generation_prefix`.

- Two ranks, batch 128 per rank, accumulation 1: global batch 256.
- Seven motion blocks; final post-norm reasoner hidden states as cross memory.
- Cross K uses learned projection, per-head RMSNorm, and native RoPE once.
- Cross V uses learned projection without V normalization; reasoner RoPE stays on.
- No additional pre-joint self-attention and no joint-attention K/V adapter.
- Same camera/head-aligned Nymeria representation and statistics.
- BONES remains 50% within T2M; T2M/TI2M weights remain 0.75/0.25.
- Subject rewriting is unchanged. Caption-only changes packing, not caption text policy.
- LR 2e-4; warmup 1000; cosine floor zero; seed 0.
- Feature/centered-joint/smoothness losses 1/1/5; contact/velocity/height 0.05/1/10.
- Save only positive 10k checkpoints; reuse the baseline fixed visualization suite.
- Existing automatic evaluation discovers this run only once a checkpoint reaches 100k.

## Yonsei launch

Reuse allocation `2295130` on `node39` (two RTX4090 GPUs), replacing the
self-attention -> joint-attention run without canceling the allocation.
The old run and its saved checkpoints are retained. The single-GPU caption-only
Projection A run and all other training/evaluation jobs are unchanged.

```bash
export COSMOS_SITE=yonsei
export SELF_CROSS_TEXT_PACKING=nano_caption_only
export SELF_CROSS_RUN_NAME=edge_7layer_self_cross_final_hidden_knorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921
export FIXED_VIZ_SOURCE=/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_self_cross_final_hidden_knorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917/visualizations/fixed_samples
bash /home/whwjdqls99/cosmos_motion_ft/motion_expert_t2m_edge/train_self_cross_knorm_ddp.sh
```

Output directory:

`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_self_cross_final_hidden_knorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921`

`--resume auto` resumes only this new output directory if restarted; the initial
launch does not load the old architecture's motion checkpoint or the 200k baseline.
The launcher remains portable via `COSMOS_SITE=grasp` and the existing site path
configuration; choose a Grasp run name and local fixed-suite path there.

The retained allocation's tmux session is
`phase2_selfjoint256_strict3d_after2294892`; its session name predates this switch.
Its shell retains the allocation when the training command exits.

## Single-token CFG compatibility

The first launch stopped before any optimizer step because required step-zero
visualization encountered a one-token caption-only CFG memory. The reasoner
already supported that case, but NATTEN rejected maximum KV length one in motion
cross-attention. `self_cross.py` now uses the exact one-key softmax result there:
repeat each sample's own V over its motion queries with the existing GQA head
mapping. Q/K retain zero-gradient graph connections for DDP. Normal multi-key
attention, prompts, masks, RoPE, projections, losses, and visualizations are unchanged.
