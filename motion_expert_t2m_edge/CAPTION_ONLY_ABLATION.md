# Projection A + V norm: Nano-style T2M caption packing

Requested on 2026-09-19. This is a fresh B128 mixed-data run, not a
continuation of the native-prefix baseline.

## Only experimental change

- Baseline T2M: `[BOS, caption IDs, EOS, start_of_generation]`.
- Ablation T2M: caption IDs only, with `add_special_tokens=False`.
- Empty, whitespace-only, or token-empty T2M input uses one EOS, matching
  `FrozenCosmos.tokenize` and the historical Nano motion-only path.
- Edge retains its own tokenizer, token IDs, pretrained weights, and native
  positional encoding. Motion positions follow the actual prefix length.
- TI2M keeps the original image/chat template, including the image-only CFG
  condition. Nano also uses a multimodal chat template for TI2M.
- Caption subject rewriting and both dataset/statistics contracts are unchanged.

`EDGE_TEXT_PACKING=nano_caption_only` is opt-in. The default remains
`edge_native_generation_prefix`. The existing `text_packing` checkpoint
contract records this choice; inference configuration restores it and rejects
explicit conflicts. Strict training resume rejects packing mismatches.
There is no persistent T2M embedding cache to invalidate. TI2M conditions and
their image cache are unchanged.

The NATTEN CUTLASS backend rejects `max_seqlen=1` during its configuration
checks, including inference. For an all-single-token reasoner batch, causal
attention is exactly V with the usual grouped-query head repetition. The
shared reasoner attention helper implements that identity for both joint and
reasoner-only blocks. This preserves the single-EOS CFG input, native token
positions, and attention mathematics; longer reasoner streams and motion
attention keep the existing backend. The original startup failure occurred
inside the sampler's null prediction, not in the skeleton renderer.

## Unchanged recipe

Seven joint-attention motion blocks with Projection A and learned K/V norms;
global batch 128, accumulation 1; T2M/TI2M weights 0.75/0.25 and BONES 50%
within T2M; camera/head-aligned Nymeria representation and statistics;
centered reconstruction losses 1/1/5 and contact/velocity/height 0.05/1/10;
LR 2e-4, 1k warmup, cosine floor 0, AdamW beta2 0.99, seed 0, 200k steps.
Save positive multiples of 10k only. Keep fixed Nymeria and BONES
visualizations, UniPC35, CFG2, and the existing renderer.

## Launch on either site

Inside an owned one-GPU allocation on a compute node, select `COSMOS_SITE`
as `yonsei` or `grasp`, then run:

```bash
bash motion_expert_t2m_edge/train_norm_ablation.sh reasoner_kv_project_vnorm_captiononly mixed
```

Yonsei launch allocation: retained job `2281723`, node35, one RTX4090,
8 CPUs, 64G RAM. Tmux session:
`phase2_projA_vnorm_captiononly_b128_20260919`.
The launcher writes a separate run under
`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge`:

```text
edge_7layer_reasoner_kv_project_vnorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260919
```

Do not cancel the retained allocation on trainer exit. The independent
Nymeria evaluator remains on its A6000 allocation and replacement queue.
