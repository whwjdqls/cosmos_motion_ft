# Projection A + K/V norm: LR, duration and self-attention ablations

Three independent mixed Nymeria/BONES experiments were requested.
The existing B128 reference uses peak LR 2e-4 and a 200k cosine horizon.
Existing runs and checkpoint files are not modified.

| Variant | Global batch | GPUs | Peak LR | Target | Motion block |
| --- | ---: | ---: | ---: | ---: | --- |
| lr10 | 128 | 1 | 2e-3 | 200k | JA -> FFN |
| long300k | 128 | 1 | 2e-4 | 300k | JA -> FFN |
| self_joint | 256 | 2 x 128 | 2e-4 | 200k | SA -> JA -> FFN |

At the user's follow-up request, the 300k experiment branches the completed
B128 K/V-norm 200k checkpoint, restoring optimizer and saved RNG state into a
separate output directory. The existing cosine scheduler is extended to300k:
LR restarts around5e-5 at200k, then decays tozero at300k, without new warmup.
This tests additional training with an explicit LR restart, not unchanged
near-zero-LR continuation. The other two experiments start fresh.
LR10 retains the normal guards and clipping;
divergence is an experimental outcome, not permission to silently lower its LR.

## Shared recipe

- Same camera/head Nymeria data, camhead mean/std and original BONES motions.
- 75% T2M / 25% TI2M; BONES is 50% within T2M.
- Native Edge prompt, subject rewrite, seed0, warmup1000, cosine floor zero.
- FP32 motion I/O, BF16 block norms, learned FP32 reasoner-interface K/V gains.
- Centered losses1/1/5 plus contact0.05/1/10; AdamW beta2=.99, TF32 off.
- Variable T2M lengths capped at200; TI2M97; workers8 per GPU.
- Save positive 10k checkpoints only; no rolling or signal checkpoint files.
- Same fixed Nymeria T2M/TI2M and two BONES visualizations, every5k and at start.
- Existing generator/evaluator statistics and frozen reasoner are unchanged.

## Self-attention architecture

Set EDGE_MOTION_SELF_ATTENTION_BEFORE_JOINT=1, with joint attention mode
and no legacy gap blocks. Each of the seven selected layers gets an independent
pre-norm motion-only attention residual, followed by the existing joint-attention
residual and its single SwiGLU FFN residual. There is no FFN between SA and JA.
The added SA uses shape plus valid motion tokens, the existing per-sample offsets,
native motion RoPE, fresh Q/K/V/O projections and Q/K norms. V is not rotated.
It uses activation recomputation during training.

Projection A still adapts reasoner pre-RoPE K and V; learned K norm precedes
one native key rotation, and learned V norm does not rotate values.
The frozen reasoner's causal path never consumes motion or adapted K/V.
This increases capacity and compute; it is not a parameter-matched comparison.

The new setting defaults to false. Saved architecture contracts record both
the flag and SA -> JA -> FFN order. Inference resolves the saved flag before
model construction. Legacy checkpoints imply false and remain loadable;
cross-architecture loads are rejected.

## Launch and resume

Use an owned GPU allocation, never a login node:

```bash
export COSMOS_SITE=yonsei  # or grasp
bash motion_expert_t2m_edge/train_kv_capacity_ablation.sh lr10
bash motion_expert_t2m_edge/train_kv_capacity_ablation.sh long300k
# In a two-GPU allocation:
bash motion_expert_t2m_edge/train_kv_capacity_ablation.sh self_joint
```

Yonsei outputs use /scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/
motion_expert_t2m_edge, with distinct variant and 20260920 names.
Grasp defaults to RUN_ROOT/motion_expert_t2m_edge.
KV_CAPACITY_OUTPUT_ROOT overrides only output placement. FIXED_VIZ_ROOT can
point to the baseline visualization directory containing fixed_samples and
fixed_samples_bones. Recipes auto-resume their own output directory. Only the
first long300k launch falls back to the baseline step_000200000.pt, including
optimizer and saved per-rank RNG; worker cursors are not replayed
bit-for-bit. Interruption can lose fewer than10k updates.

## Yonsei allocation plan

- lr10: reuse idle one-4090 allocation2291664/node32.
- long300k: reuse idle one-4090 allocation2293368/node38.
- self_joint: new two-4090 srun2294892, queued after user-approved release of
  idle2291683. Start automatically after implementation is ready and allocation.
- Existing Projection A K/V norm B256 job2291900 is not canceled; its queued
  successor2293349 already has afterany:2291900 and its original resume recipe.
- Exclude cs-gpu-01 and previously problematic nodes31,33,34,35,36,37,40.
- Retain interactive allocations after trainers exit.

These entries describe implementation and launch intent, not verified optimizer
progress. Current Slurm state and training logs are authoritative. No standalone
tests or smoke runs were requested or run.
