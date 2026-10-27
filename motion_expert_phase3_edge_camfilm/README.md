# Four-task Edge Phase 3 with context-aware camera FiLM

Isolated fresh ablation of `motion_expert_phase3_edge_fourtask`. Existing
trainers/checkpoints are not modified. Four tasks remain equal at25%:
first image+motion -> video; RGB -> motion; camera -> motion;
camera+RGB -> motion. All captions are empty.

At each of seven bridge layers, retain masked attention and then apply
`h + gamma(context)*LN(h) + beta(context)` to target motion frames only.
The ordered incoming/outgoing camera-transition hidden states, with boundary
validity flags, are projected to256 channels. Concatenate this normalized
camera context with normalized current motion hidden state; a256-wide MLP
predicts2048 channel scales and2048 shifts. LayerNorm/MLPs areFP32; return
activations in the incoming dtype. Final weights/biases start atzero; there
is no additional zero gate. Shape tokens and generator outputs are unchanged
by FiLM. Camera attention is retained. Camera hidden states can already include
RGB context in the combined task. FiLM uses no ground-truth target motion.

Each FiLM adds2,692,096 parameters; seven add18,844,672. Total trainable
attention bridges plus FiLM:136,313,870. Only these modules train. Initialization
preserves the baseline attention and subsequent random-noise RNG stream.

Same frozen Phase1 EMA200k, Phase2 self-cross/K-norm caption-only B256/200k;
validated Nymeria112503-window T97/20fps cohort, neutral shape, camhead stats,
losses video10 and motion1/1/5 plus contacts0.05/1/10; global32, microbatch16
perGPU on twoGPUs, accumulation1, LR2e-4, warmup1k, cosine0, target200k.
Save every10k plus graceful recovery; retain latest and100k. Fresh bridges,
not a continuation of the four-task control or disposable preflight weights.

The package-local acceptance gate checks exact initial identity, incoming/
outgoing transition alignment and boundaries, motion-context sensitivity,
unchanged non-camera tasks/shape/generator, DDP gradients for attention and
FiLM, fixed-batch loss reduction, frozen specialists, and checkpoint reload.
Production requires a matching acceptance file. Standard non-FiLM inference
loaders must not be used: checkpoint `phase3_architecture` explicitly identifies
this package, and strict loading rejects the old architecture.

Launch inside an owned allocation with `run.sh -m torch.distributed.run
--standalone --nproc_per_node=2 -m motion_expert_phase3_edge_camfilm.train`.
Use `COSMOS_SITE=yonsei` or `grasp`; path flags override site defaults.
Yonsei uses the revalidated camera cache without modifying original arrays.
Outputs/operations go toScratch2; never run model/data work on login nodes,
cs-gpu-01 ornode37. Preserve allocations after worker exit.

Operations: `/scratch2/whwjdqls99/cosmos/operations/phase3_camfilm_20261021`.
This document describes the recipe; runtime logs/acceptance/optimizer progress
are authoritative for whether training actually started and passed.
