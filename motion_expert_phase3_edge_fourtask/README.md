# Edge Phase 3: four tasks, no captions

Fresh, isolated bridge-only ablation of the camera-conditioned Phase 3 recipe.

| Task | Probability |
|---|---:|
| First image + motion -> video | 25% |
| RGB video -> motion | 25% |
| Camera trajectory -> motion | 25% |
| Camera trajectory + RGB video -> motion | 25% |

Captions are discarded in the dataset, noise preparation, and both specialist
conditioning paths. Native structural prompt tokens are not removed. There is
no camera+text task, no joint-generation task, and no text CFG dropout.
The old camera-mix M2V task DID use caption content; this run does not.
Equal sampling changes expected video/motion supervision from 50/50 to 25/75.
These are random draw probabilities, not exact per-batch quotas.

Otherwise retain frozen Phase 1 EMA200k and Phase 2 self-cross final-hidden
K-norm caption-only B256/200k, seven fresh zero-gated bridges, validated Nymeria
T97/20fps, camera/head statistics, neutral shape and native camera actions.
Losses remain video10, motion1/1/5, contacts0.05/1/10. Global batch32,
LR2e-4, warmup1000, cosine to zero, 200k updates. Save every10k and on graceful
stop; retain latest plus100k. No old bridge checkpoint initializes this run.

Use `run.sh -m motion_expert_phase3_edge_fourtask.train` within allocated GPUs.
Production requires a matching GPU/DDP acceptance file from `--preflight`.
The acceptance test exercises four-task packing, frozen specialist parity,
backward/overfit, camera influence, omitted-caption/video invariance, and reload.
Preflight weights are disposable; production starts fresh.

Yonsei operation: `/scratch2/whwjdqls99/cosmos/operations/phase3_fourtask_equal_20261020`.
Output: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_fourtask_equal_notext_T97_gbs32_wv10_yonsei_20261020`.
Request: tmux `phase3_fourtask_equal_20261020`, job2583743, two4090,16CPUs,
128GiB,three days; exclude cs-gpu-01 and node37. Microbatch16/rank,accumulation1.
The allocation is retained after worker exit. No existing trainer is stopped.
The site wrapper retains Grasp support via COSMOS_SITE and explicit path overrides.

This document records the recipe, not evidence that preflight or training passed.
Use the operation logs, acceptance.json and output status.json for actual state.
