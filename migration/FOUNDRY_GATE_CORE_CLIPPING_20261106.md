# Gate/core clipping and actual AdamW updates — sep-1, 2026-11-06

User requests global max_norm1 versus separate gate/core clipping, measured
actual parameter updates and an assessment accounting for AdamW scale invariance.
No production training setting/process is changed by the experiment.

Active operation: `/mnt/local/jinhyung.kim/cosmos/operations/phase3_gate_clip_adamw_20261106/experiment_20261106T032640Z`. Read launches.json, STUDY_PLAN.json, per-arm
results/status.json and COMPLETE.json before launching/resuming anything.
Do not start duplicate probes; both isolated workers completed128 updates.

## Completed findings and publication

Report, three plots, per-tensor CSV and JSON measurements are published at
https://app.notion.com/p/410d2a44f44b80a8b2aec94037f851d5 . Publication receipt:
`NOTION_PUBLISHED.json` and exact page body `REPORT_PUBLISHED.md` in the operation.

Final32 identical-gradient separate/global actual QKV/O update ratios:
directional1.52/1.54x; bidirectional1.78/1.85x. AdamW substantially compensates
raw gradient scaling, but cancellation is incomplete with actual epsilon and
changing moment history. Full-tensor controls verify scale invariance when
moments and epsilon are rescaled consistently. Native-separate total held-out
loss is0.58%/0.72% worse than global after128 updates, despite slightly better
motion loss. This16-window/single-seed finite-horizon study does not show a
generation-quality or long-run learning benefit from separate clipping.
Production remains unchanged at global max_norm1. Both production models were
still running when publication completed. Probe workers exited normally.

## Design

- Both current sep-1 caption-free50/50/no-joint10k checkpoints, matching immutable
  source snapshots, frozen Phase1/2 checkpoints and full native AdamW state.
-128 global-batch32 updates per policy, single GPU/microbatch4/accumulation8,
  same global data-draw order and cached per-step noise/targets across arms.
  This is not bit-exact continuation of the live two-rank noise stream.
-Three arms: global clip1 native training; separate replay receiving EXACTLY
  the global arm's raw gradients; separate native training receiving its own
  gradients on the same inputs. Separate means cap1 for14 gates and cap1
  for84 core tensors. Combined clipped norm may exceed1; actual values logged.
-Replay isolates optimizer behavior and is not a self-consistent training
  trajectory; its loss must not be interpreted as the separate-native result.
-AdamW LR/schedule, betas0.9/0.99, epsilon1e-8, weight_decay0 match production.
  Both gate/core optimizer groups share these same hyperparameters. All step,
  first-moment and second-moment tensors are independently cloned per arm.
-Actual FP32 weight differences are measured around torch.optim.AdamW.step:
  norm of delta, norm of pre-step weights, ratio, coordinates changed and descent
  alignment, by tensor/projection/modality/layer. Do not use raw-gradient scaling
  or an SGD formula as a proxy for AdamW updates.
-Controls: warm moment-rescaled same/scaled-epsilon comparisons;32 cold-state
  constant-gradient updates. Four CPU tests passed for optimizer-state isolation,
  scale invariance and epsilon-limited behavior. Production weights are untouched.
-Fixed16 test windows x V2M/M2V, same noised inputs, native denoising losses at
  baseline/32/64/96/128. This is not35-step generation, FID or a full benchmark.

## Interpretation cautions

Initial step differences can reflect switching from historically globally clipped
moments to larger gradients.128 updates retain coefficient0.99^128≈0.276 on the initial second-moment
tensor, not necessarily27.6% of final v;
these are finite-horizon intervention results, not asymptotic convergence claims.
The same-gradient arm and native branch answer different questions. Larger
updates alone do not establish better learning or better generated samples.

Intermediate epsilon sampling uses a deterministic stride. For trustworthy
coordinate-wide conclusions, use EXACT_EPSILON_INITIAL.json and, after completion,
EXACT_EPSILON_FINAL.json from exact_epsilon.py, which inspect ALL coordinates.

## Analysis after both workers complete

```
CUDA_VISIBLE_DEVICES='' /mnt/local/jinhyung.kim/cosmos/envs/cosmos/bin/python /mnt/local/jinhyung.kim/cosmos/operations/phase3_gate_clip_adamw_20261106/analyze.py
CUDA_VISIBLE_DEVICES='' /mnt/local/jinhyung.kim/cosmos/envs/cosmos/bin/python /mnt/local/jinhyung.kim/cosmos/operations/phase3_gate_clip_adamw_20261106/exact_epsilon.py
```

The isolated final optimizer files have TWO parameter groups: all gates first,
then all core tensors, each preserving original relative order. Bridge state
keys remain interleaved by layer. Do not map final optimizer states by simply
zipping all bridge keys against flattened group IDs; use the producer's group
layout. exact_epsilon.py implements the correct map. These are isolated forks,
not automatic production resume checkpoints.

User explicitly selected the new Stability check v2 page:
https://app.notion.com/p/410d2a44f44b80a8b2aec94037f851d5 . State observed update
ratios, Adam controls, exact epsilon diagnostics, native held-out loss and limits.
Keep caption-free training unchanged unless explicitly instructed otherwise.

Primary mathematical references: PyTorch2.10 AdamW documentation and original
Adam paper (https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html ;
https://arxiv.org/abs/1412.6980). Current source is scripts/phase3_clip_update_probe.py;
the operation runs its frozen copy. Tests: scripts/test_phase3_clip_update_probe.py.
