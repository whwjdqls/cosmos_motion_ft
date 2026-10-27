# Edge Phase 3: camera-conditioned motion mixture

This is a fresh bridge-only ablation of the original two-task Phase 3 run,
not a continuation of the joint video/motion-generation experiment.

| Task | Sampling probability |
|---|---:|
| Motion + first image + text -> video | 50% |
| Video -> motion | 12.5% |
| Camera -> motion | 12.5% |
| Camera + text -> motion | 12.5% |
| Camera + video -> motion | 12.5% |

Camera + video + text was explicitly excluded. Probabilities are per draw,
not exact task quotas in each batch. Text-bearing tasks retain 10% caption
dropout. No-text modes discard caption content in both specialist paths.
Camera-only modes have no vision tokens, including no first image.

## Unchanged baseline

- Frozen Phase 1 Edge EMA200k and Phase 2 self-cross/final-hidden/K-norm,
  caption-only B256/200k; seven newly initialized zero-gated bridges.
- Original validated Nymeria T97/20fps training rows, neutral shape inputs,
  camera/head-aligned motion mean/std, 256-square Wan latents.
- Global batch32, LR2e-4, warmup1000, cosine floor0, target200k.
- Native video loss10 and original centered motion loss1/1/5 plus
  contact/foot-velocity/foot-height0.05/1/10. No camera prediction or loss.
- Motion and video each retain 50% expected supervision probability.
- Save every10k and on graceful stop; retain latest plus100k.

Camera conditions come from the recorded upright RGB-camera poses in the
same 97-frame window, not from predicted or decoded body motion. The native
Phase 1 backward-framewise SE(3) conversion emits96 raw9-D actions, padded
to64 with domain2 and no z-score. Every action is clean. Frozen native
action projections and generator blocks encode it; the same existing
bridges receive those hidden rows. Action i connects to source motion
frames i and i+1. Video bridges retain causal Wan source intervals.

New camera tasks use raw caption-only or empty generator text; the original
M2V native I2V prompt and plain V2M empty prompt are unchanged. Camera data
are cached separately, with the complete baseline window order retained;
missing/invalid camera data fail preparation rather than silently dropping
or substituting training cases.

## Launch

Run only inside an owned Slurm allocation. The shared site wrapper supports
`COSMOS_SITE=yonsei` or `grasp`; explicit path arguments override defaults.
Yonsei cache/run outputs go to Scratch2, not Lustre. Do not use login nodes,
cs-gpu-01 or node37. Never cancel a retained allocation merely on exit.

```bash
export COSMOS_SITE=yonsei
bash motion_expert_phase3_edge_camcond/run.sh \
  -m motion_expert_phase3_edge_camcond.data --workers 4
bash motion_expert_phase3_edge_camcond/run.sh \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m motion_expert_phase3_edge_camcond.train --microbatch 16 --workers 4 \
  --preflight --out "$PREFLIGHT"
bash motion_expert_phase3_edge_camcond/run.sh \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m motion_expert_phase3_edge_camcond.train --microbatch 16 --workers 4 \
  --out "$RUN" --resume auto --acceptance "$PREFLIGHT/acceptance.json"
```

Production refuses missing/stale acceptance. Preflight covers native
packing, all five modes, temporal masks, original-task and specialist parity,
unchanged video loss reduction, real mixed-task DDP backward, gate/core
gradients, frozen specialists, fixed-batch overfit, nonzero camera influence,
omitted-input invariance after learning, and optimizer/RNG checkpoint reload.
Its weights are disposable. `PHASE3_STOP_EPOCH` optionally requests a graceful
checkpoint before the retained allocation expires. `metrics.jsonl` includes
actual per-task sample counts; `status.json` records current progress.

Implementation is not evidence of a successful preflight or training run.
The operations launcher, acceptance JSON and optimizer logs are authoritative.
Evaluate all four motion-conditioning modes separately, always with matching
GT metrics. A mixed run is not a causal isolation of each task's contribution,
and does not by itself establish improved camera/body correlation or contact.
