# Edge Phase 3: joint generation ablation

This is a fresh, isolated variant of `motion_expert_phase3_edge`. The completed
two-task baseline, its checkpoints, and its implementation snapshot are unchanged.

## Task contract

- `video2motion`, probability 1/3: clean video conditions noisy motion; no caption.
- `motimg2video`, probability 1/3: clean motion, caption, and first image condition noisy future video.
- `joint`, probability 1/3: caption, neutral body shape, and first image condition both noisy motion and noisy future video.

Joint generation does not receive clean motion or camera-action tokens. Motion and
video noise levels are drawn independently from their existing specialist training
distributions. The first video latent remains clean and is excluded from video loss.
Motion retains all 97 supervised frames. Caption dropout remains 0.1.

Both specialist losses apply to joint examples without halving either loss:
video weight 10; motion feature/joint/smooth weights 1/1/5; contact/foot-velocity/
foot-height weights 0.05/1/10. Each loss retains its original normalization and is
weighted by its number of active samples divided by the full batch size. Therefore
joint examples contribute to both `motion_samples` and `video_samples`; use the
separate `joint_samples` field rather than summing those counts to infer batch size.

## Unchanged configuration

- Frozen Phase 1 Edge EMA200k and Phase 2 Self-to-cross final-hidden K-norm caption-only B256 200k.
- Seven newly zero-gated modality bridges at layers 3, 7, 11, 15, 19, 23, 27.
- Same validated Nymeria cache, camera/head-aligned statistics, T97, 20 FPS, and 256-pixel video.
- Global batch 32; on four GPUs use microbatch 8 and accumulation 1.
- Fresh 200k schedule: AdamW, LR 2e-4, warmup 1000, cosine floor zero.
- Save every 10k, retain latest and 100k; graceful shutdown writes a recovery checkpoint.
- No in-training sampling or rendering.

The immutable data cache is reused because the represented examples are unchanged;
the new training recipe and source fingerprint separately record the task change.
Do not run cache preparation over the existing cache with a changed recipe.

## Launching

Use only an owned compute allocation, never a login node, `cs-gpu-01`, or `node37`.
The wrapper preserves the existing Yonsei/Grasp site configuration and path overrides.

```bash
COSMOS_SITE=yonsei bash motion_expert_phase3_edge_joint/run.sh \
  -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m motion_expert_phase3_edge_joint.train --preflight \
  --microbatch 8 --workers 4 --out /path/to/disposable/preflight

COSMOS_SITE=yonsei bash motion_expert_phase3_edge_joint/run.sh \
  -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m motion_expert_phase3_edge_joint.train --microbatch 8 --workers 4 \
  --out /path/to/new/run --resume auto \
  --acceptance /path/to/disposable/preflight/acceptance.json
```

Production requires its own passed acceptance fingerprint. Never reuse the baseline
acceptance file or bypass the gate. No baseline Phase 3 bridge weights are loaded.

## Offline joint sampling

`sample.py --mode joint` runs simultaneous UniPC-35 schedules: motion shift 3 and
video shift 10, using the same denoiser call at each paired step. Both predictions
are computed before either stream advances. The first video latent is re-pinned
after every step. Joint NPZ files contain `generated_motion` and `generated_video`;
single-target files retain `generated`. Joint outputs need a paired renderer rather
than the baseline single-target `generated` reader.
