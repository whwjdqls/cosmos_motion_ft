# Shape-input UniEgoMotion: three reconstruction conditioning modes

The user requested one two-GPU training run with RGB-only, trajectory-only,
and RGB+trajectory conditioning. This supersedes the initial inference-only
interpretation. Each example samples one mode uniformly; the full clean actor
shape is supplied in every mode. Validation uses window index modulo three.

Entrypoint: `python -m nymeria_baselines.train_uem_shape_modalities`.
The shared upstream shape overlay and all old runs remain unchanged.

## Recipe

- Fresh random initialization, seed62; no checkpoint warm start.
- Same floor-clean Nymeria/SOMA cache and recording split as the earlier
  supplied-shape runs; T97,20fps, no captions, body-only.
- Centered neutral SOMA30 skeleton `[30,3]` in meters, matching Phase3's
  input. No shape prediction, diffusion, loss or PCA truncation.
- Native220-D motion target, native train statistics, diffusion MSE,
  1000-step native inference, AdamW3e-5/weight decay0.01, EMA0.999,
  clipping1.0 and350epochs with the original epoch300 LR decay.
- Global64=two GPUs x32, accumulation1; existing supplied-shape UEM architecture.
- Both observation streams remain in the batch dictionary. Full-stream
  masks select the native missing-input tokens. Exactly one of the three
  modes is sampled; no extra modality dropout or forecasting task.
- Same frozen cached DINOv2 features: causal5fps hold at20fps,1024-D CLS.
- Rolling last checkpoint each epoch and archival saves every10epochs.

## Operation and acceptance

Output root:
`/scratch2/whwjdqls99/cosmos/operations/uem_shape_threeway_ddp_20261104`.
Production checkpoints and TensorBoard events are under `training/`.
Read `training/progress.json`, `training/DDP_LAYOUT.json`, `training.log`,
and live Slurm state; a launch note alone is not proof of optimizer progress.

Job2673623 requests two4090s together on one node,16CPUs,128GB,three days.
It was allocated node32. Tmux: `uem_shape_threeway_joint_20261104`.
The allocation driver runs3-update and resume-to5 smoke checks, then starts
fresh production automatically on success. It retains the allocation on exit.

The first attempt to join two separate single-GPU jobs2671155/2671156 failed
at NCCL initialization because their GPU namespaces were isolated. Both idle
allocations were released and replaced by the joint allocation. Do not use
the failed split-allocation pair launcher for production.

`DATA_ACCEPTED.json` records real-data target identity, mixed-mode collation,
finite native loss/backward, nonzero shape gradient, and agreement between
explicit masks and absent-modality inference keys. `joint_smoke_*` holds the
actual two-GPU save/resume evidence. Acceptance verifies distinct GPU UUIDs
on the same physical node and global batch64.

The earlier inference preparation is preserved separately at
`operations/uem_shape_input_ablation_20261104`: a pinned shape-input RGB+traj
checkpoint at52404updates/epoch32, three-mode forward/shape acceptance, and
one full-sampler smoke prediction. The full213-prediction sweep was not started.
Its single-modality modes would be missing-input ablations of a model trained
with both streams. They must not be confused with this new mixed-mode trainer.
