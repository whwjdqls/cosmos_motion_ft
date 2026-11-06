# Phase 3 bidirectional MMDiT + joint objective, video weight 1

Fresh user-requested Yonsei experiment, 2026-11-05. This extends the newest
image-only joint MMDiT recipe; it does not resume a historical Phase3 model.

## Scientific settings

- Tasks: **40% V2M / 40% M2V / 20% joint**. This retains the existing joint
  recipe's proportions; the user selected bidirectionality, joint objective,
  and video weight1. Joint samples contribute to both native losses.
- Joint input: first video image latent and supplied neutral shape; no caption.
  V2M has no caption; M2V keeps its existing caption/dropout0.1 policy.
- Seven MMDiT-style bridges: independent per-modality LayerNorm, Q/K/V and
  output projections, a joint softmax and zero-initialized residual gates.
  Within/cross-modal attention uses the existing temporal Wan groups, with
  bidirectional conditioning. The shape token bypasses bridge attention;
  shape conditioning inside the frozen motion specialist is retained.
- Grouped BF16 projections/attention, FP32 norms/softmax/master weights/Adam.
  No additional specialist LoRA or EMA. TF32 disabled.
- Video weight **1**, motion feature/centered-joint/smooth weights1/1/5,
  contact/foot-velocity/foot-height0.05/1/10. Independent native noise clocks;
  video first latent clean and excluded from its full-denominator loss.
- T97,20fps,256×256, Nymeria-only112503-window immutable cache.
- Fresh200k updates, global32=2×16, accumulation1,4workers/rank.
  AdamW2e-4,1000-step warmup, cosine floor0, betas0.9/0.99,WD0,clip1,seed0.
- Save every10k, retain latest and100k; graceful stop saves recovery state.
- Periodic fixed71 V2M/M2V monitoring every10k, UniPC35/CFG1, motion shift3,
  video shift10, initial sampling batch32/GPU with OOM backoff, decode batch4.
  These held-out metrics are monitoring, not test-based checkpoint selection.

## Yonsei launch and sources

- Allocation2616204, node58,2×RTX PRO6000 Blackwell96GB,16CPU,128GB RAM.
  Retained parent srun, renamed `p3-mmdit-joint-w1`; no other job canceled.
- Tmux: `phase3_mmdit_bidir_joint_wv1_yonsei_20261105`.
- Run root: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/`.
- Run name: `edge_p3_mmdit_bidir_joint402020_groupedbf16_wv1_T97_gbs32_yonsei_20261105`.
- Operation: `/scratch2/whwjdqls99/cosmos/operations/phase3_mmdit_bidir_joint_wv1_yonsei_20261105`.
- Frozen Phase1 is the staged **video-I/O EMA200k**, metadata SHA256
  `b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8`,
  including all four trained video projection tensors. Full paths appear in
  `launch/plan.json` and `sources.json`.
- Frozen Phase2 is the original completed **Self→cross final-hidden K norm,
  caption-only B256,200k**, run20260921, with its exact camera/head statistics.
  It is not the separately running no-shape Phase2 experiment.

The launcher snapshots the entire training/evaluation source before its fresh
two-rank preflight, disposable4-update fresh probe,2-update reload probe and
fresh production. Production never uses the probes' weights. W&B streams to
`scene_motion/cosmos-motion-ft`; use `launch/status.json` and `launch/wandb.log`
to verify the actual run URL and synchronization.

## Implementation and checks

The previous native loss and training logger hard-coded video weight10.
They now both use `RECIPE['video_weight']`; legacy recipes still default10.
This package pins weight1, bidirectionality, joint task/caption policy and its
own source fingerprint in the trainer and spawned workers. Its sampler checks
the saved recipe before using the same native35-step inference implementation.

Yonsei/Grasp launch and validation paths now use explicit site storage roots
and require owned Slurm allocations. Foundary keeps `/mnt/local` and its host
guards. GPU occupancy, source snapshots, cache completeness and matching
native acceptance remain mandatory. Native Yonsei cache paths use an empty
translation map; no immutable cache files are rewritten.

The base Cosmos interpreter remains unchanged, with DreamSim imported from
the already existing isolated metric package directory. Using the metric venv
as the interpreter initially broke Transformer Engine's cuDNN lookup; the
base interpreter plus extra package path passed both imports. No Torch/CUDA
or shared training environment installation was modified.

26 focused CPU/DDP tests passed on owned compute, including real native
video-loss/gradient scaling, unchanged motion-gradient scale, checkpoint
recipe rejection, worker policy, path boundaries, precision and validation.
Pinned config `validation_v2.json` verifies all71 bundle SHA256s and evaluator
assets. The earlier failed `validation.json` is not used.

Actual startup GT metric smoke: full71 motion GT MPJPE/rootADE/FID0,
PA-MPJPE2.57e-5cm numerical residual, skating18.510cm/s,
penetration1.306cm, R@3(32-case pools,3seeds)70.833%. A video identity
pair measured PSNR100dB/SSIM1/LPIPS0/DreamSim≈0, with6×400 I3D features;
native VAE decode produced finite1×3×97×256×256 output. Detailed measured
reference: `METRIC_PROBE_PASSED.json` in the operation.

## Progress and recovery

2026-11-06: first10k validation exposed a rank0-only scoring VAE constructor
using a world broadcast. Its runtime-only synchronization fix and exact10k
resume are documented in
[VALIDATION_VAE_SYNC_FIX_20261106.md](../motion_expert_phase3_edge_joint/VALIDATION_VAE_SYNC_FIX_20261106.md).
For this run, use that operation's bootstrap and active `training.log`, with
tmux `phase3_mmdit_joint_wv1_resume3_20261106`, rather than the original launcher.

Production started fresh on2026-11-05 around21:47KST. Actual updates reached
step20 with finite loss0.423452 and gradient0.507893; observed per-step
time≈2.3seconds, peak PyTorch allocation17.83GiB and device memory≈20GB/GPU,
both GPUs99% utilized. These are early measurements, not a steady-state ETA.
At this rate200k exceeds the current allocation; a later checkpoint resume
will be needed. Source checkpoint/data identities and global32 must be kept.

Current W&B: https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-eed7a56cdcf72f3f4569

[Validation dashboard](https://wandb.ai/jungbinc-upenn/cosmos-motion-ft?nw=xikr5ab126f)
shows explicit point plots against `val/step`, latest values and GT references.
See [visibility audit](WANDB_VALIDATION_VIEW_20261106.md).
The user selected this account after the original launch; see the
[destination/recovery update](../migration/WANDB_DESTINATION_20261106.md).

Consult the run's `launch/status.json`, `launch/training.log`, `metrics.jsonl`,
W&B and operation `PRODUCTION_RUNNING.json` for actual progress. A preflight
alone is not evidence of real production optimizer updates.

Resume with the saved `launch/source_snapshot`, same interpreter/package path,
validation config, package and full original scientific flags, own acceptance
receipt and `--resume auto`. Only the allocation/GPU count/microbatch may
change while preserving global32; data cursor is restored, and topology
changes reseed rank noise as documented by the inherited trainer. Do not
resume with a later edited checkout or borrow another run's acceptance.
