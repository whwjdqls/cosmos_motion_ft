# New Phase 3 training readiness on Yonsei

Read-only audit on 2026-11-05, after merging master1430dae. This is readiness
evidence, not a fresh training acceptance receipt or a production launch.

Later the same evening, the user authorized the bidirectional+joint/video1
variant. The site/dependency issues below were addressed and its native DDP
acceptance passed. Actual fresh production started on2616204/node58; details
and W&B are in
[`motion_expert_phase3_edge_image_joint_wv1/README.md`](../motion_expert_phase3_edge_image_joint_wv1/README.md).
The audit below preserves the earlier state; it does not describe an ongoing
blocker for that newly launched variant.

## Available resources and inputs

- Owned allocation2616204 on node58: two RTX PRO6000 Blackwell96GB,
  16 CPUs,128GB host RAM. Both GPUs reported zero memory/processes before
  this inspection. Expires2026-11-08 11:28 KST (approximately62hours left).
- Separate allocation2622173 continues the new no-shape Phase2 training.
  No allocation or trainer was stopped, canceled or restarted.
- Correct frozen Phase1 **video-I/O EMA200k** is staged under
  `/scratch2/whwjdqls99/cosmos/operations/idle_gpu_followups_20261104/assets/phase1_video_io_ema200k/`.
  Its DCP metadata SHA256 is exactly
  `b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8`;
  all DCP shard paths exist, including the four required trained video-I/O
  EMA tensors. Use `phase1_checkpoint.json` beside that assets directory for
  the complete selected checkpoint path, not the older September Phase1 default.
- Correct completed, supplied-shape **Self→cross final-hidden K norm,
  caption-only B256 Phase2 200k** is present. This is the20260921 reference,
  not the new in-progress no-shape ablation. Its attention/text/statistics
  contract was read on compute and matches the required source.
- Existing immutable cache `cosmos/caches/phase3_edge_camhead_T97_v1` on
  Scratch2 declares a complete training index with112503rows. First/middle/last
  samples loaded successfully, with motion97×283 and video48×25×16×16.
  This was a representative cache probe, not an all-row revalidation.
- Original fixed71 cases have71unique test recordings and all71bundles exist.
  Nymeria C45 evaluator, evaluator statistics, text cache and body adapter exist.

Full paths/results are saved at
`/scratch2/whwjdqls99/cosmos/operations/phase3_readiness_yonsei_20261105/READINESS.json`.
Model/checkpoint/data inspection ran on owned2616204compute CPUs; no numerical
work ran on the login node. No GPU forward/backward or memory-fit claim is made.

## Newest recipe and remaining work

`motion_expert_phase3_edge_mmdit` selects fresh seven per-modality MMDiT
bridges, temporally local grouped BF16 projection/attention, shape excluded
from bridge attention, and directional50/50 V2M/M2V with no joint objective.
The original frozen Phase2 expert still uses shape conditioning. No new
specialist LoRA or EMA is enabled. Global batch32 means16/rank on two GPUs.
Periodic matched fixed71 V2M/M2V metrics are configured every10k, with GT
references; production must have its own fresh matching acceptance receipt.

The merged launcher is **not yet runnable unchanged on Yonsei**:

1. `motion_expert_phase3_edge_image_joint/launch.py` explicitly accepts only
   the two Foundary hostnames, requires `/mnt/local` paths, and creates locks
   there. It must support owned Yonsei Slurm allocations and their visible GPUs
   without weakening the existing occupancy/ownership/acceptance guards.
2. `motion_expert_phase3_edge_joint/validation.py:local_path` and
   `scripts/prepare_phase3_validation.py` reject non-`/mnt/local` paths.
   They need explicit site-aware path handling and a new checksum-pinned
   Yonsei validation configuration. The assets above are not such a config.
3. `configure_local_paths()` inherits Foundary restored-path defaults from
   a storage root. On native Yonsei paths use
   `PHASE3_IMAGE_JOINT_PATH_MAP='{}'`, including spawned workers.
4. The training Python lacks `dreamsim`; other probed dependencies are
   discoverable. DreamSim exists in the separate evaluation environment,
   but synchronous periodic metrics need it importable in the training
   environment too. Avoid replacing Torch or otherwise disturbing Phase2.
5. After porting, run the existing fresh two-rank preflight and disposable
   update/save/reload probes, then check full-batch VRAM and actual remote
   W&B training/validation logging before claiming production readiness.

No core launcher, training environment or scientific settings were changed by
this inspection. Other bridge parameter-sharing variants are also available,
but this audit identifies the latest dedicated50/50MMDiT recipe rather than
silently selecting multiple new experiments.
