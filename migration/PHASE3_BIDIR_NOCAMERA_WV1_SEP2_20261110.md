# sep-2 bidirectional W1 without camera prediction

User authorized a fresh bidirectional replacement on 2026-11-10, then stopping
both directional-global/camera and directional-cross-only/camera trainers after
the replacement is verified. The directional temporal-local/camera trainer and
existing cross-only100k evaluation remain untouched.

Recipe: temporal-local MMDiT, bidirectional bridge attention, camera prediction
absent, video loss multiplier1, EMA0.999 from step0, no captions, V2M50/M2V50,
no joint generation, global batch32=2x16, no accumulation, fresh200k. Same frozen
Phase1 video/action projection EMA200k and selected Phase2 checkpoint as the
existing local camera trainer. Fixed71 EMA validation every10k.

New package: motion_expert_phase3_edge_mmdit_bidir_wv1. Inherits the proven native
two-task architecture and acceptance tests; its isolated policy pins W1 and
bidirectional attention. Two CPU task/worker tests passed. The production
launcher requires actual two-rank native preflight, caption poison audit, and
full-state fresh4/reload6 probe before starting production.

Placement: GPUs2/4, initially sharing only owned cross-only ranks1494232/1494233.
No foreign process may be shared or stopped. The guarded launcher checks exact
PID identities/free memory before every stage. handover.py requires finite
production >=30 updates, matching source/contract/EMA/caption settings and W&B
before SIGUSR1 to the two original rank0s. Full recovery/optimizer/RNG/data cursor
and EMA are verified and retained under this operation's preserved/ directory.

Exact100k is exempt from trainer rotation and an independent milestone watcher
is armed at cosmos_storage/preserved_milestones/phase3_bidir_nocamera_wv1_100k_20261110.
No shared-storage transfer or old checkpoint deletion is authorized/performed.

Authoritative status: status.json, FAILED.json, REPLACEMENT_HEALTHY.json,
COMPLETE.json. COMPLETE requires actual replacement progress and BOTH original
trainers stopped with verified full recovery states. Do not infer completion
from launcher startup alone. Older stopped/deleted runs must not be resumed.

New output: `/mnt/local/jinhyung.kim/cosmos_storage/runs/motion_expert_phase3_edge/edge_p3_mmdit_bidirectional_nojoint5050_nocamera_wv1_ema0999_temporal_local_bf16_T97_gbs32_sep2_20261110`

## Handover completed

Production passed native two-rank and no-caption gates, full-state fresh4/reload6
checks, finite step30 and W&B verification.425 source files verified. New run
`sep2_bidir5050_w1_nocamera_EMA0` is training on GPUs2/4; W&B:
https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-c0d0cecdf7475e3c7881

Old global trainer stopped134959; old cross-only trainer stopped116015. Both
SIGUSR1 checkpointed stops exited cleanly. Raw/EMA/optimizer/scheduler/two-rank
RNG/global draw cursor verified; checkpoint SHA receipts in COMPLETE.json.
Independent100k states are retained. New100k copy watcher PID2802225 remains
armed. Local directional/camera on3/5 and cross-only100k evaluation on1 remain
running. No foreign process touched. Code is local/uncommitted at this handover.
