# Edge Phase3 final200k checkpoints on Drive

Both uploads are complete and verified. Index checked 2026-11-02T06:14:04.085170+09:00.

| Model | Training task probabilities | Drive folder |
|---|---|---|
| Original September24 two-task baseline | V2M50%, M2V50% | [Download](https://drive.google.com/drive/folders/1jRLJ-Sn5FBByIi5QJ-qBaE6WrKSeXYQB) |
| October2 joint-generation model | V2M1/3, M2V1/3, joint1/3 | [Download](https://drive.google.com/drive/folders/1bXbbtDaLu_rteP6h8pn5hbXHMk3rYCLI) |

## Original two-task baseline

Archive: `original_phase3_edge_200k.tar` (1416427520 bytes).
SHA256: `5ebf5bcca805799aef3233bbbe2e2e24397dbdfa618e631dbb71689f4c464f7e`.
Rclone: `gggdrive:cosmos_data/phase1_twotask_prefix1_phase23_20261031/original_phase3_200k/original_phase3_edge_200k.tar`.

## Joint generation

Archive: `joint_phase3_edge_200k.tar` (1417031680 bytes).
SHA256: `c735ea9d80aabdd3cf0ebcdf82a9d2d5342e981e54db94e24dd7a951f61da60d`.
Rclone: `gggdrive:cosmos_data/phase1_twotask_prefix1_phase23_20261031/joint_phase3_200k/joint_phase3_edge_200k.tar`.
Source checkpoint: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_joint3_equal_T97_gbs32_wv10_yonsei_20261002/checkpoints/step_000200000.pt`.

Joint generation conditions both noisy motion and noisy future video on the
first observed image, caption and neutral shape. Training used independent
specialist noise schedules. Its sampler is
`motion_expert_phase3_edge_joint/sample.py`, including `--mode joint` for both
outputs; the two-task sampler does not implement that mode.

## Shared dependencies and restore

Both contain the complete200k bridge checkpoint with optimizer/scheduler/RNG
state, contracts and source metadata. They reference the same original four-task
Phase1EMA200k (`030_frozen_phase1_ema200k.tar`) and frozen Self-cross/K-norm/
caption-only B256 Phase2 200k (`010_frozen_phase2_crosscaption200k.tar`).
Do not substitute the newer two-task Phase1 checkpoint from the parent handoff.
Frozen specialists are separate dependencies, not embedded in the bridge PT.

Each folder includes RESTORE.md, PROVENANCE.json, DEPENDENCIES.json,
CHECKPOINT_VALIDATION.json, VERIFICATION.json and READY.json. Inspect the PAX
archive before extracting into staging. Preserve original source identities,
normalization, timestamps and the correct model variant. Phase2 timestamp drift
is documented separately and its content verified against the archived specialist.
The original200k recipe has no remaining updates; further training requires an
explicit extended schedule and normal destination acceptance.

Drive byte counts/MD5 were rechecked for both. Each archive has passed a complete
downloaded SHA256 comparison. No source checkpoint, old Drive object, active
training run or pre-existing allocation was modified. CPU-only transfer and
state validation do not establish destination GPU loading/inference readiness.

Local evidence: `/scratch2/whwjdqls99/cosmos/operations/joint_phase3_drive_20261102` and `/scratch2/whwjdqls99/cosmos/operations/original_phase3_drive_20261031`.
