# Original Edge Phase3 baseline checkpoint handoff

Verified upload completed 2026-10-31T16:59:14.619122+09:00.

- Drive folder: https://drive.google.com/drive/folders/1jRLJ-Sn5FBByIi5QJ-qBaE6WrKSeXYQB
- Rclone: `gggdrive:cosmos_data/phase1_twotask_prefix1_phase23_20261031/original_phase3_200k`
- Archive: `original_phase3_edge_200k.tar`, 1416427520 bytes.
- SHA256: `5ebf5bcca805799aef3233bbbe2e2e24397dbdfa618e631dbb71689f4c464f7e`.
- Local evidence: `/scratch2/whwjdqls99/cosmos/operations/original_phase3_drive_20261031`.

The checkpoint is the original20260924 Edge baseline at step200000, with
50%video2motion/50%motimg2video,97frames and seven modality bridges. It is not
the joint-generation, camera-conditioned or FiLM variant. Full bridge weights,
optimizer, scheduler, RNG and source identities are retained.

Source checkpoint:
`/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_T97_gbs32_wv10_yonsei_20260924/checkpoints/step_000200000.pt`.

The archive was checked against Drive size/MD5 and a complete downloaded
SHA256. Eight existing dependency archives were independently checked by Drive
size/MD5 and referenced instead of copied again. Original source files and
existing uploads are unchanged. No GPU was required or allocated for this job.

## Restore the matching specialists

Use `030_frozen_phase1_ema200k.tar` from
`gggdrive:cosmos_data/yonsei_training_20261027/archives`:this is the original
four-task20260905 Phase1EMA200k. The newer two-task prefix1 Phase1 in the parent
handoff is a different specialist and cannot substitute for it.

Use `010_frozen_phase2_crosscaption200k.tar` for Self→cross final-hidden/K-norm/
caption-only B256 Phase2 200k. The baseline bridge checkpoint does not include
frozen specialist weights. Other dependencies are listed in DEPENDENCIES.json.

The Phase2 file's mtime changed from the original Phase3 contract. Its contents
were checked against the SHA256-verified prior Drive archive and match exactly.
PHASE2_IDENTITY.json records saved/current timestamps and the content hash. No
original source timestamp or contract was changed. For strict destination
resume identities, verify content first and restore the destination copy's
recorded timestamp as needed. Inference checks the frozen-specialist contracts.

Inspect the PAX archive before extracting into staging, not blindly into `/`.
RESTORE.md inside the Drive folder explains the explicit paths, optional motion
cache/validation inputs and sampler constraints. Use the current GitHub source,
with upload-time revision `880cae02b151926c839b6949f61cbb5fc1002273` recorded for provenance.
Destination GPU loading/sampling is a separate acceptance step; it was not run
by this upload. This is already the final200k of its original schedule.
