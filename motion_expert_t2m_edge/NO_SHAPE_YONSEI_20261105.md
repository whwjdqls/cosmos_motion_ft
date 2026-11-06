# Yonsei Phase 2 without shape conditioning

User requested a fresh no-shape Phase 2 run on **two GPUs**. This is a new
Yonsei experiment, not a resume of the stopped Foundary no-shape experiment.

## Run and recovery

- Slurm allocation: **2622173**, node58, 2×RTX PRO6000 Blackwell 96GB,
  16 CPUs, 128GB host RAM; allocation expires 2026-11-08 13:03 KST.
- Production tmux: `phase2_no_shape_pro2_live_20261105`.
- Run name: `edge_selfcross_finalhidden_knorm_captiononly_noshape_nymeria_bones50_gbs256_camhead_w1_1_5_contact_yonsei_20261105`.
- Output root: `/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge/`.
- Operation: `/scratch2/whwjdqls99/cosmos/operations/phase2_no_shape_pro2_20261105`.
- Current W&B: https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/cfj632ue
  (live transaction/media replay without restarting training; see
  [account update](../migration/WANDB_DESTINATION_20261106.md)).
- Source revision: `0d7f6c2d01ba45b5ec2505a30c42b08a6f853dcd`.
- Launcher: unchanged `train_self_cross_knorm_ddp.sh`, with operation-local
  `env.sh` and `production.sh`. For an owned replacement allocation, source
  the same environment and attach `production.sh` to that allocation with
  two GPUs/16 CPUs/128GB. `--resume auto` resumes only this output directory.
  Never initialize production from the disposable preflight checkpoints.

## Controlled recipe

Reference: completed Self→cross final-hidden + K norm, caption-only B256,
200k, run ending `subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260921`.
`RECIPE_PARITY.json` records its exact path and statistics hashes.

The only scientific change is `EDGE_PHASE2_SHAPE_CONDITIONING=none`:
no neutral-joint token reaches attention, and the unused shape encoder/type
embedding is frozen and excluded from the optimizer. These frozen tensors
remain in checkpoints for strict reload. Motion targets, source skeletons,
normalization, physical losses and per-frame rotary positions are retained.
Shape may still be used by the GT/reference renderer; it is not model input.

- Fresh motion expert, frozen official Edge reasoner; **200,000 updates**.
- Seven SA → final-hidden CA → FFN blocks, hidden2048/FFN3072, SwiGLU.
- Cross keys: learned projection → per-head RMSNorm → native RoPE.
  Cross values: learned projection, no value normalization.
- Caption-only; T2M max200 variable frames, TI2M97 aligned frames.
- T2M/TI2M .75/.25; BONES50% within T2M. Expected source mass:
  Nymeria-T2M37.5%, BONES-T2M37.5%, Nymeria-TI2M25%.
- Camera/head-recanonicalized Nymeria UniEgo283 and legacy original BONES;
  same camera/head statistics, floor calibration and rejection guards.
- Global256 = two ranks × batch128 × accumulation1; 8 workers/rank.
- AdamW LR2e-4, warmup1000, cosine floor0, betas.9/.99, decay0, clip1,
  seed0, TF32off. Original T2M/TI2M caption dropout behavior preserved.
- Feature/centered-decoded-joint/smooth losses1/1/5;
  contact/planted-foot-velocity/foot-height .05/1/10.
- Checkpoints only at positive10k steps; visuals at start/every5k,
  five Nymeria cases per task plus two BONES, UniPC35/CFG2.

## Acceptance and startup corrections

All numerical/data checks ran on owned compute, not the login node.
Existing no-shape unit tests passed. Real DDP global256 updates1–2, save,
strict optimizer/RNG reload, then update3 passed with nonzero weight changes.
Actual T2M/TI2M predictions after reload were bit-for-bit identical for real,
absent and NaN shape inputs. `GATE_PASSED.json` records these checks.

W&B membership was verified and the production entity set to `scene_motion`.
Preflight deliberately handed off to a new production process after its
checks, retaining the parent srun allocation.

The first production attempt stopped before any update because baseline
fixed-sample manifests lacked the no-shape ablation field. Operation-local
`prepare_fixed_viz.py` now copies the exact suites and labels only these new
run-local manifests with the no-shape contract. It verifies every sample
record and NPZ SHA256 against the unchanged baseline; `FIXED_VIZ_PARITY.json`
records this. The existing strict visualization contract guard remains active.
The relaunch reuses the same W&B ID. It starts fresh from step0.

## BONES storage limitation

Full manifest/source audit: **125,933 unique training sources / 585,312 rows,
zero missing training files**. Thus the requested training mixture is intact.
However, **6,849 of 6,993 unique BONES validation sources are absent** from
the current Scratch2 mirror. Do not claim complete BONES validation coverage.
This run reuses the baseline's exact saved held-out visualization arrays,
which exist, rather than selecting different cases. Full future BONES
benchmarking requires restoring the missing held-out files.
See `BONES_SOURCE_AUDIT.json`, `DATA_READY.json` and missing-source lists.

## Production verification

Consult `production_relaunch.log`, the output's `train_job2622173.log`,
W&B and `PRODUCTION_RUNNING.json` for actual progress. Startup/preflight
success alone is not evidence of completed production optimizer updates.

2026-11-05 approximately21:00 KST: production reached **step20/200000**,
finite loss1.233869 and nonzero gradient norm36.1178, zero guarded batches
and zero skipped updates. All three source/task combinations were observed.
Recent speed over steps2–20 was0.4141 updates/s (2.41s/update); this is an
early measurement with cold/stale motion-cache fallbacks, not a steady-state
throughput claim. Peak PyTorch allocation21.13GiB on rank0, observed device
memory approximately23GB each. At this initial rate, the full200k exceeds
this allocation's remaining lifetime; resume from this run's latest10k
checkpoint on a replacement allocation rather than changing its recipe.
