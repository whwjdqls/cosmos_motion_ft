# Yonsei Phase 2 handover and completed evaluations

Status snapshot: **2026-09-18 15:19:43 KST**.
Latest completed evaluation: Projection A mixed batch128 final200k, completed
2026-09-18 04:03:50 KST. A newer training checkpoint is not necessarily evaluated.
This document is a portable repository-local summary, not a live dashboard.

## Server scope and operating rules

- This snapshot is from the Yonsei datacenter, not Grasp, A100, or H200 servers.
- Run model loading, dataset inspection, training, evaluation, and rendering only
  inside owned compute allocations. Never do heavy work on a login node.
- The active Codex goal checks current Slurm jobs and trainer logs every30minutes.
  The prepared helper files alone are not an independent automatic scheduler.
- Confirm actual job/process termination before replacing an allocation. A
  command observation timeout is not evidence that a job ended.
- Resume the same run and checkpoint contract. Do not restart a fresh run or
  change model, losses, LR, batch, sampler, or data merely during handover.
- Keep retained allocations unless the user explicitly requests cancellation.
  Job2270089 was canceled at the user's request to replace node37.
- Do not resurrect the intentionally canceled native/as-is run2250537 or the
  intentionally stopped Phase1 runs.
- Exclude cs-gpu-01, node31, and node33 from new Yonsei requests. The replacement
  final-hidden DDP request additionally excludes node37 after observed slowdown
  and repeated visualization-related NCCL timeouts there.
- Training launchers support COSMOS_SITE=yonsei or COSMOS_SITE=grasp. Use each
  server's restored_env.sh settings. Do not copy Yonsei absolute paths, Slurm
  job IDs, partitions, or node exclusions into Grasp commands.

## Current training snapshot

| Run | Job | Node | GPUs | Global batch | Observed step | Latest saved step | Recent steps/s |
|---|---:|---|---:|---:|---:|---:|---:|
| Self-cross final hidden, batch128 | 2275411 | node36 | 1 | 128 | 163860 | 160000 | 1.124 |
| Self-cross per-layer K/V, batch128 | 2263514 | node34 | 1 | 128 | 184120 | 180000 | 1.156 |
| Projection A mixed, batch256 | 2270088 | node36 | 2 | 256 | 130520 | 130000 | 1.219 |
| Self-cross final hidden + cross-K norm, batch256 | 2279743 | node34 | 2 | 256 | 52940 | 50000 | 1.075 |
| Projection A + projected K/V learnable norms | 2278800 | node40 | 1 | 128 | 88620 | 80000 | 1.249 |

All five runs target200000 optimizer steps. The observed logs report zero
skipped updates. Seven allocated RTX4090 GPUs are in use; no evaluation or
otherwise-idle allocation was present at this snapshot.
Single-GPU allocations use8CPUs/64GiB; two-GPU allocations use16CPUs/128GiB.

Completed runs:
- Projection A BONES-only batch128 finished200k on September17 and was evaluated.
- Projection A mixed batch128 finished200k on September18 and was evaluated.
- Their original allocations2248640/2248639 have ended. Do not resume completed
  trainers just because their old job IDs still appear in historical records.

Pending requests at the snapshot:
-2251809: Nano Phase2, two PRO6000 GPUs, Priority; existing resume watcher retained.
-2194983 and2194977: legacy four-PRO6000 requests, Priority. Do not automatically
  restart intentionally stopped Phase1 training if these requests allocate.

## Allocation handovers and node37 observations

- final_hidden_128: old2231899 expired; replacement2275411 on node36 resumed the
  same run from90000. Current training remains in the original output directory.
- projection_a_kvnorm_mixed_128: started September17 after BONES-only Projection A
  finished and the missing-latest evaluation sweep completed. It initially used
  job2248640. Dependent replacement2278800 started on node40 at
  2026-09-18 04:17:46 and restored40000 after2248640 expired.
- final_hidden_knorm_256: job2270089 on node37 hit the600-second NCCL collective
  timeout twice while rank0 saved/rendered and the other rank waited. The second
  failure occurred at15000; the latest saved checkpoint then was10000.
- The user requested a replacement avoiding node37. Job2279743 started on node34,
  restoring the same10000 checkpoint with all training settings unchanged.
  Its20000 checkpoint and all12 visualizations completed; training advanced
  afterward. It has since advanced past50000 without an observed recurrence.
- Recent training speed changed from approximately0.34steps/s on node37 to
  approximately1.08steps/s on node34. Previous passive observations showed Lustre
  waits and greater major-page-fault activity on node37. This supports a
  node-dependent I/O/cache contribution, not proof of defective GPUs or a
  definitive isolation of the storage bottleneck.
- The default600-second distributed timeout remains unchanged. The proposed
  configurable60-minute timeout was NOT implemented.
- Checkpoints remain10k-only. Allocation expiry can therefore discard progress
  since the last10k checkpoint; no off-interval recovery checkpoint was enabled.

## Run names and locations

Current and completed tracked Edge runs use this Yonsei output root:

```text
/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge
```

Older runs also reside under:

```text
/lustre/whwjdqls99/cosmos/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge
```

| Run ID | Exact directory name under the output root |
|---|---|
| final_hidden_128 | `edge_7layer_self_cross_final_hidden_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916` |
| layer_kv_128 | `edge_7layer_self_cross_layer_kv_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916` |
| projection_a_mixed_128 | `edge_7layer_reasoner_kv_project_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916` |
| projection_a_bones_128 | `edge_7layer_reasoner_kv_project_bones100_t2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916` |
| projection_a_mixed_256 | `edge_7layer_reasoner_kv_project_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917` |
| final_hidden_knorm_256 | `edge_7layer_self_cross_final_hidden_knorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917` |
| projection_a_kvnorm_mixed_128 | `edge_7layer_reasoner_kv_project_vnorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917` |

Trainer logs are run-local train_jobJOBID.log. Saved checkpoints use
checkpoints/step_000NNNNNN.pt; periodic visualizations use
visualizations/step_000NNNNNN. Always resolve the actual available checkpoint
rather than inferring it from the live training step.

## Reproduction and resume recipes

Run these from the repository root inside an owned allocation with the required
GPU count. The launchers use resume-auto and run/job locks. For Grasp, replace
COSMOS_SITE=yonsei with COSMOS_SITE=grasp and provide that site's allocation;
Yonsei checkpoints/data are not automatically copied to Grasp.

```bash
# final_hidden_128
COSMOS_SITE=yonsei bash motion_expert_t2m_edge/train_norm_ablation.sh self_cross_final_hidden mixed

# layer_kv_128
COSMOS_SITE=yonsei bash motion_expert_t2m_edge/train_norm_ablation.sh self_cross_layer_kv mixed

# projection_a_mixed_128
COSMOS_SITE=yonsei bash motion_expert_t2m_edge/train_norm_ablation.sh reasoner_kv_project mixed

# projection_a_bones_128
COSMOS_SITE=yonsei bash motion_expert_t2m_edge/train_norm_ablation.sh reasoner_kv_project bones

# projection_a_mixed_256
COSMOS_SITE=yonsei bash motion_expert_t2m_edge/train_kv_projection_ddp.sh

# final_hidden_knorm_256
COSMOS_SITE=yonsei bash motion_expert_t2m_edge/train_self_cross_knorm_ddp.sh

# projection_a_kvnorm_mixed_128
COSMOS_SITE=yonsei bash motion_expert_t2m_edge/train_norm_ablation.sh reasoner_kv_project_vnorm mixed
```

Do not launch the two completed Projection A batch128 runs again unless explicitly
requested. Their commands above are retained for provenance.

Shared current mixed-run recipe:
- Seven motion blocks; frozen Edge reasoner and frozen vision encoder.
- Batch128/GPU, accumulation1; two ranks give global256.
- T2M T=200; TI2M frames=97; eight loader workers/rank.
- T2M/TI2M task weights0.75/0.25. BONES fraction0.5 is WITHIN T2M, not50% of all
  tasks. BONES-only Projection A uses only BONES T2M instead.
- Camera/head-aligned Nymeria representation and camera/head-aligned statistics;
  main_actor_camera_wearer_v1 BONES caption policy.
- Feature/joint/smoothness weights1/1/5, centered decoded-joint loss,
  contact weight0.05, foot-velocity weight1, foot-height weight10.
- Extra global-rotation and global-translation weights are0 in these current runs.
- LR2e-4, warmup1000, cosine floor0, Adam betas0.9/0.99, weight decay0,
  gradient clip1, CFG dropout0.1, seed0.
- Per-task loss reduction, Nano loss guard, float32 motion I/O, TF32 disabled.
- Positive10k checkpoint interval, no off-interval recovery saves.
- Visualization every5k with the fixed suite:5 Nymeria T2M,5 Nymeria TI2M,
  and2 BONES T2M examples for newly launched/resumed processes. The older
  already-running Projection A mixed128 process retained its earlier10-case
  imported visualization code through completion.
- Default renderer: first-frame head-rotation axes, correct above/below-floor
  occlusion, wider framing and thinner skeleton, without numeric top overlays.
  Do not ground generated motions to make floating disappear.

Architecture distinctions:
- Projection A: reasoner K/V -> separate learned projections; projected K ->
  learned interface RMSNorm -> native RoPE. Projected V has no norm in original A.
- Requested new variant: the same Projection A plus a fresh trainable native
  RMSNorm on each projected128-D V head, with FP32 gains. Existing K norm is
  unchanged. Seven blocks add896parameters. V is not rotated; the frozen
  reasoner's own tensors/computation are not modified.
- Enable it with EDGE_REASONER_V_NORM_TRAINABLE=1 together with
  EDGE_REASONER_KV_PROJECTION=linear_identity_v1 and learned reasoner K norm.
  The default is0; old checkpoint contracts imply false. Strict checkpoint
  loading and inference contract resolution preserve architecture compatibility.
- Self-cross final hidden: motion self-attention then cross-attention to projected
  final reasoner hidden states, then FFN. The batch256 run additionally has
  EDGE_CROSS_ATTENTION_K_NORM=rmsnorm; the original batch128 run does not.
  Therefore these two runs are NOT a pure batch-size-only comparison.
- Self-cross per-layer K/V: motion self-attention then separate cross-attention
  using corresponding reasoner-layer K/V instead of the joint stream softmax.
- Projection A mixed128 versus mixed256 retains the original A interface; the
  added projected-V-norm variant is a separate run.

Important Yonsei paths:
```text
Environment: /lustre/whwjdqls99/cosmos/envs/cosmos
Python: /lustre/whwjdqls99/cosmos/envs/cosmos/bin/python3.13
Edge framework: /home/whwjdqls99/cosmos-framework-edge
Edge reasoner weights: /lustre/whwjdqls99/cosmos/weka/cosmos3_edge_dcp/model
Vision bundle: /lustre/whwjdqls99/cosmos/weka/Cosmos3-Edge/vision_encoder
Nymeria representation: /lustre/whwjdqls99/cosmos/weka/nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1
Generator mean: /lustre/whwjdqls99/cosmos/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_mean.npy
Generator std: /lustre/whwjdqls99/cosmos/weka/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_std.npy
Motion cache: /scratch2/whwjdqls99/cosmos/cache/uniego_arrays_v1
Image/condition cache: /scratch2/whwjdqls99/cosmos/cache/phase2_images_v1
Evaluator bundle: /lustre/whwjdqls99/cosmos/weka/shape_aware_motion_eval_c45_20260715
```

The cache paths are on shared Lustre-backed storage, not node-local SSD.

## Evaluation contract and latest completed results

-2374 fixed BONES repetition/overview cases, existing per-case seeds.
-UniPC35, CFG2, shift3; inference batch32; no generated grounding.
-Full-cohort C45 retrieval with the established similar-caption positives and
 average-tie policy.
-Generator normalization comes from each checkpoint's own contract. The
 evaluator uses its separate unchanged C45 mean/std; do not interchange them.
-Always report GT retrieval and physical metrics alongside generated results.
 GT FID is0 by definition because GT is compared with itself.
-Airborne is the percentage of frames with all feet above5cm, not a direct
 causal diagnosis of floating. GT can naturally contain airborne frames.
-Checkpoint steps differ. Do not treat the following as step-matched architectural
 or batch-size ablations, or claim current training quality from an old evaluation.

| Model | Evaluated step | R@3 % | FID | Contact skate m/s | Foot penetration cm | Airborne % |
|---|---:|---:|---:|---:|---:|---:|
| GT | - | 93.93 | 0.00000 | 0.0213 | 0.0352 | 2.34 |
| Nano reference | 200000 | 82.31 | 0.00930 | 0.0426 | 0.0335 | 4.43 |
| Projection A BONES-only, batch128 | 200000 | 66.68 | 0.01933 | 0.0425 | 0.0687 | 3.94 |
| Projection A mixed, batch128 | 200000 | 47.77 | 0.04050 | 0.0447 | 0.0620 | 2.49 |
| Self-cross final hidden, batch128 | 110000 | 45.20 | 0.09952 | 0.0679 | 4.2929 | 30.17 |
| Projection A mixed, batch256 | 70000 | 39.64 | 0.12447 | 0.0616 | 9.7042 | 19.84 |
| Self-cross per-layer K/V, batch128 | 130000 | 33.11 | 0.08538 | 0.0524 | 2.1752 | 33.77 |
| Learned K norm, BONES-only | 200000 | 32.48 | 0.04788 | 0.0421 | 0.0650 | 8.18 |
| Parity baseline, batch256 | 175000 | 25.32 | 0.05655 | 0.0389 | 0.0449 | 10.82 |
| FP32 motion norm, BONES-only | 150000 | 21.36 | 0.05788 | 0.0440 | 0.1645 | 26.96 |
| Learned K norm, mixed | 200000 | 18.20 | 0.07049 | 0.0455 | 0.0604 | 4.95 |
| Native/as-is K/V, mixed | 180000 | 16.68 | 0.07249 | 0.0441 | 0.2893 | 3.04 |
| Parity baseline, batch128 | 200000 | 15.42 | 0.07580 | 0.0441 | 0.1355 | 2.56 |
| Original BONES-only | 200000 | 14.74 | 0.08272 | 0.0513 | 1.5224 | 14.85 |
| Projection A + projected K/V learnable norms | 30000 | 13.44 | 0.41743 | 0.1319 | 10.8299 | 66.28 |
| Earlier DDP batch256 | 200000 | 13.14 | 0.07879 | 0.0511 | 0.0938 | 3.37 |
| FP32 motion norm, mixed | 110000 | 10.36 | 0.12032 | 0.0546 | 5.2866 | 18.97 |
| Global-feature weight10 | 200000 | 7.75 | 0.14050 | 0.0643 | 4.8537 | 21.90 |
| 7 joint-attention +6 self-attention | 200000 | 6.32 | 0.17093 | 0.0547 | 0.0361 | 73.96 |
| Original representation + Nano stats | 200000 | 5.39 | 0.12429 | 0.0661 | 5.8733 | 3.63 |
| Self-cross final hidden + cross-K norm, batch256 | 10000 | 3.92 | 1.04457 | 0.1171 | 0.4525 | 93.66 |
| Absolute-joint loss | 200000 | 3.16 | 0.29178 | 0.0641 | 0.9214 | 18.80 |
| Original camera/head baseline | 90000 | 2.99 | 0.31309 | 0.0772 | 14.0850 | 10.40 |

Projection A mixed190k had R@3=48.36,FID=0.04079,contact skate=0.0444m/s,
foot penetration=0.0757cm,airborne=4.47%. At200k R@3 is47.77 while FID and
ground-contact metrics improve. No uncertainty estimate establishes whether the
small retrieval difference is meaningful.

## Evaluation provenance and operational helper cautions

All absolute artifact paths below are Yonsei-only. This repository-local table
preserves the numbers for Grasp readers; generated batches, manifests, source
measurements, and checkpoints are not copied by a Git pull.

```text
Manifest root:
/scratch2/whwjdqls99/cosmos/operations/phase2_quant200k_yonsei_20260914/manifests

Completed six-checkpoint sweep:
/scratch2/whwjdqls99/cosmos/operations/phase2_autopilot_eval_20260918_0226_job2270089

Completed Projection A mixed final200k:
/scratch2/whwjdqls99/cosmos/operations/phase2_projectionA_mixed128_final200k_20260918_job2248639

Consolidated completed-result table:
/scratch2/whwjdqls99/cosmos/operations/phase2_autopilot_20260917/LATEST_EVALUATIONS.md

Timestamped runtime snapshot:
/scratch2/whwjdqls99/cosmos/operations/phase2_autopilot_20260917/runtime_state.json
```

Each evaluation root retains configuration/request JSON, per-checkpoint results,
COMPARISON.md, comparison.json, and completion status. Exact-protocol and
checkpoint-identity matching governs reuse, including generator mean/std hashes.

Operational cautions:
- registry.json is the INITIAL September17 inventory, not current job state.
  runtime_state.json is also a timestamped snapshot, not a live daemon.
  Current squeue state and advancing trainer logs are authoritative.
- The correct idle-evaluation controller is evaluate_idle.py in the operations
  directory. Invoke it inside an owned idle single-GPU compute step with the
  established evaluation environment, using a new output root to pin a fresh
  snapshot or the same root to resume an interrupted pinned sweep.
- Do NOT use the prepared run_eval.sh wrapper: its final command still points
  at an older followup.py. All successful sweeps reported here bypassed it.
  That unused wrapper was not silently repaired during this documentation update.
- allocate_resume.sh was prepared for the original six run IDs. It does not
  cover the added V-norm run or include node37 in its default exclusion list.
  Use the recorded matching launcher, explicitly exclude the required nodes,
  and check for an existing queued replacement before submitting another.
- No standalone tests or model-code changes were performed for this documentation
  refresh. Git commit/push has not been performed as part of this refresh.

## Exact evaluated run identifiers

| Short label | Exact run name |
|---|---|
| Nano reference | `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35` |
| Projection A BONES-only, batch128 | `edge_7layer_reasoner_kv_project_bones100_t2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916` |
| Projection A mixed, batch128 | `edge_7layer_reasoner_kv_project_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916` |
| Self-cross final hidden, batch128 | `edge_7layer_self_cross_final_hidden_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916` |
| Projection A mixed, batch256 | `edge_7layer_reasoner_kv_project_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917` |
| Self-cross per-layer K/V, batch128 | `edge_7layer_self_cross_layer_kv_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916` |
| Learned K norm, BONES-only | `edge_7layer_reasoner_k_fp32_bones100_t2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915` |
| Parity baseline, batch256 | `edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_nanoparity_gbs256_lr0_w1_1_5_contact_c0p05_v1_h10_yonsei_20260913` |
| FP32 motion norm, BONES-only | `edge_7layer_motion_fp32_bones100_t2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915` |
| Learned K norm, mixed | `edge_7layer_reasoner_k_fp32_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915` |
| Native/as-is K/V, mixed | `edge_7layer_reasoner_kv_as_is_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915` |
| Parity baseline, batch128 | `edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_nanoparity_gbs128_lr0_w1_1_5_contact_c0p05_v1_h10_yonsei_20260913` |
| Original BONES-only | `edge_7layer_bones100_t2m_camheadstats_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912` |
| Projection A + projected K/V learnable norms | `edge_7layer_reasoner_kv_project_vnorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917` |
| Earlier DDP batch256 | `edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_ddp2_bs128_gbs256_lr0_w1_1_5_contact_c0p05_v1_h10_yonsei_20260913` |
| FP32 motion norm, mixed | `edge_7layer_motion_fp32_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915` |
| Global-feature weight10 | `edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_globalfeat10_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912` |
| 7 joint-attention +6 self-attention | `edge_7ja6sa_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912` |
| Original representation + Nano stats | `edge_7layer_nymeria_bones50_t2m_ti2m_uniego283_nanostats_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260910` |
| Self-cross final hidden + cross-K norm, batch256 | `edge_7layer_self_cross_final_hidden_knorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917` |
| Absolute-joint loss | `edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_absjoint_w1_1_5_contact_c0p05_v1_h10_yonsei_20260911` |
| Original camera/head baseline | `edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260908` |

