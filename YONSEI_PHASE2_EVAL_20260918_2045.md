# Yonsei Phase-2 evaluation follow-up, September18 at20:45 KST

Scheduled as the five-hour follow-up to Nymeria evaluator implementation. This is a completed BONES evaluation snapshot, not a live training dashboard.

Updated September 18, 2026 after the scheduled 20:45 KST sweep. Five new checkpoint evaluations completed at 21:16:33 KST, exit0, with no missing results.
Checkpoints were pinned at 20:46:48 KST; checkpoints written later are not included.
2374 fixed BONES repetition/overview cases; UniPC35, CFG2, shift3, fixed per-case seeds.
Generator statistics are checkpoint-specific; independent BONES-trained C45 evaluator statistics are unchanged.
Different steps are labeled, not treated as step-matched ablations. Nano and unchanged rows reuse earlier completed results.

| Run | Step | R@3 % | FID | Contact skate m/s | Foot penetration cm | Airborne % |
|---|---:|---:|---:|---:|---:|---:|
| GT | - | 93.93 | 0.00000 | 0.0213 | 0.0352 | 2.34 |
| ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35 | 200000 | 82.31 | 0.00930 | 0.0426 | 0.0335 | 4.43 |
| edge_7layer_reasoner_kv_project_bones100_t2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916 | 200000 | 66.68 | 0.01933 | 0.0425 | 0.0687 | 3.94 |
| edge_7layer_reasoner_kv_project_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917 | 150000 | 61.50 | 0.02487 | 0.0419 | 0.6408 | 6.39 |
| edge_7layer_self_cross_final_hidden_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916 | 180000 | 57.29 | 0.03552 | 0.0448 | 0.2908 | 2.62 |
| edge_7layer_reasoner_kv_project_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916 | 200000 | 47.77 | 0.04050 | 0.0447 | 0.0620 | 2.49 |
| edge_7layer_reasoner_kv_project_vnorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917 | 110000 | 45.49 | 0.06893 | 0.0536 | 2.4349 | 26.74 |
| edge_7layer_self_cross_final_hidden_knorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917 | 70000 | 44.99 | 0.08790 | 0.0604 | 2.5354 | 23.93 |
| edge_7layer_self_cross_layer_kv_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916 | 200000 | 38.04 | 0.04984 | 0.0445 | 0.0607 | 3.41 |
| edge_7layer_reasoner_k_fp32_bones100_t2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915 | 200000 | 32.48 | 0.04788 | 0.0421 | 0.0650 | 8.18 |
| edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_nanoparity_gbs256_lr0_w1_1_5_contact_c0p05_v1_h10_yonsei_20260913 | 175000 | 25.32 | 0.05655 | 0.0389 | 0.0449 | 10.82 |
| edge_7layer_motion_fp32_bones100_t2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915 | 150000 | 21.36 | 0.05788 | 0.0440 | 0.1645 | 26.96 |
| edge_7layer_reasoner_k_fp32_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915 | 200000 | 18.20 | 0.07049 | 0.0455 | 0.0604 | 4.95 |
| edge_7layer_reasoner_kv_as_is_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915 | 180000 | 16.68 | 0.07249 | 0.0441 | 0.2893 | 3.04 |
| edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_nanoparity_gbs128_lr0_w1_1_5_contact_c0p05_v1_h10_yonsei_20260913 | 200000 | 15.42 | 0.07580 | 0.0441 | 0.1355 | 2.56 |
| edge_7layer_bones100_t2m_camheadstats_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912 | 200000 | 14.74 | 0.08272 | 0.0513 | 1.5224 | 14.85 |
| edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_ddp2_bs128_gbs256_lr0_w1_1_5_contact_c0p05_v1_h10_yonsei_20260913 | 200000 | 13.14 | 0.07879 | 0.0511 | 0.0938 | 3.37 |
| edge_7layer_motion_fp32_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260915 | 110000 | 10.36 | 0.12032 | 0.0546 | 5.2866 | 18.97 |
| edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_globalfeat10_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912 | 200000 | 7.75 | 0.14050 | 0.0643 | 4.8537 | 21.90 |
| edge_7ja6sa_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912 | 200000 | 6.32 | 0.17093 | 0.0547 | 0.0361 | 73.96 |
| edge_7layer_nymeria_bones50_t2m_ti2m_uniego283_nanostats_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260910 | 200000 | 5.39 | 0.12429 | 0.0661 | 5.8733 | 3.63 |
| edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_absjoint_w1_1_5_contact_c0p05_v1_h10_yonsei_20260911 | 200000 | 3.16 | 0.29178 | 0.0641 | 0.9214 | 18.80 |
| edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260908 | 90000 | 2.99 | 0.31309 | 0.0772 | 14.0850 | 10.40 |

Missing/incomplete results: 0.

## Changes since the previous completed report

| Run | Previous step | New step | R@3 %: previous -> new | FID: previous -> new | Foot penetration cm: previous -> new | Airborne %: previous -> new |
|---|---:|---:|---:|---:|---:|---:|
| Projection A mixed, batch256 | 70000 | 150000 | 39.64 -> 61.50 | 0.12447 -> 0.02487 | 9.7042 -> 0.6408 | 19.84 -> 6.39 |
| Projected K/V norm mixed, batch128 | 30000 | 110000 | 13.44 -> 45.49 | 0.41743 -> 0.06893 | 10.8299 -> 2.4349 | 66.28 -> 26.74 |
| Self-cross final hidden + cross-K norm, batch256 | 10000 | 70000 | 3.92 -> 44.99 | 1.04457 -> 0.08790 | 0.4525 -> 2.5354 | 93.66 -> 23.93 |
| Self-cross final hidden, batch128 | 110000 | 180000 | 45.20 -> 57.29 | 0.09952 -> 0.03552 | 4.2929 -> 0.2908 | 30.17 -> 2.62 |
| Self-cross layer K/V, batch128 | 130000 | 200000 | 33.11 -> 38.04 | 0.08538 -> 0.04984 | 2.1752 -> 0.0607 | 33.77 -> 3.41 |

GT reference for the same cases: R@3=93.93%, self-FID=0, contact skating=0.0213m/s, foot penetration=0.0352cm, airborne=2.34%.

- Projection A mixed batch256 is the strongest newly evaluated mixed run: R@3 61.50, FID0.02487. It still trails Nano's82.31 and the Projection A BONES-only200k reference66.68.
- Final-hidden batch128 improves to R@3 57.29 and FID0.03552. Its airborne rate is2.62%, close to GT2.34%, but mean foot penetration0.2908cm remains above GT0.0352cm.
- Layer-K/V200k improves modestly in retrieval to38.04 while its physical metrics improve substantially: foot penetration0.0607cm and airborne3.41%. Retrieval and floor behavior are distinct outcomes.
- Cross-K-norm70k and projected-K/V-norm110k improve retrieval but still have approximately24-27% airborne frames and2.4-2.5cm foot penetration. They do not establish that floating is solved.
- Cross-K-norm foot penetration increased from0.4525cm at10k to2.5354cm at70k while the airborne fraction fell. A retrieval gain is not uniformly improved physical behavior.
- These are different-step, different-exposure comparisons, not causal or significance-tested architecture/batch-size conclusions.

## Sources

- [Scheduled five-checkpoint sweep](/scratch2/whwjdqls99/cosmos/operations/nymeria_c45_20260918/phase2_latest_due20260918_2045_job2281723/COMPARISON.md)
- [Machine-readable comparison, including each run's GT](/scratch2/whwjdqls99/cosmos/operations/nymeria_c45_20260918/phase2_latest_due20260918_2045_job2281723/comparison.json)
- [Previous six-checkpoint sweep](/scratch2/whwjdqls99/cosmos/operations/phase2_autopilot_eval_20260918_0226_job2270089/COMPARISON.md)
- [Projection A mixed128 final200k](/scratch2/whwjdqls99/cosmos/operations/phase2_projectionA_mixed128_final200k_20260918_job2248639/COMPARISON.md)

## Separate Nymeria evaluator experiment

The new Nymeria C45 pipeline is implemented and official initialization weights were downloaded, but training has NOT started. Preparation stopped on identical duplicate source windows before any optimizer update. The deduplication correction is awaiting user approval. There is no Nymeria-trained evaluator checkpoint or test score yet; the requested80-90% target has not been achieved. These BONES Phase-2 results are not a substitute for that experiment.

## Exact newly evaluated checkpoint provenance

- Projection A mixed, batch256: step150000; run `edge_7layer_reasoner_kv_project_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917`; summary `/scratch2/whwjdqls99/cosmos/operations/nymeria_c45_20260918/phase2_latest_due20260918_2045_job2281723/results/edge_7layer_reasoner_kv_project_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917_step150000__bones_repetition_overview/summary.json`.
- Projected K/V norm mixed, batch128: step110000; run `edge_7layer_reasoner_kv_project_vnorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917`; summary `/scratch2/whwjdqls99/cosmos/operations/nymeria_c45_20260918/phase2_latest_due20260918_2045_job2281723/results/edge_7layer_reasoner_kv_project_vnorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917_step110000__bones_repetition_overview/summary.json`.
- Self-cross final hidden + cross-K norm, batch256: step70000; run `edge_7layer_self_cross_final_hidden_knorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917`; summary `/scratch2/whwjdqls99/cosmos/operations/nymeria_c45_20260918/phase2_latest_due20260918_2045_job2281723/results/edge_7layer_self_cross_final_hidden_knorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260917_step70000__bones_repetition_overview/summary.json`.
- Self-cross final hidden, batch128: step180000; run `edge_7layer_self_cross_final_hidden_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916`; summary `/scratch2/whwjdqls99/cosmos/operations/nymeria_c45_20260918/phase2_latest_due20260918_2045_job2281723/results/edge_7layer_self_cross_final_hidden_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916_step180000__bones_repetition_overview/summary.json`.
- Self-cross layer K/V, batch128: step200000; run `edge_7layer_self_cross_layer_kv_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916`; summary `/scratch2/whwjdqls99/cosmos/operations/nymeria_c45_20260918/phase2_latest_due20260918_2045_job2281723/results/edge_7layer_self_cross_layer_kv_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260916_step200000__bones_repetition_overview/summary.json`.
