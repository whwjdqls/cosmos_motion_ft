# Directional 70k EMA: fixed71 CFG sweep

User-selected checkpoint: directional Phase 3, video weight 10, step 70,000,
EMA decay 0.999 (19,181 updates since step 50,819). The newer checkpoints and
the additional video-weight-1 model were not substituted.

Operation:
`/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_directional70k_cfg_sweep_20261107T092224Z`

Checkpoint SHA-256:
`23a0f51176455b685b135ab50e2b6e83a65d16f9a3bfce3c518acc70f535ab9d`

## Best tested settings

| Task | Reference scheme | Constant CFG | Selection metric |
| --- | --- | ---: | ---: |
| V2M | All seven whole bridges disabled | 2.5 | MPJPE 18.001958 cm |
| M2V | All seven whole bridges disabled | 1.1 | DreamSim 0.138729 |

V2M: PA-MPJPE 8.814480 cm, root ADE/FDE 16.143032/23.630016 cm,
TMR FID 0.246010, R@1/2/3 31.25/45.833333/58.333333%.
M2V: PSNR 19.441488 dB, SSIM 0.604408, LPIPS-Alex 0.258496,
I3D-FVD 289.312293, VideoMAE CD-FVD 300.363554.

V2M whole-off2 has MPJPE 18.465373 cm and better acceleration error/TMR FID
than whole-off2.5. The paired MPJPE difference (2.5 minus 2) is -0.463415 cm,
95% bootstrap CI [-1.804987, 0.482782]; improvement over 2 is uncertain.
Against native1, MPJPE improves by 3.860976 cm, CI [-7.141422, -0.945575].

M2V whole-off1.1 minus native1 DreamSim is -0.003279,
CI [-0.004984, -0.001683]; versus previous whole-off2 it is -0.020279,
CI [-0.024179, -0.016562]. Its difference from best cross-only1.5 is
-0.001192, CI [-0.003520, 0.000922], so the two schemes are not clearly
separated on this cohort. All intervals use 5,000 paired recording resamples,
seed 20261029, at fixed generation seeds.

## Protocol

- Exactly the original 71 windows (71 recordings), order and seeds 0–70.
  No dataset resampling. Captions empty in both specialists and every pass.
- Native 35-step UniPC; motion shift 3, video shift 10. EMA weights,
  supplied neutral shape, BF16 sampling. V2M batch 64, M2V batch 71.
- V2M receives the full clean video. M2V receives full clean 283-D motion
  and one initial image latent, preserved after every solver update.
- Coarse: shared native1 plus whole-off and cross-edges-off at 0.5/1.5/2/3.
  Refine: V2M whole2.5/4/6, cross0.9/1.1, whole3 interval; M2V cross1.25/1.75,
  whole0.9/1.1, cross1.5 interval. Interval sigma [0.2,0.8], native1 outside.
  Total 15 V2M and 14 M2V settings, each on all 71 cases. Whole-off2 reuses
  the previous verified exact71 results. There are 2,059 output records.
- Whole reference disables all seven bridge modules including within-modal
  updates. Cross reference removes only active target-query/source-key edges,
  retaining all seven modules and unchanged within-modal edges/shape bypass.
  Both pass native1 bit-exact, dense/grouped-mask and hook acceptance checks.
- Matched TMR 91f9d8a… and original reconstruction/physics pipeline.
  Video metrics score lossless future RGB frames 1–96; canonical FP32 I3D400
  and VideoMAE1408 features, all71 population-covariance FVD. Current seven-case
  GT feature parity is exact against the verified cached full71 reference.

These are the best point estimates among tested settings on this fixed test
cohort. Tuning and selection use this cohort; no independent validation or
global continuous optimum is claimed. Other metric winners and Pareto
tradeoffs are in the full report.

## Publication and checks

[Notion report](https://app.notion.com/p/410d2a44f44b8098b1b7c7af42d2353b),
new top toggle “Directional 70k EMA — CFG scale, reference and interval search
(2026-11-07)”. Earlier evaluations preserved. Includes all 85 table rows,
four-panel sweep plot, six videos for fixed cases 0/28/70, and two evidence ZIPs.

Authoritative completion: operation `COMPLETE.json`, `notion_publication.json`,
and nine `verified_uploads/` size/SHA receipts. `summary_final.json`, `report.md`,
all per-case metrics, embeddings/features, source/array manifests and sampler
ledgers remain local. GPU4 released; all three trainers left running unchanged.
