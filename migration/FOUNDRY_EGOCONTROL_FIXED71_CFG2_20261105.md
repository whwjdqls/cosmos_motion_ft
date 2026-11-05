# EgoControl EMA200k: fixed71 native image-CFG2 evaluation

[Verified quantitative report](https://app.notion.com/p/40fd2a44f44b80489a0bdecc5d9410bb). Completed 2026-11-05T02:33:41.176413+00:00 on sep-1.

All 71 original fixed windows were freshly generated and scored, one per held-out recording, zero exclusions. Seeds follow the matched71 manifest, 0–70; the earlier full sweep used seed0 throughout. RGB and motion arrays match the full-test preparation exactly. This is a fresh run, not a subset aggregate of old predictions.

| PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ | DreamSim ↓ | Video FVD ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 20.301307 | 0.644898 | 0.221591 | 0.144942 | 376.908678 |

Use the trained EgoControl Edge EMA200k with strict native DCP loading, first RGB frame plus full GT Body23 controls, no semantic caption or camera-action tokens. Encode only frame0. Native UniPC35, video shift10, T97/20fps/256², batch1, TF32 off, native precision with no added autocast wrapper. The unchanged pose representation constructs 96×138 controls from the original normalized SOMA motion and saved statistics.

**Reference CFG2 is image guidance with pose retained in both passes.** Null-image zeros the first image latent while preserving the clean timestep/mask and identical adapted caption metadata. Formula: `v_cond + 2*(v_cond-v_null_image)`, equivalent to Edge standard guidance3. Every sample executes35 conditional +35 null-image calls. This is distinct from motion/shape bridge guidance. The Edge adaptation lacks the original Predict2 negative-text encoder.

Native RNG is unchanged; equal integer seeds do not assert identical Gaussian realizations across different samplers/devices. The first window uses seed0 in both this run and the previous full sweep and reproduces its saved uint8 output bit-for-bit (pixel RMS0).

Metrics score only lossless generated RGB frames1–96, excluding observed frame0, before MP4 compression. Same PSNR/SSIM/LPIPS/DreamSim backend and weights as the existing M2V report. FVD uses I3D Kinetics400 logits: six16-frame suffix segments/window,224 crop,[-1,1], one global float64 Fréchet over426 features. All71 GT I3D comparisons pass against the prior fixed71 preparation; maximum absolute difference is 3.3378601e-05. Do not compare this FVD directly to the whole-test70,062-segment result.

The report includes population SD and5000-recording-bootstrap95% intervals for means, seed20261105. No FVD interval or motion-adherence claim. EgoControl has a different trained adaptation/conditioning representation and M2V exposure from the frozen-specialist Phase3 models.

Scientific Python/TOML fingerprints match the training contract. A pre-existing recorded host launcher differs and was not invoked. All trainers continued during temporary sharing of our GPUs1/5; both evaluation workers exited and released GPU memory. All repeated reads and outputs are local NVMe, with no shared transfers.

Checkpoint: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/cosmos3_camera/egocontrol_edge/egocontrol_edge_7cond_m2v_p1ema200k_phase1recipe_T97_gbs32_200k_yonsei_20261026/checkpoints/iter_000200000`. Model metadata SHA256 `2985d773da18d1d8d7bb999da8eb98b1896f1aa473097de167385ba826136f41`.

Local operation: `/mnt/local/jinhyung.kim/cosmos/operations/eval_egocontrol_matched71_cfg2_20261105`. Root COMPLETE.json is published_verified: 8 numeric table rows and the evidence ZIP were read/downloaded back and checked. ZIP SHA256 `f348f39cb50c4ff63f4a6b7a81181adaa8ff47d62d0c5d6ff49f73671dcbcb3e` (1,624,937 bytes). Exact cohort/seeds/mapping, per-window/per-frame metrics, I3D features, resolved config, source/CFG identities and reproducible code are attached; fresh prediction arrays remain local.
