# Camera evaluation and video-weight-one ablation on sep-1

The user requested a fresh copy of the pretrained-motion camera-prediction
experiment with video loss weight1 instead of10, then explicitly asked to
evaluate the existing camera model and publish its results before starting
the new ablation. All substantial reads/writes are on node-local NVMe.

## Completed interim camera evaluation

Operation:
`/mnt/local/jinhyung.kim/cosmos/operations/eval_camera_interim40k_video_text_matched71_20261103`.
`COMPLETE.json` is `published_verified` at2026-11-03 22:05UTC. The latest
available saved checkpoint was40k, while training was beyond45k. Its hardlink
is preserved locally, SHA256
`77d83d2c50388a8fe9c3e6725b6c368912f596d6f7b0d628e7cc7b421c3032b9`.

Same71 held-out recording windows, original case seeds, supplied neutral
shapes, T97/20fps, native guidance1 and UniPC35/motion shift3 as the earlier
matched reports. GPUs1,7 evaluated disjoint case shards without sharing a
trainer.142 predictions, zero skipped cases; captions reach both reasoners,
zero motion-target-template invariance passes, and no GT camera is supplied.
Matched TMR91f9d8…, original normalization/text rows and train-only Body23→C45
completion reproduce the reference GT retrieval/diagnostics.

| Camera40k condition | MPJPE cm | PA-MPJPE cm | Motion FID | R@1 % |
|---|---:|---:|---:|---:|
| Video |30.0712|15.6146|0.442398|18.7500|
| Video+text |22.5612|12.3882|0.146880|40.1042|

Text MPJPE difference−7.5100cm, paired recording-bootstrap95% interval
[−11.6202,−3.5498]cm,5000 draws/seed20261103. Full nine physical metrics,
FID and R@1/2/3, per-case evidence and scripts are published on
https://app.notion.com/p/40dd2a44f44b80459383d6df25184a73 . Both saved numeric
tables and the attached536051-byte ZIP were read back and verified; archive
SHA256 `e9c3a49b0b38f9b530788cbe34d901124665ffa0962a38b108eb8041d9a25929`.
The baseline on that page is50k, so this interim40k comparison is not matched
in training exposure. The independent original exact50k camera watcher and
evaluation/publication queue remain active.

## Video loss weight-one experiment

Production and remote W&B configuration were verified at2026-11-03 22:10UTC:
fresh step30, approximately2.02s/update, GPUs1,7, finite loss/gradients.
W&B: https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-8545d27c753b644f9ce1 .
Original baseline/camera/scratch jobs continue on0,4/2,3/5,6 respectively.
The new production step1 also exactly matches the original run's unweighted
video/motion losses and task partition, with the expected new total loss.

Code: `motion_expert_phase3_edge_image_joint_camera_wv1/`.
Operation:
`/mnt/local/jinhyung.kim/cosmos/operations/phase3_camera_prediction_video_weight1_20261103`.
Run under local `weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/`:
`edge_p3_imagejoint402020_bidir_camera_prediction_nocamloss_wv1_T97_gbs32_sep1_20261103`.

The original camera contract/source is the control. Only experiment identity
and video multiplier10→1 differ. Same frozen Phase1 video-I/O EMA200k and
selected pretrained Phase2 200k, original117469198 bridge parameters, noisy GT
camera in native attention, no camera loss,112503-window cohort and camera
cache.40/40/20 V2M/M2V/joint, caption policy and camera/noise RNG unchanged.
Fresh global32=2x16 on GPUs1,7, AdamW2e-4/betas0.9,0.99/no decay/clip1,
1000-step warmup then cosine to zero at200k; BF16 model, FP32 bridges and
motion I/O, TF32 off. No existing run is stopped or reconfigured.

The video multiplier applies to M2V and joint video velocity targets. Motion
physical weights, including foot-height10, remain identical. Source-guarded
adapters reuse the original formulas and trainer without editing the original
camera/shared packages; weighted-video logging also reflects weight1. A new
contract fingerprints this adapter and rejects weight10 checkpoint resumes.

Two CPU tests cover all four task partitions with actual native loss/gradient
comparisons, unchanged motion gradients/clean image masks, absent camera
gradients, recipe parity and wrong-checkpoint rejection. Two-GPU/microbatch16
acceptance passes native specialist parity, freeze/backward/replica checks,
12 disposable overfit updates and all camera audits. Fresh4/reload2 probes
pass; production starts fresh at0, never from disposable weights.
`FIRST_UPDATE_PARITY.json` confirms the original weight10 and new weight1
first updates have identical raw video/motion losses and task counts, with
the expected total-loss change.

The launch gate waits for the interim evaluation's verified Notion publication
and release of both GPUs before entering acceptance. See live `status.json`,
`launch/status.json`, operation `RUNNING_VERIFIED.json` and W&B for production
status. W&B deterministic ID: `p3-8545d27c753b644f9ce1`, project
`jungbinc-upenn/cosmos-motion-ft`. Code/documentation additions are local until
explicitly committed/pushed; running jobs use their pinned source snapshot.
