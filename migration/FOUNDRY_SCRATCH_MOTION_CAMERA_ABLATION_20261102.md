# Camera ablation without learned Phase2 motion weights

Production started fresh on sep-1 GPUs5,6; verified through step30 at
2026-11-02 21:05UTC with finite losses/gradients and remotely verified W&B.

- W&B: https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-b0122564c04ef74531b6
- Operation: `/mnt/local/jinhyung.kim/cosmos/operations/phase3_camera_prediction_scratch_motion_20261102/retry_with_production_warmup`
- Run: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_imagejoint402020_camera_prediction_scratchmotion_T97_gbs32_sep1_20261102_warmup`
- Evidence: operation `RUNNING_VERIFIED.json`, `plan.json`, pinned
  `source_snapshot`; run `contract.json`, `sources.json`, `launch/status.json`.

User explicitly selected training the fresh motion expert with the bridges.
Trainable:318605083 motion parameters and117469198 bridge parameters. The
Phase1 Cosmos reasoner, generator, LoRA and video/camera projections stay
pretrained and frozen. Motion starts from the original Phase2 constructor,
including its zero-initialized output projection. No learned Phase2 motion
weights are loaded. The5639-byte architecture artifact contains tensor-free
metadata from the reference camera run's saved JSON contract, not a checkpoint.

All other camera-ablation settings match:112503-window cohort; T97/20fps/256;
V2M/M2V/joint40/40/20; original caption/shape policies and native losses;
bidirectional bridges and clean motion E(0); global32=2x16; AdamW2e-4,
betas0.9/0.99, no decay, global clip1;1000-update warmup then cosine to zero
at200k; noisy-GT camera tokens in native generator attention, with camera
output detached and camera loss0. Same video-motion bridge token scope.

Twelve CPU checks passed. Real two-rank preflight verified native parity,
fresh motion updates, all unchanged camera audits, frozen Cosmos and complete
motion/bridge/optimizer save-reload-continue. Fresh4/reload2 probes passed;
production starts again at step0. Full checkpoints now include `motion_state`
alongside bridges, optimizer, schedule, per-rank RNG and global data cursor.
Use `motion_expert_phase3_edge_image_joint_camera_scratch/sample.py` for loading
both trained components and the same inference sampler.

The first disposable preflight inherited an immediate peak learning rate from
the frozen-motion diagnostic and failed its overfit check. That failure remains
preserved in the parent operation. The successful scratch preflight uses the
unchanged production warmup for all13 disposable updates, including its resumed
update. Its loss decreased4.16353→4.15284. Production learning rates and all
other recipe values were never tuned or changed. The `_warmup` suffix identifies
the corrected preflight attempt, not a different production schedule.

The original joint baseline on0,4 and pretrained-motion camera ablation on2,3
continue unchanged. All repeated reads and artifacts remain on local NVMe.
No existing process was stopped, shared data transferred, or Git change pushed.
