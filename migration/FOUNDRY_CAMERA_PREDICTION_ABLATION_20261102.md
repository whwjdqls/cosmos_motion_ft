# sep-1 camera-prediction ablation

Started fresh on2026-11-02, GPUs2,3. Production was verified through step10
at20:00UTC, with finite losses/gradients and live W&B metrics. Consult the
operation receipts and live logs for current progress.

- Operation: `/mnt/local/jinhyung.kim/cosmos/operations/phase3_camera_prediction_launch_20261102`
- Output: `/mnt/local/jinhyung.kim/cosmos/weka/cosmos_motion_ft_runs/motion_expert_phase3_edge/edge_p3_imagejoint402020_bidir_camera_prediction_nocamloss_T97_gbs32_sep1_20261102_2gpu`
- W&B: https://wandb.ai/jungbinc-upenn/cosmos-motion-ft/runs/p3-b0c2529274be38864cbb
- Evidence: operation `RUNNING_VERIFIED.json`, `plan.json`, pinned
  `source_snapshot`, and output `contract.json` / `launch/status.json`.

Matches the running image-joint baseline's saved source hashes, frozen Phase1
LoRA+video-projection EMA200k, selected Phase2-200k,112503-window cohort, caption
policy and original objectives. Baseline GPUs0,4 continue unchanged.

Recipe: V2M40%, M2V40%, image-only joint20%; bidirectional conditioning,
clean motion timestep zero; global32=2x16; AdamW2e-4, betas0.9/0.99, no weight
decay, clip1; warmup1000 followed by cosine to zero at fresh200k updates.

Every task adds96 noisy GT camera tokens to native generator attention,
raw relative SE(3)9 padded64, domain2, no clean camera tokens. Camera time uses
motion sigma for V2M and video sigma for M2V/joint. Original seven video-motion
bridges retain their1600/98 token scope. Only those117469198 bridge parameters
train; specialists and camera projections are frozen. Camera readout is
detached and excluded from the loss. Noisy GT camera supplies partial camera
information; do not attribute improvements solely to an unused output head.

Real two-GPU/microbatch16 preflight passed native parity, finite gradients,
frozen specialists, twelve disposable overfit updates, all-three-task camera
audits and zero camera loss/readout gradients. Fresh4-update and reload2-update
probes passed. Production starts again at step0 using the original specialists,
never disposable probe weights. All repeated reads/checkpoints are node-local.

Earlier queued single-GPU attempt passed preflight but aborted at the next
resource check; no production updates occurred. That attempt and its failed
monitor are preserved under `phase3_camera_prediction_idle_queue_20261102`.
The new launcher allows up to60seconds for GPU process release between stages
and still refuses occupied GPUs. It never terminates another job.
