# Sep-2 handoff: bidirectional W1 camera-prediction EMA200k

Model ID: `sep1_bidir5050_w1_camera_nocamloss_EMA0`.
This is the completed camera base model at step **200000**, before trajectory
FiLM post-training. Use its **EMA weights** (decay 0.999; 200000 EMA updates).

## Shared checkpoint

```text
/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_models/phase3/bidirectional_w1_camera_prediction_ema200k_20261114/checkpoints/step_000200000.pt
```

Bytes: `3759189535`.
SHA256: `53e0f5f5e95594436e7a0dbb6f6c3b9a3838f54378e9fa2d32cb2a75dd486a65`.

The directory also contains `README_SEP2.md`, `HANDOFF.json`, `DEPENDENCIES.json`,
`contract.json`, `sources.json`, the launch plan/source manifest,
`source_snapshot.tar.gz`, `SHA256SUMS`, and the completion marker `READY.json`.
All 428 captured Python source files were checked against the launch manifest
before packaging. Shared payload checksums were verified after transfer;
`READY.json` is written only after this verification succeeds.

## Recipe identity

| Setting | Value |
|---|---|
| Bridge | MMDiT, per-modality normalization and Q/K/V/O |
| Attention | Bidirectional, temporal-local, grouped BF16 |
| Tasks | 50% V2M / 50% M2V / 0% joint |
| Video loss weight | 1 |
| Camera prediction | Enabled in native generator; loss weight 0 |
| Camera-motion bridge edges | None |
| Training captions | Empty for both tasks |
| FiLM post-training | None in this checkpoint |
| Frozen specialists | Phase1 video-I/O EMA200k + selected Phase2 200k |

Full optimization and architecture settings are recorded in `contract.json`.
The checkpoint includes raw/EMA trainable state and native resume state;
frozen specialist weights are separate dependencies.

## Copy to Sep-2 local NVMe

```bash
hostname
SRC=/mnt/sep-50tb/jinhyung.kim/cosmos/cosmos_models/phase3/bidirectional_w1_camera_prediction_ema200k_20261114
DEST=/mnt/local/jinhyung.kim/cosmos/checkpoints/bidirectional_w1_camera_prediction_ema200k_20261114
test -f "$SRC/READY.json"
df -h /mnt/local
mkdir -p "$DEST"
rsync -a --partial --info=progress2 "$SRC/" "$DEST/"
cd "$DEST"
sha256sum -c SHA256SUMS
```

Inference/training must use node-local paths. This handoff does not launch a
Sep-2 job or claim GPU acceptance there.

## Dependency and implementation requirements

Required Phase1 is the **October28 video-I/O EMA200k**, originally from Sep-2.
Its `model/.metadata` SHA256 is
`b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8`.
Keep its run-root `config.yaml` and `native_phase1_contract.json`.
The older four-task Phase1 archive with metadata `0b5a35...` is incompatible.
See [Phase1 download handoff](PHASE1_VIDEO_IO_EMA200K_DRIVE_20261104.md).

Required shape-conditioned Phase2 checkpoint SHA256:
`6a8d58d746560c0af3d340f8d711dcfe23ce163bb741912afcffa8ebfdbadc3a`.
Existing shared Phase2/base-model/VAE/stat/cache archives are listed in
`DEPENDENCIES.json`; they are not duplicated in this checkpoint directory.

Before model construction, call
`motion_expert_phase3_edge_camera_directional_nojoint.policy.configure_checkpoint(payload)`.
Construct `motion_expert_phase3_edge_image_joint_camera.model.EdgePhase3` with
Sep-2-local dependency paths and load trainable state using
`motion_expert_phase3_edge_joint.adapters.load_trainable_state(model, payload, weights='ema')`.
Despite the historical policy module name, retain the checkpoint's **bidirectional**
attention and `temporal_local` scope. Use the included frozen source snapshot in
an isolated local directory if current code has diverged.

The native camera-aware sampler is
`motion_expert_phase3_edge_camera_directional_nojoint.sample.sample`: 35 UniPC
steps, motion shift 3, video shift 10, empty generator captions, independent
private camera noise. Its native guidance argument accepts CFG1; CFG above 1
uses the existing whole-bridge evaluator instead. Never feed ground-truth camera
trajectories to this base-model evaluation.

Sep-1 staging and verification operation:
`/mnt/local/jinhyung.kim/cosmos/operations/stage_bidir_w1_camera_ema200k_shared_20261114/`.
