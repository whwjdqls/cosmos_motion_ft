# Official egomotion baselines on Nymeria-SOMA

Both experiments use the same original Nymeria camera/body recordings at 20 FPS and fixed 97-frame windows. The cache is `/scratch2/whwjdqls99/cosmos/caches/nymeria_soma_official_baselines_T97_20fps_camhead_v1_20261015`. Its `index.json` records the source hashes and window provenance. The original 642 training recordings were split at recording level into 578 train and 64 validation; the 71 official test recordings remain untouched. The train, validation, and test windows number 108030, 11602, and 12613, respectively. Neither baseline uses the five-name Phase 1 exclusion.

## Common body contract

The source is the existing SOMA-30 camera/head-aligned UniEgo representation, not fitted SMPL. We keep 23 body joints, including the wrist/hand bases and foot/toe bases, and omit jaw, eyes, and finger endpoints. `body_shape.py` defines the ordered joint subset and kinematic parents. Its training-only PCA of actor neutral-joint offsets supplies shape targets; these coefficients are **not SMPL betas**. Both baselines receive measured Nymeria RGB-camera trajectories. Hands are not predicted or conditioned in either baseline.

## UniEgoMotion

Official source SHA: `580c92c6d70a91672c4106bab30ca82cdd80f379`. Working source and configuration: `/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/upstream/UniEgoMotion`. The adapter `dataset/nymeria_soma_dataset.py` supplies an 18-D camera trajectory and a 230-D target: 220 body-motion channels plus the first 10 train-only SOMA shape coefficients. Official `v4_beta` predicts shape even though shape is not an input condition; this adaptation retains that direction. The official diffusion model, optimizer, LR, weight decay, and 350-epoch setting are retained. Batch size is 32 on one A6000. The pinned PyTorch 2.5.1 CUDA 11.8 run is under `/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/runs/uniegomotion_nymeria_soma_body230_pinned_20261015`; the earlier overlay-runtime partial epoch is separate.

## EgoAllo

Official source SHA: `0c20ae8c8d48be4ff682803ea523bcb00a6a708d`. Working source: `/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/upstream/egoallo`. The Nymeria adapter provides measured z-up camera poses, relative poses, 22 non-root body-joint local rotations, four observed foot contacts, and 16 train-only SOMA shape PCA coefficients. Hands and hand-position conditioning are disabled. The official 512-wide denoiser, cosine schedule, loss terms and weights, AdamW, and learning-rate warmup remain; only the skeleton/contact dimensionality and training-data path changed. Batch size is 16 on one 4090 because the official batch 256 does not fit that GPU. Checkpoints are under the working source's `experiments/nymeria_soma_body_t97_shape16_nohands_20261015/v1`.

The training loss imports a training-only copy of the unchanged official cosine schedule because the official sampling module imports Project Aria guidance code that is unused by training. The source tarball lacks `.git`, so the trainer records the pinned upstream SHA instead of invoking git.

**Current comparability caveat:** the EgoAllo adapter passes the Nymeria world-Z camera pose directly. The official conditioning reads camera Z as floor height, but Nymeria provides a per-window `ground_offset_y`. A corrected restart should subtract that offset from camera Z; until then, treat the current EgoAllo run as provisional for floor-relative comparisons. Its separate `height_from_floor` metadata is corrected, but the denoiser does not read that field.

EgoAllo currently uses an isolated dependency overlay on the Cosmos Python/Torch 2.10 CUDA 12.8 runtime while its floor-height correction is pending. UniEgoMotion was restarted from scratch under its pinned PyTorch 2.5.1 CUDA 11.8 environment after that environment finished installing. The earlier partial epoch used the overlay runtime and is not part of the pinned run.

## EgoAllo floor-height correction (2026-10-15)

This section supersedes the provisional EgoAllo runtime and floor-height caveat
above. The adapter now copies the cached z-up camera poses and subtracts each
window's `ground_offset_y` from world Z before packing `T_world_cpf`. Official
`EgoDenoiserConfig.make_cond()` reads this corrected Z directly as camera height
above the floor. `height_from_floor` contains the same value, not a second
subtraction. Camera rotations and relative camera transforms are unchanged;
source caches are not modified. No loss or network changes accompany this fix.

The focused check ran on RTX4090/node35 in allocation2490722 and passed for
positive and negative ground offsets, including unchanged relative transforms,
source-cache immutability and finite official GPU conditioning. Evidence:
`/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/egoallo_floor_check.log`.

The provisional training process was stopped; its allocation and checkpoints
were retained. The corrected experiment starts fresh, rather than resuming
weights trained with the incorrect absolute-height conditioning. Its name is
`nymeria_soma_body_t97_shape16_floor_pinned_20261015`, under the upstream
`egoallo/experiments/` directory in the same Scratch2 operations root.

Runtime: `env_egoallo/bin/python`, Python3.12.15, pinned PyTorch2.7.1+cu126.
The temporary Cosmos runtime/dependency overlay is no longer used. Launch
session: `egoallo_soma_floor_pinned_20261015`; training log:
`/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/egoallo_floor_pinned_train.log`.
Actual optimizer progress and completion must be read from this log; the
launch record itself is not proof of training completion. UniEgoMotion and
unrelated Edge trainers were not restarted.

## Official batch-size correction and active runs (2026-10-15)

The initial B32 UniEgoMotion and B16 EgoAllo launches were provisional startup
runs, not the final baseline recipe. They are stopped and their outputs are
preserved. This section supersedes their active-run paths and batch settings.

- UniEgoMotion now uses B64, matching upstream `config/uem.yaml`, with the
  same LR3e-5, weight decay0.01, 350epochs and official scheduler/EMA. Camera-only
  conditioning, T97/20fps and body230 remain the documented Nymeria adaptations.
  Allocation2490842/node51; session `uem_soma_body_b64_pinned_20261015`.
  Output: operations root `/runs/uniegomotion_nymeria_soma_body230_b64_pinned_20261015`.
  Log: operations root `/uem_b64_pinned_train.log`.
- EgoAllo now uses B256, matching `EgoAlloTrainConfig.batch_size`, with the
  floor correction and pinned runtime above. LR1e-4, weight decay1e-4 and
  1000-step warmup are unchanged. Allocation2490722/node35; session
  `egoallo_soma_b256_pinned_20261015`.
  Output: operations root `/upstream/egoallo/experiments/nymeria_soma_body_t97_shape16_floor_b256_pinned_20261015/v0`.
  Log: operations root `/egoallo_b256_pinned_train.log`.
- The upstream EgoAllo loop is unbounded. A finite stopping budget has not yet
  been implemented; do not describe it as a completed or fixed-length run.
- The former UniEgoMotion allocation2477396/node47 is retained for separate
  data checks, not duplicate training. Unrelated Edge trainers are untouched.

Read-only compute-node audits checked 24 recordings from each of train, val
and test (72 windows per adapter), including nonzero-start windows. Recording
splits are disjoint. Results: `audit_uem_data.json` and
`audit_egoallo_data.json` in the operations root. UEM normalization round-trip
maximum error was3.815e-6; EgoAllo local-rotation reconstruction maximum error
was1.448e-7, and neutral-skeleton FK maximum position error was0.0615mm.
These are representation checks, not generated-motion evaluation results or
a claim that every cached window has been audited. Large normalized feature
magnitudes are being inspected separately before claiming the data is clean.

### Geometry edge cases and EgoAllo metadata

The large UEM normalized features were traced to source geometry, not shape
PCA or a failed normalization round-trip. In the inspected Elizabeth Tucker
window, canonical heading changes69.88degrees while the actual camera changes
1.55degrees in that step. This is consistent with sensitivity of yaw-only
canonicalization, not an equivalent physical camera rotation. No clipping or
representation change was applied.

In the inspected held-out Megan Mejia window, the camera is25.05-25.09m above
the recording-wide floor plane and feet are at least23.36m above that same
plane. Camera-to-head distance remains0.14-0.15m. This is an input/GT floor
reference limitation, not a generated-motion error. Do not silently replace
it with a local floor or remove the test case. Detailed evidence is in
`audit_geometry_edge_cases.json` in the operations root.

An additional adapter issue was found in fields unused by the body-only
EgoAllo training loss: `T_world_root` contained canonical coordinates and
`joints_wrt_cpf` was a zero placeholder. The adapter now loads correctly
precomputed world-root poses and camera-relative body joint positions from
`egoallo_metadata_v1/` under the shared cache, applying the same per-window
floor offset to root Z as camera Z. The preparation command is:

```bash
python -m nymeria_baselines.prepare_egoallo_metadata --cache /path/to/shared/cache
```

Preparation runs separately on node47; its completion marker is
`egoallo_metadata_v1/index.json`, not this documentation entry. The preparation
log is `egoallo_metadata_prepare.log` in the operations root. Source arrays
are unchanged. Already-running training workers retain the previous unused
metadata fields until restart, but their body rotations, shape targets,
contacts, masks, camera conditioning and losses are unchanged by this fix.
New evaluation/loader processes use the corrected metadata after preparation.

### Finite EgoAllo training budget (2026-10-15)

The unbounded upstream loop is now supervised by a separate stop controller,
with an initial target at the native `checkpoints_200000` save. This is an
operational budget chosen for this Nymeria baseline, not a published upstream
schedule. No LR, loss, optimizer or training-loop changes implement the stop.
The official counter has its own documented initial-step off-by-one convention;
200k here names its native checkpoint, not a claim to change that convention.

The controller waits for the exact run's successful-save log, then signals only
its pinned trainer PID, checking process start time against PID reuse. It
retains the srun allocation and does not force-kill a slow shutdown. A trainer
exiting early is recorded as early exit, not success. Session:
`egoallo_stop200k_20261015`; log and status in the operations root:
`egoallo_stop200k.log`, `egoallo_stop200k_status.json`.

UniEgoMotion keeps the official350-epoch horizon. Both current allocations
have roughly three days available at launch; ongoing training, final saves,
and shutdown must be established from runtime logs rather than this plan.

### Completed metadata acceptance check (2026-10-15)

Preparation completed for all713 recordings. A fresh EgoAllo loader then
passed independent FK/coordinate checks on one middle window per recording:
578train,64validation,71test. The check confirmed source-manifest SHA256 and
full cache-record coverage, then composed root-world rotation with local body
rotations, reconstructed joints using the neutral skeleton, and transformed
those positions into the measured camera frame. Maximum disagreement with the
corrected metadata was0.0797mm. Per-recording mean head-to-camera distances
ranged0.130-0.395m; none exceeded0.5m. These are GT/data checks, not generated
motion scores. Evidence: `egoallo_metadata_check.json` and
`egoallo_metadata_check.log` in the operations root.

Both B64 UniEgoMotion and B256 EgoAllo continued training during this check.
The check does not claim validation of every frame or final model quality.

### First EgoAllo checkpoint and resume-schedule guard (2026-10-15)

EgoAllo B256 saved its native `checkpoints_5000` at16:03:30KST. A separate
A6000 process loaded that checkpoint through the official Accelerate path,
restored populated optimizer state and scheduler counter5001 (native counter
convention), and performed one finite forward/backward/optimizer update at
LR1e-4 on two real training samples with corrected metadata.196 parameter
tensors received finite gradients. Production was neither restarted nor
modified by this check. Evidence: `egoallo_checkpoint_5000_check.json` and
`egoallo_checkpoint_5000_check.log` in the operations root. This establishes
checkpoint usability, not model accuracy or bitwise replay of data sampling.

UniEgoMotion's Nymeria config now sets the official `TRAIN.USE_CKPT_LR=True`
option. Without it, upstream `on_train_start()` resets LR to the configured
initial value on resume, which could undo its late-training LR decay. Fresh
runs return before that branch, so their recipe is unchanged. Already-running
training continues unchanged; future resumes preserve the saved optimizer LR.

### UniEgoMotion epoch-10 checkpoint acceptance (2026-10-15)

UniEgoMotion B64 saved its first scheduled checkpoint at16:38:05KST, after
10 completed epochs (native epoch9/global step16880). On the spare A6000,
a separate Lightning trainer restored the saved model tensors, all228 Adam
parameter states, all233 EMA tensors, scheduler epoch10 and LR3e-5. Each
restored model/EMA tensor and Adam state tensor was compared exactly with
the checkpoint before updating. One real B64 training update then advanced
only the isolated trainer to16881, with finite loss/gradients/parameters and
EMA state. A separate EMA-weighted denoising forward on a validation batch
was also finite. This is not a motion-quality benchmark or a bitwise replay
claim for dataloader sampling.

Evidence: `uem_checkpoint_epoch10_check.json` and
`uem_checkpoint_epoch10_check.log` in the operations root. Production stayed
on node51 and continued uninterrupted. Both official training paths now have
real saved checkpoints that passed isolated resume/update checks.
