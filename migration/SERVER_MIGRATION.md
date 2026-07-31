# Cosmos Motion Restore Contract

Status date: 2026-07-31.

This is the canonical server hand-off runbook. It covers the source,
environments, derived data, normalization, evaluator state, selected
checkpoints, and exact restore/verification commands needed by active Phase
1/2/3 work.

Cloud roots:

```text
GCS:   gs://mm-jinhyung_kim/jungbin_cho
Drive: data:
```

GCS is the comprehensive archive and includes base runtime weights, the
256-tier latent cache, full original Phase-1 DCP, source bundles, and selected
run trees. Drive is an independently verifiable supplementary archive for the
active derived data, selected checkpoints, evaluator weights, and run metadata.
Neither upload path deletes local data.

## Source And Environments

| component | pinned state |
|---|---|
| this repository | GitHub `whwjdqls/cosmos_motion_ft`, final migration commit on `master` |
| Cosmos Framework | NVIDIA commit `82f82293ffd8983651cd51d8191287da3973f534` plus `external/cosmos_framework_patches/` |
| Kimodo | `5e3daacf09887a6c8581a8f496629b008b6ba4d5` |
| Nymeria pipeline | `cd1b3bc4048957b2ddd3a20fc1baacd29c5d6643` |
| Cosmos conda environment | `external/cosmos_env.yml` plus `external/cosmos_env_pip_freeze.txt` |
| preprocessing environments | `external/nymeria_kimodo_pipeline/envs/{audio,kimodo,nymeria_plus,soma}.yml` |

All YAML exports are relocatable: machine-specific conda `prefix` fields were
removed. The Cosmos pip freeze pins the two editable Cosmos Framework packages
to the base commit. The native Phase-1 features in this repository additionally
require the vendored local patch.

Fresh Cosmos Framework setup:

```bash
git clone https://github.com/NVIDIA/cosmos-framework
cd cosmos-framework
git checkout 82f82293ffd8983651cd51d8191287da3973f534
git apply /path/to/cosmos_motion_ft/external/cosmos_framework_patches/local_changes.patch
cp -r /path/to/cosmos_motion_ft/external/cosmos_framework_patches/untracked/* .
```

Create the environment, then install the patched checkout in editable mode so
it overrides the `cosmos-framework==1.2.2` wheel recorded in the export:

```bash
conda env create -f external/cosmos_env.yml
conda activate cosmos
pip install -e /path/to/cosmos-framework
pip install -e /path/to/cosmos-framework/packages/diffusers-cosmos3
```

The `motion_expert/bs_*` BONES POC remains A100-machine-only and uses the
`kimodo` environment. Native Phase 1 and `motion_expert_joint_attention/` were
validated on the H200 a3ultra machines.

## Archive Coverage

The exact Drive inventory is `DRIVE_ARTIFACT_MANIFEST.tsv`; the exact GCS
inventory is `GCS_ARTIFACT_MANIFEST.tsv`.

| requirement | Drive | GCS | regeneration |
|---|---|---|---|
| derived Nymeria motion/video/camera/metadata/split | verified | verified | raw access plus licensed pipeline, expensive |
| 256-tier T97 Wan latents `(48,25,16,16)` fp16 | no | verified, 127,956 files | `precompute_latents.py` |
| 720-tier T97 Wan latents `(48,25,40,40)` fp16 | contract only | no | `precompute_latents.py`; 115,583 train windows |
| BONES proportional motion and 283-D UniEgo | verified | verified | SEED release plus vendored converters |
| BONES uniform motion | upload active on status date | not required by active Phase 2/3 | SEED release |
| BONES uniform 369-D stats | verified | not required by active Phase 2/3 | recomputable from uniform motion |
| 20-fps Kimodo benchmark | verified | splits retained | Kimodo benchmark conversion |
| Cosmos3-Nano materialized HF snapshot | no | verified | pinned HF download |
| native Cosmos3-Nano base DCP | no | verified | pinned HF conversion |
| Wan2.2 VAE | no | verified | pinned HF download |
| DreamSim/content-debiased FVD/LPIPS | verified | verified | public downloads |
| C45 evaluator bundle and stats | verified | verified | retained bundle is preferred |
| fixed Phase-1/3 held-out manifests | verified core fixtures | verified | scripts in repo |
| selected run configs/logs/numeric eval results | verified archives | verified selected runs | not all are cheap to reproduce |

The 720-tier latent cache contract and completion record are tracked under
`cache_contracts/`; the cache itself is intentionally not duplicated because it
is about 415 GB and can be regenerated from archived video, manifest, split,
and VAE.

## Normalization Contract

Do not interchange these arrays merely because they are called mean/std.

| role | location | shape/dtype | SHA-256 |
|---|---|---|---|
| active Phase-2/3 shared generator motion mean | `motion_expert/stats/uniego283_mean.npy` and byte-identical joint-attention copy | `(283,)`, float32 | `bd1d6bdc9a3b026fe1e5b28899441655ee36672c69c3e6e6389e9baff4b400d3` |
| active Phase-2/3 shared generator motion std | `motion_expert/stats/uniego283_std.npy` and byte-identical joint-attention copy | `(283,)`, float32 | `ee069e3aa9f3cd1a1e70135cc00bc751030f8045fae6bbfb7b4f5b32fa65f28c` |
| BONES proportional source mean | `seed/soma_proportional_uniegomotion_20fps/Mean_uniego.npy` | `(283,)`, float32 | `f4f32d4f03cede93b35c46a3aeaef7282dabc9d57b428a4b672acfcf064a79d5` |
| BONES proportional source std | `seed/soma_proportional_uniegomotion_20fps/Std_uniego.npy` | `(283,)`, float32 | `559948ffc1d665a9e5c8e3a53f5b9ea024294fcb51b536c0f47bc7fb00ac9471` |
| C45 evaluator motion mean | `shape_aware_motion_eval_c45_20260715/artifacts/evaluator/stats/motion/mean.npy` | `(190,)`, float64 | `9ac9d47414dc8e777a2c0212c350b5dbbea5f43a31564bcc4b748ea9ae218f07` |
| C45 evaluator motion std | `shape_aware_motion_eval_c45_20260715/artifacts/evaluator/stats/motion/std.npy` | `(190,)`, float64 | `8f662b94d580ad5e522235a07a237a354c2b66ca508902f4377a6d45a265147f` |

Current joint-attention training defaults to the repo's Nymeria-grounded
283-D arrays for both Nymeria and BONES samples. BONES source stats are used
only by explicit proportional-stat alternatives and the standalone BONES POC.
The C45 190-D arrays normalize evaluator features only.

The C45 bundle also contains `artifacts/generator/Mean_uniego.npy` and
`Std_uniego.npy`; those are copies of the BONES proportional source stats, not
the active shared Phase-2/3 training stats and not the evaluator stats.

Native Phase 1 has no dataset camera mean/std. Camera actions remain raw metric
relative SE(3) values (`position[3] + rotation6d[6]`), while video is represented
in the Wan VAE latent space. Pixel normalization and VAE scaling are model
operations, not dataset statistics.

Root 369-D text-to-motion work uses
`seed/stats/soma_uniform_motions_20fps/`; that experiment is separate from the
283-D UniEgo Phase-2/3 path.

## Phase 1 Requirements

Native camera/video training needs:

1. `/weka/jungbin/cosmos3_nano_dcp` native base DCP.
2. `/weka/jungbin/wan22_vae/Wan2.2_VAE.pth`.
3. materialized Cosmos3-Nano processor/text tokenizer files.
4. derived Nymeria video manifest, split, metadata, and quality filter.
5. the 256-tier cached T97 latent tree, or a regenerated cache matching its
   contract.
6. patched Cosmos Framework and the `cosmos` environment.

The 256-tier cache stores one compressed reference frame plus 24 temporal
latent frames: shape `(48 channels, 25 latent frames, 16, 16)` for a 97-frame
clip. Prefix boundaries are mapped from RGB frames into that VAE timeline by
the native Phase-1 dataset/model code.

Canonical Phase-1 evaluation needs:

- `native_phase1_eval_inputs_full71_256_T97_v2` for the 71 held-out sequences;
- `native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1` for fixed
  prefixes 1/9/17/33/49;
- Wan VAE for decoding;
- DreamSim, content-debiased FVD, and LPIPS weights for video metrics;
- the selected native DCP.

Native checkpoints are directory trees. Exact resume requires `model`, `optim`,
`scheduler`, and `trainer`, not only model shards.

## Phase 2 Requirements

T2M/TI2M Motion Expert training needs:

1. materialized Cosmos3-Nano weights, processor/tokenizer, and
   `vision_encoder/model.safetensors`;
2. Nymeria `uniego_rep`, video, metadata, floor calibration, and split;
3. BONES proportional 283-D UniEgo tree;
4. exact BONES pair/index files;
5. the repo's active 283-D motion mean/std;
6. `skeleton_soma30.npz`.

`bones_pairs_train.jsonl` implements the fourth-overview-caption policy;
single- and multi-timeline captions remain unchanged.

Shape-aware evaluation uses the complete
`shape_aware_motion_eval_c45_20260715` bundle. After restore:

```bash
cd /weka/jungbin/shape_aware_motion_eval_c45_20260715
sha256sum -c SHA256SUMS
```

The bundle check passed for all 462 files on 2026-07-31.

## Phase 3 Requirements

The modality bridge needs:

1. the frozen base Cosmos Nano model;
2. Phase-1 EMA LoRA/action initialization;
3. a selected Phase-2 motion checkpoint;
4. Nymeria video/camera-RGB/UniEgo/metadata/floor/split and T97 latents;
5. active repo motion stats;
6. `full71_windows.json` for canonical evaluation.

Drive stores a compact exact original Phase-1 EMA export:

```text
data:cosmos_ckpts/native_phase1_baseline/iter_000100000_ema_gen_delta.pt
```

It contains 293 LoRA/action tensors and 32,249,856 parameters. SHA-256:

```text
4f55361baaf60a8e9438b2828d43649c4ff1e7f67608f6baae92b8c908abc393
```

It is sufficient for Phase-3 initialization but cannot resume native Phase-1
training or run official native-DCP inference. Use the GCS full DCP for those
operations. Phase-3 launchers accept the compact file through
`PHASE1_INIT=/path/to/delta.pt`; `PHASE2_INIT` similarly overrides the motion
checkpoint.

## Retained Checkpoints

| phase | checkpoint | Drive | GCS |
|---|---|---:|---:|
| Phase 1 original | `native_phase1_camera_json.../iter_000100000` | compact EMA init only | full resumable DCP |
| Phase 1 A | `native_phase1_vq_A.../iter_000100000` | full DCP | full DCP |
| Phase 1 B | `native_phase1_vq_B.../iter_000100000` | full DCP | full DCP |
| Phase 1 D | `native_phase1_vq_D.../iter_000100000` | full DCP | GCS retained iter 55000 contract |
| Phase 2 custom | `ja_t2m_ti2m_reasonerimg_x0_T200_mrope3d/ckpt_step130000.pt` | no | yes |
| Phase 2 native | `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt` | yes | yes |
| Phase 2 contact | `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35/ckpt_step200000.pt` | yes | yes |
| Phase 3 vanilla | `ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/ckpt_step200000.pt` | yes | yes |
| Phase 3 head-camera | corresponding `headcam/ckpt_step115000.pt` | yes | yes |
| Phase 3 multitask | corresponding `multitask/ckpt_step065000.pt` | yes | yes |
| Phase 3 contact-init | corresponding `p2contact200k/ckpt_step035000.pt` | yes | yes |

GCS also retains the two selected historical native `iter_000007000` runs and
their evaluations. Drive run metadata is stored as 13 checked `tar.zst`
archives with SHA-256 sidecars. It excludes checkpoint shards and media but
retains configs, logs, manifests, and numeric evaluation outputs.

## Small Critical Files

These are part of the restore contract and must not be replaced casually:

| artifact | role |
|---|---|
| `train_test_split.json` | sequence-level held-out partition |
| `floor_calibration.json` | motion floor offsets and exclusions |
| `camera_motion_quality_filter_v1_T97.json` | exact filtered Phase-1 windows |
| `manifest_video.jsonl` | video/camera/motion/text alignment |
| `bones_pairs_{train,val}.jsonl` | exact BONES caption policy |
| `bones_index_{train,val}.json` | resolved BONES data index |
| `motion_expert/pairs_{train,val}.jsonl` | Nymeria POC pairs; ignored by Git, Drive-backed |
| `full71_windows.json` | canonical joint-attention held-out windows |
| repo `uniego283_{mean,std}.npy` | active Phase-2/3 normalization |
| C45 `stats/motion/{mean,std}.npy` | evaluator-only normalization |
| `head_camera_calibration_train.json` | train-split rigid head-camera fit |

## Legacy And Optional Paths

The codebase contains older launchers with hard-coded `/mnt/shared/...` paths.
Those are A100-side BONES POC/shape-TMR experiments, not dependencies of the
active H200 Phase-1/2/3 pipeline. Their training caches and historical
checkpoints must be restored from the A100 project archive if those exact runs
are revived.

Direct evaluation with NVIDIA's public `TMR-SOMA-RP-v1` requires downloading HF
snapshot `e427752ae3446dedba49e928c93ddc9f0e413401`. Current reported Phase-2
evaluation instead uses the archived C45 bundle, which contains its own
checkpoint, 190-D evaluator stats, benchmark LLM2Vec cache, code, and reference
manifests.

The 57 GB `/weka/jungbin/kimodo_caches/bones_seed_proportional_raw.pt` cache is
not archived because it is regenerated from the verified proportional motion
tree. The old VGGT camera baseline checkpoint, audio/ImageBind experiments,
T33 video latent cache, and `nymeria_world` scratch outputs are also outside
the active restore contract.

## Restore Order

Prerequisites: configure `gcloud` access to the GCS project and configure an
`rclone` Google Drive remote named `data` (or set `DRIVE_REMOTE`).

1. Clone this repository from GitHub at the final migration commit.
2. Recreate the Cosmos environment and patched Cosmos Framework checkout.
3. Restore GCS source/core/data/runs when GCS credentials are available:

   ```bash
   bash migration/restore_from_gcs.sh all
   ```

4. Restore Drive data/evaluators/checkpoints/run metadata:

   ```bash
   bash migration/restore_from_drive.sh all
   ```

5. If GCS is unavailable, download the pinned HF models and regenerate the
   native base DCP and T97 latent cache using the recorded contracts.
6. Verify local critical state and both cloud archives:

   ```bash
   python migration/verify_migration.py local
   python migration/verify_drive.py --include-trees
   python migration/verify_migration.py gcs
   ```

7. Run the smallest Phase-1, Phase-2, and Phase-3 contract/smoke tests before a
   production launch.

All restore roots are overridable:

```bash
WEKA_ROOT=/mnt/weka/jungbin \
RUN_ROOT=/mnt/weka/jungbin/cosmos_motion_ft_runs \
HF_HOME=/home/user/.cache/huggingface \
TORCH_HOME=/home/user/.cache/torch \
bash migration/restore_from_gcs.sh all
```

Use the same `WEKA_ROOT`, `RUN_ROOT`, `TORCH_HOME`, and `REPO_ROOT` overrides
with `restore_from_drive.sh`.

## Verification Record

GCS passed the complete verifier on 2026-07-23:

```text
camera: 729 objects
camera_rgb: 735 objects
uniego_rep: 733 objects
video: 1480 objects
joint_latents_T97: 127956 objects
required objects, hashes, and counts passed
```

On 2026-07-31:

- local required-file/hash verification passed;
- the C45 bundle passed all 462 `SHA256SUMS`;
- Drive Nymeria core matched `4,474` files and `350,742,352,390` bytes;
- Drive model cache matched all 30 local files exactly;
- Drive A/B/D DCPs matched exact local file counts and byte totals;
- selected Phase-2/3 checkpoint MD5 values matched local files;
- Drive run metadata matched all 26 archive/sidecar files, totaling
  `1,431,747,113` bytes;
- the 20-fps benchmark matched all `94,045` local files and
  `17,888,724,977` bytes on Drive;
- `python migration/verify_drive.py --include-trees` passed all 40 completed
  file/tree entries;
- all six uniform-motion statistics files matched Drive checksums;
- the Drive upload remained in progress only for the optional uniform BONES
  motion tree.

The current GCS account requires interactive `gcloud auth login`, so the
2026-07-31 GCS recheck is blocked only by expired credentials. Do not interpret
that authentication failure as missing archive data.

## Raw Data Boundary

The historical `gs://.../nymeriaplus/` raw upload is incomplete. Raw VRS
captures are access-gated. A true raw-from-scratch rebuild also requires
licensed SMPL assets and external SOMA/Kimodo dependencies. The active derived
tree is therefore the operational backup and is intentionally preserved in
both GCS and Drive.
