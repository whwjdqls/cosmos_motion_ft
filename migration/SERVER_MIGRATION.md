# Cosmos Motion Server Migration

Status date: 2026-07-23.

This document separates assets required for actual training/evaluation from raw
sources and obsolete experiment debris. The canonical bucket root is:

```text
gs://mm-jinhyung_kim/jungbin_cho
```

## Runtime Source Contract

- Offline Git bundles for all four repositories are under `source/` in GCS.
  Restore them with `bash migration/restore_from_gcs.sh source`, then use
  `git clone <name>.bundle <destination>`.
- `cosmos_motion_ft`: GitHub `whwjdqls/cosmos_motion_ft`, use the commit named in
  the final migration verification record.
- `cosmos-framework`: NVIDIA base commit
  `82f82293ffd8983651cd51d8191287da3973f534`; apply
  `external/cosmos_framework_patches/local_changes.patch` and copy the files
  under `external/cosmos_framework_patches/untracked/`.
- `kimodo_open`: commit `5e3daacf09887a6c8581a8f496629b008b6ba4d5`.
- `nymeria_kimodo_pipeline`: commit
  `cd1b3bc4048957b2ddd3a20fc1baacd29c5d6643`.
- Cosmos environment: `external/cosmos_env.yml` plus
  `external/cosmos_env_pip_freeze.txt`.

## Phase 1: Native Camera/Video

Training requires:

1. `runtime/cosmos3_nano_dcp`: native base DCP.
2. `runtime/wan22_vae`: local Wan2.2 VAE.
3. `nymeriaplus_proportional/joint_latents_T97`: 127,956 cached
   `[48,25,16,16]` fp16 latent windows with `[96,9]` camera actions.
4. `nymeriaplus_proportional/video/manifest_video.jsonl`.
5. `nymeriaplus_proportional/train_test_split.json`.
6. `nymeriaplus_proportional/metadata`, especially
   `camera_motion_quality_filter_v1_T97.json`.
7. The Cosmos text tokenizer under the materialized Nano HF snapshot.

Evaluation requires:

1. Wan VAE and the selected native DCP checkpoint.
2. `native_phase1_eval_inputs_full71_256_T97_v2` for canonical 71-sequence
   forward/inverse reporting.
3. `native_phase1_eval_inputs_viz5_256_T97_v2` for legacy fixed-prefix
   qualitative checks.
4. `native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1` for
   prefixes 1/9/17/33/49.
5. `runtime/model_cache` for DreamSim and content-debiased FVD.
6. `runtime/torch_hub/checkpoints/alexnet-owt-7be5be79.pth` for offline
   LPIPS evaluation.

Native checkpoints are full DCP directories. Keep `model`, `optim`, `scheduler`,
and `trainer`; model-only copies can evaluate but cannot exactly resume.

## Phase 2: T2M/TI2M Motion Expert

Training requires:

1. Materialized Cosmos3-Nano HF snapshot, including all seven transformer
   shards, processor/tokenizer files, and `vision_encoder/model.safetensors`.
2. `nymeriaplus_proportional/uniego_rep` for Nymeria 283-D motion.
3. `nymeriaplus_proportional/video` for the aligned TI2M image/window.
4. `nymeriaplus_proportional/metadata/floor_calibration.json`.
5. `seed/soma_proportional_uniegomotion_20fps`.
6. Exact BONES train/val pair and index files under
   `cosmos_motion_ft_runs/joint_attention`.
7. `benchmarks/Kimodo-Motion-Gen-Benchmark/splits` to rebuild BONES pairs.
8. Generator normalization in Git:
   `motion_expert_joint_attention/uniego283_{mean,std}.npy`, SHA-256
   `bd1d6bdc...` and `ee069e3a...`.

Shape/TMR evaluation requires the complete
`evaluators/shape_aware_motion_eval_c45_20260715` bundle. Its
`SHA256SUMS` covers the C45 checkpoint, evaluator-only motion statistics,
generator statistics, text embeddings, reference cases, and vendored runtime.
Do not substitute trainer statistics for
`artifacts/evaluator/stats/motion/{mean,std}.npy`.
The exact generated BONES cohort manifests used for reported evaluation are
retained under the Phase-2 `eval_shape_tmr_c45_native_unipc35` directory. The
raw 20-fps benchmark test-suite is not required to rerun those fixed manifests.

## Phase 3: Modality Bridge

Training requires both selected specialist checkpoints plus all Phase-2 data:

- Phase-1 native DCP:
  `native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k/`
  `checkpoints/iter_000100000`.
- Phase-2 native schedule:
  `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/`
  `ckpt_step200000.pt`, or the contact-loss variant for that ablation.
- Nymeria video, camera-RGB, UniEgo motion, metadata, floor calibration, and
  cached T97 latents.
- Git-tracked `head_camera_calibration_train.json` for the head-camera run.

Evaluation uses `cosmos_motion_ft_runs/joint_attention/full71_windows.json`,
the Nymeria train/test split, motion normalization arrays in Git, and the Wan
VAE for video decoding.

## Retained Checkpoints

Only the following checkpoints are required by the migration contract:

| phase | run | retained checkpoint |
|---|---|---|
| historical | `world_camera_nymeria_97f_cont` | `iter_000007000` |
| historical | `world_camera_nymeria_97f_hung_iter6000` | `iter_000007000` |
| Phase 1 | `native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k` | `iter_000100000` |
| Phase 1 | `native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person` | `iter_000100000` |
| Phase 1 | `native_phase1_vq_B_varprefix_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person` | `iter_000100000` |
| Phase 1 | `native_phase1_vq_D_varprefix_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person` | `iter_000055000` |
| Phase 2 | `ja_t2m_ti2m_reasonerimg_x0_T200_mrope3d` | `ckpt_step130000.pt` |
| Phase 2 | `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d` | `ckpt_step200000.pt` |
| Phase 2 | `ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35` | `ckpt_step200000.pt` |
| Phase 3 | `ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k` | `ckpt_step200000.pt` |
| Phase 3 | `ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam` | `ckpt_step115000.pt` |
| Phase 3 | `ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask` | `ckpt_step065000.pt` |
| Phase 3 | `ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k` | `ckpt_step035000.pt` |

Native checkpoints retain all DCP `model`, `optim`, `scheduler`, and `trainer`
components. Joint-attention `.pt` files retain the trainer state written by
`train.py`. Run configs/logs and `eval*` directories are retained. Standalone
`viz_step*` directories and intermediate checkpoints are not part of the
required archive.

Some extra objects were uploaded before this scope was narrowed, including the
filtered Phase-1 `qfilterv1/iter_000035000` checkpoint and selected
visualizations. They are optional and are not checked by the verifier. The
non-deleting migration policy leaves them in GCS.

## Small Critical State

These files are not optional metadata:

| artifact | role |
|---|---|
| `train_test_split.json` | sequence-level held-out partition |
| `floor_calibration.json` | per-sequence motion floor offsets and exclusions |
| `camera_motion_quality_filter_v1_T97.json` | exact Phase-1 filtered windows |
| `bones_pairs_{train,val}.jsonl` | fourth-caption BONES sampling policy |
| `bones_index_{train,val}.json` | resolved BONES motion/caption index |
| `full71_windows.json` | canonical joint-attention held-out windows |
| `uniego283_{mean,std}.npy` | generator motion normalization |
| C45 `stats/motion/**/{mean,std}.npy` | evaluator-only normalization |
| `head_camera_calibration_train.json` | train-only rigid head-camera fit |

The three Nymeria split/filter/calibration hashes are enforced by
`verify_migration.py`. The repo-tracked motion statistics and head-camera
calibration are protected by Git.

Repo-state SHA-256 values at migration:

```text
bd1d6bdc9a3b026fe1e5b28899441655ee36672c69c3e6e6389e9baff4b400d3  uniego283_mean.npy
ee069e3aa9f3cd1a1e70135cc00bc751030f8045fae6bbfb7b4f5b32fa65f28c  uniego283_std.npy
17bd180d105f447b6aa05eaba5c12339a78605a8f2d53e01380000c1f2c01404  head_camera_calibration_train.json
52c02eb2b83e830bbf0bb6a20504ce82a8aca920d7b9b09b2b25a172c68fe636  cosmos-framework local_changes.patch
```

## Final Dependency Sweep

A code-path sweep covered Python, shell, TOML, YAML, dataset loaders, native
launchers, samplers, and evaluators. The active external dependencies found
there are covered by this contract. The following referenced paths are
intentionally not archived as active requirements:

- raw/access-gated `/weka/jungbin/nymeriaplus`;
- obsolete T33 `joint_latents`, because production uses `joint_latents_T97`;
- `nymeriaplus_audio`, VGGT checkpoints, and `nymeria_world` baselines;
- old proof-of-concept motion checkpoints and A100-only BONES caches;
- temporary merge/smoke directories and intermediate checkpoints;
- Qwen tokenizer fallback, because the exact Cosmos Nano text tokenizer is in
  the materialized HF snapshot.

The small Kimodo skeleton buffer used only to regenerate proportional motion is
archived under `runtime/kimodo_caches`. Benchmark split files are archived to
rebuild BONES train/validation pairs; fixed evaluator manifests are retained
with the Phase-2 evaluation.

## Raw Data Boundary

`gs://.../nymeriaplus/` is an incomplete historical raw-source upload and must
not be presented as a complete Nymeria backup. Raw VRS captures are
access-gated and are not needed by current trainers once the archived derived
`video`, `camera`, `camera_rgb`, `uniego_rep`, metadata, and latent trees have
been restored.

The active derived tree is the operational backup. Rebuilding it from raw data
would require the external pipeline and substantial GPU time.

## Restore Order

1. Clone this repository at the migration commit.
2. Clone Cosmos Framework at the pinned commit and apply the vendored patch.
3. Recreate the `cosmos` environment.
4. Run `bash migration/restore_from_gcs.sh data`.
5. Run `bash migration/restore_from_gcs.sh core`.
6. Run `bash migration/restore_from_gcs.sh runs`.
7. Run `python migration/verify_migration.py local`.
8. Run the smallest Phase-1, Phase-2, and Phase-3 smoke tests before launching.

Use environment overrides when the new server does not use `/weka/jungbin`.
The launchers still contain historical absolute defaults, so either restore
that layout or export the documented path variables.

## Final Verification

Verified on 2026-07-23 with `python migration/verify_migration.py gcs`:

```text
camera: 729 objects
camera_rgb: 735 objects
uniego_rep: 733 objects
video: 1480 objects
joint_latents_T97: 127956 objects
required objects, hashes, and counts passed
```

The retained native DCPs were additionally compared as local-vs-GCS
file-count/byte totals: baseline `36/91,398,229,408`, A
`36/91,398,229,476`, B `36/91,398,229,504`, and D
`36/91,263,133,825`. Every retained Phase-2/3 `.pt` checkpoint matched its
local byte size exactly. Source Git bundles passed `git bundle verify`; the
Cosmos Motion bundle is refreshed after the final migration commit.
