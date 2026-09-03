# Site Storage Map

Last updated: 2026-09-04

This document maps physical storage for the Yonsei datacenter and the Grasp lab
server. It complements `SERVER_MIGRATION.md`, which remains the restore
runbook, and the artifact manifests, which remain the authority for required
files, sizes, and hashes.

Do not infer the site from old absolute paths in experiment notes. Set the site
explicitly and resolve all active paths from that site's roots.

## Site identities

| Site | Repository | Persistent storage base | Artifact root |
| --- | --- | --- | --- |
| Yonsei datacenter | `/home/whwjdqls99/cosmos_motion_ft` | `/lustre/whwjdqls99/cosmos` | `/lustre/whwjdqls99/cosmos/weka` |
| Grasp lab | `/home/jungbinc/cosmos_motion_ft` | `/mnt/projects/ll/jungbinc` | `/mnt/projects/ll/jungbinc` |

The following paths belong to other machines and are not aliases for either
site above:

- `/home/jungbin_cho` and `/weka/jungbin`: historical H200/a3ultra paths.
- `/mnt/shared/jungbin_cho`: historical A100 paths.

Those paths remain in experiment records for provenance. Do not use them as
portable defaults and do not rewrite historical result locations.

## Explicit site environments

Use these variables as the shell-level path contract. Individual scripts may
have additional arguments, but active launchers should derive their physical
paths from these roots or receive explicit overrides.

### Yonsei datacenter

```bash
export COSMOS_SITE=yonsei
export COSMOS_REPO_ROOT=/home/whwjdqls99/cosmos_motion_ft
export COSMOS_STORAGE_ROOT=/lustre/whwjdqls99/cosmos
export WEKA_ROOT=${COSMOS_STORAGE_ROOT}/weka
export COSMOS_RUNS_ROOT=${WEKA_ROOT}/cosmos_motion_ft_runs
export HF_HOME=${COSMOS_STORAGE_ROOT}/.cache/huggingface
export TORCH_HOME=${COSMOS_STORAGE_ROOT}/.cache/torch
export WAN_VAE_PATH=${WEKA_ROOT}/wan22_vae/Wan2.2_VAE.pth
```

Set `COSMOS_SITE=yonsei` before sourcing `restored_env.sh` or importing
`runtime_paths.py`. Both selectors reject an unset or unknown site instead of
silently falling back to another server. Set any one-off path override after
loading the site defaults.

The Yonsei `cosmos` conda environment is installed at
`${COSMOS_STORAGE_ROOT}/envs/cosmos` and is exposed to conda as `cosmos` through
`/home/whwjdqls99/miniconda3/envs/cosmos`. Activate it with `conda activate
cosmos`. It uses the pinned Edge framework checkout at
`/home/whwjdqls99/cosmos-framework-edge`; PyTorch 2.10.0+cu128 import, CUDA
12.8 access, and a CUDA tensor operation were validated on an RTX 3090 on
2026-09-04. The `${COSMOS_STORAGE_ROOT}/tools/hf-download-venv` environment is
only for model downloads and is not a Cosmos training or inference environment.

### Grasp lab

```bash
export COSMOS_SITE=grasp
export COSMOS_REPO_ROOT=/home/jungbinc/cosmos_motion_ft
export COSMOS_STORAGE_ROOT=/mnt/projects/ll/jungbinc
export WEKA_ROOT=${COSMOS_STORAGE_ROOT}
export COSMOS_RUNS_ROOT=${WEKA_ROOT}/cosmos_motion_ft_runs
export HF_HOME=${COSMOS_STORAGE_ROOT}/.cache/huggingface
export TORCH_HOME=${COSMOS_STORAGE_ROOT}/.cache/torch
export WAN_VAE_PATH=${WEKA_ROOT}/wan22_vae/Wan2.2_VAE.pth
```

Framework checkout and conda-environment locations are site dependencies, not
repository data roots. Use the framework selected by the site's existing
launcher. Never silently fall back from a missing Yonsei path to an H200,
A100, or Grasp path.

## Yonsei directory layout

```text
/lustre/whwjdqls99/cosmos/
|-- weka/                         datasets, models, checkpoints, and runs
|-- .cache/
|   |-- huggingface/              Hugging Face cache
|   `-- torch/                    Torch hub and evaluation caches
|-- logs/                         transfer and verification logs
`-- tools/                        migration-only download tooling
```

`/lustre/whwjdqls99/cosmos/weka` is a compatibility name inside the project
root. It is not the H200 `/weka` filesystem.

## Yonsei artifact inventory

The states below are the migration snapshot from 2026-09-04. A `SYNCING` or
`PENDING` item is not safe for training or evaluation. The final readiness
authority is a successful transfer followed by the checks described below.

| Artifact | Yonsei location | State | Source or evidence |
| --- | --- | --- | --- |
| Nymeria+ proportional corpus | `${WEKA_ROOT}/nymeriaplus_kimodo_proportional` | `READY` | 4,474 files and 350,742,352,390 bytes matched the Drive source |
| Restored joint-attention archive | `${COSMOS_RUNS_ROOT}/joint_attention` | `READY` | Scratch-to-Lustre one-way rsync comparison passed |
| BONES SEED | `${WEKA_ROOT}/seed` | `SYNCING` | `gggdrive:cosmos_data/seed` |
| BONES 20 fps benchmark | `${WEKA_ROOT}/Kimodo-Motion-Gen-Benchmark-20fps` | `READY` | Drive comparison passed 43,425/43,425 checks |
| BONES benchmark splits | `${WEKA_ROOT}/Kimodo-Motion-Gen-Benchmark/splits` | `READY` | Drive copy exited successfully |
| Evaluation model cache | `${WEKA_ROOT}/model_cache` | `READY` | 30 files and 5,222,030,288 bytes match the manifest |
| Shape-aware C45 evaluation tree | `${WEKA_ROOT}/shape_aware_motion_eval_c45_20260715` | `SYNCING` | Drive evaluation fixture |
| Native full71 evaluation inputs | `${COSMOS_RUNS_ROOT}/native_phase1_eval_inputs_full71_256_T97_v2` | `PENDING` | Drive evaluation fixture |
| Native VQ evaluation inputs | `${COSMOS_RUNS_ROOT}/native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1` | `PENDING` | Drive evaluation fixture |
| LPIPS AlexNet weights | `${TORCH_HOME}/hub/checkpoints/alexnet-owt-7be5be79.pth` | `PENDING` | Drive evaluation model archive |
| Cosmos3-Edge raw repository | `${WEKA_ROOT}/Cosmos3-Edge` | `READY` | `hf download` exited successfully at revision `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba` |
| Cosmos3-Nano raw repository | `${WEKA_ROOT}/Cosmos3-Nano` | `SYNCING` | Hugging Face revision `7a312c868bcce8e40b3eb40861300a9d0ba3fde1` |
| Wan 2.2 VAE | `${WAN_VAE_PATH}` | `READY` | SHA-256 `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |
| Cosmos3-Edge DCP | `${WEKA_ROOT}/cosmos3_edge_dcp` | `NOT_BUILT` | Derived locally from the raw Edge repository; it is not a separate download |

Do not change a state to `READY` merely because a destination path exists.
Hugging Face and multi-stream rclone downloads create partial or preallocated
files whose apparent size can look complete before the transfer exits.

## Active checkpoint locations on Yonsei

| Phase | Yonsei checkpoint | Migration state |
| --- | --- | --- |
| Phase 1 EMA delta | `${COSMOS_RUNS_ROOT}/portable/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_iter100000_ema_gen_delta.pt` | `READY`, manifest SHA-256 verified |
| Phase 2 native | `${COSMOS_RUNS_ROOT}/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt` | `SYNCING` |
| Phase 2 contact | `${COSMOS_RUNS_ROOT}/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35/ckpt_step200000.pt` | `SYNCING` |
| Phase 3 native | `${COSMOS_RUNS_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/ckpt_step200000.pt` | `SYNCING` |
| Phase 3 head camera | `${COSMOS_RUNS_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam/ckpt_step115000.pt` | `SYNCING` |
| Phase 3 multitask | `${COSMOS_RUNS_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask/ckpt_step065000.pt` | `SYNCING` |
| Phase 3 contact | `${COSMOS_RUNS_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k/ckpt_step035000.pt` | `SYNCING` |

Historical Phase 1 A/B/D ablation DCPs are intentionally not part of the
Yonsei active set. They add about 274 GB and are not required by the active
Phase 1/2/3 paths. Restore one only when a specific historical reproduction
requires it and quota has been checked first.

## Readiness checks

Use the manifest hashes for individual checkpoints. For Drive trees, require a
successful copy and a one-way source-to-destination comparison so that local
site-only files do not cause false failures:

```bash
rclone check gggdrive:cosmos_data/seed \
  /lustre/whwjdqls99/cosmos/weka/seed \
  --size-only --one-way
```

For Hugging Face repositories, require `hf download` to exit successfully at
the pinned revision. Do not use directory size alone as completion evidence.

Transfer and verification logs live in:

```text
/lustre/whwjdqls99/cosmos/logs/
```

The current migration supervisor writes
`transfer_supervisor_ysdc.log`. The large Drive-tree jobs also write
`drive_bones_seed_ysdc.log` and `drive_bones_benchmark_ysdc.log`.

## Scratch policy on Yonsei

`/scratch2/whwjdqls99/cosmos` is not a canonical project root. The verified
351,380,573,883-byte migration was moved to Lustre and the Scratch source was
removed. Scratch may be used as temporary staging, but its retention policy
requires periodic access. Always copy to Lustre, compare source to destination,
and only then remove the staged copy.

## Lustre quota

The measured Yonsei user soft quota is 1 TiB. No hard byte limit was reported,
but the project should remain below the soft quota. Check both user quota and
filesystem capacity before restoring historical artifacts:

```bash
lfs quota -u whwjdqls99 /lustre
df -hT /lustre/whwjdqls99
df -ih /lustre/whwjdqls99
```

## Portability rules

- Keep Git-tracked code and small configuration in the site repository path.
- Keep datasets, model weights, run outputs, caches, and logs under the site's
  persistent storage root.
- Pass `WEKA_ROOT`, cache roots, checkpoint paths, and framework roots
  explicitly at launch time.
- Treat old absolute paths in result records as provenance, not defaults.
- Fail when a required site path is absent. Never search another site's root
  automatically.
- Record new artifacts in the migration manifests before relying on them from
  both sites.
