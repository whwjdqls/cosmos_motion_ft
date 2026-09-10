#!/usr/bin/env bash
# Original-UniEgo/Nano-statistics BONES50 ablation for a retained Yonsei 4090 allocation.
set -euo pipefail

: "${SLURM_JOB_ID:?Run this launcher inside a retained Yonsei GPU allocation}"
export COSMOS_SITE=yonsei
export REPO_ROOT=/home/whwjdqls99/cosmos_motion_ft
export COSMOS_REPO_ROOT=${REPO_ROOT}
export PROJECT_ROOT=/lustre/whwjdqls99/cosmos
export COSMOS_STORAGE_ROOT=${PROJECT_ROOT}
export WEKA_ROOT=${PROJECT_ROOT}/weka
export RUN_ROOT=${WEKA_ROOT}/cosmos_motion_ft_runs
export COSMOS_RUNS_ROOT=${RUN_ROOT}
export COSMOS_ENV_ROOT=${PROJECT_ROOT}/envs/cosmos
export COSMOS_PYTHON=${COSMOS_ENV_ROOT}/bin/python3.13
export COSMOS_FRAMEWORK_EDGE_ROOT=/home/whwjdqls99/cosmos-framework-edge
export COSMOS_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT}
export EDGE_MODEL_ROOT=${WEKA_ROOT}/Cosmos3-Edge
export EDGE_DCP_ROOT=${WEKA_ROOT}/cosmos3_edge_dcp
export BASE_CHECKPOINT_PATH=${EDGE_DCP_ROOT}
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export PATH=${CUDA_HOME}/bin:${COSMOS_ENV_ROOT}/bin:${PATH}
export HF_HOME=${PROJECT_ROOT}/.cache/huggingface
export TORCH_HOME=${PROJECT_ROOT}/.cache/torch
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export EDGE_MOTION_REPRESENTATION=legacy_uniego283
export NYMERIA_UNIEGO_ROOT=${WEKA_ROOT}/nymeriaplus_kimodo_proportional/uniego_rep
export MOTION_STATS_MEAN=${REPO_ROOT}/motion_expert/stats/uniego283_mean.npy
export MOTION_STATS_STD=${REPO_ROOT}/motion_expert/stats/uniego283_std.npy
unset MOTION_STATS_SUMMARY
export COSMOS_MOTION_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/uniego_arrays_v1
export COSMOS_MOTION_CACHE_SOURCE_ROOT=${WEKA_ROOT}
export COSMOS_MOTION_CACHE_MAX_OPEN=64

export EDGE_BONES_CAPTION_POLICY=main_actor_camera_wearer_v1
export BATCH_SIZE=128
export GRAD_ACCUM=1
export BONES_FRAC=0.5
export W_FEAT=1
export W_JOINT=1
export W_SMOOTH=5
export W_CONTACT=0.05
export W_FOOT_VEL=1
export W_FOOT_HEIGHT=10

RUN_NAME=edge_7layer_nymeria_bones50_t2m_ti2m_uniego283_nanostats_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260910
export OUT=${RUN_ROOT}/motion_expert_t2m_edge/${RUN_NAME}
export RESUME=auto
export VIZ_EVERY=5000
export VIZ_SAMPLES_PER_TASK=5
export WANDB_MODE=online
export WANDB_PROJECT=cosmos-motion-ft
export WANDB_ENTITY=scene_motion
export WANDB_RUN_NAME=${RUN_NAME}
export WANDB_GROUP=cosmos3-edge-phase2-bones50-contact-legacy-uniego
export WANDB_TAGS=cosmos3-edge,phase2,nymeria,bones50,legacy-uniego283,nano-stats,subject-v1,poc-contact,yonsei
export WANDB_STORAGE_ROOT=${RUN_ROOT}/.wandb_runtime
export WANDB_DIR=${WANDB_STORAGE_ROOT}
mkdir -p "${PROJECT_ROOT}/logs" "${WANDB_STORAGE_ROOT}"

exec 9>"/home/whwjdqls99/cosmos_phase2_edge_legacy_bones50_4090_${SLURM_JOB_ID}.lock"
flock -n 9 || { printf 'Another legacy Phase 2 launcher holds this allocation lock.\n' >&2; exit 1; }
exec > >(tee -a "${PROJECT_ROOT}/logs/${RUN_NAME}_train_job${SLURM_JOB_ID}.log") 2>&1
printf '[edge-phase2-legacy] job=%s node=%s date=%s\n' "${SLURM_JOB_ID}" "$(hostname)" "$(date --iso-8601=seconds)"
printf '[edge-phase2-legacy] output=%s\n' "${OUT}"
printf '[edge-phase2-legacy] representation=%s mean=%s std=%s\n' "${EDGE_MOTION_REPRESENTATION}" "${MOTION_STATS_MEAN}" "${MOTION_STATS_STD}"
printf '[edge-phase2-legacy] BONES T2M=37.5%% Nymeria T2M=37.5%% Nymeria TI2M=25%%\n'
exec bash "${REPO_ROOT}/motion_expert_t2m_edge/sbatch_train_1gpu.sh"
