#!/usr/bin/env bash
# Run inside a retained Yonsei GPU srun allocation; do not submit this with sbatch.
# Fresh global-feature ablation of the 2026-09-08 centered-joint contact run.
set -euo pipefail

: "${SLURM_JOB_ID:?Run this launcher inside the retained Yonsei GPU allocation}"
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

export EDGE_MOTION_REPRESENTATION=camera_head_recanonicalization_v1
export EDGE_BONES_CAPTION_POLICY=main_actor_camera_wearer_v1
export JOINT_LOSS_MODE=centered
# Add 10 per normalized global feature coordinate, on top of W_FEAT=1.
export W_GLOBAL_ROT=10
export W_GLOBAL_TRANS=10

# Shared, representation-preserving caches; paths are Yonsei-specific.
export COSMOS_MOTION_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/uniego_arrays_v1
export COSMOS_MOTION_CACHE_SOURCE_ROOT=${WEKA_ROOT}
export COSMOS_REASONER_IMAGE_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/phase2_images_v1
export COSMOS_EDGE_CONDITION_CACHE_ROOT=${COSMOS_REASONER_IMAGE_CACHE_ROOT}
export COSMOS_EDGE_CONDITION_CACHE_ID=edge-d4599e2e43fbd06168e9884205b9b66c3902d8f6-pinned-weights-processor-v1
export COSMOS_EDGE_CONDITION_PREFETCH_THREADS=4
export COSMOS_EDGE_CONDITION_ASYNC_WRITES=1
export BATCH_SIZE=128
export GRAD_ACCUM=1
export BONES_FRAC=0.5
# POC/Nano contact-aware recipe. The shared loss helper uses logit scale 2, fps 20.
export W_FEAT=1
export W_JOINT=1
export W_SMOOTH=5
export W_CONTACT=0.05
export W_FOOT_VEL=1
export W_FOOT_HEIGHT=10

RUN_NAME=edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_globalfeat10_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912
export OUT=${RUN_ROOT}/motion_expert_t2m_edge/${RUN_NAME}
export RESUME=auto
export VIZ_EVERY=5000
export VIZ_SAMPLES_PER_TASK=5
export WANDB_MODE=online
export WANDB_PROJECT=cosmos-motion-ft
export WANDB_ENTITY=scene_motion
export WANDB_RUN_NAME=${RUN_NAME}
export WANDB_GROUP=cosmos3-edge-phase2-bones50-globalfeat10
export WANDB_TAGS=cosmos3-edge,phase2,nymeria,bones50,camera-head-v1,subject-v1,poc-contact,global-feature-extra10,centered-joint,yonsei
export WANDB_STORAGE_ROOT=${RUN_ROOT}/.wandb_runtime
export WANDB_DIR=${WANDB_STORAGE_ROOT}
mkdir -p "${PROJECT_ROOT}/logs" "${WANDB_STORAGE_ROOT}"

# Reuse the previous preparation launcher's allocation lock to avoid overlap.
exec 9>"/home/whwjdqls99/cosmos_phase2_edge_bones50_4090_${SLURM_JOB_ID}.lock"
flock -n 9 || { printf 'Another Phase 2 launcher holds this allocation lock.\n' >&2; exit 1; }
mkdir -p "${OUT}"
exec 8>"${OUT}/.training.lock"
flock -n 8 || { printf 'Another trainer holds this output directory lock.\n' >&2; exit 1; }
exec > >(tee -a "${PROJECT_ROOT}/logs/${RUN_NAME}_train_job${SLURM_JOB_ID}.log") 2>&1
printf '[edge-phase2-globalfeat10] job=%s node=%s date=%s\n' "${SLURM_JOB_ID}" "$(hostname)" "$(date --iso-8601=seconds)"
printf '[edge-phase2-globalfeat10] output=%s\n' "${OUT}"
printf '[edge-phase2-globalfeat10] joint_loss_mode=centered; normalized global rotation[270:276] and translation[276:279] each +10 (total 11), full-283 mean denominator\n'
printf '[edge-phase2-globalfeat10] baseline=edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260908\n'
printf '[edge-phase2-globalfeat10] feat/joint/smooth=1/1/5 contact/foot_vel/foot_height=0.05/1/10 scale=2 fps=20\n'
printf '[edge-phase2-globalfeat10] batch=128 steps=200000; BONES T2M=37.5%% Nymeria T2M=37.5%% Nymeria TI2M=25%%\n'
printf '[edge-phase2-globalfeat10] fixed visualization at start/every 5000; checkpoint every 5000; recovery every 250\n'
exec bash "${REPO_ROOT}/motion_expert_t2m_edge/sbatch_train_1gpu.sh"
