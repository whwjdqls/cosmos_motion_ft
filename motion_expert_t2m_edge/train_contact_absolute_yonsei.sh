#!/usr/bin/env bash
# Fresh absolute-joint ablation; invoke inside a retained Yonsei GPU allocation.
set -euo pipefail

: "${SLURM_JOB_ID:?Run inside a retained Yonsei GPU allocation}"
export COSMOS_SITE=yonsei
export REPO_ROOT=/home/whwjdqls99/cosmos_motion_ft
export COSMOS_REPO_ROOT=$REPO_ROOT
export PROJECT_ROOT=/lustre/whwjdqls99/cosmos
export COSMOS_STORAGE_ROOT=$PROJECT_ROOT
export WEKA_ROOT=$PROJECT_ROOT/weka
export RUN_ROOT=$WEKA_ROOT/cosmos_motion_ft_runs
export COSMOS_RUNS_ROOT=$RUN_ROOT
export COSMOS_ENV_ROOT=$PROJECT_ROOT/envs/cosmos
export COSMOS_PYTHON=$COSMOS_ENV_ROOT/bin/python3.13
export COSMOS_FRAMEWORK_EDGE_ROOT=/home/whwjdqls99/cosmos-framework-edge
export COSMOS_FRAMEWORK_ROOT=$COSMOS_FRAMEWORK_EDGE_ROOT
export EDGE_MODEL_ROOT=$WEKA_ROOT/Cosmos3-Edge
export EDGE_DCP_ROOT=$WEKA_ROOT/cosmos3_edge_dcp
export BASE_CHECKPOINT_PATH=$EDGE_DCP_ROOT
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export PATH=$CUDA_HOME/bin:$COSMOS_ENV_ROOT/bin:$PATH
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export TORCH_HOME=$PROJECT_ROOT/.cache/torch
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
unset WANDB_RUN_ID WANDB_RESUME

# Both datasets use the pinned camera/head-aligned normalization arrays.
# BONES remains motion-only UniEgo, with no synthetic camera/head alignment.
export EDGE_MOTION_REPRESENTATION=camera_head_recanonicalization_v1
export NYMERIA_MOTION_REPRESENTATION=camera_head_recanonicalization_v1
export NYMERIA_UNIEGO_ROOT=$WEKA_ROOT/nymeriaplus_kimodo_proportional/uniego_rep_camhead_v1
export MOTION_STATS_MEAN=$RUN_ROOT/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_mean.npy
export MOTION_STATS_STD=$RUN_ROOT/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_std.npy
export MOTION_STATS_SUMMARY=$RUN_ROOT/nymeria_camera_head_recanonicalization_v1/stats/summary.json
export EDGE_BONES_CAPTION_POLICY=main_actor_camera_wearer_v1
export NYMERIA_REPLACE_STANDALONE_C=1 NYMERIA_STANDALONE_C_SUBJECT=camera_wearer

export COSMOS_MOTION_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/uniego_arrays_v1
export COSMOS_MOTION_CACHE_SOURCE_ROOT=$WEKA_ROOT
export COSMOS_REASONER_IMAGE_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/phase2_images_v1
export COSMOS_EDGE_CONDITION_CACHE_ROOT=$COSMOS_REASONER_IMAGE_CACHE_ROOT
export COSMOS_EDGE_CONDITION_CACHE_ID=edge-d4599e2e43fbd06168e9884205b9b66c3902d8f6-pinned-weights-processor-v1
export COSMOS_EDGE_CONDITION_PREFETCH_THREADS=4 COSMOS_EDGE_CONDITION_ASYNC_WRITES=1

export BATCH_SIZE=128 GRAD_ACCUM=1 BONES_FRAC=0.5
export W_FEAT=1 W_JOINT=1 W_SMOOTH=5 JOINT_LOSS_MODE=absolute
export W_CONTACT=0.05 W_FOOT_VEL=1 W_FOOT_HEIGHT=10
RUN_NAME=edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_absjoint_w1_1_5_contact_c0p05_v1_h10_yonsei_20260911
export OUT=$RUN_ROOT/motion_expert_t2m_edge/$RUN_NAME
# This distinct output starts at step zero and can resume only its own checkpoints.
export RESUME=auto
export VIZ_EVERY=5000 VIZ_SAMPLES_PER_TASK=5
export WANDB_MODE=online WANDB_PROJECT=cosmos-motion-ft WANDB_ENTITY=scene_motion
export WANDB_RUN_NAME=$RUN_NAME
export WANDB_GROUP=cosmos3-edge-phase2-bones50-camhead-absolute-joint
export WANDB_TAGS=cosmos3-edge,phase2,nymeria,bones50,camera-head-v1,subject-v1,absolute-joint,poc-contact,yonsei
export WANDB_STORAGE_ROOT=$RUN_ROOT/.wandb_runtime
export WANDB_DIR=$WANDB_STORAGE_ROOT
export WANDB_DATA_DIR=$WANDB_STORAGE_ROOT/data WANDB_CACHE_DIR=$WANDB_STORAGE_ROOT/cache
mkdir -p "$PROJECT_ROOT/logs" "$WANDB_STORAGE_ROOT"

exec 8>/home/whwjdqls99/cosmos_phase2_edge_absolute_camhead_run.lock
flock -n 8 || { printf 'The absolute-joint run already has a trainer.\n' >&2; exit 1; }
exec 9>"/home/whwjdqls99/cosmos_phase2_edge_bones50_4090_${SLURM_JOB_ID}.lock"
flock -n 9 || { printf 'Another Phase-2 launcher holds this allocation.\n' >&2; exit 1; }
exec > >(tee -a "$PROJECT_ROOT/logs/${RUN_NAME}_train_job${SLURM_JOB_ID}.log") 2>&1
printf '[edge-absjoint] job=%s node=%s date=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Iseconds)"
printf '[edge-absjoint] output=%s\n' "$OUT"
printf '[edge-absjoint] representation=%s mean=%s std=%s\n' "$EDGE_MOTION_REPRESENTATION" "$MOTION_STATS_MEAN" "$MOTION_STATS_STD"
printf '[edge-absjoint] absolute decoded-joint MSE replaces centered MSE; feat/joint/smooth=1/1/5; contact/foot_vel/foot_height=0.05/1/10\n'
printf '[edge-absjoint] batch=128 steps=200000; fresh LR warmup=1000 peak=2e-4 cosine minimum=2e-5\n'
printf '[edge-absjoint] BONES T2M=37.5%% Nymeria T2M=37.5%% Nymeria TI2M=25%%; num_workers=8; fast caches enabled\n'
exec bash "$REPO_ROOT/motion_expert_t2m_edge/sbatch_train_1gpu.sh"
