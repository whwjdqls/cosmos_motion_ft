#!/usr/bin/env bash
#SBATCH -p a2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=bsnatp2
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-bsnatp2-%j.out

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert
RUN_NAME=${BS_NATIVE_RUN_NAME:-bs_native_x0_logitnormal_shift3_w1_1_5_200k}
SHIFT=${BS_NATIVE_SHIFT:-3}
STEPS=${BS_NATIVE_STEPS:-200000}
START_STEP=${BS_NATIVE_START_STEP:-0}
INIT_CKPT=${BS_NATIVE_INIT_CKPT:-}
LR=${BS_NATIVE_LR:-0.0002}
SEED=${BS_NATIVE_SEED:-0}
BATCH_SIZE=${BS_NATIVE_BATCH_SIZE:-128}
DATALOADER_RNG=${BS_NATIVE_DATALOADER_RNG:-dedicated}
STEP_INDEXING=${BS_NATIVE_STEP_INDEXING:-completed_updates}
INDEX_CACHE=${BS_NATIVE_INDEX_CACHE:-/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_incontext_v1/bs_train_index.json}
MEAN=${BS_NATIVE_MEAN:-/mnt/shared/jungbin_cho/seed/soma_proportional_uniegomotion_20fps/Mean_uniego.npy}
STD=${BS_NATIVE_STD:-/mnt/shared/jungbin_cho/seed/soma_proportional_uniegomotion_20fps/Std_uniego.npy}
NORMALIZATION_TAG=${BS_NATIVE_NORMALIZATION_TAG:-bones_seed_proportional_20fps}
SMOKE=${BS_NATIVE_SMOKE:-0}
W_FEAT=${BS_NATIVE_W_FEAT:-1}
W_JOINT=${BS_NATIVE_W_JOINT:-1}
W_SMOOTH=${BS_NATIVE_W_SMOOTH:-5}
W_CONTACT=${BS_NATIVE_W_CONTACT:-0}
W_FOOT_VEL=${BS_NATIVE_W_FOOT_VEL:-0}
W_FOOT_HEIGHT=${BS_NATIVE_W_FOOT_HEIGHT:-0}
CONTACT_LOGIT_SCALE=${BS_NATIVE_CONTACT_LOGIT_SCALE:-10}
INLINE_EVAL_EVERY=${BS_NATIVE_INLINE_EVAL_EVERY:-10000}
INLINE_EVAL_MAX_CASES=${BS_NATIVE_INLINE_EVAL_MAX_CASES:-0}
INLINE_EVAL_SHAPE_CF=${BS_NATIVE_INLINE_EVAL_SHAPE_CF:-farthest}

echo "[bsnatp2] node=$(hostname) date=$(date)"
echo "[bsnatp2] run=${RUN_NAME} shift=${SHIFT} steps=${START_STEP}->${STEPS} batch=${BATCH_SIZE} lr=${LR} seed=${SEED} dataloader_rng=${DATALOADER_RNG} step_indexing=${STEP_INDEXING} smoke=${SMOKE}"
echo "[bsnatp2] init_ckpt=${INIT_CKPT:-none}"
echo "[bsnatp2] losses=${W_FEAT}/${W_JOINT}/${W_SMOOTH} contact=${W_CONTACT} foot_vel=${W_FOOT_VEL} foot_height=${W_FOOT_HEIGHT}"
echo "[bsnatp2] inline_eval_every=${INLINE_EVAL_EVERY} inline_eval_max_cases=${INLINE_EVAL_MAX_CASES} shape_cf=${INLINE_EVAL_SHAPE_CF}"
echo "[bsnatp2] index_cache=${INDEX_CACHE}"
echo "[bsnatp2] normalization=${NORMALIZATION_TAG} mean=${MEAN} std=${STD}"
sha256sum "${MEAN}" "${STD}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader

ARGS=(
  --native_shift "${SHIFT}"
  --native_num_train_timesteps 1000
  --steps "${STEPS}"
  --start_step "${START_STEP}"
  --lr "${LR}"
  --seed "${SEED}"
  --batch_size "${BATCH_SIZE}"
  --dataloader_rng "${DATALOADER_RNG}"
  --step_indexing "${STEP_INDEXING}"
  --w_feat "${W_FEAT}"
  --w_joint "${W_JOINT}"
  --w_smooth "${W_SMOOTH}"
  --w_contact "${W_CONTACT}"
  --w_foot_vel "${W_FOOT_VEL}"
  --w_foot_height "${W_FOOT_HEIGHT}"
  --contact_logit_scale "${CONTACT_LOGIT_SCALE}"
  --mean "${MEAN}"
  --std "${STD}"
  --normalization_tag "${NORMALIZATION_TAG}"
  --index_cache "${INDEX_CACHE}"
  --num_workers "${SLURM_CPUS_PER_TASK:-8}"
  --viz_n 4
  --viz_steps 35
  --inline_eval_every "${INLINE_EVAL_EVERY}"
  --inline_eval_max_cases "${INLINE_EVAL_MAX_CASES}"
  --inline_eval_steps 35
  --inline_eval_solver unipc
  --inline_eval_shape_counterfactual "${INLINE_EVAL_SHAPE_CF}"
)

if [[ -n "${INIT_CKPT}" ]]; then
  ARGS+=(--init_ckpt "${INIT_CKPT}")
fi

if [[ "${SMOKE}" == "1" ]]; then
  ARGS+=(--smoke --viz_n 0 --inline_eval_every 0)
else
  ARGS+=(
    --run_name "${RUN_NAME}"
    --save_every 10000
    --viz_every 10000
  )
fi

bash "${D}/bs_run.sh" bs_native_train.py "${ARGS[@]}"

echo "[bsnatp2] done date=$(date)"
