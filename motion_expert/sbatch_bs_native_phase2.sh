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
BATCH_SIZE=${BS_NATIVE_BATCH_SIZE:-128}
INDEX_CACHE=${BS_NATIVE_INDEX_CACHE:-/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/bs_incontext_v1/bs_train_index.json}
SMOKE=${BS_NATIVE_SMOKE:-0}

echo "[bsnatp2] node=$(hostname) date=$(date)"
echo "[bsnatp2] run=${RUN_NAME} shift=${SHIFT} steps=${STEPS} batch=${BATCH_SIZE} smoke=${SMOKE}"
echo "[bsnatp2] index_cache=${INDEX_CACHE}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader

ARGS=(
  --native_shift "${SHIFT}"
  --native_num_train_timesteps 1000
  --batch_size "${BATCH_SIZE}"
  --w_feat 1
  --w_joint 1
  --w_smooth 5
  --index_cache "${INDEX_CACHE}"
  --num_workers "${SLURM_CPUS_PER_TASK:-8}"
  --viz_n 4
  --viz_steps 50
)

if [[ "${SMOKE}" == "1" ]]; then
  ARGS+=(--smoke --viz_n 0)
else
  ARGS+=(
    --steps "${STEPS}"
    --run_name "${RUN_NAME}"
    --save_every 5000
    --viz_every 5000
  )
fi

bash "${D}/bs_run.sh" bs_native_train.py "${ARGS[@]}"

echo "[bsnatp2] done date=$(date)"
