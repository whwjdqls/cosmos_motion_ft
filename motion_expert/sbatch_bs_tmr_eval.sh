#!/usr/bin/env bash
#SBATCH --job-name=bstmreval
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-bstmreval-%j.out

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert
RUN_ROOT=/mnt/shared/jungbin_cho/cosmos_motion_ft_runs
NATIVE_CKPT=${BS_NATIVE_CKPT:-${RUN_ROOT}/bs_native_x0_logitnormal_shift3_w1_1_5_200k/latest.pt}
LEGACY_CKPT=${BS_LEGACY_CKPT:-${RUN_ROOT}/bs_incontext_v1/latest.pt}
TMR_RUN=${RUN_ROOT}/shape_tmr/c45_official30fps_balanced_10k
TMR_CKPT=${BS_TMR_CKPT:-${TMR_RUN}/step_00005000.pt}
TMR_STATS=${BS_TMR_STATS:-/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/snapshots/e427752ae3446dedba49e928c93ddc9f0e413401/stats/motion}
MAX_CASES=${BS_TMR_MAX_CASES:-0}
STEPS=${BS_TMR_STEPS:-100}
BATCH_SIZE=${BS_TMR_BATCH_SIZE:-16}
NATIVE_SOLVER=${BS_TMR_NATIVE_SOLVER:-euler}
SHAPE_COUNTERFACTUAL=${BS_TMR_SHAPE_COUNTERFACTUAL:-farthest}
NATIVE_LABEL=${BS_TMR_NATIVE_LABEL:-native}
ONLY_NATIVE=${BS_TMR_ONLY_NATIVE:-0}
OUT=${BS_TMR_OUT:-${RUN_ROOT}/bs_tmr_eval/c45_step5k_content_overview_native_vs_legacy.json}

echo "[bstmreval] job=${SLURM_JOB_ID:-none} host=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

ARGS=(
  --generator "${NATIVE_LABEL}=${NATIVE_CKPT}"
  --tmr-ckpt "${TMR_CKPT}"
  --tmr-stats "${TMR_STATS}"
  --out "${OUT}"
  --steps "${STEPS}"
  --native-solver "${NATIVE_SOLVER}"
  --batch-size "${BATCH_SIZE}"
  --max-cases "${MAX_CASES}"
  --shape-counterfactual "${SHAPE_COUNTERFACTUAL}"
)
if [[ "${ONLY_NATIVE}" != "1" ]]; then
  ARGS+=(--generator "legacy=${LEGACY_CKPT}")
fi

echo "[bstmreval] native_solver=${NATIVE_SOLVER} steps=${STEPS} only_native=${ONLY_NATIVE} shape_cf=${SHAPE_COUNTERFACTUAL}"
bash "${D}/bs_run.sh" bs_tmr_eval.py "${ARGS[@]}"

echo "[bstmreval] done date=$(date) out=${OUT}"
