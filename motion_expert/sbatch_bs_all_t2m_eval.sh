#!/usr/bin/env bash
#SBATCH --job-name=bsalltmr
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-bsalltmr-%j.out

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert
RUN_ROOT=/mnt/shared/jungbin_cho/cosmos_motion_ft_runs
CHECKPOINT=${BS_ALL_CHECKPOINT:?set BS_ALL_CHECKPOINT to the generator checkpoint}
LABEL=${BS_ALL_LABEL:-full_contact}
OUT_DIR=${BS_ALL_OUT_DIR:-${RUN_ROOT}/bs_tmr_eval/${LABEL}_all_text2motion}
TMR_RUN=${RUN_ROOT}/shape_tmr/c45_official30fps_balanced_10k
TMR_CKPT=${BS_TMR_CKPT:-${TMR_RUN}/step_00005000.pt}
TMR_STATS=${BS_TMR_STATS:-/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/snapshots/e427752ae3446dedba49e928c93ddc9f0e413401/stats/motion}
STEPS=${BS_TMR_STEPS:-35}
BATCH_SIZE=${BS_TMR_BATCH_SIZE:-16}
MAX_CASES=${BS_TMR_MAX_CASES:-0}
NATIVE_SOLVER=${BS_TMR_NATIVE_SOLVER:-unipc}
SHAPE_COUNTERFACTUAL=${BS_TMR_SHAPE_COUNTERFACTUAL:-farthest}

mkdir -p "${OUT_DIR}"
echo "[bsalltmr] job=${SLURM_JOB_ID:-none} host=$(hostname) date=$(date)"
echo "[bsalltmr] checkpoint=${CHECKPOINT} label=${LABEL} out=${OUT_DIR}"
echo "[bsalltmr] solver=${NATIVE_SOLVER} steps=${STEPS} batch=${BATCH_SIZE} max_cases=${MAX_CASES} shape_cf=${SHAPE_COUNTERFACTUAL}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

SUITES=(
  "content overview"
  "content timeline_single"
  "content timeline_multi"
  "repetition overview"
  "repetition timeline_single"
  "repetition timeline_multi"
)

for suite in "${SUITES[@]}"; do
  read -r split group <<< "${suite}"
  out="${OUT_DIR}/${split}_${group}.json"
  echo "[bsalltmr] start ${split}/${group} date=$(date)"
  bash "${D}/bs_run.sh" bs_tmr_eval.py \
    --generator "${LABEL}=${CHECKPOINT}" \
    --tmr-ckpt "${TMR_CKPT}" \
    --tmr-stats "${TMR_STATS}" \
    --out "${out}" \
    --split "${split}" \
    --group "${group}" \
    --steps "${STEPS}" \
    --native-solver "${NATIVE_SOLVER}" \
    --batch-size "${BATCH_SIZE}" \
    --max-cases "${MAX_CASES}" \
    --shape-counterfactual "${SHAPE_COUNTERFACTUAL}"
done

bash "${D}/bs_run.sh" bs_all_benchmark_summary.py \
  --input-dir "${OUT_DIR}" \
  --out "${OUT_DIR}/summary.json"

echo "[bsalltmr] done date=$(date) summary=${OUT_DIR}/summary.json"
