#!/usr/bin/env bash
#SBATCH --job-name=eval_shapeaware
#SBATCH --partition=a2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /home/jungbin_cho/cosmos_motion_ft/shape_aware_TMR
mkdir -p slurm_logs

OFFICIAL=/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/snapshots/e427752ae3446dedba49e928c93ddc9f0e413401
UNIFORM=/home/jungbin_cho/seed/soma_uniform_motions_20fps
PROPORTIONAL=/mnt/shared/jungbin_cho/seed/soma_proportional_uniegomotion_20fps
RUN_ROOT=/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr
C45=${RUN_ROOT}/c45_official30fps_balanced_10k
MATRIX=${RUN_ROOT}/shape_awareness_matrix
mkdir -p "${MATRIX}"

echo "== official TMR on uniform uniego motions =="
if [[ -s "${MATRIX}/official_tmr_uniform_full.json" ]]; then
  echo "skip existing ${MATRIX}/official_tmr_uniform_full.json"
else
  bash st_run.sh official_tmr_eval.py \
    --uniego-root "${UNIFORM}" \
    --out "${MATRIX}/official_tmr_uniform_full.json" \
    --batch-size 64 \
    --cases 0
fi

echo "== official TMR on proportional uniego motions =="
if [[ -s "${MATRIX}/official_tmr_proportional_full.json" ]]; then
  echo "skip existing ${MATRIX}/official_tmr_proportional_full.json"
else
  bash st_run.sh official_tmr_eval.py \
    --uniego-root "${PROPORTIONAL}" \
    --out "${MATRIX}/official_tmr_proportional_full.json" \
    --batch-size 64 \
    --cases 0
fi

echo "== C45 shape-aware TMR on uniform uniego motions =="
if [[ -s "${C45}/full6_uniform_step_00005000.json" ]]; then
  echo "skip existing ${C45}/full6_uniform_step_00005000.json"
else
  bash st_run.sh st_eval.py \
    --ckpt "${C45}/step_00005000.pt" \
    --stats-path "${OFFICIAL}/stats/motion" \
    --uniego-root "${UNIFORM}" \
    --out "${C45}/full6_uniform_step_00005000.json" \
    --cases 100000
fi

echo "== C45 shape-aware TMR on proportional uniego motions =="
if [[ -s "${C45}/full6_proportional_step_00005000_rerun.json" ]]; then
  echo "skip existing ${C45}/full6_proportional_step_00005000_rerun.json"
else
  bash st_run.sh st_eval.py \
    --ckpt "${C45}/step_00005000.pt" \
    --stats-path "${OFFICIAL}/stats/motion" \
    --uniego-root "${PROPORTIONAL}" \
    --out "${C45}/full6_proportional_step_00005000_rerun.json" \
    --cases 100000
fi
