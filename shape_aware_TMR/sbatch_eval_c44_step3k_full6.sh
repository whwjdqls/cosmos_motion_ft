#!/usr/bin/env bash
#SBATCH --job-name=eval_c44_3k
#SBATCH --partition=a2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /home/jungbin_cho/cosmos_motion_ft/shape_aware_TMR
mkdir -p slurm_logs

OFFICIAL=/home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/snapshots/e427752ae3446dedba49e928c93ddc9f0e413401
RUN=/mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c44_official30fps_shape_10k

bash st_run.sh st_eval.py \
  --ckpt "${RUN}/step_00003000.pt" \
  --stats-path "${OFFICIAL}/stats/motion" \
  --out "${RUN}/full6_step_00003000.json"
