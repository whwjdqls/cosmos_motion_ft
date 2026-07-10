#!/usr/bin/env bash
#SBATCH --job-name=eval_c32_full6
#SBATCH --partition=a2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /home/jungbin_cho/cosmos_motion_ft/shape_aware_TMR

bash st_run.sh st_eval.py \
  --ckpt /mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c32_offtxt_frz_mu_rec001_klm3e6/last.pt \
  --stats-path /mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/stats_v0 \
  --out /mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c32_offtxt_frz_mu_rec001_klm3e6/full6_eval.json \
  --cases 100000 \
  --device cuda
