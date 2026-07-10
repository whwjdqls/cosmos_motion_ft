#!/usr/bin/env bash
#SBATCH --job-name=c31_offtxt_rec1
#SBATCH --partition=a2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /home/jungbin_cho/cosmos_motion_ft/shape_aware_TMR

bash st_run.sh st_train.py \
  --out-dir /mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c31_offtxt_frz_mu_rec1 \
  --stats-path /mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/stats_v0 \
  --pretrained-text-encoder /home/jungbin_cho/.cache/huggingface/hub/models--nvidia--TMR-SOMA-RP-v1/snapshots/e427752ae3446dedba49e928c93ddc9f0e413401/last_weights/text_encoder.pt \
  --freeze-text-encoder \
  --text-use-mean \
  --lambda-recon 1.0 \
  --lambda-kl-motion 1e-4 \
  --lambda-kl-text 0.0 \
  --max-steps 10000 \
  --warmup 1000 \
  --lr 3e-4 \
  --batch 256 \
  --num-workers 6 \
  --ckpt-every 1000 \
  --eval-every 1000 \
  --eval-cases 0 \
  --natural-desc4-only \
  --natural-weight 2 \
  --frozen-dup-filter \
  --text-dup-threshold 0.95 \
  --aug-feat-noise-std 0.02 \
  --info-nce-temp 0.1
