#!/usr/bin/env bash
#SBATCH --job-name=c46_off30_ret
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

bash st_run.sh st_train.py \
  --out-dir /mnt/shared/jungbin_cho/cosmos_motion_ft_runs/shape_tmr/c46_official30fps_retrieval_only_balanced_10k \
  --stats-path "${OFFICIAL}/stats/motion" \
  --fps 30 \
  --data-fps 20 \
  --pretrained-text-encoder "${OFFICIAL}/last_weights/text_encoder.pt" \
  --pretrained-motion-encoder "${OFFICIAL}/last_weights/motion_encoder.pt" \
  --freeze-text-encoder \
  --text-use-mean \
  --motion-use-mean \
  --lambda-recon 0.0 \
  --lambda-kl-motion 0.0 \
  --lambda-kl-text 0.0 \
  --max-steps 10000 \
  --warmup 1000 \
  --lr 5e-5 \
  --batch 128 \
  --num-workers 6 \
  --ckpt-every 1000 \
  --eval-every 1000 \
  --eval-cases 0 \
  --natural-desc4-only \
  --natural-weight 1 \
  --frozen-dup-filter \
  --text-dup-threshold 0.95 \
  --aug-feat-noise-std 0.02 \
  --info-nce-temp 0.1
