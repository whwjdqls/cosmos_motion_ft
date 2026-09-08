#!/usr/bin/env bash
# Production Cosmos3-Edge T97 Wan-latent shards 1-3 on Yonsei RTX 4090s.
# Shard 0 runs in the persistent smoke-test allocation.
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node36
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-3
#SBATCH --job-name=p1latent-t97
#SBATCH --output=/lustre/whwjdqls99/cosmos/logs/slurm-wan-latents-T97-%A_%a.out

set -euo pipefail

export TOTAL_SHARDS=4
export SHARD_ID=${SLURM_ARRAY_TASK_ID}

exec bash /home/whwjdqls99/cosmos_motion_ft/native_phase_training/run_precompute_latents_256tier_yonsei.sh shard
