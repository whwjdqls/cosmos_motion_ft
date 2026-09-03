#!/usr/bin/env bash
# Shared-batch alternative for the canonical one-GPU Phase-2 launcher.
# The called launcher supplies the training/W&B/checkpoint environment; SBATCH
# directives in that called file are comments once it is executed with bash.
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|l40"
#SBATCH --exclude=ll-l40-1.grasp.maas
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=edge-phase2-7l-1g
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edge-phase2-7l-1g-%j.out

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
exec bash "${REPO_ROOT}/motion_expert_t2m_edge/sbatch_train_1gpu.sh"
