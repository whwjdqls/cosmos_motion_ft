#!/bin/bash
# Four-GPU, eight-clips-per-rank Cosmos3-Edge Phase 1 production run.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=96:00:00
#SBATCH --job-name=edgep1-4g8b
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgep1-4g8b-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
export NATIVEP1_WORLD_SIZE=4
export NATIVEP1_CLIPS_PER_GPU=8
export NATIVEP1_SMOKE_CLIPS_PER_GPU=8
export NATIVEP1_SMOKE_RUN_NAME=${NATIVEP1_SMOKE_RUN_NAME:-edge_phase1_T97_20fps_bs8_camera_wearer_lora_smoke_gate1}
export NATIVEP1_RUN_NAME=${NATIVEP1_RUN_NAME:-edge_phase1_T97_20fps_bs32_4gpu_bs8_camera_wearer_global_lora_100k_v1}

exec bash "${REPO_ROOT}/native_phase_training/sbatch_phase1_edge_8gpu.sh"
