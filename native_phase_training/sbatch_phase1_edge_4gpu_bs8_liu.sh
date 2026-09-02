#!/bin/bash
# Preemption-hardened continuation of the four-GPU Edge Phase-1 run on L40.
#SBATCH --partition=liu-compute
#SBATCH --qos=ll-med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:4
#SBATCH --exclude=ll-l40-1.grasp.maas
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --job-name=edgep1-4g8b-liu
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgep1-4g8b-liu-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
export NATIVEP1_WORLD_SIZE=4
export NATIVEP1_CLIPS_PER_GPU=8
export NATIVEP1_SMOKE_CLIPS_PER_GPU=8
export NATIVEP1_SMOKE_RUN_NAME=${NATIVEP1_SMOKE_RUN_NAME:-edge_phase1_T97_20fps_bs8_camera_wearer_lora_smoke_gate1}
export NATIVEP1_RUN_NAME=${NATIVEP1_RUN_NAME:-edge_phase1_T97_20fps_bs32_4gpu_bs8_camera_wearer_global_lora_100k_v1}
export NATIVEP1_WANDB_MODE=${NATIVEP1_WANDB_MODE:-online}

# This supplements the permanent 5k checkpoints. The framework keeps the last
# three wall-clock checkpoints and never prunes a save_iter milestone.
export COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES=${COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES:-15}

echo "[edge-p1-liu] partition=liu-compute qos=ll-med excluded_node=ll-l40-1.grasp.maas"
echo "[edge-p1-liu] rolling_checkpoint_minutes=${COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES}"

exec bash "${REPO_ROOT}/native_phase_training/sbatch_phase1_edge_8gpu.sh"
