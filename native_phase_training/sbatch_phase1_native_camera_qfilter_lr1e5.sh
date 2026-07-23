#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=np1qf4l1
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-np1qf4l1-%j.out
#SBATCH --exclusive

set -euo pipefail

export NYMERIA_QUALITY_FILTER=/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/camera_motion_quality_filter_v1_T97.json
export NATIVEP1_QUALITY_FILTER_SHA256=1fd6465890cbf175068db839beb8bb220f6964090ff2c583cbf50d5001989848
export NYMERIA_DROP_MODES=
export NATIVEP1_LORA_LR=1e-5
export NATIVEP1_ACTION_LR_MULT=4.0
export NATIVEP1_RUN_NAME=native_phase1_camera_json_bs4_lora1e5_action4x_ema_100k_qfilterv1
# Match the filtered control's manual selected-checkpoint evaluation policy.
export NATIVEP1_AUTO_EVAL=0

exec bash /home/jungbin_cho/cosmos_motion_ft/native_phase_training/sbatch_phase1_native_camera.sh
