#!/bin/bash
# Controlled high-tier counterpart of the completed Original Phase-1 run.
# The released Nano contract is model tier 720, shift 10, and raw 640x640 pixels.
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10-00:00:00
#SBATCH --job-name=p1cam720
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p1cam720-%j.out
#SBATCH --exclusive
#SBATCH --exclude=a3ultravis-a3ultranodeset-2

set -euo pipefail

export NYMERIA_NUM_FRAMES=97
export NYMERIA_RESOLUTION=720
export NYMERIA_LATENT_ROOT=/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97_720tier_640
export NATIVEP1_EXPECTED_LATENT_HW=40
export NATIVEP1_EXPECTED_IMAGE_HW=640
export NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=1
export NATIVEP1_CACHE_VALIDATION_SAMPLES=256

# Match the historical Original run except for spatial tier and its released
# resolution-adaptive RF shift.
export NYMERIA_QUALITY_FILTER=
export NATIVEP1_QUALITY_FILTER_SHA256=
export NYMERIA_DROP_MODES=
export NYMERIA_REPLACE_STANDALONE_C=0
export NATIVEP1_ADAPTATION_MODE=global_lora
export NATIVEP1_PREFIX_LENGTHS=1
export NATIVEP1_PREFIX_SAMPLING_WEIGHTS=
export NATIVEP1_ACTION_LOSS_WEIGHT=10.0
export NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=0
export NATIVEP1_LORA_LR=5e-5
export NATIVEP1_ACTION_LR_MULT=4.0
export NATIVEP1_SHIFT_OVERRIDE=10.0
export NATIVEP1_CLIPS_PER_GPU=4
export NATIVEP1_MAX_ITER=100000
export NATIVEP1_SAVE_ITER=5000
export NATIVEP1_PREFLIGHT_STEPS=2
export NATIVEP1_RUN_NAME=native_phase1_camera_json_720tier640_bs4_lora5e5_action4x_ema_100k

export NATIVEP1_AUTO_EVAL=1
export NATIVEP1_AUTO_EVAL_EVERY=10000
export NATIVEP1_VIZ_N=5
export NATIVEP1_EVAL_PREFIX_LENGTHS=1
export NATIVEP1_EVAL_INPUT_DIR=/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_viz5_720_T97_release_s10_v1
export NATIVEP1_AUTO_EVAL_FULL71=1
export NATIVEP1_FULL71_EVAL_EVERY=10000
export NATIVEP1_FULL71_EVAL_INPUT_DIR=/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_720_T97_release_s10_v1

exec bash /home/jungbin_cho/cosmos_motion_ft/native_phase_training/sbatch_phase1_native_camera.sh
