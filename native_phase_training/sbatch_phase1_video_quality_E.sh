#!/bin/bash
# Optional factorization E: single-frame prefix with camera-token-only K/V LoRA.
# Smoke-test only by default; do not submit with the A-D suite.
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=np1vqE
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-np1vqE-%j.out
#SBATCH --exclusive

set -euo pipefail

export NYMERIA_QUALITY_FILTER=/weka/jungbin/nymeriaplus_kimodo_proportional/metadata/camera_motion_quality_filter_v1_T97.json
export NATIVEP1_QUALITY_FILTER_SHA256=1fd6465890cbf175068db839beb8bb220f6964090ff2c583cbf50d5001989848
export NYMERIA_DROP_MODES=image2video
export NYMERIA_REPLACE_STANDALONE_C=1
export NATIVEP1_LORA_LR=5e-5
export NATIVEP1_ACTION_LR_MULT=4.0
export NATIVEP1_ACTION_LOSS_WEIGHT=2.0
export NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=1
export NATIVEP1_ADAPTATION_MODE=camera_kv_lora
export NATIVEP1_PREFIX_LENGTHS=1
export NATIVEP1_PREFIX_SAMPLING_WEIGHTS=
export NATIVEP1_RUN_NAME=native_phase1_vq_E_p1_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person
export NATIVEP1_AUTO_EVAL=1
export NATIVEP1_AUTO_EVAL_EVERY=10000
export NATIVEP1_PREFLIGHT_STEPS=2
export NATIVEP1_VIZ_N=5
export NATIVEP1_EVAL_INPUT_DIR=/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1
export NATIVEP1_EVAL_PREFIX_LENGTHS=1,9,17,33,49
export NATIVEP1_AUTO_EVAL_FULL71=1
export NATIVEP1_FULL71_EVAL_EVERY=10000
export NATIVEP1_FULL71_EVAL_INPUT_DIR=/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2

exec bash /home/jungbin_cho/cosmos_motion_ft/native_phase_training/sbatch_phase1_native_camera.sh
