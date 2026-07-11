#!/usr/bin/env bash
set -euo pipefail

cd /home/jungbin_cho/cosmos-framework
export PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework:${PYTHONPATH:-}
export BASE_CHECKPOINT_PATH=${BASE_CHECKPOINT_PATH:-/weka/jungbin/cosmos3_nano_dcp}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/weka/jungbin/wan22_vae/Wan2.2_VAE.pth}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/weka/jungbin/cosmos_motion_ft_runs}
export NYMERIA_NUM_FRAMES=${NYMERIA_NUM_FRAMES:-97}
export NYMERIA_LATENT_ROOT=${NYMERIA_LATENT_ROOT:-/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T${NYMERIA_NUM_FRAMES}}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

exec /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
  /home/jungbin_cho/cosmos_motion_ft/native_phase_training/run_latent_train.py \
  --sft-toml /home/jungbin_cho/cosmos_motion_ft/native_phase_training/world_camera_nymeria_latent.toml \
  "$@"
