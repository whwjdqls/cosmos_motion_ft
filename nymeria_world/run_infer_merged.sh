#!/usr/bin/env bash
# Run native inference with a MERGED full DCP (LoRA+heads merged into base).
# Usage: run_infer_merged.sh <MERGED_DCP> <INPUT_JSONL> <OUT_DIR> <NUM_FRAMES> <CUDA_DEV>
set -e
cd /home/jungbin_cho/cosmos-framework
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft/nymeria_world
unset LD_LIBRARY_PATH
MERGED=$1; INPUT=$2; OUT=$3; export NYMERIA_NUM_FRAMES=${4:-97}; DEV=${5:-0}

CUDA_VISIBLE_DEVICES="$DEV" /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
  -m cosmos_framework.scripts.inference \
  -i "$INPUT" -o "$OUT" --no-guardrails \
  --experiment world_camera_nymeria_nano \
  --experiment-overrides model.config.lora_enabled=false \
    model.config.tokenizer.vae_path=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth \
  --checkpoint-path "$MERGED"
