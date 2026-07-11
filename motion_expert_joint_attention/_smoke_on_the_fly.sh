#!/usr/bin/env bash
# =============================================================================
# _smoke_on_the_fly.sh — GPU smoke for ON-THE-FLY Wan-VAE latent encoding.
#
# Diagnostic (one-off, "_"-prefixed) — verifies the T=97 on-the-fly path:
#   (A) T=97 --force_on_the_fly: camera tasks (inverse_dynamics/policy) VAE-encode raw frames
#       live -> (C,25,h,w) latents, forward runs, camera/vision losses finite.
#   (B) T=33 (default cache): confirms the fast path hits the precomputed cache (no VAE load).
# Run on a node (ssh in first): bash _smoke_on_the_fly.sh
# =============================================================================
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
cd /home/jungbin_cho/cosmos-framework
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/motion_expert:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
PY=~/miniforge3/envs/cosmos/bin/python
TRAIN=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/train.py

echo "############## (A) T=97 --force_on_the_fly, camera tasks ##############"
"$PY" "$TRAIN" --smoke --T 97 --force_on_the_fly --num_workers 2 \
    --tasks inverse_dynamics policy forward_dynamics video2motion \
    --gen_lora --bones_frac 0.0
echo "EXIT_A=$?"

echo "############## (B) T=33 default cache, fast path (no VAE) ##############"
"$PY" "$TRAIN" --smoke --T 33 --num_workers 2 \
    --tasks inverse_dynamics policy text2motion \
    --gen_lora --bones_frac 0.0
echo "EXIT_B=$?"
