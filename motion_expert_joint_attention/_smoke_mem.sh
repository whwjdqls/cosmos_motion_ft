#!/usr/bin/env bash
# Short real-step timing/memory probe (NOT the smoke path): 12 steps each.
#   (A) T=97 --force_on_the_fly, batch=4  -> on-the-fly VAE cost + peak mem
#   (B) T=33 precomputed cache,   batch=4  -> fast-path baseline
# Video-heavy mixture so most steps actually encode/pack video.
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
VID='{"inverse_dynamics":0.34,"policy":0.33,"forward_dynamics":0.33}'

echo "############## (A) T=97 --force_on_the_fly batch=4 ##############"
"$PY" "$TRAIN" --T 97 --force_on_the_fly --batch_size 4 --steps 12 --num_workers 4 \
    --task_weights "$VID" --tasks inverse_dynamics policy forward_dynamics \
    --gen_lora --bones_frac 0.0 --log_every 1 --save_every 100000 --viz_every 100000 --viz_n 0 --out ja_otf_memA
echo "EXIT_A=$?"

echo "############## (B) T=33 cache batch=4 ##############"
"$PY" "$TRAIN" --T 33 --batch_size 4 --steps 12 --num_workers 4 \
    --task_weights "$VID" --tasks inverse_dynamics policy forward_dynamics \
    --gen_lora --bones_frac 0.0 --log_every 1 --save_every 100000 --viz_every 100000 --viz_n 0 --out ja_otf_memB
echo "EXIT_B=$?"
