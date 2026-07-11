#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --job-name=smoke7
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-smoke7-%j.out
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
cd /home/jungbin_cho/cosmos-framework
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
LAT=/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents
echo "[smoke7] node=$(hostname) $(date)"
echo "===== STEP 1: precompute latents (subset of 16 windows) ====="
CUDA_VISIBLE_DEVICES=0 python $D/precompute_latents.py --limit 16; echo "[smoke7] precompute exit=$?"
echo "===== STEP 2: smoke MOTION path (text2motion + motion tasks, gen_lora) ====="
CUDA_VISIBLE_DEVICES=0 python $D/smoke_7task.py --gen_lora --steps 3; echo "[smoke7] smoke-motion exit=$?"
echo "===== STEP 3: smoke VIDEO path (video2motion, real latents — the real test) ====="
CUDA_VISIBLE_DEVICES=0 python $D/smoke_7task.py --gen_lora --steps 3 --latents_dir $LAT --video_mode video2motion; echo "[smoke7] smoke-video exit=$?"
echo "===== DONE $(date) ====="
