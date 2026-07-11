#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --job-name=resmoke7
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-resmoke7-%j.out
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
cd /home/jungbin_cho/cosmos-framework
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
LAT=/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents
echo "[resmoke7] node=$(hostname) $(date) — sparse-depth motion expert (stride=3, 12 blocks)"
echo "===== CPU mask self-tests =====" && python $D/mot_joint_attention.py && python $D/mot_joint_layer.py
echo "===== MOTION path (text2motion; tests empty full-block at plain layers) ====="
CUDA_VISIBLE_DEVICES=0 python $D/smoke_7task.py --gen_lora --steps 3; echo "[resmoke7] motion exit=$?"
echo "===== VIDEO path (video2motion, real latents) ====="
CUDA_VISIBLE_DEVICES=0 python $D/smoke_7task.py --gen_lora --steps 3 --latents_dir $LAT --video_mode video2motion; echo "[resmoke7] video exit=$?"
echo "===== DONE $(date) ====="
