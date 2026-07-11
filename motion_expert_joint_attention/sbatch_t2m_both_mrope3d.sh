#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=t2m3d
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-t2m3d-%j.out

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[t2m3d] node=$(hostname) $(date) - text2motion Nymeria+BONES, x0, T=200, motion_mrope=cosmos3d"

# Same recipe as sbatch_t2m_both.sh / ja_t2m_x0_T200, except motion rotary positions use
# official-style 3D-mRoPE (T x 1 x 1) so motion frames share the video/camera temporal frame.
bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp --tasks text2motion --bones_frac 0.5 --T 200 \
  --motion_mrope cosmos3d \
  --viz_n 8 --viz_every 2000 --resume auto \
  --out ja_t2m_x0_T200_mrope3d --steps 200000 --batch_size 32
