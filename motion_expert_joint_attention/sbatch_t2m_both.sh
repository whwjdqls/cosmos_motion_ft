#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=t2m_both
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-t2mboth-%j.out
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[t2m_both] node=$(hostname) $(date) — text2motion on Nymeria + BONES-SEED (Nymeria stats, raw text -> reasoner)"
# --tasks text2motion + default --bones_frac 0.5 => trains on BOTH sources; Nymeria stats are the default
# (MOTION_STATS_MEAN/STD). --viz_n 8 => ~4 Nymeria-test + ~4 BONES-test held-out samples each viz step.
bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp --tasks text2motion --bones_frac 0.5 --T 200 --viz_n 8 --viz_every 2000 --resume auto \
  --out ja_t2m_x0_T200 --steps 200000 --batch_size 32
