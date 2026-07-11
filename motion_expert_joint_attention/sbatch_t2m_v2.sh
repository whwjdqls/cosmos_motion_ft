#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=t2m_v2
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-t2mv2-%j.out
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[t2m_v2] node=$(hostname) $(date) — text2motion on the NEW sparse-depth(12)/fresh-init/3072 motion expert"
bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp --tasks text2motion --out ja_t2m_v2_sparse12 \
  --steps 200000 --batch_size 64
