#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=ja7_nym
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-ja7nym-%j.out
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[ja7_nym] node=$(hostname) $(date) — FULL 7-task, NYMERIA ONLY (bones_frac=0), gen-LoRA, sparse-12 motion expert"
bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp --bones_frac 0 --gen_lora --viz_n 8 \
  --out ja_7task_nymeria_sparse12 --steps 200000 --batch_size 16
