#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=ja7_3d
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-ja7full3d-%j.out

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[ja7_3d] node=$(hostname) $(date) -- FULL 7-task from scratch, gen-LoRA + motion expert, motion_mrope=cosmos3d"

bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp --bones_frac 0.5 --gen_lora --precomputed_latents --viz_n 8 --T 97 \
  --motion_mrope cosmos3d \
  --resume auto --out ja_7task_full_mrope3d --steps 200000 --batch_size 4
