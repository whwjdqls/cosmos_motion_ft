#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=ja7_full
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-ja7full-%j.out
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[ja7_full] node=$(hostname) $(date) — FULL 7-task, ALL FIXES (timestep, gen-LoRA+action heads, action_gen, sparse-12, in-env viz)"
# all 7 tasks (default); bones_frac 0.5 -> t2m on Nymeria+BONES; gen_lora now trains attn-LoRA + action heads
bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp --bones_frac 0.5 --gen_lora --precomputed_latents --viz_n 8 --T 97 \
  --resume auto --out ja_7task_full --steps 200000 --batch_size 4
