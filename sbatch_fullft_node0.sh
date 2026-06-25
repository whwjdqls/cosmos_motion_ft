#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodelist=a3ultravis-a3ultranodeset-0
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --job-name=cmot_fullft
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-fullft-%j.out

# Full-generator finetune (no LoRA): batch 32 x 8 = 256/step, 200k steps, lr 2e-5
# constant (no cosine), same LR for heads+backbone, ckpt+viz every 10k.
bash /home/jungbin_cho/cosmos_motion_ft/run_train_node.sh 8 32 200000 full 2e-5 51234
