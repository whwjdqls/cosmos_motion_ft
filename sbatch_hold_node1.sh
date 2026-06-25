#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodelist=a3ultravis-a3ultranodeset-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --job-name=hold_node1_lora
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-hold1-%j.out

# Placeholder: the LoRA finetune runs on node 1 via ssh+tmux (outside Slurm), which
# leaves node 1 marked 'idle' and schedulable by other users -> collision risk. This
# job reserves node 1 in Slurm to reflect the real occupancy. It does NO compute (just
# sleeps); the actual LoRA training keeps using the GPUs. scancel when the LoRA run ends.
echo "[hold] node=$(hostname) reserving node 1 for the ssh+tmux LoRA run. start=$(date)"
sleep 108000   # ~30h, covers the 200k-step LoRA run; Slurm --time also bounds it
echo "[hold] done=$(date)"
