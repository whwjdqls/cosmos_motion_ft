#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=cosmos_motion_ft
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-%j.out

set -euo pipefail

# ---- env (the cosmos venv; LD_LIBRARY_PATH MUST be cleared or torch import fails) ----
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# run from repo root so `import cosmos_framework` resolves
cd /home/jungbin_cho/cosmos-framework

# ---- paths ----
DATA_DIR=/weka/jungbin/seed/cosmos_text_motion_subset
RUN_NAME=full_generator_fsdp8_$(date +%Y%m%d_%H%M%S)
OUT_DIR=/weka/jungbin/cosmos_motion_ft_runs/${RUN_NAME}
mkdir -p "${OUT_DIR}"

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi -L | wc -l) data=${DATA_DIR} out=${OUT_DIR}"

# ---- launch: full-generator finetune, FSDP2 across 8 GPUs ----
# (If ${DATA_DIR} is missing the script auto-falls-back to synthetic data and
#  logs a loud warning -- swap DATA_DIR to the full export when it is ready.)
torchrun --standalone --nproc_per_node=8 \
    /home/jungbin_cho/cosmos_motion_ft/train_motion_ft.py \
    --fsdp \
    --data "${DATA_DIR}" \
    --out "${OUT_DIR}" \
    --steps 2000 \
    --batch_size 2 \
    --lr 1e-4 \
    --max_frames 200 \
    --save_every 500 \
    --log_every 10 \
    --grad_clip 1.0 \
    --num_workers 4

echo "[sbatch] done -> ${OUT_DIR}"
