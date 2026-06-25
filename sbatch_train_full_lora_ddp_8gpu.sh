#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=cmot_lora_ddp
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-lora-ddp-%j.out

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd /home/jungbin_cho/cosmos-framework

DATA_DIR="${1:-/weka/jungbin/seed/cosmos_text_motion_full}"
STEPS="${2:-20000}"
BATCH="${3:-8}"
RUN_NAME=full_lora_ddp_kimodofk_$(date +%Y%m%d_%H%M%S)
OUT_DIR=/weka/jungbin/cosmos_motion_ft_runs/${RUN_NAME}
mkdir -p "${OUT_DIR}"

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi -L | wc -l) data=${DATA_DIR} steps=${STEPS} batch=${BATCH} out=${OUT_DIR}"

# LoRA + projection finetune (reasoner frozen), plain DDP (no FSDP — model fits per GPU),
# kimodo bones_seed loss (per-block weighted smooth-L1 + FK consistency).
torchrun --standalone --nproc_per_node=8 \
    /home/jungbin_cho/cosmos_motion_ft/train_motion_ft.py \
    --ddp --lora --loss kimodo \
    --data "${DATA_DIR}" \
    --out "${OUT_DIR}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH}" \
    --lr 2e-4 \
    --max_frames 200 \
    --save_every 2000 \
    --log_every 20 \
    --grad_clip 1.0 \
    --num_workers 8

echo "[sbatch] done -> ${OUT_DIR}"
