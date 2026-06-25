#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=cosmos_motion_ft_full
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-fullft-%j.out

set -euo pipefail

# ---- env (cosmos venv; LD_LIBRARY_PATH MUST be cleared or torch import fails) ----
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd /home/jungbin_cho/cosmos-framework

# ---- paths ----
# Full multi-shard export (parent dir containing shard_*/). The trainer's
# TextMotionDataset must support the multi-shard parent dir (added by the export agent).
DATA_DIR="${1:-/weka/jungbin/seed/cosmos_text_motion_full}"
STEPS="${2:-20000}"
RUN_NAME=full_data_kimodo_fk_fsdp8_$(date +%Y%m%d_%H%M%S)
OUT_DIR=/weka/jungbin/cosmos_motion_ft_runs/${RUN_NAME}
mkdir -p "${OUT_DIR}"

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi -L | wc -l) data=${DATA_DIR} steps=${STEPS} out=${OUT_DIR}"

# ---- launch: full-generator finetune (reasoner frozen), FSDP2 x8, kimodo FK loss ----
torchrun --standalone --nproc_per_node=8 \
    /home/jungbin_cho/cosmos_motion_ft/train_motion_ft.py \
    --fsdp \
    --loss kimodo \
    --data "${DATA_DIR}" \
    --out "${OUT_DIR}" \
    --steps "${STEPS}" \
    --batch_size 2 \
    --lr 1e-4 \
    --max_frames 200 \
    --save_every 2000 \
    --log_every 20 \
    --grad_clip 1.0 \
    --num_workers 8

echo "[sbatch] done -> ${OUT_DIR}"
