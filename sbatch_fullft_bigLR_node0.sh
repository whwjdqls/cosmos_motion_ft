#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodelist=a3ultravis-a3ultranodeset-0
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=cmot_full_bigLR
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-fullbig-%j.out

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd /home/jungbin_cho/cosmos-framework

# Full-generator finetune: backbone lr 5e-5, projector (motion heads) 3x => 1.5e-4,
# 10% text-dropout (valid CFG), kimodo+FK loss, DDP x8, batch 32 (=256 eff), 200k steps.
RUN=full_full_lr5e5_proj3x_drop10_$(date +%Y%m%d_%H%M%S)
OUT=/weka/jungbin/cosmos_motion_ft_runs/${RUN}
mkdir -p "${OUT}"
echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi -L | wc -l) out=${OUT}"

torchrun --standalone --nproc_per_node=8 --master_port=51234 \
    /home/jungbin_cho/cosmos_motion_ft/train_motion_ft.py \
    --ddp --loss kimodo --data /weka/jungbin/seed/cosmos_text_motion_full \
    --out "${OUT}" \
    --lr 5e-5 --head_lr_mult 3.0 --text_dropout 0.1 \
    --lr_schedule constant --warmup_steps 500 \
    --batch_size 32 --steps 200000 --max_frames 200 \
    --save_every 10000 --viz_every 10000 --log_every 20 --grad_clip 1.0 --num_workers 8

echo "[sbatch] done -> ${OUT}"
