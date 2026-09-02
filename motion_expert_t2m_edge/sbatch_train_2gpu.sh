#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=96:00:00
#SBATCH --job-name=edge-phase2-7l
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edge-phase2-7l-%j.out

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
OUT=${OUT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_nymeria_t2m_ti2m_2gpu_v1}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}

exec bash "${REPO_ROOT}/motion_expert_t2m_edge/run.sh" \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    "${REPO_ROOT}/motion_expert_t2m_edge/train.py" \
    --out "${OUT}" --T 200 --batch-size "${BATCH_SIZE}" \
    --ti2m-frames 97 --task-weights '{"text2motion":0.75,"textimg2motion":0.25}' \
    --bones-frac 0 --reasoner-image-size 256 \
    --grad-accum "${GRAD_ACCUM}" --num-workers 8
