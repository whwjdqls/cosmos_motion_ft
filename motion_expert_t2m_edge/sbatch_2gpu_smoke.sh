#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --job-name=edge-phase2-ddp-smoke
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edge-phase2-ddp-smoke-%j.out

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
OUT=${OUT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/ddp_smoke_${SLURM_JOB_ID:-manual}}

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
exec bash "${REPO_ROOT}/motion_expert_t2m_edge/run.sh" \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    "${REPO_ROOT}/motion_expert_t2m_edge/train.py" \
    --smoke --T 200 --out "${OUT}"
