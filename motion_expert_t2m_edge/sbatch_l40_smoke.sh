#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --job-name=edge-phase2-smoke
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edge-phase2-smoke-%j.out

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
OUT=${OUT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/smoke_${SLURM_JOB_ID:-manual}}
mkdir -p "${OUT}"

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
bash "${REPO_ROOT}/motion_expert_t2m_edge/run.sh" \
    "${REPO_ROOT}/motion_expert_t2m_edge/train.py" \
    --smoke --T 16 --out "${OUT}/train"

CHECKPOINT=${OUT}/train/checkpoints/step_000000002.pt
test -s "${CHECKPOINT}"
bash "${REPO_ROOT}/motion_expert_t2m_edge/run.sh" \
    "${REPO_ROOT}/motion_expert_t2m_edge/sample.py" \
    --checkpoint "${CHECKPOINT}" --out "${OUT}/sample" \
    --mode textimg2motion --T 16 --steps 2 --guidance 1 --no-render
test -s "${OUT}/sample/manifest.json"
echo "[edge-phase2-smoke] PASS ${OUT}"
