#!/usr/bin/env bash
# One-L40S end-to-end gate for W&B scalars plus fixed T2M/TI2M media.
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=edge-p2-wbviz
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edge-p2-wbviz-%j.out

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
OUT=${OUT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/_smoke_wandb_viz_${SLURM_JOB_ID}}
WANDB_PROJECT=${WANDB_PROJECT:-cosmos-motion-ft}
WANDB_ENTITY=${WANDB_ENTITY:-jungbinc-upenn}

exec bash "${REPO_ROOT}/motion_expert_t2m_edge/run.sh" \
    "${REPO_ROOT}/motion_expert_t2m_edge/train.py" \
    --smoke --out "${OUT}" --T 16 --ti2m-frames 16 \
    --task-weights '{"text2motion":0.75,"textimg2motion":0.25}' \
    --bones-frac 0 --reasoner-image-size 256 \
    --wandb-mode online --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-run-name "edge-phase2-wandb-viz-smoke-${SLURM_JOB_ID}" \
    --wandb-group cosmos3-edge-phase2-smoke --require-wandb \
    --viz-every 1000000 --viz-samples-per-task 1 \
    --viz-steps 2 --viz-guidance 1 --viz-frame-stride 2 \
    --viz-at-start --no-viz-final --require-viz
