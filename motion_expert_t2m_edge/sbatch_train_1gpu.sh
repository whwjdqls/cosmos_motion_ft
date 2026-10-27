#!/usr/bin/env bash
# Canonical one-L40 Cosmos-3 Edge Phase-2 Nymeria-only training run.
# Auto-resumes the newest rolling/regular checkpoint; RESUME may override it.
#SBATCH --partition=liu-compute
#SBATCH --qos=ll-med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --exclude=ll-l40-1.grasp.maas
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=edge-phase2-7l-1g
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edge-phase2-7l-1g-%j.out

set -euo pipefail
export COSMOS_SITE=${COSMOS_SITE:-grasp}
REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
OUT=${OUT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/motion_expert_t2m_edge/edge_7layer_nymeria_t2m_ti2m_v1_wandb_viz_preemptsafe}
BATCH_SIZE=${BATCH_SIZE:-128}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'NPROC_PER_NODE must be a positive integer, got %s\n' "${NPROC_PER_NODE}" >&2
    exit 2
fi
TORCHRUN_ARGS=()
if (( NPROC_PER_NODE > 1 )); then
    TORCHRUN_ARGS=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node "${NPROC_PER_NODE}" --max_restarts 0)
fi
BONES_FRAC=${BONES_FRAC:-0}
TASK_WEIGHTS=${TASK_WEIGHTS:-}
if [[ -z "${TASK_WEIGHTS}" ]]; then
    TASK_WEIGHTS='{"text2motion":0.75,"textimg2motion":0.25}'
fi
WANDB_PROJECT=${WANDB_PROJECT:-cosmos-motion-ft}
WANDB_ENTITY=${WANDB_ENTITY:-jungbinc-upenn}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-edge-phase2-7l-nymeria-t2m-ti2m-v1-preemptsafe}
WANDB_GROUP=${WANDB_GROUP:-cosmos3-edge-phase2}
WANDB_STORAGE_ROOT=${WANDB_STORAGE_ROOT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/.wandb_runtime}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${WANDB_STORAGE_ROOT}/data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${WANDB_STORAGE_ROOT}/cache}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
export WANDB_INIT_ATTEMPTS=${WANDB_INIT_ATTEMPTS:-3}
export WANDB_INIT_RETRY_DELAY=${WANDB_INIT_RETRY_DELAY:-10}
mkdir -p "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}"
VIZ_EVERY=${VIZ_EVERY:-5000}
VIZ_SAMPLES_PER_TASK=${VIZ_SAMPLES_PER_TASK:-5}
RESUME=${RESUME:-auto}
RESUME_ARGS=()
if [[ -n "${RESUME}" ]]; then
    RESUME_ARGS=(--resume "${RESUME}")
fi

exec bash "${REPO_ROOT}/motion_expert_t2m_edge/run.sh" \
    "${TORCHRUN_ARGS[@]}" \
    "${REPO_ROOT}/motion_expert_t2m_edge/train.py" \
    --out "${OUT}" --T 200 --batch-size "${BATCH_SIZE}" \
    --ti2m-frames 97 --task-weights "${TASK_WEIGHTS}" \
    --bones-frac "${BONES_FRAC}" --reasoner-image-size 256 \
    --grad-accum "${GRAD_ACCUM}" --num-workers "${NUM_WORKERS:-8}" --recovery-save-every 250 \
    --min-lr-ratio "${MIN_LR_RATIO}" \
    --adam-beta2 "${ADAM_BETA2:-0.95}" --loss-reduction "${LOSS_REDUCTION:-pooled}" \
    --loss-guard "${LOSS_GUARD:-strict}" --allow-tf32 "${ALLOW_TF32:-1}" \
    --w-feat "${W_FEAT:-1}" --w-joint "${W_JOINT:-10}" --w-smooth "${W_SMOOTH:-50}" \
    --w-global-rot "${W_GLOBAL_ROT:-0}" --w-global-trans "${W_GLOBAL_TRANS:-0}" \
    --joint-loss-mode "${JOINT_LOSS_MODE:-centered}" \
    --w-contact "${W_CONTACT:-0}" --w-foot-vel "${W_FOOT_VEL:-0}" \
    --w-foot-height "${W_FOOT_HEIGHT:-0}" \
    --wandb-mode online --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" --wandb-run-name "${WANDB_RUN_NAME}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-service-wait "${WANDB__SERVICE_WAIT}" \
    --wandb-init-attempts "${WANDB_INIT_ATTEMPTS}" \
    --wandb-init-retry-delay "${WANDB_INIT_RETRY_DELAY}" --require-wandb \
    --viz-every "${VIZ_EVERY}" --viz-samples-per-task "${VIZ_SAMPLES_PER_TASK}" \
    --viz-source "${VIZ_SOURCE:-nymeria}" \
    --viz-steps 35 --viz-guidance 2 --viz-frame-stride 2 \
    --viz-at-start --require-viz \
    "${RESUME_ARGS[@]}"
