#!/bin/bash
# Canonical Phase-1 benchmark: one prefix-1 forward and inverse sample for each
# of the 71 held-out sequences. The five-prefix qualitative suite is separate.
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --job-name=native71
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-native71-%j.out

set -euo pipefail

: "${CHECKPOINT_PATH:?CHECKPOINT_PATH must point to an iter_XXXXXXXXX DCP checkpoint}"
: "${EVAL_INPUT_DIR:?EVAL_INPUT_DIR must contain the canonical full-71 JSONLs}"
: "${EVAL_OUTPUT_DIR:?EVAL_OUTPUT_DIR is required}"

iteration=$(basename "${CHECKPOINT_PATH}")
run_dir=$(dirname "$(dirname "${CHECKPOINT_PATH}")")
eval_root=$(dirname "${EVAL_OUTPUT_DIR}")
if [[ "${EVAL_OUTPUT_DIR}" != "${eval_root}/${iteration}" ]]; then
    echo "[native71] output/checkpoint mismatch: ${EVAL_OUTPUT_DIR} vs ${iteration}" >&2
    exit 1
fi

export RUN_DIR="${run_dir}"
export EVAL_INPUT_DIR
export EVAL_ROOT="${eval_root}"
export FULL71_ITERATION="${iteration}"
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export DP_SHARD_SIZE=${DP_SHARD_SIZE:-${NPROC_PER_NODE}}

exec bash /home/jungbin_cho/cosmos_motion_ft/native_phase_training/run_full71_all_checkpoints.sh
