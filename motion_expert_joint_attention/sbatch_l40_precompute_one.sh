#!/usr/bin/env bash
# Encode one canonical held-out T97 window for restored-server Phase-3 smoke tests.
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=cosmos-l40-vae

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck source=../restored_env.sh
source "${REPO_ROOT}/restored_env.sh"

OUT_ROOT=${OUT_ROOT:-${RUN_ROOT}/l40_smoke/latents_T97_256}
WINDOWS_JSON=${WINDOWS_JSON:-${RUN_ROOT}/joint_attention/full71_windows.json}
LOG_DIR=${RUN_ROOT}/l40_smoke/precompute_${SLURM_JOB_ID:-manual}
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"
exec > >(tee "${LOG_DIR}/run.log") 2>&1

echo "[l40-vae] date=$(date -Is) node=$(hostname) out=${OUT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

bash "${REPO_ROOT}/motion_expert_joint_attention/run.sh" \
  "${REPO_ROOT}/motion_expert_joint_attention/precompute_latents.py" \
  --out_root "${OUT_ROOT}" \
  --num_frames 97 \
  --resolution 256 \
  --split test \
  --windows_json "${WINDOWS_JSON}" \
  --limit 1 \
  --log_every 1 \
  --fail_on_error \
  --device cuda

find "${OUT_ROOT}" -type f -name '*.npz' -size +0c -print -quit | grep -q .
echo "[l40-vae] COMPLETE ${OUT_ROOT}"
