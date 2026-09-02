#!/usr/bin/env bash
# One-GPU restored-server sampling probe. Override CKPT/T/STEPS/PROMPT as needed.
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=cosmos-l40-smoke

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck source=../restored_env.sh
source "${REPO_ROOT}/restored_env.sh"

CKPT=${CKPT:-${RUN_ROOT}/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt}
T=${T:-33}
STEPS=${STEPS:-2}
PROMPT=${PROMPT:-a person walks forward}
OUT=${OUT:-${RUN_ROOT}/l40_smoke/phase2_t2m_T${T}_steps${STEPS}_${SLURM_JOB_ID:-manual}}

mkdir -p "${OUT}"
exec > >(tee "${OUT}/run.log") 2>&1

echo "[l40-smoke] date=$(date -Is) node=$(hostname) checkpoint=${CKPT}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

bash "${REPO_ROOT}/motion_expert_joint_attention/run.sh" \
  "${REPO_ROOT}/motion_expert_joint_attention/sample.py" \
  --ckpt "${CKPT}" \
  --out "${OUT}" \
  --prompts "${PROMPT}" \
  --T "${T}" \
  --steps "${STEPS}" \
  --cfg 1 \
  --seed 0

test -s "${OUT}/manifest.json"
echo "[l40-smoke] COMPLETE ${OUT}"
