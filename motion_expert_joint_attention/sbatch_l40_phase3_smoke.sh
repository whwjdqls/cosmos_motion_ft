#!/usr/bin/env bash
# One-window Phase-3 V2M sampling probe using the restored portable specialists.
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=cosmos-l40-p3

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck source=../restored_env.sh
source "${REPO_ROOT}/restored_env.sh"

CKPT=${CKPT:-${RUN_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/ckpt_step200000.pt}
LATENT_ROOT=${LATENT_ROOT:-${RUN_ROOT}/l40_smoke/latents_T97_256}
FULL_WINDOWS=${FULL_WINDOWS:-${RUN_ROOT}/joint_attention/full71_windows.json}
TASK=${TASK:-video2motion}
STEPS=${STEPS:-2}
DECODE_VIDEO=${DECODE_VIDEO:-0}
OUT=${OUT:-${RUN_ROOT}/l40_smoke/phase3_${TASK}_T97_steps${STEPS}_${SLURM_JOB_ID:-manual}}
ONE_WINDOW=${OUT}/window.json

mkdir -p "${OUT}"
jq '.[0:1]' "${FULL_WINDOWS}" > "${ONE_WINDOW}"
exec > >(tee "${OUT}/run.log") 2>&1

echo "[l40-p3] date=$(date -Is) node=$(hostname) checkpoint=${CKPT}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

video_args=(--no_video)
if [[ "${DECODE_VIDEO}" == "1" ]]; then
  video_args=()
fi

bash "${REPO_ROOT}/motion_expert_joint_attention/run.sh" \
  "${REPO_ROOT}/motion_expert_joint_attention/eval_all.py" \
  --ckpt "${CKPT}" \
  --out_dir "${OUT}" \
  --n 1 \
  --tasks "${TASK}" \
  --windows_json "${ONE_WINDOW}" \
  --latent_root "${LATENT_ROOT}" \
  --steps "${STEPS}" \
  --cfg 1 \
  --seed 0 \
  --motion_native_solver unipc \
  --split test \
  --num_frames 97 \
  --resolution 256 \
  "${video_args[@]}" \
  --motion_viz_limit 0 \
  --device cuda

test -s "${OUT}/summary.json"
echo "[l40-p3] COMPLETE ${OUT}"
