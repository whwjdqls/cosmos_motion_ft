#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CKPT OUT_DIR WINDOWS_JSON HIGHRES_LATENT_ROOT" >&2
  exit 2
fi

CKPT=$1
OUT_DIR=$2
WINDOWS_JSON=$3
LATENT_ROOT=$4
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../restored_env.sh
source "${REPO_ROOT}/restored_env.sh"
D="${REPO_ROOT}/motion_expert_joint_attention"

for path in "${CKPT}" "${WINDOWS_JSON}"; do
  test -s "${path}"
done
test -d "${LATENT_ROOT}"
N=$(jq 'length' "${WINDOWS_JSON}")
if [[ ! ${N} =~ ^[1-9][0-9]*$ ]]; then
  echo "windows JSON must contain a nonempty array: ${WINDOWS_JSON}" >&2
  exit 2
fi

if [[ -f ${OUT_DIR}/EVALUATION_COMPLETE ]]; then
  echo "[phase3-m2v-720] SKIP completed ${OUT_DIR}"
  exit 0
fi

mkdir -p "${OUT_DIR}"
echo "[phase3-m2v-720] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[phase3-m2v-720] ckpt=${CKPT}"
echo "[phase3-m2v-720] windows=${WINDOWS_JSON} n=${N}"
echo "[phase3-m2v-720] contract=T97 VAE-bucket-480 latent-40x40 shift10 UniPC30 CFG1 seed0"

bash "${D}/run.sh" "${D}/eval_all.py" \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  --n "${N}" \
  --tasks motimg2video \
  --windows_json "${WINDOWS_JSON}" \
  --latent_root "${LATENT_ROOT}" \
  --steps 30 \
  --cfg 1 \
  --seed 0 \
  --split test \
  --num_frames 97 \
  --resolution 480 \
  --gen_shift_override 10 \
  --expected_m2v_latent_hw 40 \
  --motion_viz_limit 0 \
  --device cuda 2>&1 | tee "${OUT_DIR}/eval.log"

touch "${OUT_DIR}/EVALUATION_COMPLETE"
echo "[phase3-m2v-720] COMPLETE ${OUT_DIR}"
