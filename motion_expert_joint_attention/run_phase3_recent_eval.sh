#!/usr/bin/env bash

# Serial held-out V2M/M2V evaluation for native Phase-3 bridge checkpoints.
# Usage: run_phase3_recent_eval.sh GPU_INDEX STEP [STEP ...]

set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 GPU_INDEX STEP [STEP ...]" >&2
  exit 2
fi

GPU_INDEX=$1
shift

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../restored_env.sh
source "${REPO_ROOT}/restored_env.sh"
D="${REPO_ROOT}/motion_expert_joint_attention"
RUN="${RUN_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k"
FULL_WINDOWS="${RUN_ROOT}/joint_attention/full71_windows.json"
REPLACEMENT_WINDOWS=${RUN}/eval_full71_step110000_unipc30/motion_clean_replacement5_windows.json

export CUDA_VISIBLE_DEVICES=${GPU_INDEX}

for STEP in "$@"; do
  if [[ ! ${STEP} =~ ^[0-9]{6}$ ]]; then
    echo "step must be a zero-padded six-digit integer, got: ${STEP}" >&2
    exit 2
  fi

  CKPT=${RUN}/ckpt_step${STEP}.pt
  FULL_OUT=${RUN}/eval_full71_step${STEP}_unipc30
  REPL_OUT=${RUN}/eval_motion_clean_replacement5_step${STEP}_unipc30

  test -s "${CKPT}"
  test -s "${FULL_WINDOWS}"
  test -s "${REPLACEMENT_WINDOWS}"

  if [[ -f ${REPL_OUT}/EVALUATION_COMPLETE ]]; then
    echo "[recent-eval] $(date) SKIP completed step=${STEP}"
    continue
  fi

  mkdir -p "${FULL_OUT}" "${REPL_OUT}"
  echo "[recent-eval] $(date) START full71 step=${STEP} gpu=${GPU_INDEX}"
  bash "${D}/run.sh" "${D}/eval_all.py" \
    --ckpt "${CKPT}" \
    --out_dir "${FULL_OUT}" \
    --n 71 \
    --tasks video2motion motimg2video \
    --windows_json "${FULL_WINDOWS}" \
    --steps 30 \
    --cfg 1 \
    --seed 0 \
    --motion_native_solver unipc \
    --split test \
    --num_frames 97 \
    --resolution 256 \
    --motion_viz_limit -1 \
    --device cuda 2>&1 | tee "${FULL_OUT}/eval.log"

  echo "[recent-eval] $(date) START replacement5 step=${STEP} gpu=${GPU_INDEX}"
  bash "${D}/run.sh" "${D}/eval_all.py" \
    --ckpt "${CKPT}" \
    --out_dir "${REPL_OUT}" \
    --n 5 \
    --tasks video2motion motimg2video \
    --windows_json "${REPLACEMENT_WINDOWS}" \
    --steps 30 \
    --cfg 1 \
    --seed 0 \
    --motion_native_solver unipc \
    --split test \
    --num_frames 97 \
    --resolution 256 \
    --motion_viz_limit -1 \
    --device cuda 2>&1 | tee "${REPL_OUT}/eval.log"

  touch "${REPL_OUT}/EVALUATION_COMPLETE"
  echo "[recent-eval] $(date) COMPLETE step=${STEP}"
done
