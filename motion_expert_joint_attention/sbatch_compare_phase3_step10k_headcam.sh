#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --job-name=p3s10cmp
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p3s10cmp-%j.out
#SBATCH --exclusive

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
BASE_RUN=/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k
HEAD_RUN=/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam
BASE_CKPT=${BASE_RUN}/ckpt_step010000.pt
HEAD_CKPT=${HEAD_RUN}/ckpt_step010000.pt
FULL_WINDOWS=/weka/jungbin/cosmos_motion_ft_runs/joint_attention/full71_windows.json
REPLACEMENT_WINDOWS=${BASE_RUN}/eval_full71_step110000_unipc30/motion_clean_replacement5_windows.json
CALIBRATION=${D}/head_camera_calibration_train.json

BASE_FULL=${BASE_RUN}/eval_full71_step010000_unipc30
BASE_REPL=${BASE_RUN}/eval_motion_clean_replacement5_step010000_unipc30
HEAD_FULL=${HEAD_RUN}/eval_full71_step010000_unipc30
HEAD_REPL=${HEAD_RUN}/eval_motion_clean_replacement5_step010000_unipc30
COMPARISON=${HEAD_RUN}/compare_step010000_vs_baseline_step010000.json

for path in "${BASE_CKPT}" "${HEAD_CKPT}" "${FULL_WINDOWS}" \
            "${REPLACEMENT_WINDOWS}" "${CALIBRATION}"; do
  test -s "${path}"
done

echo "[p3s10cmp] node=$(hostname) date=$(date)"
echo "[p3s10cmp] baseline=${BASE_CKPT}"
echo "[p3s10cmp] headcam=${HEAD_CKPT}"
echo "[p3s10cmp] contract=full71+replacement5 T97 seed0 CFG1 native-UniPC30"
echo "[p3s10cmp] V2M and M2V visualization=all windows"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

ALLOCATED_GPUS=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -r -a GPU_IDS <<< "${ALLOCATED_GPUS}"
if (( ${#GPU_IDS[@]} < 2 )); then
  echo "expected two allocated GPUs, got CUDA_VISIBLE_DEVICES=${ALLOCATED_GPUS}" >&2
  exit 1
fi

run_eval() {
  local slot=$1
  local checkpoint=$2
  local output=$3
  local count=$4
  local windows=$5
  if [[ -f ${output}/EVALUATION_COMPLETE ]]; then
    echo "[p3s10cmp] SKIP completed output=${output}"
    return
  fi
  mkdir -p "${output}"
  echo "[p3s10cmp] START gpu=${GPU_IDS[${slot}]} n=${count} output=${output} date=$(date)"
  CUDA_VISIBLE_DEVICES=${GPU_IDS[${slot}]} bash "${D}/run.sh" "${D}/eval_all.py" \
    --ckpt "${checkpoint}" \
    --out_dir "${output}" \
    --n "${count}" \
    --tasks video2motion motimg2video \
    --windows_json "${windows}" \
    --steps 30 \
    --cfg 1 \
    --seed 0 \
    --motion_native_solver unipc \
    --eval_head_camera_alignment \
    --head_camera_calibration "${CALIBRATION}" \
    --split test \
    --num_frames 97 \
    --resolution 256 \
    --motion_viz_limit -1 \
    --device cuda 2>&1 | tee "${output}/eval.log"
  touch "${output}/EVALUATION_COMPLETE"
  echo "[p3s10cmp] COMPLETE output=${output} date=$(date)"
}

run_checkpoint() {
  local slot=$1
  local checkpoint=$2
  local full_output=$3
  local replacement_output=$4
  run_eval "${slot}" "${checkpoint}" "${full_output}" 71 "${FULL_WINDOWS}"
  run_eval "${slot}" "${checkpoint}" "${replacement_output}" 5 "${REPLACEMENT_WINDOWS}"
}

run_checkpoint 0 "${BASE_CKPT}" "${BASE_FULL}" "${BASE_REPL}" &
PID_BASE=$!
run_checkpoint 1 "${HEAD_CKPT}" "${HEAD_FULL}" "${HEAD_REPL}" &
PID_HEAD=$!

cleanup() {
  kill "${PID_BASE}" "${PID_HEAD}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
wait "${PID_BASE}"
wait "${PID_HEAD}"
trap - INT TERM EXIT

bash "${D}/run.sh" "${D}/merge_phase3_clean71.py" \
  --full-root "${BASE_FULL}" \
  --replacement-root "${BASE_REPL}" \
  --template-root "${BASE_RUN}/eval_full71_step110000_unipc30"
bash "${D}/run.sh" "${D}/merge_phase3_clean71.py" \
  --full-root "${HEAD_FULL}" \
  --replacement-root "${HEAD_REPL}" \
  --template-root "${BASE_RUN}/eval_full71_step110000_unipc30"

bash "${D}/run.sh" "${D}/compare_phase3_evals.py" \
  --baseline-root "${BASE_FULL}" \
  --candidate-root "${HEAD_FULL}" \
  --candidate-label headcam \
  --out "${COMPARISON}"

echo "[p3s10cmp] PASS comparison=${COMPARISON} date=$(date)"
