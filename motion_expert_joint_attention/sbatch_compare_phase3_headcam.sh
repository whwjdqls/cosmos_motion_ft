#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=3:00:00
#SBATCH --job-name=p3hdcmp
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p3hdcmp-%j.out
#SBATCH --exclusive

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
BASE_RUN=/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k
HEAD_RUN=/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam
HEAD_STEP=${PHASE3_HEAD_STEP:-030000}
BASE_STEP=${PHASE3_BASE_STEP:-200000}
BASE_CONTROL_STEP=${PHASE3_BASE_CONTROL_STEP:-${HEAD_STEP}}
for value in "${HEAD_STEP}" "${BASE_STEP}" "${BASE_CONTROL_STEP}"; do
  if [[ ! ${value} =~ ^[0-9]{6}$ ]]; then
    echo "checkpoint steps must be six digits, got ${value}" >&2
    exit 2
  fi
done

HEAD_CKPT=${HEAD_RUN}/ckpt_step${HEAD_STEP}.pt
BASE_CKPT=${BASE_RUN}/ckpt_step${BASE_STEP}.pt
BASE_CONTROL_CKPT=${BASE_RUN}/ckpt_step${BASE_CONTROL_STEP}.pt
FULL_WINDOWS=/weka/jungbin/cosmos_motion_ft_runs/joint_attention/full71_windows.json
REPLACEMENT_WINDOWS=${BASE_RUN}/eval_full71_step110000_unipc30/motion_clean_replacement5_windows.json
CALIBRATION=${D}/head_camera_calibration_train.json

HEAD_FULL=${HEAD_RUN}/eval_full71_step${HEAD_STEP}_unipc30
HEAD_REPL=${HEAD_RUN}/eval_motion_clean_replacement5_step${HEAD_STEP}_unipc30
BASE_FULL=${BASE_RUN}/eval_full71_step${BASE_STEP}_unipc30
BASE_REPL=${BASE_RUN}/eval_motion_clean_replacement5_step${BASE_STEP}_unipc30
CONTROL_FULL=${BASE_RUN}/eval_full71_step${BASE_CONTROL_STEP}_unipc30
CONTROL_REPL=${BASE_RUN}/eval_motion_clean_replacement5_step${BASE_CONTROL_STEP}_unipc30
LATEST_COMPARISON=${HEAD_RUN}/compare_headcam_step${HEAD_STEP}_vs_baseline_step${BASE_STEP}.json
CONTROL_COMPARISON=${HEAD_RUN}/compare_step${HEAD_STEP}_vs_baseline_step${BASE_CONTROL_STEP}.json

for path in "${BASE_CKPT}" "${BASE_CONTROL_CKPT}" "${HEAD_CKPT}" "${FULL_WINDOWS}" \
            "${REPLACEMENT_WINDOWS}" "${CALIBRATION}"; do
  test -s "${path}"
done

echo "[p3hdcmp] node=$(hostname) date=$(date)"
echo "[p3hdcmp] headcam=${HEAD_CKPT}"
echo "[p3hdcmp] baseline_latest=${BASE_CKPT}"
echo "[p3hdcmp] baseline_same_step=${BASE_CONTROL_CKPT}"
echo "[p3hdcmp] contract=full71+replacement5 T97 seed0 CFG1 native-UniPC30"
echo "[p3hdcmp] V2M and M2V visualization=all windows"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

ALLOCATED_GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
IFS=',' read -r -a GPU_IDS <<< "${ALLOCATED_GPUS}"
if (( ${#GPU_IDS[@]} < 6 )); then
  echo "expected six allocated GPUs, got CUDA_VISIBLE_DEVICES=${ALLOCATED_GPUS}" >&2
  exit 1
fi

run_eval() {
  local slot=$1
  local checkpoint=$2
  local output=$3
  local count=$4
  local windows=$5
  if [[ -f ${output}/EVALUATION_COMPLETE ]]; then
    echo "[p3hdcmp] SKIP completed output=${output}"
    return
  fi
  mkdir -p "${output}"
  echo "[p3hdcmp] START gpu=${GPU_IDS[${slot}]} n=${count} output=${output} date=$(date)"
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
  echo "[p3hdcmp] COMPLETE output=${output} date=$(date)"
}

declare -a PIDS=()
run_eval 0 "${HEAD_CKPT}" "${HEAD_FULL}" 71 "${FULL_WINDOWS}" & PIDS+=("$!")
run_eval 1 "${BASE_CKPT}" "${BASE_FULL}" 71 "${FULL_WINDOWS}" & PIDS+=("$!")
run_eval 2 "${BASE_CONTROL_CKPT}" "${CONTROL_FULL}" 71 "${FULL_WINDOWS}" & PIDS+=("$!")
run_eval 3 "${HEAD_CKPT}" "${HEAD_REPL}" 5 "${REPLACEMENT_WINDOWS}" & PIDS+=("$!")
run_eval 4 "${BASE_CKPT}" "${BASE_REPL}" 5 "${REPLACEMENT_WINDOWS}" & PIDS+=("$!")
run_eval 5 "${BASE_CONTROL_CKPT}" "${CONTROL_REPL}" 5 "${REPLACEMENT_WINDOWS}" & PIDS+=("$!")

cleanup() {
  kill "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done
trap - INT TERM EXIT

bash "${D}/run.sh" "${D}/merge_phase3_clean71.py" \
  --full-root "${HEAD_FULL}" \
  --replacement-root "${HEAD_REPL}" \
  --template-root "${BASE_RUN}/eval_full71_step110000_unipc30"
bash "${D}/run.sh" "${D}/merge_phase3_clean71.py" \
  --full-root "${BASE_FULL}" \
  --replacement-root "${BASE_REPL}" \
  --template-root "${BASE_RUN}/eval_full71_step110000_unipc30"
bash "${D}/run.sh" "${D}/merge_phase3_clean71.py" \
  --full-root "${CONTROL_FULL}" \
  --replacement-root "${CONTROL_REPL}" \
  --template-root "${BASE_RUN}/eval_full71_step110000_unipc30"

bash "${D}/run.sh" "${D}/compare_phase3_evals.py" \
  --baseline-root "${BASE_FULL}" \
  --candidate-root "${HEAD_FULL}" \
  --candidate-label headcam \
  --out "${LATEST_COMPARISON}"
bash "${D}/run.sh" "${D}/compare_phase3_evals.py" \
  --baseline-root "${CONTROL_FULL}" \
  --candidate-root "${HEAD_FULL}" \
  --candidate-label headcam \
  --out "${CONTROL_COMPARISON}"

echo "[p3hdcmp] PASS latest_comparison=${LATEST_COMPARISON} date=$(date)"
echo "[p3hdcmp] PASS same_step_comparison=${CONTROL_COMPARISON} date=$(date)"
