#!/usr/bin/env bash
# Two-L40 FSDP smoke for one official Phase-1 forward-dynamics sample.
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --job-name=cosmos-l40-native

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck source=../restored_env.sh
source "${REPO_ROOT}/restored_env.sh"

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${RUN_ROOT}/cosmos3_camera/camera_world/native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000}
INPUT_SOURCE=${INPUT_SOURCE:-${RUN_ROOT}/native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1/fd_input.jsonl}
OUT=${OUT:-${RUN_ROOT}/l40_smoke/native_phase1_vqA_fd_${NPROC_PER_NODE}gpu_${SLURM_JOB_ID:-manual}}
INPUT=${OUT}/fd_one.jsonl
MEMORY_LOG=${OUT}/gpu_memory.csv

mkdir -p "${OUT}"
head -n 1 "${INPUT_SOURCE}" | jq -c --arg root "${WEKA_ROOT}" \
  '(.vision_path, .action_path) |= sub("^/weka/jungbin"; $root)
   | del(.rgb_prefix_length, .latent_prefix_length, .source_name)' > "${INPUT}"
exec > >(tee "${OUT}/run.log") 2>&1

test -s "${CHECKPOINT_PATH}/model/.metadata"
test -s "${INPUT}"
test -s "${WAN_VAE_PATH}"

echo "[l40-native] date=$(date -Is) node=$(hostname) checkpoint=${CHECKPOINT_PATH}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader

(
  while true; do
    printf '%s,' "$(date +%s)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -
    sleep 1
  done
) > "${MEMORY_LOG}" &
monitor_pid=$!

cd "${COSMOS_FRAMEWORK_ROOT}"
set +e
"${COSMOS_ENV_ROOT}/bin/torchrun" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m cosmos_framework.scripts.inference \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-file native_phase_training/inference_config.py \
  --experiment world_camera_nymeria_latent_nano \
  --sampler unipc \
  --use-ema-weights \
  --parallelism-preset throughput \
  --dp-shard-size "${NPROC_PER_NODE}" --dp-replicate-size 1 --cp-size 1 --cfgp-size 1 \
  --no-use-torch-compile --no-use-cuda-graphs --no-guardrails \
  -o "${OUT}" \
  -i "${INPUT}"
status=$?
set -e

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true

awk -F, '
  { for (i=2; i<=NF; i++) if (($i+0)>max[i]) max[i]=$i+0 }
  END { for (i=2; i<=NF; i++) printf "[l40-native] gpu%d_peak_nvidia_smi_mib=%d\n", i-2, max[i] }
' "${MEMORY_LOG}"

if [[ ${status} -ne 0 ]]; then
  echo "[l40-native] FAILED status=${status}" >&2
  exit "${status}"
fi

find "${OUT}" -type f -name vision.mp4 -size +0c -print -quit | grep -q .
echo "[l40-native] COMPLETE ${OUT}"
