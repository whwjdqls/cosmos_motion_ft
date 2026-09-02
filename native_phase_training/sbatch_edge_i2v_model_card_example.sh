#!/bin/bash
# NVIDIA's bundled Cosmos3-Edge Diffusers I2V example on its bundled assets.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=edgei2vref
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgei2vref-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
DIFFUSERS_ROOT=${DIFFUSERS_ROOT:-/mnt/projects/ll/jungbinc/cosmos3_edge_diffusers_diag/diffusers}
DIFFUSERS_VENV=${DIFFUSERS_VENV:-/mnt/projects/ll/jungbinc/cosmos3_edge_diffusers_diag/venv}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_model_card_i2v_diffusers_v040_seed0}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
DISABLE_SAFETY_CHECKER=${DISABLE_SAFETY_CHECKER:-0}
INPUT_IMAGE=${INPUT_IMAGE:-${EDGE_MODEL_ROOT}/assets/example_i2v_input.jpg}
PROMPT_JSON=${PROMPT_JSON:-${EDGE_MODEL_ROOT}/assets/example_i2v_prompt.json}
PROMPT_TEXT_FILE=${PROMPT_TEXT_FILE:-}
NUM_FRAMES=${NUM_FRAMES:-121}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
FPS=${FPS:-24}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-20}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-6}
FLOW_SHIFT=${FLOW_SHIFT:-12}
SEED=${SEED:-0}
USE_NYMERIA_TEMPLATE=${USE_NYMERIA_TEMPLATE:-0}
NYMERIA_TEMPLATE_JSON=${NYMERIA_TEMPLATE_JSON:-${REPO_ROOT}/native_phase_training/prompts/nymeria_i2v_prompt_template_v2_1.json}
USE_NYMERIA_NEGATIVE_TEMPLATE=${USE_NYMERIA_NEGATIVE_TEMPLATE:-${USE_NYMERIA_TEMPLATE}}
if [[ "${USE_NYMERIA_NEGATIVE_TEMPLATE}" == 1 ]]; then
  NEGATIVE_PROMPT_JSON=${NEGATIVE_PROMPT_JSON:-${REPO_ROOT}/native_phase_training/prompts/nymeria_i2v_negative_prompt_template_v2.json}
else
  NEGATIVE_PROMPT_JSON=${NEGATIVE_PROMPT_JSON:-${EDGE_MODEL_ROOT}/assets/negative_prompt.json}
fi

if [[ -n "${PROMPT_TEXT_FILE}" ]]; then
  PROMPT_PATH=${PROMPT_TEXT_FILE}
else
  PROMPT_PATH=${PROMPT_JSON}
fi

for required in \
  "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${INPUT_IMAGE}" \
  "${PROMPT_PATH}" \
  "${NEGATIVE_PROMPT_JSON}" \
  "${DIFFUSERS_ROOT}/src/diffusers" \
  "${DIFFUSERS_VENV}/bin/python"; do
  [[ -e "${required}" ]] || { echo "[edge-i2v-ref] missing required artifact: ${required}" >&2; exit 1; }
done

CUDA_PYTHON_LIB_ROOT=${COSMOS_ENV_ROOT}/lib/python3.13/site-packages/nvidia
export LD_LIBRARY_PATH=${CUDA_PYTHON_LIB_ROOT}/cudnn/lib:${CUDA_PYTHON_LIB_ROOT}/cublas/lib:${CUDA_PYTHON_LIB_ROOT}/curand/lib:${CUDA_PYTHON_LIB_ROOT}/npp/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_nvrtc/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_runtime/lib:${CUDA_PYTHON_LIB_ROOT}/nvjitlink/lib
# PEFT only probes Transformer Engine while importing. The shared TE wheel was
# built against a newer glibc than these nodes; this documented TE build-mode
# switch leaves that optional integration unavailable without touching the
# shared cosmos environment.
export NVTE_PROJECT_BUILDING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${DIFFUSERS_ROOT}/src:${PYTHONPATH:-}

echo "[edge-i2v-ref] node=$(hostname) model=${EDGE_MODEL_ROOT}"
echo "[edge-i2v-ref] Diffusers commit=$(git -C "${DIFFUSERS_ROOT}" rev-parse HEAD)"
echo "[edge-i2v-ref] input=${INPUT_IMAGE} prompt=${PROMPT_PATH} negative=${NEGATIVE_PROMPT_JSON}"
echo "[edge-i2v-ref] recipe: ${WIDTH}x${HEIGHT}, T${NUM_FRAMES}, ${FPS} FPS, shift${FLOW_SHIFT}/${NUM_INFERENCE_STEPS}, guidance${GUIDANCE_SCALE}, seed${SEED}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

PROMPT_ARGS=(--prompt-json "${PROMPT_JSON}")
if [[ -n "${PROMPT_TEXT_FILE}" ]]; then
  PROMPT_ARGS=(--prompt-text-file "${PROMPT_TEXT_FILE}")
fi
TEMPLATE_ARGS=()
if [[ "${USE_NYMERIA_TEMPLATE}" == 1 ]]; then
  [[ -n "${PROMPT_TEXT_FILE}" ]] || {
    echo "[edge-i2v-ref] USE_NYMERIA_TEMPLATE=1 requires PROMPT_TEXT_FILE" >&2
    exit 1
  }
  [[ -f "${NYMERIA_TEMPLATE_JSON}" ]] || {
    echo "[edge-i2v-ref] missing Nymeria positive template: ${NYMERIA_TEMPLATE_JSON}" >&2
    exit 1
  }
  TEMPLATE_ARGS+=(--use-nymeria-template)
  TEMPLATE_ARGS+=(--nymeria-template-json "${NYMERIA_TEMPLATE_JSON}")
elif [[ "${USE_NYMERIA_TEMPLATE}" != 0 ]]; then
  echo "[edge-i2v-ref] USE_NYMERIA_TEMPLATE must be 0 or 1" >&2
  exit 1
fi
if [[ "${USE_NYMERIA_NEGATIVE_TEMPLATE}" == 1 ]]; then
  TEMPLATE_ARGS+=(--use-nymeria-negative-template)
elif [[ "${USE_NYMERIA_NEGATIVE_TEMPLATE}" != 0 ]]; then
  echo "[edge-i2v-ref] USE_NYMERIA_NEGATIVE_TEMPLATE must be 0 or 1" >&2
  exit 1
fi
SAFETY_ARGS=()
if [[ "${DISABLE_SAFETY_CHECKER}" == 1 ]]; then
  SAFETY_ARGS+=(--disable-safety-checker)
elif [[ "${DISABLE_SAFETY_CHECKER}" != 0 ]]; then
  echo "[edge-i2v-ref] DISABLE_SAFETY_CHECKER must be 0 or 1" >&2
  exit 1
fi
"${DIFFUSERS_VENV}/bin/python" \
  "${REPO_ROOT}/native_phase_training/run_edge_i2v_model_card_example.py" \
  --model "${EDGE_MODEL_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --diffusers-source "${DIFFUSERS_ROOT}" \
  --image "${INPUT_IMAGE}" \
  "${PROMPT_ARGS[@]}" \
  --negative-prompt-json "${NEGATIVE_PROMPT_JSON}" \
  --num-frames "${NUM_FRAMES}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --fps "${FPS}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --flow-shift "${FLOW_SHIFT}" \
  --seed "${SEED}" \
  "${TEMPLATE_ARGS[@]}" \
  "${SAFETY_ARGS[@]}"

echo "[edge-i2v-ref] complete ${OUTPUT_DIR}"
