#!/bin/bash
# Compact four-mode official EMA/UniPC evaluation for Cosmos3-Edge checkpoints.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=edgep1eval
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgep1eval-%j.out

set -euo pipefail

: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
: "${EVAL_INPUT_DIR:?EVAL_INPUT_DIR is required}"
: "${EVAL_OUTPUT_DIR:?EVAL_OUTPUT_DIR is required}"

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
export EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
CUDA_PYTHON_LIB_ROOT=${COSMOS_ENV_ROOT}/lib/python3.13/site-packages/nvidia
for cuda_lib_dir in npp cuda_nvrtc cuda_runtime nvjitlink; do
  [[ -d "${CUDA_PYTHON_LIB_ROOT}/${cuda_lib_dir}/lib" ]] || {
    echo "missing TorchCodec CUDA library directory: ${CUDA_PYTHON_LIB_ROOT}/${cuda_lib_dir}/lib" >&2
    exit 1
  }
done
export LD_LIBRARY_PATH=${CUDA_PYTHON_LIB_ROOT}/npp/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_nvrtc/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_runtime/lib:${CUDA_PYTHON_LIB_ROOT}/nvjitlink/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${EDGE_FRAMEWORK_ROOT}:${PYTHONPATH:-}

"${COSMOS_ENV_ROOT}/bin/python" -c \
  'from torchcodec.decoders import VideoDecoder; print("[edge-eval] TorchCodec libraries ready")'

mkdir -p "${EVAL_OUTPUT_DIR}"
"${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/run_contract.py" \
  --checkpoint-path "${CHECKPOINT_PATH}" --output-dir "${EVAL_OUTPUT_DIR}"
source "${EVAL_OUTPUT_DIR}/resolved_run_contract.env"
[[ "${NATIVEP1_MODEL_FAMILY}" == "edge" ]] || { echo "not an Edge checkpoint" >&2; exit 1; }

"${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/validate_eval_inputs.py" \
  --input-dir "${EVAL_INPUT_DIR}" \
  --expected-shift "${NATIVEP1_EFFECTIVE_SHIFT}" \
  --expected-resolution "${NYMERIA_RESOLUTION}" \
  --expected-num-frames "${NYMERIA_NUM_FRAMES}" \
  fd_input.jsonl invdyn_input.jsonl policy_input.jsonl i2v_input.jsonl

SANITIZED_INPUT_DIR=${EVAL_OUTPUT_DIR}/inference_inputs
"${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/sanitize_prefix_inference_inputs.py" \
  --input-dir "${EVAL_INPUT_DIR}" --output-dir "${SANITIZED_INPUT_DIR}" \
  --model-family edge --replace-standalone-c --standalone-c-subject camera_wearer

cd "${EDGE_FRAMEWORK_ROOT}"
"${COSMOS_ENV_ROOT}/bin/torchrun" --standalone --nproc_per_node=1 \
  -m cosmos_framework.scripts.inference \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-file native_phase_training/inference_config.py \
  --experiment world_camera_nymeria_latent_edge \
  --sampler unipc --use-ema-weights --parallelism-preset latency \
  --dp-shard-size 1 --dp-replicate-size 1 --cp-size 1 --cfgp-size 1 \
  --no-use-torch-compile --no-use-cuda-graphs --no-guardrails \
  -o "${EVAL_OUTPUT_DIR}" \
  -i "${SANITIZED_INPUT_DIR}/fd_input.jsonl" \
     "${SANITIZED_INPUT_DIR}/invdyn_input.jsonl" \
     "${SANITIZED_INPUT_DIR}/policy_input.jsonl" \
     "${SANITIZED_INPUT_DIR}/i2v_input.jsonl"

cd "${REPO_ROOT}"
"${COSMOS_ENV_ROOT}/bin/python" native_phase_training/save_inference_prompts.py \
  --inference-root "${EVAL_OUTPUT_DIR}"
"${COSMOS_ENV_ROOT}/bin/python" native_phase_training/visualize_checkpoint.py \
  --inference-root "${EVAL_OUTPUT_DIR}" --eval-root "${EVAL_INPUT_DIR}"
"${COSMOS_ENV_ROOT}/bin/python" native_phase_training/evaluate_prefix_suite.py \
  --inference-root "${EVAL_OUTPUT_DIR}" --eval-root "${EVAL_INPUT_DIR}" \
  --out "${EVAL_OUTPUT_DIR}/metrics" --prefix-lengths "${NATIVEP1_EVAL_PREFIX_LENGTHS:-1}" \
  --expected-sources "${NATIVEP1_VIZ_N:-5}"

"${COSMOS_ENV_ROOT}/bin/python" - "${EVAL_OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = (root / "resolved_run_contract.json", root / "viz" / "manifest.json", root / "metrics" / "METRICS_COMPLETE.json")
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"incomplete Edge checkpoint evaluation: {missing}")
(root / "COMPLETE.json").write_text(json.dumps({"status": "complete", "model_family": "edge"}, indent=2) + "\n")
PY

echo "[edge-eval] complete ${EVAL_OUTPUT_DIR}"
