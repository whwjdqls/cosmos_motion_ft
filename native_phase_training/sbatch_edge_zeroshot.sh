#!/bin/bash
# Untouched Cosmos3-Edge baseline on one Nymeria sample in all four modes.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=edgezs
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgezs-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
EVAL_INPUT_DIR=${EVAL_INPUT_DIR:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/native_phase1_eval_inputs_viz1_edge_256_T97_s10_smoke_v1}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_zeroshot_base_20260825_256_T97_20fps_seed0}

for required in \
  "${EDGE_FRAMEWORK_ROOT}/cosmos_framework" \
  "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${EDGE_MODEL_ROOT}/transformer/diffusion_pytorch_model.safetensors.index.json" \
  "${EDGE_MODEL_ROOT}/vision_encoder/model.safetensors" \
  "${WAN_VAE_PATH}"; do
  [[ -e "${required}" ]] || { echo "[edge-zs] missing required artifact: ${required}" >&2; exit 1; }
done

for name in fd_input.jsonl invdyn_input.jsonl policy_input.jsonl i2v_input.jsonl; do
  [[ -s "${EVAL_INPUT_DIR}/${name}" ]] || { echo "[edge-zs] missing input: ${name}" >&2; exit 1; }
  [[ "$(wc -l < "${EVAL_INPUT_DIR}/${name}")" -eq 1 ]] || {
    echo "[edge-zs] expected exactly one record in ${name}" >&2
    exit 1
  }
done

if [[ -s "${EVAL_OUTPUT_DIR}/COMPLETE.json" ]]; then
  echo "[edge-zs] already complete: ${EVAL_OUTPUT_DIR}"
  exit 0
fi
if [[ -d "${EVAL_OUTPUT_DIR}" ]] && find "${EVAL_OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "[edge-zs] refusing to mix with incomplete output: ${EVAL_OUTPUT_DIR}" >&2
  exit 1
fi

CUDA_PYTHON_LIB_ROOT=${COSMOS_ENV_ROOT}/lib/python3.13/site-packages/nvidia
for cuda_lib_dir in npp cuda_nvrtc cuda_runtime nvjitlink; do
  [[ -d "${CUDA_PYTHON_LIB_ROOT}/${cuda_lib_dir}/lib" ]] || {
    echo "[edge-zs] missing CUDA library directory: ${cuda_lib_dir}" >&2
    exit 1
  }
done
export LD_LIBRARY_PATH=${CUDA_PYTHON_LIB_ROOT}/npp/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_nvrtc/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_runtime/lib:${CUDA_PYTHON_LIB_ROOT}/nvjitlink/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${EDGE_FRAMEWORK_ROOT}:${PYTHONPATH:-}

mkdir -p "${EVAL_OUTPUT_DIR}"
"${COSMOS_ENV_ROOT}/bin/python" -c \
  'from torchcodec.decoders import VideoDecoder; print("[edge-zs] TorchCodec libraries ready")'
"${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/validate_eval_inputs.py" \
  --input-dir "${EVAL_INPUT_DIR}" \
  --expected-shift 10 --expected-resolution 256 --expected-num-frames 97 --expected-fps 20 \
  fd_input.jsonl invdyn_input.jsonl policy_input.jsonl i2v_input.jsonl

SANITIZED_INPUT_DIR=${EVAL_OUTPUT_DIR}/inference_inputs
"${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/sanitize_prefix_inference_inputs.py" \
  --input-dir "${EVAL_INPUT_DIR}" --output-dir "${SANITIZED_INPUT_DIR}" \
  --model-family edge --replace-standalone-c --standalone-c-subject camera_wearer

echo "[edge-zs] node=$(hostname) checkpoint=${EDGE_MODEL_ROOT}"
echo "[edge-zs] untouched base weights; 256 tier, T97, 20 FPS, seed 0"
echo "[edge-zs] action: UniPC shift10/30 steps/guidance1; I2V: shift10/35/guidance6"
echo "[edge-zs] caption subject: action modes C -> the person; I2V C -> the camera wearer"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

cd "${EDGE_FRAMEWORK_ROOT}"
"${COSMOS_ENV_ROOT}/bin/torchrun" --standalone --nproc_per_node=1 \
  -m cosmos_framework.scripts.inference \
  --checkpoint-path "${EDGE_MODEL_ROOT}" \
  --experiment-overrides \
    "model.config.tokenizer.bucket_name=\"\"" \
    "model.config.tokenizer.vae_path=${WAN_VAE_PATH}" \
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
  --out "${EVAL_OUTPUT_DIR}/metrics" --prefix-lengths 1 --expected-sources 1

"${COSMOS_ENV_ROOT}/bin/python" - \
  "${EVAL_OUTPUT_DIR}" "${EDGE_MODEL_ROOT}" "${EDGE_FRAMEWORK_ROOT}" "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
framework = Path(sys.argv[3])
job_id = sys.argv[4]
required = (root / "viz" / "manifest.json", root / "metrics" / "METRICS_COMPLETE.json")
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"incomplete Edge zero-shot evaluation: {missing}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

framework_commit = subprocess.check_output(
    ["git", "-C", str(framework), "rev-parse", "HEAD"], text=True
).strip()
record = {
    "status": "complete",
    "kind": "cosmos3_edge_zeroshot",
    "slurm_job_id": job_id,
    "checkpoint_path": str(checkpoint.resolve()),
    "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
    "checkpoint_transformer_index_sha256": sha256(
        checkpoint / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    ),
    "framework_path": str(framework.resolve()),
    "framework_commit": framework_commit,
    "weights_modified": False,
    "fps": 20,
    "resolution": "256",
    "num_frames": 97,
    "seed": 0,
    "caption_replacement": {
        "action_modes": "(?<!\\w)C(?!\\w) -> the person (sentence-aware capitalization)",
        "image2video": "(?<!\\w)C(?!\\w) -> the camera wearer (sentence-aware capitalization)",
    },
    "modes": {
        "forward_dynamics": {"shift": 10.0, "num_steps": 30, "guidance": 1.0},
        "inverse_dynamics": {"shift": 10.0, "num_steps": 30, "guidance": 1.0},
        "policy": {"runtime_mode": "wam", "shift": 10.0, "num_steps": 30, "guidance": 1.0},
        "image2video": {"shift": 10.0, "num_steps": 35, "guidance": 6.0},
    },
}
(root / "COMPLETE.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

echo "[edge-zs] complete ${EVAL_OUTPUT_DIR}"
