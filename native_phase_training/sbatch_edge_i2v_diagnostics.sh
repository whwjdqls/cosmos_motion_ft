#!/bin/bash
# Controlled single-GPU Cosmos3-Edge I2V ablations on one fixed Nymeria sample.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=edgei2vdiag
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgei2vdiag-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
SOURCE_INPUT=${SOURCE_INPUT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/native_phase1_eval_inputs_viz1_edge_256_T97_s10_smoke_v1/i2v_input.jsonl}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_i2v_diagnostics_20260901_256_T97_20fps_seed0}
REFRESHED_NEGATIVE=${REFRESHED_NEGATIVE:-${EDGE_MODEL_ROOT}/assets/negative_prompt.json}
DIAGNOSTIC_VARIANTS=${DIAGNOSTIC_VARIANTS:-current sampler refreshed guidance1}
INCLUDE_VISUAL=${INCLUDE_VISUAL:-0}
export INCLUDE_VISUAL
read -r -a VARIANT_KEYS <<< "${DIAGNOSTIC_VARIANTS}"
[[ "${#VARIANT_KEYS[@]}" -gt 0 ]] || { echo "[edge-i2v-diag] no variants requested" >&2; exit 1; }

for required in \
  "${EDGE_FRAMEWORK_ROOT}/cosmos_framework" \
  "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${EDGE_MODEL_ROOT}/transformer/diffusion_pytorch_model.safetensors.index.json" \
  "${EDGE_MODEL_ROOT}/vision_encoder/model.safetensors" \
  "${REFRESHED_NEGATIVE}" \
  "${SOURCE_INPUT}" \
  "${WAN_VAE_PATH}"; do
  [[ -e "${required}" ]] || { echo "[edge-i2v-diag] missing required artifact: ${required}" >&2; exit 1; }
done

if [[ -s "${EVAL_OUTPUT_DIR}/COMPLETE.json" ]]; then
  echo "[edge-i2v-diag] already complete: ${EVAL_OUTPUT_DIR}"
  exit 0
fi
if [[ -d "${EVAL_OUTPUT_DIR}" ]] && find "${EVAL_OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "[edge-i2v-diag] refusing to mix with incomplete output: ${EVAL_OUTPUT_DIR}" >&2
  exit 1
fi

CUDA_PYTHON_LIB_ROOT=${COSMOS_ENV_ROOT}/lib/python3.13/site-packages/nvidia
for cuda_lib_dir in npp cuda_nvrtc cuda_runtime nvjitlink; do
  [[ -d "${CUDA_PYTHON_LIB_ROOT}/${cuda_lib_dir}/lib" ]] || {
    echo "[edge-i2v-diag] missing CUDA library directory: ${cuda_lib_dir}" >&2
    exit 1
  }
done
export LD_LIBRARY_PATH=${CUDA_PYTHON_LIB_ROOT}/npp/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_nvrtc/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_runtime/lib:${CUDA_PYTHON_LIB_ROOT}/nvjitlink/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${EDGE_FRAMEWORK_ROOT}:${PYTHONPATH:-}

mkdir -p "${EVAL_OUTPUT_DIR}/inference_inputs"
INPUT_JSONL=${EVAL_OUTPUT_DIR}/inference_inputs/i2v_diagnostics.jsonl
INPUT_MANIFEST=${EVAL_OUTPUT_DIR}/inference_inputs/manifest.json
"${COSMOS_ENV_ROOT}/bin/python" \
  "${REPO_ROOT}/native_phase_training/prepare_edge_i2v_diagnostics.py" \
  --source-input "${SOURCE_INPUT}" \
  --negative-prompt "${REFRESHED_NEGATIVE}" \
  --output "${INPUT_JSONL}" \
  --manifest "${INPUT_MANIFEST}" \
  --variants "${VARIANT_KEYS[@]}"

[[ "$(wc -l < "${INPUT_JSONL}")" -eq "${#VARIANT_KEYS[@]}" ]] || {
  echo "[edge-i2v-diag] diagnostic record count mismatch" >&2
  exit 1
}

echo "[edge-i2v-diag] node=$(hostname) checkpoint=${EDGE_MODEL_ROOT}"
echo "[edge-i2v-diag] fixed: Nymeria frame, prompt, seed0, 256x256, T97, 20 FPS"
echo "[edge-i2v-diag] global diffusion cache: disabled"
echo "[edge-i2v-diag] variants: ${DIAGNOSTIC_VARIANTS}; include_visual=${INCLUDE_VISUAL}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

cd "${EDGE_FRAMEWORK_ROOT}"
EXPERIMENT_OVERRIDES=(
  "model.config.tokenizer.bucket_name=\"\""
  "model.config.tokenizer.vae_path=${WAN_VAE_PATH}"
)
if [[ "${INCLUDE_VISUAL}" == 1 ]]; then
  EXPERIMENT_OVERRIDES+=("model.config.vlm_config.model_instance.config.include_visual=true")
elif [[ "${INCLUDE_VISUAL}" != 0 ]]; then
  echo "[edge-i2v-diag] INCLUDE_VISUAL must be 0 or 1, got ${INCLUDE_VISUAL}" >&2
  exit 1
fi
"${COSMOS_ENV_ROOT}/bin/torchrun" --standalone --nproc_per_node=1 \
  -m cosmos_framework.scripts.inference \
  --checkpoint-path "${EDGE_MODEL_ROOT}" \
  --experiment-overrides "${EXPERIMENT_OVERRIDES[@]}" \
  --sampler unipc --use-ema-weights --parallelism-preset latency \
  --dp-shard-size 1 --dp-replicate-size 1 --cp-size 1 --cfgp-size 1 \
  --no-diffusion-cache --no-use-torch-compile --no-use-cuda-graphs --no-guardrails \
  -o "${EVAL_OUTPUT_DIR}" -i "${INPUT_JSONL}"

cd "${REPO_ROOT}"
"${COSMOS_ENV_ROOT}/bin/python" native_phase_training/save_inference_prompts.py \
  --inference-root "${EVAL_OUTPUT_DIR}"
"${COSMOS_ENV_ROOT}/bin/python" - \
  "${EVAL_OUTPUT_DIR}" "${EDGE_MODEL_ROOT}" "${EDGE_FRAMEWORK_ROOT}" "${SLURM_JOB_ID}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
framework = Path(sys.argv[3])
job_id = sys.argv[4]
manifest = json.loads((root / "inference_inputs" / "manifest.json").read_text())
missing = []
for variant in manifest["variants"]:
    sample_dir = root / variant["name"]
    for filename in ("sample_args.json", "sample_outputs.json", "vision.mp4"):
        path = sample_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
if missing:
    raise SystemExit(f"incomplete Edge I2V diagnostics: {missing}")

record = {
    "status": "complete",
    "kind": "cosmos3_edge_i2v_controlled_diagnostics",
    "slurm_job_id": job_id,
    "checkpoint_path": str(checkpoint.resolve()),
    "framework_path": str(framework.resolve()),
    "framework_commit": subprocess.check_output(
        ["git", "-C", str(framework), "rev-parse", "HEAD"], text=True
    ).strip(),
    "weights_modified": False,
    "diffusion_cache": False,
    "include_visual": bool(int(__import__("os").environ.get("INCLUDE_VISUAL", "0"))),
    "input_manifest": manifest,
}
(root / "COMPLETE.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

echo "[edge-i2v-diag] complete ${EVAL_OUTPUT_DIR}"
