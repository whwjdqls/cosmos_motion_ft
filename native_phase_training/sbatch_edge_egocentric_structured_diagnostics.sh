#!/bin/bash
# Frozen-JSON I2V sampler comparison plus GT-action forward controls.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=edgeegodiag
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgeegodiag-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
SOURCE_DIR=${SOURCE_DIR:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/native_phase1_eval_inputs_viz1_edge_256_T97_s10_smoke_v1}
STRUCTURED_PROMPT=${STRUCTURED_PROMPT:-${REPO_ROOT}/native_phase_training/prompts/edge_egocentric_s07_structured.json}
NEGATIVE_PROMPT=${NEGATIVE_PROMPT:-${EDGE_MODEL_ROOT}/assets/negative_prompt.json}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_egocentric_structured_20260901_256_T97_20fps_seed0}

for required in \
  "${EDGE_FRAMEWORK_ROOT}/cosmos_framework" \
  "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${EDGE_MODEL_ROOT}/transformer/diffusion_pytorch_model.safetensors.index.json" \
  "${WAN_VAE_PATH}" \
  "${SOURCE_DIR}/i2v_input.jsonl" \
  "${SOURCE_DIR}/fd_input.jsonl" \
  "${STRUCTURED_PROMPT}" \
  "${NEGATIVE_PROMPT}"; do
  [[ -e "${required}" ]] || { echo "[edge-ego] missing required artifact: ${required}" >&2; exit 1; }
done

if [[ -s "${EVAL_OUTPUT_DIR}/COMPLETE.json" ]]; then
  echo "[edge-ego] already complete: ${EVAL_OUTPUT_DIR}"
  exit 0
fi
if [[ -d "${EVAL_OUTPUT_DIR}" ]] && find "${EVAL_OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "[edge-ego] refusing to mix with incomplete output: ${EVAL_OUTPUT_DIR}" >&2
  exit 1
fi

CUDA_PYTHON_LIB_ROOT=${COSMOS_ENV_ROOT}/lib/python3.13/site-packages/nvidia
for cuda_lib_dir in npp cuda_nvrtc cuda_runtime nvjitlink; do
  [[ -d "${CUDA_PYTHON_LIB_ROOT}/${cuda_lib_dir}/lib" ]] || {
    echo "[edge-ego] missing CUDA library directory: ${cuda_lib_dir}" >&2
    exit 1
  }
done
export LD_LIBRARY_PATH=${CUDA_PYTHON_LIB_ROOT}/npp/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_nvrtc/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_runtime/lib:${CUDA_PYTHON_LIB_ROOT}/nvjitlink/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${EDGE_FRAMEWORK_ROOT}:${PYTHONPATH:-}

INPUT_DIR=${EVAL_OUTPUT_DIR}/inference_inputs
mkdir -p "${INPUT_DIR}"
"${COSMOS_ENV_ROOT}/bin/python" \
  "${REPO_ROOT}/native_phase_training/prepare_edge_egocentric_structured_diagnostics.py" \
  --source-i2v "${SOURCE_DIR}/i2v_input.jsonl" \
  --source-forward "${SOURCE_DIR}/fd_input.jsonl" \
  --structured-prompt "${STRUCTURED_PROMPT}" \
  --negative-prompt "${NEGATIVE_PROMPT}" \
  --output-dir "${INPUT_DIR}"

[[ "$(wc -l < "${INPUT_DIR}/i2v_structured.jsonl")" -eq 2 ]] || {
  echo "[edge-ego] expected two structured I2V records" >&2
  exit 1
}
[[ "$(wc -l < "${INPUT_DIR}/forward_controls.jsonl")" -eq 2 ]] || {
  echo "[edge-ego] expected two forward controls" >&2
  exit 1
}

echo "[edge-ego] node=$(hostname) checkpoint=${EDGE_MODEL_ROOT}"
echo "[edge-ego] fixed structured JSON across I2V shift10/35 and shift12/20"
echo "[edge-ego] GT-action forward controls: person versus camera wearer"
echo "[edge-ego] fixed: seed0, 256x256, T97, 20 FPS, cache disabled"
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
  --no-diffusion-cache --no-use-torch-compile --no-use-cuda-graphs --no-guardrails \
  -o "${EVAL_OUTPUT_DIR}" \
  -i "${INPUT_DIR}/i2v_structured.jsonl" "${INPUT_DIR}/forward_controls.jsonl"

cd "${REPO_ROOT}"
"${COSMOS_ENV_ROOT}/bin/python" native_phase_training/save_inference_prompts.py \
  --inference-root "${EVAL_OUTPUT_DIR}"
METRICS=${EVAL_OUTPUT_DIR}/metrics/egocentric_diagnostics.json
"${COSMOS_ENV_ROOT}/bin/python" \
  "${REPO_ROOT}/native_phase_training/analyze_edge_egocentric_diagnostics.py" \
  --inference-root "${EVAL_OUTPUT_DIR}" --output "${METRICS}"

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
manifest = json.loads((root / "inference_inputs" / "manifest.json").read_text())
required = [root / "metrics" / "egocentric_diagnostics.json"]
for variant in manifest["variants"]:
    sample_dir = root / variant["name"]
    required.extend(sample_dir / filename for filename in ("sample_args.json", "sample_outputs.json", "vision.mp4"))
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"incomplete Edge egocentric diagnostics: {missing}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

dirty = bool(subprocess.check_output(["git", "-C", str(framework), "status", "--porcelain"], text=True).strip())
record = {
    "status": "complete",
    "kind": manifest["kind"],
    "slurm_job_id": job_id,
    "checkpoint_path": str(checkpoint.resolve()),
    "framework_path": str(framework.resolve()),
    "framework_commit": subprocess.check_output(
        ["git", "-C", str(framework), "rev-parse", "HEAD"], text=True
    ).strip(),
    "framework_dirty": dirty,
    "weights_modified": False,
    "diffusion_cache": False,
    "input_manifest": str((root / "inference_inputs" / "manifest.json").resolve()),
    "input_manifest_sha256": sha256(root / "inference_inputs" / "manifest.json"),
    "metrics": str((root / "metrics" / "egocentric_diagnostics.json").resolve()),
    "metrics_sha256": sha256(root / "metrics" / "egocentric_diagnostics.json"),
}
(root / "COMPLETE.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

echo "[edge-ego] complete ${EVAL_OUTPUT_DIR}"
