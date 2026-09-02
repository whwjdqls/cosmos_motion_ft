#!/bin/bash
# Frozen first-20 Edge qualitative suite: 4 native tasks + matched Diffusers I2V.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=edgequal20
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgequal20-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
DIFFUSERS_ROOT=${DIFFUSERS_ROOT:-/mnt/projects/ll/jungbinc/cosmos3_edge_diffusers_diag/diffusers}
DIFFUSERS_VENV=${DIFFUSERS_VENV:-/mnt/projects/ll/jungbinc/cosmos3_edge_diffusers_diag/venv}
COHORT=${COHORT:-${REPO_ROOT}/native_phase_training/edge_qualitative20_cohort_v1.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_zeroshot_qualitative20_v1_20260901_256_T97_20fps_seed0}

PREPARED_ROOT=${OUTPUT_ROOT}/prepared
NATIVE_ROOT=${OUTPUT_ROOT}/native
DIFFUSERS_OUTPUT_ROOT=${OUTPUT_ROOT}/diffusers_i2v
VIZ_ROOT=${OUTPUT_ROOT}/viz
VALIDATION_PATH=${OUTPUT_ROOT}/validation.json

for required in \
  "${EDGE_FRAMEWORK_ROOT}/cosmos_framework" \
  "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${EDGE_MODEL_ROOT}/transformer/diffusion_pytorch_model.safetensors.index.json" \
  "${WAN_VAE_PATH}" \
  "${DIFFUSERS_ROOT}/src/diffusers" \
  "${DIFFUSERS_VENV}/bin/python" \
  "${COHORT}"; do
  [[ -e "${required}" ]] || { echo "[edge-qual20] missing: ${required}" >&2; exit 1; }
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NVTE_PROJECT_BUILDING=1

mkdir -p "${OUTPUT_ROOT}"
PYTHONPATH=${REPO_ROOT} "${COSMOS_ENV_ROOT}/bin/python" \
  -m native_phase_training.prepare_edge_qualitative20 \
  --cohort "${COHORT}" --output-dir "${PREPARED_ROOT}"

echo "[edge-qual20] node=$(hostname) output=${OUTPUT_ROOT}"
echo "[edge-qual20] frozen cohort=first 20/71; T97, 256x256, 20 FPS, seed0"
echo "[edge-qual20] native FD/ID/WAM=shift10/30/g1; both I2V=shift12/20/g6"
echo "[edge-qual20] I2V uses one shared JSONL, native-normalized prompt text, and native UniPC scheduler"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

CUDA_PYTHON_LIB_ROOT=${COSMOS_ENV_ROOT}/lib/python3.13/site-packages/nvidia
export LD_LIBRARY_PATH=${CUDA_PYTHON_LIB_ROOT}/cudnn/lib:${CUDA_PYTHON_LIB_ROOT}/cublas/lib:${CUDA_PYTHON_LIB_ROOT}/curand/lib:${CUDA_PYTHON_LIB_ROOT}/npp/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_nvrtc/lib:${CUDA_PYTHON_LIB_ROOT}/cuda_runtime/lib:${CUDA_PYTHON_LIB_ROOT}/nvjitlink/lib

if [[ ! -s "${NATIVE_ROOT}/NATIVE_COMPLETE.json" ]]; then
  mkdir -p "${NATIVE_ROOT}"
  cd "${EDGE_FRAMEWORK_ROOT}"
  PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${EDGE_FRAMEWORK_ROOT} \
    "${COSMOS_ENV_ROOT}/bin/torchrun" --standalone --nproc_per_node=1 \
    -m cosmos_framework.scripts.inference \
    --checkpoint-path "${EDGE_MODEL_ROOT}" \
    --experiment-overrides \
      "model.config.tokenizer.bucket_name=\"\"" \
      "model.config.tokenizer.vae_path=${WAN_VAE_PATH}" \
    --sampler unipc --use-ema-weights --parallelism-preset latency \
    --dp-shard-size 1 --dp-replicate-size 1 --cp-size 1 --cfgp-size 1 \
    --no-use-torch-compile --no-use-cuda-graphs --no-guardrails \
    --no-diffusion-cache \
    -o "${NATIVE_ROOT}" \
    -i "${PREPARED_ROOT}/inference_inputs/fd_input.jsonl" \
       "${PREPARED_ROOT}/inference_inputs/invdyn_input.jsonl" \
       "${PREPARED_ROOT}/inference_inputs/policy_input.jsonl" \
       "${PREPARED_ROOT}/inference_inputs/i2v_input.jsonl"
  cd "${REPO_ROOT}"
  PYTHONPATH=${REPO_ROOT} "${COSMOS_ENV_ROOT}/bin/python" - \
    "${PREPARED_ROOT}" "${NATIVE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

prepared, root = map(Path, sys.argv[1:])
names = []
for filename in ("fd_input.jsonl", "invdyn_input.jsonl", "policy_input.jsonl", "i2v_input.jsonl"):
    names.extend(json.loads(line)["name"] for line in (prepared / "inference_inputs" / filename).read_text().splitlines() if line.strip())
if len(names) != 80 or len(set(names)) != 80:
    raise SystemExit("native input names are not exactly 80 unique records")
for name in names:
    path = root / name / "sample_outputs.json"
    if not path.is_file() or json.loads(path.read_text()).get("status") != "success":
        raise SystemExit(f"missing successful native output: {path}")
(root / "NATIVE_COMPLETE.json").write_text(json.dumps({"status": "complete", "count": 80, "samples": names}, indent=2) + "\n")
PY
else
  echo "[edge-qual20] native outputs already complete"
fi

cd "${REPO_ROOT}"
PYTHONPATH=${REPO_ROOT} "${COSMOS_ENV_ROOT}/bin/python" \
  native_phase_training/save_inference_prompts.py --inference-root "${NATIVE_ROOT}"

PYTHONPATH=${DIFFUSERS_ROOT}/src:${EDGE_FRAMEWORK_ROOT}:${REPO_ROOT} \
  "${DIFFUSERS_VENV}/bin/python" native_phase_training/run_edge_i2v_batch.py \
  --model "${EDGE_MODEL_ROOT}" \
  --input "${PREPARED_ROOT}/inference_inputs/i2v_input.jsonl" \
  --output-root "${DIFFUSERS_OUTPUT_ROOT}" \
  --diffusers-source "${DIFFUSERS_ROOT}" \
  --native-framework "${EDGE_FRAMEWORK_ROOT}" \
  --disable-safety-checker

PYTHONPATH=${REPO_ROOT} "${COSMOS_ENV_ROOT}/bin/python" \
  native_phase_training/validate_edge_qualitative20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --native-root "${NATIVE_ROOT}" \
  --diffusers-root "${DIFFUSERS_OUTPUT_ROOT}" \
  --out "${VALIDATION_PATH}"

PYTHONPATH=${REPO_ROOT} "${COSMOS_ENV_ROOT}/bin/python" \
  native_phase_training/visualize_checkpoint.py \
  --inference-root "${NATIVE_ROOT}" \
  --eval-root "${PREPARED_ROOT}/canonical_inputs" \
  --out "${VIZ_ROOT}/native_mode_pairs"
PYTHONPATH=${REPO_ROOT} "${COSMOS_ENV_ROOT}/bin/python" \
  native_phase_training/visualize_edge_qualitative20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --native-root "${NATIVE_ROOT}" \
  --diffusers-root "${DIFFUSERS_OUTPUT_ROOT}" \
  --out "${VIZ_ROOT}/five_way"

PYTHONPATH=${REPO_ROOT} "${COSMOS_ENV_ROOT}/bin/python" - \
  "${OUTPUT_ROOT}" "${EDGE_MODEL_ROOT}" "${EDGE_FRAMEWORK_ROOT}" "${DIFFUSERS_ROOT}" "${SLURM_JOB_ID:-interactive}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root, model, framework, diffusers = map(Path, sys.argv[1:5])
job_id = sys.argv[5]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

required = (
    root / "prepared" / "cohort_contract.json",
    root / "native" / "NATIVE_COMPLETE.json",
    root / "diffusers_i2v" / "I2V_DIFFUSERS_COMPLETE.json",
    root / "validation.json",
    root / "viz" / "native_mode_pairs" / "manifest.json",
    root / "viz" / "five_way" / "manifest.json",
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"qualitative suite is incomplete: {missing}")
record = {
    "status": "complete",
    "kind": "cosmos3_edge_frozen_qualitative20_v1",
    "slurm_job_id": job_id,
    "checkpoint": str(model.resolve()),
    "checkpoint_config_sha256": sha256(model / "config.json"),
    "native_framework": str(framework.resolve()),
    "native_framework_commit": subprocess.check_output(["git", "-C", str(framework), "rev-parse", "HEAD"], text=True).strip(),
    "diffusers_source": str(diffusers.resolve()),
    "diffusers_commit": subprocess.check_output(["git", "-C", str(diffusers), "rev-parse", "HEAD"], text=True).strip(),
    "cohort_contract_sha256": sha256(root / "prepared" / "cohort_contract.json"),
    "validation_sha256": sha256(root / "validation.json"),
    "counts": {"cohort": 20, "native": 80, "diffusers_i2v": 20, "five_way_grids": 20},
    "media": {"num_frames": 97, "fps": 20, "resolution": [256, 256], "seed": 0},
    "samplers": {
        "forward_dynamics": {"shift": 10.0, "num_steps": 30, "guidance": 1.0},
        "inverse_dynamics": {"shift": 10.0, "num_steps": 30, "guidance": 1.0},
        "policy": {"runtime_mode": "wam", "shift": 10.0, "num_steps": 30, "guidance": 1.0},
        "native_image2video": {"shift": 12.0, "num_steps": 20, "guidance": 6.0},
        "diffusers_image2video": {"shift": 12.0, "num_steps": 20, "guidance": 6.0},
    },
    "i2v_matched_contract": {
        "shared_input_jsonl": True,
        "positive_prompt_exact_text": True,
        "negative_prompt_exact_text": True,
        "conditioning_image_path_and_hash": True,
        "native_prompt_upsampling": False,
        "scheduler": "native FlowUniPCMultistepScheduler",
        "backend_implementation_is_the_only_intended_variable": True,
    },
}
(root / "COMPLETE.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

echo "[edge-qual20] complete ${OUTPUT_ROOT}"
