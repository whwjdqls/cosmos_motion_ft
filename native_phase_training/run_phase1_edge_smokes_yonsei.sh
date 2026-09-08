#!/usr/bin/env bash
# Cosmos3-Edge Phase 1 save/resume and four-GPU smoke gates on Yonsei Pro 6000.

set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/opt/ohpc/pub/apps/cuda/12.8}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export EDGE_DCP_ROOT

REPO_ROOT=/home/whwjdqls99/cosmos_motion_ft
STORAGE_ROOT=/lustre/whwjdqls99/cosmos
WEKA_ROOT=${STORAGE_ROOT}/weka
COSMOS_ENV_ROOT=${STORAGE_ROOT}/envs/cosmos
EDGE_FRAMEWORK_ROOT=/home/whwjdqls99/cosmos-framework-edge
LEGACY_FRAMEWORK_ROOT=/home/whwjdqls99/cosmos-framework
EDGE_MODEL_ROOT=${WEKA_ROOT}/Cosmos3-Edge
EDGE_DCP_ROOT=${WEKA_ROOT}/cosmos3_edge_dcp
WAN_VAE_PATH=${WEKA_ROOT}/wan22_vae/Wan2.2_VAE.pth
FULL_LATENT_ROOT=${WEKA_ROOT}/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256
SMOKE_LATENT_ROOT=${WEKA_ROOT}/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256_smoke_yonsei_4090
IMAGINAIRE_OUTPUT_ROOT=${WEKA_ROOT}/cosmos_motion_ft_runs
LOG=${STORAGE_ROOT}/logs/phase1_edge_pro6000_smoke.log

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

for required in \
  "${COSMOS_ENV_ROOT}/bin/python" \
  "${COSMOS_ENV_ROOT}/bin/torchrun" \
  "${EDGE_FRAMEWORK_ROOT}/cosmos_framework" \
  "${LEGACY_FRAMEWORK_ROOT}/cosmos_framework" \
  "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${WAN_VAE_PATH}" \
  "${FULL_LATENT_ROOT}/latent_cache_complete.json" \
  "${SMOKE_LATENT_ROOT}/smoke_validation_complete.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[edge-p1-ysdc] ERROR: missing required artifact: ${required}" >&2
    exit 1
  fi
done

export COSMOS_SITE=yonsei
export REPO_ROOT COSMOS_ENV_ROOT WEKA_ROOT WAN_VAE_PATH IMAGINAIRE_OUTPUT_ROOT
export COSMOS_FRAMEWORK_EDGE_ROOT=${EDGE_FRAMEWORK_ROOT}
export COSMOS_FRAMEWORK_ROOT=${LEGACY_FRAMEWORK_ROOT}
export EDGE_MODEL_ROOT
export BASE_CHECKPOINT_PATH=${EDGE_DCP_ROOT}
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NATIVEP1_WANDB_MODE=disabled
export WANDB_PROJECT=cosmos3_camera

echo "[$(date --iso-8601=seconds)] validating four-GPU Pro 6000 allocation"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
"${COSMOS_ENV_ROOT}/bin/python" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 4, f"expected 4 GPUs, found {torch.cuda.device_count()}"
for index in range(4):
    value = torch.ones(1, device=f"cuda:{index}").item()
    print(f"CUDA_GATE_OK index={index} name={torch.cuda.get_device_name(index)} value={value}")
PY

visible_devices=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
first_device=${visible_devices%%,*}

if [[ ! -s "${EDGE_DCP_ROOT}/model/.metadata" ]]; then
  echo "[edge-p1-ysdc] converting the pinned Edge snapshot to DCP on GPU ${first_device}"
  CUDA_VISIBLE_DEVICES=${first_device} \
    bash "${REPO_ROOT}/native_phase_training/sbatch_convert_edge_to_dcp.sh"
fi

single_run_name=edge_phase1_yonsei_pro6000_bs8_save_resume_smoke_${SLURM_JOB_ID}
echo "[edge-p1-ysdc] running single-GPU batch-8 save/resume gate: ${single_run_name}"
CUDA_VISIBLE_DEVICES=${first_device} \
NYMERIA_LATENT_ROOT=${SMOKE_LATENT_ROOT} \
NATIVEP1_SMOKE_CLIPS_PER_GPU=8 \
NATIVEP1_RUN_NAME=${single_run_name} \
NATIVEP1_WANDB_MODE=disabled \
  bash "${REPO_ROOT}/native_phase_training/sbatch_phase1_edge_smoke.sh"

distributed_run_name=edge_phase1_yonsei_pro6000_4gpu_bs8_smoke_${SLURM_JOB_ID}
echo "[edge-p1-ysdc] running four-GPU batch-8 distributed gate: ${distributed_run_name}"
export CUDA_VISIBLE_DEVICES=${visible_devices}
export NYMERIA_LATENT_ROOT=${FULL_LATENT_ROOT}
export NATIVEP1_WORLD_SIZE=4
export NATIVEP1_CLIPS_PER_GPU=8
export NATIVEP1_SMOKE_CLIPS_PER_GPU=8
export NATIVEP1_SMOKE_RUN_NAME=${single_run_name}
export NATIVEP1_RUN_NAME=${distributed_run_name}
export NATIVEP1_MAX_ITER=3
export NATIVEP1_SAVE_ITER=2
export NATIVEP1_AUTO_EVAL=0
export NATIVEP1_AUTO_EVAL_FULL71=0
export NATIVEP1_WANDB_MODE=disabled
bash "${REPO_ROOT}/native_phase_training/sbatch_phase1_edge_8gpu.sh"

distributed_run_dir=${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world_edge/${distributed_run_name}
[[ -s "${distributed_run_dir}/checkpoints/iter_000000003/model/.metadata" ]] || {
  echo "[edge-p1-ysdc] ERROR: four-GPU smoke produced no step-3 DCP" >&2
  exit 1
}

"${COSMOS_ENV_ROOT}/bin/python" - "${distributed_run_dir}/FOUR_GPU_SMOKE_COMPLETE.json" "${SLURM_JOB_ID}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "slurm_job_id": sys.argv[2],
    "world_size": 4,
    "clips_per_gpu": 8,
    "effective_batch": 32,
    "completed_optimizer_steps": 3,
    "model_family": "cosmos3_edge",
    "num_frames": 97,
    "fps": 20,
}, indent=2, sort_keys=True) + "\n")
PY

echo "[$(date --iso-8601=seconds)] Cosmos3-Edge Phase 1 Pro 6000 smoke gates passed"
