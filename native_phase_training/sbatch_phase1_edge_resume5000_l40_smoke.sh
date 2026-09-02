#!/bin/bash
# One-L40 exact-state resume gate for the preempted four-rank Edge Phase-1 run.
# The source DCP is read-only; the resumed update is saved under a separate run.
#SBATCH --partition=liu-compute
#SBATCH --qos=ll-med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --exclude=ll-l40-1.grasp.maas
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --job-name=edgep1-r5k-smk
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgep1-r5k-smk-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
LEGACY_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
export EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
export NYMERIA_LATENT_ROOT=${NYMERIA_LATENT_ROOT:-/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs}

SOURCE_RUN_NAME=${NATIVEP1_SOURCE_RUN_NAME:-edge_phase1_T97_20fps_bs32_4gpu_bs8_camera_wearer_global_lora_100k_v1}
SOURCE_ITERATION=${NATIVEP1_SOURCE_ITERATION:-5000}
if ! [[ "${SOURCE_ITERATION}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[edge-resume-smoke] ERROR: NATIVEP1_SOURCE_ITERATION must be a positive integer" >&2
  exit 1
fi
SOURCE_ITER_NAME=$(printf 'iter_%09d' "${SOURCE_ITERATION}")
SOURCE_RUN_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world_edge/${SOURCE_RUN_NAME}
SOURCE_CHECKPOINT=${SOURCE_RUN_DIR}/checkpoints/${SOURCE_ITER_NAME}

SMOKE_RUN_NAME=${NATIVEP1_RESUME_SMOKE_RUN_NAME:-edge_phase1_T97_20fps_bs8_camera_wearer_resume5000_l40_smoke_v1}
SMOKE_RUN_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world_edge/${SMOKE_RUN_NAME}
TARGET_ITERATION=$((SOURCE_ITERATION + 1))
TARGET_ITER_NAME=$(printf 'iter_%09d' "${TARGET_ITERATION}")
TARGET_CHECKPOINT=${SMOKE_RUN_DIR}/checkpoints/${TARGET_ITER_NAME}
MARKER=${SMOKE_RUN_DIR}/L40_RESUME_SMOKE_COMPLETE.json

for component in model optim scheduler trainer; do
  metadata=${SOURCE_CHECKPOINT}/${component}/.metadata
  if [[ ! -s "${metadata}" ]]; then
    echo "[edge-resume-smoke] ERROR: incomplete source checkpoint component: ${metadata}" >&2
    exit 1
  fi
done
for required in \
  "${EDGE_FRAMEWORK_ROOT}/cosmos_framework" \
  "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${WAN_VAE_PATH}" \
  "${NYMERIA_LATENT_ROOT}/latent_cache_contract.json" \
  "${NYMERIA_LATENT_ROOT}/latent_cache_complete.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[edge-resume-smoke] ERROR: required artifact is missing: ${required}" >&2
    exit 1
  fi
done

if [[ -s "${MARKER}" ]]; then
  for component in model optim scheduler trainer; do
    [[ -s "${TARGET_CHECKPOINT}/${component}/.metadata" ]] || {
      echo "[edge-resume-smoke] ERROR: marker exists but ${component} checkpoint metadata is missing" >&2
      exit 1
    }
  done
  echo "[edge-resume-smoke] PASS (existing) marker=${MARKER}"
  exit 0
fi
if [[ -e "${SMOKE_RUN_DIR}/checkpoints/latest_checkpoint.txt" ]]; then
  echo "[edge-resume-smoke] ERROR: incomplete prior smoke state exists without ${MARKER}" >&2
  echo "[edge-resume-smoke] Inspect ${SMOKE_RUN_DIR}; it will not be overwritten automatically." >&2
  exit 1
fi

export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${EDGE_FRAMEWORK_ROOT}:${PYTHONPATH:-}

export NATIVEP1_MODEL_FAMILY=edge
export NYMERIA_NUM_FRAMES=97
export NYMERIA_RESOLUTION=256
export NYMERIA_MODE=mixture
export NYMERIA_DROP_MODES=
export NYMERIA_REPLACE_STANDALONE_C=1
export NYMERIA_STANDALONE_C_SUBJECT=camera_wearer
export NYMERIA_QUALITY_FILTER=
export NATIVEP1_ADAPTATION_MODE=global_lora
export NATIVEP1_PREFIX_LENGTHS=1
export NATIVEP1_CLIPS_PER_GPU=8
export NATIVEP1_EXPECTED_LATENT_HW=16
export NATIVEP1_EXPECTED_IMAGE_HW=256
export NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=1
export NATIVEP1_LORA_LR=5e-5
export NATIVEP1_ACTION_LR_MULT=4
export NATIVEP1_ACTION_LOSS_WEIGHT=10
export NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=0
export NATIVEP1_AUTO_EVAL=0
export NATIVEP1_AUTO_EVAL_FULL71=0
export NATIVEP1_WANDB_MODE=disabled

echo "[edge-resume-smoke] node=$(hostname) source=${SOURCE_CHECKPOINT}"
echo "[edge-resume-smoke] target=${TARGET_CHECKPOINT} batch=8 world_size=1"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
nvidia-smi -q -d ECC || true

env PYTHONPATH=${LEGACY_FRAMEWORK_ROOT}:${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${REPO_ROOT}/motion_expert_joint_attention \
  "${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/validate_latent_cache.py" \
  --root "${NYMERIA_LATENT_ROOT}" --sample-count 8

cd "${EDGE_FRAMEWORK_ROOT}"
"${COSMOS_ENV_ROOT}/bin/torchrun" --standalone --nproc_per_node=1 \
  "${REPO_ROOT}/native_phase_training/run_latent_train.py" \
  --sft-toml "${REPO_ROOT}/native_phase_training/world_camera_nymeria_latent_edge.toml" \
  job.name="${SMOKE_RUN_NAME}" \
  job.wandb_mode=disabled \
  trainer.max_iter="${TARGET_ITERATION}" \
  trainer.grad_accum_iter=1 \
  trainer.logging_iter=1 \
  checkpoint.load_path="${SOURCE_CHECKPOINT}" \
  checkpoint.load_training_state=true \
  checkpoint.keys_to_skip_loading=[] \
  checkpoint.save_iter="${TARGET_ITERATION}" \
  model.config.compile.enabled=false \
  model.config.ema.enabled=true \
  model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.parallelism.data_parallel_replicate_degree=1 \
  model.config.parallelism.context_parallel_shard_degree=1 \
  model.config.parallelism.cfg_parallel_shard_degree=1

for component in model optim scheduler trainer; do
  [[ -s "${TARGET_CHECKPOINT}/${component}/.metadata" ]] || {
    echo "[edge-resume-smoke] ERROR: resumed ${component} metadata was not committed" >&2
    exit 1
  }
done

"${COSMOS_ENV_ROOT}/bin/python" - "${MARKER}" "${SLURM_JOB_ID}" "${SLURMD_NODENAME:-$(hostname)}" \
  "${SOURCE_CHECKPOINT}" "${TARGET_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

marker = Path(sys.argv[1])
marker.write_text(json.dumps({
    "status": "complete",
    "slurm_job_id": sys.argv[2],
    "node": sys.argv[3],
    "gpu_type": "l40",
    "world_size": 1,
    "clips_per_gpu": 8,
    "source_checkpoint": sys.argv[4],
    "resumed_checkpoint": sys.argv[5],
    "loaded_state": ["model", "optim", "scheduler", "trainer"],
    "ema_loaded": True,
}, indent=2, sort_keys=True) + "\n")
PY

echo "[edge-resume-smoke] PASS marker=${MARKER}"
