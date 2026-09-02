#!/bin/bash
# Single-GPU Cosmos3-Edge LoRA save/resume gate.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --job-name=edgep1smoke
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgep1smoke-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
LEGACY_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
export EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
export BASE_CHECKPOINT_PATH=${BASE_CHECKPOINT_PATH:-/mnt/projects/ll/jungbinc/weka/cosmos3_edge_dcp}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs}
SMOKE_CLIPS_PER_GPU=${NATIVEP1_SMOKE_CLIPS_PER_GPU:-4}
if ! [[ "${SMOKE_CLIPS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[edge-smoke] ERROR: NATIVEP1_SMOKE_CLIPS_PER_GPU must be a positive integer" >&2
  exit 1
fi
export NYMERIA_LATENT_ROOT=${NYMERIA_LATENT_ROOT:-/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256_smoke${SMOKE_CLIPS_PER_GPU}}

for required in \
  "${EDGE_FRAMEWORK_ROOT}/cosmos_framework" \
  "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${BASE_CHECKPOINT_PATH}/model/.metadata" \
  "${WAN_VAE_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[edge-smoke] ERROR: required artifact is missing: ${required}" >&2
    exit 1
  fi
done

# Build a dedicated smoke cache on demand.  The Wan VAE is identical for
# Nano and Edge, so the audited legacy precompute path is the source of truth.
mkdir -p "${NYMERIA_LATENT_ROOT}"
cached_count=$(find "${NYMERIA_LATENT_ROOT}" -mindepth 2 -maxdepth 2 -name '*.npz' 2>/dev/null | wc -l)
if (( cached_count < SMOKE_CLIPS_PER_GPU )); then
  echo "[edge-smoke] preparing ${SMOKE_CLIPS_PER_GPU} T97/256/20-FPS cached clips"
  (
    cd "${LEGACY_FRAMEWORK_ROOT}"
    PYTHONPATH=${LEGACY_FRAMEWORK_ROOT}:${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${REPO_ROOT}/motion_expert_joint_attention \
      "${COSMOS_ENV_ROOT}/bin/python" \
      "${REPO_ROOT}/motion_expert_joint_attention/precompute_latents.py" \
      --split train --num_frames 97 --fps 20 --resolution 256 \
      --expected_image_hw 256 --expected_latent_hw 16 \
      --out_root "${NYMERIA_LATENT_ROOT}" --limit "${SMOKE_CLIPS_PER_GPU}" --fail_on_error
  )
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
export NYMERIA_STANDALONE_C_SUBJECT=${NYMERIA_STANDALONE_C_SUBJECT:-camera_wearer}
export NYMERIA_QUALITY_FILTER=
export NYMERIA_MAX_SAMPLES=${NYMERIA_MAX_SAMPLES:-${SMOKE_CLIPS_PER_GPU}}
export NATIVEP1_ADAPTATION_MODE=global_lora
export NATIVEP1_PREFIX_LENGTHS=1
export NATIVEP1_CLIPS_PER_GPU=${SMOKE_CLIPS_PER_GPU}
export NATIVEP1_EXPECTED_LATENT_HW=16
export NATIVEP1_EXPECTED_IMAGE_HW=256
export NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=0
export NATIVEP1_LORA_LR=5e-5
export NATIVEP1_ACTION_LR_MULT=4
export NATIVEP1_ACTION_LOSS_WEIGHT=10
export NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=0
export NATIVEP1_AUTO_EVAL=0
export NATIVEP1_AUTO_EVAL_FULL71=0
export NATIVEP1_WANDB_MODE=${NATIVEP1_WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-cosmos3_camera}

case "${NATIVEP1_WANDB_MODE}" in
  online|offline|disabled) ;;
  *)
    echo "[edge-smoke] ERROR: NATIVEP1_WANDB_MODE must be online, offline, or disabled" >&2
    exit 1
    ;;
esac

RUN_NAME=${NATIVEP1_RUN_NAME:-edge_phase1_T97_20fps_bs${SMOKE_CLIPS_PER_GPU}_camera_wearer_lora_smoke_gate1}
RUN_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world_edge/${RUN_NAME}
if [[ "${SMOKE_CLIPS_PER_GPU}" == "4" ]]; then
  MARKER=${RUN_DIR}/SINGLE_GPU_FOUR_CLIP_SMOKE_COMPLETE.json
else
  MARKER=${RUN_DIR}/SINGLE_GPU_${SMOKE_CLIPS_PER_GPU}_CLIP_SMOKE_COMPLETE.json
fi
TOML=${REPO_ROOT}/native_phase_training/world_camera_nymeria_latent_edge.toml
PYTHON=${COSMOS_ENV_ROOT}/bin/python
TORCHRUN=${COSMOS_ENV_ROOT}/bin/torchrun

echo "[edge-smoke] node=$(hostname) framework=${EDGE_FRAMEWORK_ROOT}"
echo "[edge-smoke] checkpoint=${BASE_CHECKPOINT_PATH} latents=${NYMERIA_LATENT_ROOT}"
echo "[edge-smoke] batch=${SMOKE_CLIPS_PER_GPU} tasks=40/25/20/15 fps=20 caption_subject='the camera wearer'"
echo "[edge-smoke] shifts: training=3 inference=10 wandb_mode=${NATIVEP1_WANDB_MODE}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

cd "${EDGE_FRAMEWORK_ROOT}"
run_train() {
  local max_iter=$1
  "${TORCHRUN}" --standalone --nproc_per_node=1 \
    "${REPO_ROOT}/native_phase_training/run_latent_train.py" \
    --sft-toml "${TOML}" \
    job.name="${RUN_NAME}" \
    job.wandb_mode="${NATIVEP1_WANDB_MODE}" \
    trainer.max_iter="${max_iter}" \
    trainer.grad_accum_iter=1 \
    trainer.logging_iter=1 \
    checkpoint.save_iter=2 \
    model.config.compile.enabled=false \
    model.config.ema.enabled=true \
    model.config.parallelism.data_parallel_shard_degree=1 \
    model.config.parallelism.data_parallel_replicate_degree=1 \
    model.config.parallelism.context_parallel_shard_degree=1 \
    model.config.parallelism.cfg_parallel_shard_degree=1
}

# Two optimizer updates and a DCP, then a fresh process must resume and make
# the third update.  A stale successful run is accepted only if its marker and
# post-resume checkpoint are both present.
if [[ ! -s "${MARKER}" ]]; then
  if [[ "${NATIVEP1_WANDB_MODE}" == "online" ]]; then
    WANDB_NETRC_FILE=${WANDB_NETRC_FILE:-/home/jungbinc/.netrc}
    if [[ -z "${WANDB_API_KEY:-}" ]] && \
       ! grep -Eq '^[[:space:]]*machine[[:space:]]+api\.wandb\.ai([[:space:]]|$)' \
         "${WANDB_NETRC_FILE}" 2>/dev/null; then
      echo "[edge-smoke] ERROR: W&B online logging is enabled but no credential was found." >&2
      echo "[edge-smoke] Set WANDB_API_KEY or run 'wandb login' before submission." >&2
      exit 1
    fi
  fi
  run_train 2
  expected_two=${RUN_DIR}/checkpoints/iter_000000002
  [[ -s "${expected_two}/model/.metadata" ]] || {
    echo "[edge-smoke] ERROR: step-2 DCP was not committed: ${expected_two}/model/.metadata" >&2
    exit 1
  }
  run_train 3
  expected_three=${RUN_DIR}/checkpoints/iter_000000003
  [[ -s "${expected_three}/model/.metadata" ]] || {
    echo "[edge-smoke] ERROR: resumed step-3 DCP was not committed: ${expected_three}/model/.metadata" >&2
    exit 1
  }

  "${PYTHON}" - "${MARKER}" "${SLURM_JOB_ID}" "${SMOKE_CLIPS_PER_GPU}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "complete",
    "slurm_job_id": sys.argv[2],
    "world_size": 1,
    "clips_per_gpu": int(sys.argv[3]),
    "effective_batch": int(sys.argv[3]),
    "completed_optimizer_steps": 3,
    "resume_verified_from_step": 2,
    "conditioning_fps": 20.0,
    "standalone_c_subject": "camera_wearer",
}, indent=2, sort_keys=True) + "\n")
PY
fi

echo "[edge-smoke] PASS marker=${MARKER}"
