#!/bin/bash
# Replicate-only Cosmos3-Edge Phase 1 production run (eight GPUs by default).
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:8
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --job-name=edgep1
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edgep1-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
LEGACY_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
export EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
export BASE_CHECKPOINT_PATH=${BASE_CHECKPOINT_PATH:-/mnt/projects/ll/jungbinc/weka/cosmos3_edge_dcp}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
export NYMERIA_LATENT_ROOT=${NYMERIA_LATENT_ROOT:-/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs}
export NATIVEP1_EVAL_INPUT_DIR=${NATIVEP1_EVAL_INPUT_DIR:-${IMAGINAIRE_OUTPUT_ROOT}/native_phase1_eval_inputs_viz5_edge_256_T97_s10_camera_wearer_v1}
FULL71_INPUT_DIR=${NATIVEP1_FULL71_EVAL_INPUT_DIR:-${IMAGINAIRE_OUTPUT_ROOT}/native_phase1_eval_inputs_full71_256_T97_camera_wearer_v1}
WORLD_SIZE=${NATIVEP1_WORLD_SIZE:-8}
CLIPS_PER_GPU=${NATIVEP1_CLIPS_PER_GPU:-4}
SMOKE_CLIPS_PER_GPU=${NATIVEP1_SMOKE_CLIPS_PER_GPU:-4}
MAX_ITER=${NATIVEP1_MAX_ITER:-100000}
SAVE_ITER=${NATIVEP1_SAVE_ITER:-5000}
for value_name in WORLD_SIZE CLIPS_PER_GPU SMOKE_CLIPS_PER_GPU MAX_ITER SAVE_ITER; do
  value=${!value_name}
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[edge-p1] ERROR: ${value_name} must be a positive integer" >&2
    exit 1
  fi
done
GLOBAL_BATCH=$((WORLD_SIZE * CLIPS_PER_GPU))

SMOKE_RUN_NAME=${NATIVEP1_SMOKE_RUN_NAME:-edge_phase1_T97_20fps_bs${SMOKE_CLIPS_PER_GPU}_camera_wearer_lora_smoke_gate1}
if [[ "${SMOKE_CLIPS_PER_GPU}" == "4" ]]; then
  SMOKE_MARKER_NAME=SINGLE_GPU_FOUR_CLIP_SMOKE_COMPLETE.json
else
  SMOKE_MARKER_NAME=SINGLE_GPU_${SMOKE_CLIPS_PER_GPU}_CLIP_SMOKE_COMPLETE.json
fi
SMOKE_MARKER=${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world_edge/${SMOKE_RUN_NAME}/${SMOKE_MARKER_NAME}
if [[ ! -s "${SMOKE_MARKER}" ]]; then
  echo "[edge-p1] ERROR: single-GPU ${SMOKE_CLIPS_PER_GPU}-clip smoke has not passed: ${SMOKE_MARKER}" >&2
  exit 1
fi

for required in \
  "${EDGE_FRAMEWORK_ROOT}/cosmos_framework" \
  "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${BASE_CHECKPOINT_PATH}/model/.metadata" \
  "${WAN_VAE_PATH}" \
  "${NYMERIA_LATENT_ROOT}/latent_cache_contract.json" \
  "${NYMERIA_LATENT_ROOT}/latent_cache_complete.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[edge-p1] ERROR: required artifact is missing: ${required}" >&2
    exit 1
  fi
done

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
export NATIVEP1_CLIPS_PER_GPU=${CLIPS_PER_GPU}
export NATIVEP1_EXPECTED_LATENT_HW=16
export NATIVEP1_EXPECTED_IMAGE_HW=256
export NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=1
export NATIVEP1_LORA_LR=5e-5
export NATIVEP1_ACTION_LR_MULT=4
export NATIVEP1_ACTION_LOSS_WEIGHT=10
export NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=0
export NATIVEP1_AUTO_EVAL=${NATIVEP1_AUTO_EVAL:-1}
export NATIVEP1_AUTO_EVAL_EVERY=${NATIVEP1_AUTO_EVAL_EVERY:-5000}
export NATIVEP1_VIZ_N=5
export NATIVEP1_EVAL_PREFIX_LENGTHS=1
export NATIVEP1_AUTO_EVAL_FULL71=${NATIVEP1_AUTO_EVAL_FULL71:-0}
export NATIVEP1_WANDB_MODE=${NATIVEP1_WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-cosmos3_camera}

case "${NATIVEP1_WANDB_MODE}" in
  online|offline|disabled) ;;
  *)
    echo "[edge-p1] ERROR: NATIVEP1_WANDB_MODE must be online, offline, or disabled" >&2
    exit 1
    ;;
esac
if [[ "${NATIVEP1_WANDB_MODE}" == "online" ]]; then
  WANDB_NETRC_FILE=${WANDB_NETRC_FILE:-/home/jungbinc/.netrc}
  if [[ -z "${WANDB_API_KEY:-}" ]] && \
     ! grep -Eq '^[[:space:]]*machine[[:space:]]+api\.wandb\.ai([[:space:]]|$)' \
       "${WANDB_NETRC_FILE}" 2>/dev/null; then
    echo "[edge-p1] ERROR: W&B online logging is enabled but no credential was found." >&2
    echo "[edge-p1] Set WANDB_API_KEY or run 'wandb login' before submission." >&2
    exit 1
  fi
fi

RUN_NAME=${NATIVEP1_RUN_NAME:-edge_phase1_T97_20fps_bs32_global_lora_100k}
TOML=${REPO_ROOT}/native_phase_training/world_camera_nymeria_latent_edge.toml

echo "[edge-p1] node=$(hostname) smoke_gate=${SMOKE_MARKER}"
echo "[edge-p1] replicate-only: shard=1 replicate=${WORLD_SIZE}; clips/GPU=${CLIPS_PER_GPU} global_batch=${GLOBAL_BATCH}"
echo "[edge-p1] tasks=40/25/20/15 fps=20 caption_subject='the camera wearer'"
echo "[edge-p1] shifts: training=3 inference=10"
echo "[edge-p1] wandb_mode=${NATIVEP1_WANDB_MODE} project=${WANDB_PROJECT} entity=${WANDB_ENTITY:-account-default}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

if [[ "${NATIVEP1_AUTO_EVAL}" == "1" ]]; then
  env RUN_ROOT="${IMAGINAIRE_OUTPUT_ROOT}" \
    "${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/prepare_phase1_eval_tier.py" \
    --source-dir "${FULL71_INPUT_DIR}" \
    --output-dir "${NATIVEP1_EVAL_INPUT_DIR}" \
    --resolution-tier 256 --shift 10 --limit 5
  for name in fd_input.jsonl invdyn_input.jsonl policy_input.jsonl i2v_input.jsonl; do
    [[ "$(wc -l < "${NATIVEP1_EVAL_INPUT_DIR}/${name}")" -eq 5 ]] || {
      echo "[edge-p1] expected five compact-eval records in ${name}" >&2
      exit 1
    }
  done
  "${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/validate_eval_inputs.py" \
    --input-dir "${NATIVEP1_EVAL_INPUT_DIR}" \
    --expected-shift 10 --expected-resolution 256 --expected-num-frames 97 --expected-fps 20 \
    fd_input.jsonl invdyn_input.jsonl policy_input.jsonl i2v_input.jsonl
fi

env PYTHONPATH=${LEGACY_FRAMEWORK_ROOT}:${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${REPO_ROOT}/motion_expert_joint_attention \
  "${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/validate_latent_cache.py" \
  --root "${NYMERIA_LATENT_ROOT}" --sample-count 256

cd "${EDGE_FRAMEWORK_ROOT}"
"${COSMOS_ENV_ROOT}/bin/torchrun" --standalone --nproc_per_node="${WORLD_SIZE}" \
  "${REPO_ROOT}/native_phase_training/run_latent_train.py" \
  --sft-toml "${TOML}" \
  job.name="${RUN_NAME}" \
  job.wandb_mode="${NATIVEP1_WANDB_MODE}" \
  trainer.max_iter="${MAX_ITER}" \
  trainer.grad_accum_iter=1 \
  trainer.logging_iter=50 \
  checkpoint.save_iter="${SAVE_ITER}" \
  model.config.compile.enabled=false \
  model.config.ema.enabled=true \
  model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.parallelism.data_parallel_replicate_degree="${WORLD_SIZE}" \
  model.config.parallelism.context_parallel_shard_degree=1 \
  model.config.parallelism.cfg_parallel_shard_degree=1

echo "[edge-p1] complete date=$(date)"
