#!/bin/bash
# Build the full Edge T97/256 cache as independent, resumable one-GPU Slurm steps.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
LEGACY_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
LATENT_ROOT=${NYMERIA_LATENT_ROOT:-/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256}

PARTITION=${LATENT_CACHE_PARTITION:-batch}
GPU_TYPE=${LATENT_CACHE_GPU_TYPE:-l40}
TOTAL_SHARDS=${LATENT_CACHE_NUM_SHARDS:-4}
CPUS_PER_SHARD=${LATENT_CACHE_CPUS_PER_SHARD:-16}
MEMORY_PER_SHARD=${LATENT_CACHE_MEMORY_PER_SHARD:-32G}
TIME_LIMIT=${LATENT_CACHE_TIME_LIMIT:-24:00:00}
BATCH_SIZE=${LATENT_CACHE_BATCH_SIZE:-1}
DECODE_WORKERS=${LATENT_CACHE_DECODE_WORKERS:-4}
PREFETCH_SIZE=${LATENT_CACHE_PREFETCH_SIZE:-8}

for required in \
  "${REPO_ROOT}/motion_expert_joint_attention/precompute_latents.py" \
  "${REPO_ROOT}/native_phase_training/validate_latent_cache.py" \
  "${LEGACY_FRAMEWORK_ROOT}/cosmos_framework" \
  "${COSMOS_ENV_ROOT}/bin/python" \
  "${WAN_VAE_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[p1pc256] ERROR: required artifact is missing: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${LATENT_ROOT}/_logs"

export REPO_ROOT LEGACY_FRAMEWORK_ROOT COSMOS_ENV_ROOT WAN_VAE_PATH LATENT_ROOT
export TOTAL_SHARDS BATCH_SIZE DECODE_WORKERS PREFETCH_SIZE
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${LEGACY_FRAMEWORK_ROOT}:${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${REPO_ROOT}/motion_expert_joint_attention

echo "[p1pc256] root=${LATENT_ROOT}"
echo "[p1pc256] launching ${TOTAL_SHARDS} independent ${GPU_TYPE} shards"
echo "[p1pc256] batch_size=${BATCH_SIZE} decode_workers=${DECODE_WORKERS} prefetch_size=${PREFETCH_SIZE}"

pids=()
for ((shard_id = 0; shard_id < TOTAL_SHARDS; shard_id++)); do
  log_path=${LATENT_ROOT}/_logs/shard_${shard_id}.log
  echo "[p1pc256] shard ${shard_id}/${TOTAL_SHARDS} -> ${log_path}"
  SHARD_ID=${shard_id} srun \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-task="${CPUS_PER_SHARD}" \
    --mem="${MEMORY_PER_SHARD}" \
    --time="${TIME_LIMIT}" \
    --job-name="p1pc256-s${shard_id}" \
    bash -lc '
      set -euo pipefail
      cd "${LEGACY_FRAMEWORK_ROOT}"
      exec "${COSMOS_ENV_ROOT}/bin/python" \
        "${REPO_ROOT}/motion_expert_joint_attention/precompute_latents.py" \
        --split train --num_frames 97 --fps 20 --resolution 256 \
        --model_resolution_tier 256 --expected_image_hw 256 --expected_latent_hw 16 \
        --write_cache_contract --fail_on_error \
        --vae_path "${WAN_VAE_PATH}" \
        --out_root "${LATENT_ROOT}" \
        --num_shards "${TOTAL_SHARDS}" --shard_id "${SHARD_ID}" \
        --batch_size "${BATCH_SIZE}" --decode_workers "${DECODE_WORKERS}" \
        --prefetch_size "${PREFETCH_SIZE}" \
        --log_every 100
    ' >"${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[${index}]}"; then
    echo "[p1pc256] ERROR: shard ${index} failed; see ${LATENT_ROOT}/_logs/shard_${index}.log" >&2
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi

echo "[p1pc256] all shards finished; validating the complete file set"
srun \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=02:00:00 \
  --job-name=p1pc256-validate \
  "${COSMOS_ENV_ROOT}/bin/python" \
  "${REPO_ROOT}/native_phase_training/validate_latent_cache.py" \
  --root "${LATENT_ROOT}" --sample-count 256

echo "[p1pc256] complete: ${LATENT_ROOT}"
