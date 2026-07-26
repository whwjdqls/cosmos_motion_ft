#!/bin/bash
# Official 720 model tier uses the transform key 480 for square 640x640 pixels.
# Three array tasks x eight local GPUs = 24 deterministic global cache shards.
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=p1pc720
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p1pc720-%A_%a.out
#SBATCH --array=0-2%3
#SBATCH --exclusive
#SBATCH --exclude=a3ultravis-a3ultranodeset-2

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH=${WAN_VAE_PATH:-/weka/jungbin/wan22_vae/Wan2.2_VAE.pth}
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention

LATENT_ROOT=${NYMERIA_LATENT_ROOT:-/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T97_720tier_640}
TOTAL_SHARDS=${PRECOMPUTE_TOTAL_SHARDS:-24}
LOCAL_SHARDS=8
ARRAY_TASKS=3
LIMIT_PER_SHARD=${PRECOMPUTE_LIMIT_PER_SHARD:-}

if (( TOTAL_SHARDS != LOCAL_SHARDS * ARRAY_TASKS )); then
  echo "[p1pc720] ERROR: TOTAL_SHARDS must equal ${LOCAL_SHARDS}x${ARRAY_TASKS}" >&2
  exit 1
fi

mkdir -p "${LATENT_ROOT}/_logs"
cd /home/jungbin_cho/cosmos-framework

echo "[p1pc720] node=$(hostname) date=$(date)"
echo "[p1pc720] array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "[p1pc720] root=${LATENT_ROOT}"
echo "[p1pc720] split=train T=97 transform_resolution=480 model_tier=720 image=640 latent_hw=40"
echo "[p1pc720] total_shards=${TOTAL_SHARDS} local_shards=${LOCAL_SHARDS} limit=${LIMIT_PER_SHARD:-none}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

pids=()
for local_rank in 0 1 2 3 4 5 6 7; do
  global_shard=$((SLURM_ARRAY_TASK_ID * LOCAL_SHARDS + local_rank))
  log_path="${LATENT_ROOT}/_logs/array_${SLURM_ARRAY_TASK_ID}_shard_${global_shard}.log"
  args=(
    --split train
    --num_frames 97
    --resolution 480
    --model_resolution_tier 720
    --expected_image_hw 640
    --expected_latent_hw 40
    --write_cache_contract
    --fail_on_error
    --out_root "${LATENT_ROOT}"
    --num_shards "${TOTAL_SHARDS}"
    --shard_id "${global_shard}"
    --log_every 100
  )
  if [[ -n "${LIMIT_PER_SHARD}" ]]; then
    args+=(--limit "${LIMIT_PER_SHARD}")
  fi
  echo "[p1pc720] GPU ${local_rank} -> global shard ${global_shard}; log=${log_path}"
  CUDA_VISIBLE_DEVICES=${local_rank} \
    /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
    /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention/precompute_latents.py \
    "${args[@]}" >"${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed )); then
  echo "[p1pc720] ERROR: at least one local shard failed" >&2
  tail -n 80 "${LATENT_ROOT}/_logs/array_${SLURM_ARRAY_TASK_ID}"_shard_*.log >&2
  exit 1
fi

echo "[p1pc720] array task complete date=$(date)"
for local_rank in 0 1 2 3 4 5 6 7; do
  global_shard=$((SLURM_ARRAY_TASK_ID * LOCAL_SHARDS + local_rank))
  tail -n 14 "${LATENT_ROOT}/_logs/array_${SLURM_ARRAY_TASK_ID}_shard_${global_shard}.log"
done
