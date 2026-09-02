#!/bin/bash
# Full T97, 20-FPS, 256x256 Wan-latent cache on one 8xL40S node.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:8
#SBATCH --exclude=kd-l40-0.grasp.maas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --job-name=p1pc256
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-p1pc256-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
LEGACY_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/mnt/projects/ll/jungbinc/weka/wan22_vae/Wan2.2_VAE.pth}
LATENT_ROOT=${NYMERIA_LATENT_ROOT:-/mnt/projects/ll/jungbinc/weka/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256}
TOTAL_SHARDS=8
BATCH_SIZE=${LATENT_CACHE_BATCH_SIZE:-1}
DECODE_WORKERS=${LATENT_CACHE_DECODE_WORKERS:-4}
PREFETCH_SIZE=${LATENT_CACHE_PREFETCH_SIZE:-8}

export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${LEGACY_FRAMEWORK_ROOT}:${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${REPO_ROOT}/motion_expert_joint_attention
mkdir -p "${LATENT_ROOT}/_logs"
cd "${LEGACY_FRAMEWORK_ROOT}"

pids=()
for local_rank in 0 1 2 3 4 5 6 7; do
  log_path=${LATENT_ROOT}/_logs/shard_${local_rank}.log
  echo "[p1pc256] GPU ${local_rank} -> shard ${local_rank}/${TOTAL_SHARDS}"
  CUDA_VISIBLE_DEVICES=${local_rank} \
    "${COSMOS_ENV_ROOT}/bin/python" \
    "${REPO_ROOT}/motion_expert_joint_attention/precompute_latents.py" \
    --split train --num_frames 97 --fps 20 --resolution 256 \
    --model_resolution_tier 256 --expected_image_hw 256 --expected_latent_hw 16 \
    --write_cache_contract --fail_on_error \
    --out_root "${LATENT_ROOT}" --num_shards "${TOTAL_SHARDS}" --shard_id "${local_rank}" \
    --batch_size "${BATCH_SIZE}" --decode_workers "${DECODE_WORKERS}" \
    --prefetch_size "${PREFETCH_SIZE}" \
    --log_every 100 >"${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if (( failed )); then
  echo "[p1pc256] ERROR: at least one shard failed" >&2
  tail -n 80 "${LATENT_ROOT}"/_logs/shard_*.log >&2
  exit 1
fi

"${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/validate_latent_cache.py" \
  --root "${LATENT_ROOT}" --sample-count 256
echo "[p1pc256] complete: ${LATENT_ROOT}"
