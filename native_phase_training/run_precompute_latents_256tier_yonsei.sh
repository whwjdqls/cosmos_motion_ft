#!/usr/bin/env bash
# Yonsei worker for the Cosmos3-Edge T97/20-FPS/256-tier Wan latent cache.

set -euo pipefail

MODE=${1:-smoke}
REPO_ROOT=${REPO_ROOT:-/home/whwjdqls99/cosmos_motion_ft}
COSMOS_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_ROOT:-/home/whwjdqls99/cosmos-framework}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/lustre/whwjdqls99/cosmos/envs/cosmos}
WEKA_ROOT=${WEKA_ROOT:-/lustre/whwjdqls99/cosmos/weka}
WAN_VAE_PATH=${WAN_VAE_PATH:-${WEKA_ROOT}/wan22_vae/Wan2.2_VAE.pth}
SOURCE_ROOT=${WEKA_ROOT}/nymeriaplus_kimodo_proportional
MANIFEST=${NYMERIA_MANIFEST:-${SOURCE_ROOT}/video/manifest_video.jsonl}
SPLIT_FILE=${NYMERIA_SPLIT_FILE:-${SOURCE_ROOT}/train_test_split.json}
LATENT_ROOT=${NYMERIA_LATENT_ROOT:-${SOURCE_ROOT}/joint_latents_T97_256tier_256}
SMOKE_ROOT=${NYMERIA_LATENT_SMOKE_ROOT:-${SOURCE_ROOT}/joint_latents_T97_256tier_256_smoke_yonsei_4090}
LOG_ROOT=${LATENT_CACHE_LOG_ROOT:-/lustre/whwjdqls99/cosmos/logs}

for required in \
  "${REPO_ROOT}/motion_expert_joint_attention/precompute_latents.py" \
  "${REPO_ROOT}/native_phase_training/validate_latent_cache.py" \
  "${COSMOS_FRAMEWORK_ROOT}/cosmos_framework" \
  "${COSMOS_ENV_ROOT}/bin/python" \
  "${WAN_VAE_PATH}" \
  "${MANIFEST}" \
  "${SPLIT_FILE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[p1pc256-ysdc] ERROR: required artifact is missing: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_ROOT}"
export COSMOS_SITE=yonsei
export REPO_ROOT COSMOS_FRAMEWORK_ROOT COSMOS_ENV_ROOT WEKA_ROOT WAN_VAE_PATH
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${COSMOS_FRAMEWORK_ROOT}:${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${REPO_ROOT}/motion_expert_joint_attention

run_extract() {
  local out_root=$1
  local num_shards=$2
  local shard_id=$3
  shift 3
  mkdir -p "${out_root}/_logs"
  cd "${COSMOS_FRAMEWORK_ROOT}"
  "${COSMOS_ENV_ROOT}/bin/python" \
    "${REPO_ROOT}/motion_expert_joint_attention/precompute_latents.py" \
    --manifest "${MANIFEST}" --split_file "${SPLIT_FILE}" --split train \
    --num_frames 97 --fps 20 --resolution 256 \
    --model_resolution_tier 256 --expected_image_hw 256 --expected_latent_hw 16 \
    --write_cache_contract --fail_on_error \
    --vae_path "${WAN_VAE_PATH}" --out_root "${out_root}" \
    --num_shards "${num_shards}" --shard_id "${shard_id}" \
    --batch_size 1 --decode_workers 4 --prefetch_size 8 --log_every 100 \
    "$@"
}

case "${MODE}" in
  smoke)
    LOG=${LOG_ROOT}/wan_latents_T97_256_smoke_4090.log
    exec > >(tee -a "${LOG}") 2>&1
    echo "[$(date --iso-8601=seconds)] starting 40-window RTX 4090 smoke"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    run_extract "${SMOKE_ROOT}" 1 0 --limit 40 --log_every 10
    "${COSMOS_ENV_ROOT}/bin/python" - "${SMOKE_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np

from native_phase_training.latent_cache_contract import (
    load_latent_cache_contract,
    validate_cached_sample,
)

root = Path(sys.argv[1])
contract = load_latent_cache_contract(root)
files = sorted(root.glob("*/*.npz"))
if contract.num_shards != 1 or contract.limit_per_shard != 40:
    raise ValueError(
        "unexpected smoke contract: "
        f"num_shards={contract.num_shards}, limit_per_shard={contract.limit_per_shard}"
    )
if len(files) != 40:
    raise ValueError(f"smoke cache has {len(files)} files; expected 40")
for path in files:
    with np.load(path, allow_pickle=False) as data:
        validate_cached_sample(
            contract,
            latents=data["latents"],
            camera_action=data["camera_action"],
            image_size=data["image_size"],
            context=str(path),
        )
        if int(data["T"]) != 97 or abs(float(data["fps"]) - 20.0) > 1e-6:
            raise ValueError(f"{path}: unexpected T/FPS metadata")
marker = root / "smoke_validation_complete.json"
temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(
        {
            "kind": "nymeria_wan_latent_smoke_validation",
            "validated_files": len(files),
            "num_frames": contract.num_frames,
            "fps": contract.fps,
            "latent_shape": list(contract.expected_latent_shape),
            "latent_dtype": contract.latent_dtype,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
os.replace(temporary, marker)
print(f"[p1pc256-ysdc] smoke validation passed: {len(files)} files; marker={marker}")
PY
    echo "[$(date --iso-8601=seconds)] smoke complete; allocation may be reused for shard 0"
    ;;
  shard)
    : "${TOTAL_SHARDS:?set TOTAL_SHARDS for production extraction}"
    : "${SHARD_ID:?set SHARD_ID for production extraction}"
    LOG=${LOG_ROOT}/wan_latents_T97_256_shard_${SHARD_ID}_of_${TOTAL_SHARDS}.log
    exec > >(tee -a "${LOG}") 2>&1
    echo "[$(date --iso-8601=seconds)] starting production shard ${SHARD_ID}/${TOTAL_SHARDS}"
    run_extract "${LATENT_ROOT}" "${TOTAL_SHARDS}" "${SHARD_ID}"
    echo "[$(date --iso-8601=seconds)] production shard ${SHARD_ID}/${TOTAL_SHARDS} complete"
    ;;
  validate)
    LOG=${LOG_ROOT}/wan_latents_T97_256_validate.log
    exec > >(tee -a "${LOG}") 2>&1
    "${COSMOS_ENV_ROOT}/bin/python" \
      "${REPO_ROOT}/native_phase_training/validate_latent_cache.py" \
      --root "${LATENT_ROOT}" --sample-count 256
    ;;
  *)
    echo "usage: $0 {smoke|shard|validate}" >&2
    exit 2
    ;;
esac
