#!/bin/bash
# After inspecting a successful smoke, build the full cache and queue production.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/mnt/projects/ll/jungbinc/weka/cosmos_motion_ft_runs}
SMOKE_RUN_NAME=${NATIVEP1_SMOKE_RUN_NAME:-edge_phase1_T97_20fps_bs8_lora_smoke_gate3}
SMOKE_MARKER=${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world_edge/${SMOKE_RUN_NAME}/SINGLE_GPU_8_CLIP_SMOKE_COMPLETE.json
export NATIVEP1_WANDB_MODE=${NATIVEP1_WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-cosmos3_camera}

if [[ ! -s "${SMOKE_MARKER}" ]]; then
  echo "[edge-submit] ERROR: inspect and complete the single-GPU smoke first: ${SMOKE_MARKER}" >&2
  exit 1
fi

case "${NATIVEP1_WANDB_MODE}" in
  online|offline|disabled) ;;
  *)
    echo "[edge-submit] ERROR: NATIVEP1_WANDB_MODE must be online, offline, or disabled" >&2
    exit 1
    ;;
esac
if [[ "${NATIVEP1_WANDB_MODE}" == "online" ]]; then
  WANDB_NETRC_FILE=${WANDB_NETRC_FILE:-/home/jungbinc/.netrc}
  if [[ -z "${WANDB_API_KEY:-}" ]] && \
     ! grep -Eq '^[[:space:]]*machine[[:space:]]+api\.wandb\.ai([[:space:]]|$)' \
       "${WANDB_NETRC_FILE}" 2>/dev/null; then
    echo "[edge-submit] ERROR: W&B online logging is enabled but no credential was found." >&2
    echo "[edge-submit] Set WANDB_API_KEY or run 'wandb login' before submission." >&2
    exit 1
  fi
fi

cache_job=$(sbatch --parsable "${REPO_ROOT}/native_phase_training/sbatch_precompute_latents_256tier_l40s.sh")
production_job=$(sbatch --parsable --dependency="afterok:${cache_job}" \
  "${REPO_ROOT}/native_phase_training/sbatch_phase1_edge_4gpu_bs8.sh")
echo "full latent-cache job: ${cache_job}"
echo "edge 4-GPU, 8-clips/GPU job (after cache): ${production_job}"
