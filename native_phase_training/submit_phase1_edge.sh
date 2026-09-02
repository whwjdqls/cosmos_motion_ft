#!/bin/bash
# Submit DCP preparation and the single-GPU eight-clip gate only.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
export NATIVEP1_WANDB_MODE=${NATIVEP1_WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-cosmos3_camera}

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

convert_job=$(sbatch --parsable "${REPO_ROOT}/native_phase_training/sbatch_convert_edge_to_dcp.sh")
smoke_job=$(sbatch --parsable --dependency="afterok:${convert_job}" \
  --export=ALL,NATIVEP1_SMOKE_CLIPS_PER_GPU=8 \
  "${REPO_ROOT}/native_phase_training/sbatch_phase1_edge_smoke.sh")
echo "edge DCP conversion job: ${convert_job}"
echo "edge smoke job: ${smoke_job}"
echo "The 4-GPU run was not submitted. Inspect the smoke marker/log first, then run submit_phase1_edge_production.sh."
