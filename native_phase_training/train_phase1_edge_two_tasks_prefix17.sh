#!/usr/bin/env bash
# Fresh two-task ablation: change only the forward RGB prefix from 1 to 17.
# Shared recipe retains full-video inverse dynamics and works on Yonsei/Grasp.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei or grasp}"
export NATIVEP1_PREFIX_LENGTHS=17
export NATIVEP1_RUN_NAME=${NATIVEP1_RUN_NAME:-edge_phase1_T97_20fps_2gpu_bs16_camera_wearer_lora_exclude5_fd_invdyn_p17_200k_${COSMOS_SITE}_20261016}
exec bash "${HERE}/train_phase1_edge_two_tasks.sh"
