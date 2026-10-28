#!/usr/bin/env bash
# The prefix1 action-only ablation with both native video projections unfrozen.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei, grasp or custom}"
export NATIVEP1_ADAPTATION_MODE=action_video_projections
export NATIVEP1_PREFIX_LENGTHS=1
export NATIVEP1_WANDB_PREVIEWS=${NATIVEP1_WANDB_PREVIEWS:-1}
export NATIVEP1_RUN_NAME=${NATIVEP1_RUN_NAME:-edge_phase1_T97_20fps_2gpu_bs16_action_video_projections_exclude5_fd_invdyn_no_text_p1_200k_${COSMOS_SITE}}
case "$NATIVEP1_RUN_NAME" in
  *_action_video_projections_*) ;;
  *) printf 'Use a distinct action_video_projections run name\n' >&2; exit 2 ;;
esac
if [[ "$COSMOS_SITE" == yonsei ]]; then
  export WANDB_ENTITY=${WANDB_ENTITY:-jungbinc-upenn}
fi
unset WANDB_DISABLED
printf '%s\n' '[action-video-projections] train action2llm, llm2action, action_modality_embed, vae2llm and llm2vae; frozen backbone; no LoRA.'
printf '%s\n' '[action-video-projections] initial effective LR: action 8e-5; video 2e-5; same LambdaLinear schedule.'
exec bash "$HERE/train_phase1_edge_two_tasks_no_text.sh"
