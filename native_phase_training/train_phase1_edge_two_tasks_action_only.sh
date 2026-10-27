#!/usr/bin/env bash
# Same prefix1/no-text two-task recipe, training only the existing action modules.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei, grasp or custom}"
export NATIVEP1_ADAPTATION_MODE=action_only
export NATIVEP1_PREFIX_LENGTHS=1
export NATIVEP1_WANDB_PREVIEWS=${NATIVEP1_WANDB_PREVIEWS:-1}
export NATIVEP1_RUN_NAME=${NATIVEP1_RUN_NAME:-edge_phase1_T97_20fps_2gpu_bs16_action_only_exclude5_fd_invdyn_no_text_p1_200k_${COSMOS_SITE}_20261027}
case "$NATIVEP1_RUN_NAME" in
  *_action_only_*) ;;
  *) printf 'Use a distinct action_only run name\n' >&2; exit 2 ;;
esac
if [[ "$COSMOS_SITE" == yonsei ]]; then
  export WANDB_ENTITY=${WANDB_ENTITY:-jungbinc-upenn}
fi
unset WANDB_DISABLED
printf '%s\n' '[action-only] train action2llm, llm2action and action_modality_embed; freeze backbone; no LoRA.'
exec bash "$HERE/train_phase1_edge_two_tasks_no_text.sh"
