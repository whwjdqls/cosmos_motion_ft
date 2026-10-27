#!/usr/bin/env bash
# Fresh image/prefix + camera -> video and video -> camera; no semantic captions.
# Use COSMOS_SITE=yonsei or grasp, with NATIVEP1_PREFIX_LENGTHS=1 or 17.
# Retain the same windows, task ratio, global32, losses, EMA, LR and 200k horizon.
# Saved config.yaml records caption_conditioning=false for both datasets.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei or grasp}"
export NATIVEP1_PREFIX_LENGTHS=${NATIVEP1_PREFIX_LENGTHS:-1}
case "$NATIVEP1_PREFIX_LENGTHS" in 1|17) ;; *) printf 'Expected prefix 1 or 17\n' >&2; exit 2;; esac
export NATIVEP1_DISABLE_CAPTIONS=1
export NATIVEP1_RUN_NAME=${NATIVEP1_RUN_NAME:-edge_phase1_T97_20fps_2gpu_bs16_lora_exclude5_fd_invdyn_no_text_p${NATIVEP1_PREFIX_LENGTHS}_200k_${COSMOS_SITE}_20261017}
case "$NATIVEP1_RUN_NAME" in *_no_text_*) ;; *) printf 'Use a distinct no_text run name\n' >&2; exit 2;; esac
# Use the saved CLI authentication, not stale credentials inherited by tmux.
unset WANDB_API_KEY WANDB_RUN_ID WANDB_RESUME WANDB_NAME
printf '[no-text] semantic captions disabled; prefix=%s; initialize=%s\n' "$NATIVEP1_PREFIX_LENGTHS" "${RESUME_CHECKPOINT:-pretrained Edge base (fresh optimizer)}"
exec bash "$HERE/train_phase1_edge_two_tasks.sh"
