#!/usr/bin/env bash
# One reference frame; baseline LoRA/action training plus both video projections.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export NATIVEP1_ADAPTATION_MODE=global_lora_video_io NATIVEP1_PREFIX_LENGTHS=1
export NATIVEP1_CLIPS_PER_GPU=${NATIVEP1_CLIPS_PER_GPU:-16}
: "${NATIVEP1_RUN_NAME:?Choose a distinct video_io run name}"
[[ "$NATIVEP1_RUN_NAME" == *video_io* ]] || { printf 'Run name must contain video_io\n' >&2; exit 2; }
exec bash "$here/train_phase1_edge_foundary.sh" "$@"
