#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
production_training=0
preflight=0
for argument in "$@"; do
    [[ "$argument" != motion_expert_phase3_edge_camtrajfilm.train ]] || production_training=1
    [[ "$argument" != --preflight ]] || preflight=1
done
if (( production_training && ! preflight )); then
    : "${PHASE3_RUN_DIR:?Set PHASE3_RUN_DIR to the trainer output directory for W&B logging}"
    exec bash "$HERE/../motion_expert_phase3_edge/run.sh" \
        "$HERE/wandb_training.py" --run-dir "$PHASE3_RUN_DIR" -- "$@"
fi
exec bash "$HERE/../motion_expert_phase3_edge/run.sh" "$@"
