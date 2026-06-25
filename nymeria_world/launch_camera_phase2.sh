#!/usr/bin/env bash
# Launch the NymeriaPlus camera-only Phase-2 training (or smoke).
# Usage: launch_camera_phase2.sh <NPROC> <MAX_ITER> <LOGFILE> [extra opts...]
set -e
cd /home/jungbin_cho/cosmos-framework
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft/nymeria_world
unset LD_LIBRARY_PATH
export BASE_CHECKPOINT_PATH=/weka/jungbin/cosmos3_nano_dcp
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/weka/jungbin/cosmos_motion_ft_runs

NPROC=${1:-8}; MAX_ITER=${2:-3}; LOG=${3:-/weka/jungbin/cosmos_motion_ft_runs/nymeria_camera_smoke.log}
shift 3 || true

/home/jungbin_cho/miniforge3/envs/cosmos/bin/torchrun --standalone --nproc_per_node="$NPROC" \
  -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/world_camera_nymeria_repro.toml \
  trainer.max_iter="$MAX_ITER" trainer.logging_iter=1 "$@" \
  > "$LOG" 2>&1
echo "SMOKE_EXIT=$?" >> "$LOG"
