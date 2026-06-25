#!/usr/bin/env bash
# Launch MotionExpert scripts with the correct cwd/env.
# train_motion_ft.build_network reads a RELATIVE QWEN_JSON, so cwd MUST be the cosmos-framework repo root.
# Usage:  ssh <node> 'bash /home/jungbin_cho/cosmos_motion_ft/motion_expert/run.sh <script.py> [args...]'
#         (set CUDA_VISIBLE_DEVICES before calling)
set -e
cd /home/jungbin_cho/cosmos-framework
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft/motion_expert
unset LD_LIBRARY_PATH
PY=/home/jungbin_cho/miniforge3/envs/cosmos/bin/python
SCRIPT="$1"; shift
exec "$PY" "/home/jungbin_cho/cosmos_motion_ft/motion_expert/$SCRIPT" "$@"
