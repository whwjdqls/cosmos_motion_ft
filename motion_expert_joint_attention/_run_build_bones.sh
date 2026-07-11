#!/bin/bash
unset LD_LIBRARY_PATH
export PYTHONPATH=/home/jungbin_cho/kimodo_open:/home/jungbin_cho/cosmos_motion_ft/motion_expert:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
cd /home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[launch] cwd=$(pwd) PYTHONPATH=$PYTHONPATH"
exec /home/jungbin_cho/miniforge3/envs/kimodo/bin/python build_bones_pairs.py "$@"
