#!/usr/bin/env bash
# Launch shape-aware-TMR scripts in the `kimodo` env with the right PYTHONPATH.
# GPUs come via Slurm (partition a2) or the interactive tmux GPU.
#
# Usage:
#   bash st_run.sh build_stats.py --out ...                                  # CPU OK
#   srun -p a2 --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --mem=96G \
#       bash st_run.sh st_train.py --out-dir ... [args]
set -e
SELF_DIR=/home/jungbin_cho/cosmos_motion_ft/shape_aware_TMR
export PYTHONPATH=/home/jungbin_cho/kimodo_open:${SELF_DIR}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/jungbin_cho/miniconda3/envs/kimodo/bin/python
cd "${SELF_DIR}"
SCRIPT="$1"; shift
exec "${PY}" "${SCRIPT}" "$@"
