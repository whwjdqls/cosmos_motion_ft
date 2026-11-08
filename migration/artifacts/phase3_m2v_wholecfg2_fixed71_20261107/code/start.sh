#!/usr/bin/env bash
set -euo pipefail
source /home/jinhyung.kim/cosmos_motion_ft/.site.local.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /mnt/local/jinhyung.kim/cosmos/envs/cosmos/bin/python -u /mnt/local/jinhyung.kim/cosmos/operations/eval_sep1_m2v_wholecfg2_fixed71_20261107T083035Z/scheduler.py
