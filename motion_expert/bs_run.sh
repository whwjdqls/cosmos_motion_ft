#!/usr/bin/env bash
# Launch BONES-SEED in-context POC scripts in the `kimodo` env with the right PYTHONPATH.
# Everything runs in kimodo (torch 2.4); GPUs come via Slurm (partition a2).
#
# Usage (set CUDA_VISIBLE_DEVICES, or wrap in srun):
#   bash bs_run.sh bs_check_cache.py --split .../train_split_paths.txt          # CPU, login node OK
#   srun -p a2 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=64G \
#       bash bs_run.sh bs_train.py --smoke
#   srun -p a2 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=64G \
#       bash bs_run.sh bs_train.py --steps 150000 --batch_size 128 --run_name bs_incontext_v1
#   sbatch sbatch_bs_native_phase2.sh  # native shifted-logitnormal x0 Phase-2 POC
set -e
SELF_DIR=/home/jungbin_cho/cosmos_motion_ft/motion_expert
export PYTHONPATH=/home/jungbin_cho/kimodo_open:${SELF_DIR}
export TOKENIZERS_PARALLELISM=false
PY=/home/jungbin_cho/miniconda3/envs/kimodo/bin/python
cd "${SELF_DIR}"
SCRIPT="$1"; shift
exec "${PY}" "${SCRIPT}" "$@"
