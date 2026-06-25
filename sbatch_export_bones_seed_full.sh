#!/bin/bash
#SBATCH --job-name=bseed_export
#SBATCH --partition=a3ultra
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/export_logs/shard_%a.out
#SBATCH --error=/home/jungbin_cho/cosmos_motion_ft/export_logs/shard_%a.err

# Full FULL BONES-SEED (raw_text, motion[T,369]) export, sharded across an array.
# Each task k of N=16 exports unique entries [k::16] to
# /weka/jungbin/seed/cosmos_text_motion_full/shard_<k>/{features.npy,index.json}.
# The pool index (~10-14 min build) is cached on weka and reused by later tasks.
set -euo pipefail

NUM_SHARDS=16
OUT=/weka/jungbin/seed/cosmos_text_motion_full
CACHE=${OUT}/bones_seed_index_cache.json
PY=/home/jungbin_cho/miniforge3/envs/kimodo/bin/python
SCRIPT=/home/jungbin_cho/cosmos_motion_ft/export_bones_seed_full.py

mkdir -p "$OUT" /home/jungbin_cho/cosmos_motion_ft/export_logs

echo "[task ${SLURM_ARRAY_TASK_ID}] host=$(hostname) start=$(date)"
"$PY" "$SCRIPT" \
    --out "$OUT" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "$NUM_SHARDS" \
    --cache-index "$CACHE" \
    --max-samples -1
echo "[task ${SLURM_ARRAY_TASK_ID}] done=$(date)"
