#!/bin/bash
# Run a contiguous block of export shards as parallel background processes on ONE node.
# usage: run_shards_node.sh NUM_SHARDS K_START K_END
set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate kimodo
export PYTHONPATH=/home/jungbin_cho/kimodo_open:${PYTHONPATH:-}
# export is python+IO bound, not BLAS-bound; 1 thread/proc avoids oversubscription
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY=/home/jungbin_cho/miniforge3/envs/kimodo/bin/python
SCRIPT=/home/jungbin_cho/cosmos_motion_ft/export_bones_seed_full.py
OUT=/weka/jungbin/seed/cosmos_text_motion_full
CACHE=$OUT/bones_seed_index_cache.json
LOGD=/home/jungbin_cho/cosmos_motion_ft/export_logs
mkdir -p "$LOGD"

NS=$1; A=$2; B=$3
echo "[driver $(hostname)] launching shards $A..$B of $NS  start=$(date)"
for k in $(seq "$A" "$B"); do
  nohup "$PY" "$SCRIPT" --out "$OUT" --shard "$k" --num-shards "$NS" \
      --cache-index "$CACHE" --max-samples -1 \
      > "$LOGD/shard_${k}.out" 2>&1 &
done
wait
echo "[driver $(hostname)] DONE shards $A..$B  end=$(date)"
