#!/usr/bin/env bash
# Stage 3 SAM-Audio separation driver for a GPU node (tmux), no slurm.
# Spawns N shards of separate.py, round-robin pinned across the node's GPUs.
# Each shard ~7 min/recording; 735 recs / 8 shards ~= 11 h. Resumable (skips existing).
set -uo pipefail
source /home/jungbin_cho/miniforge3/etc/profile.d/conda.sh
conda activate audio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=/home/jungbin_cho/nymeria_kimodo_pipeline/audio
LOG=/weka/jungbin/nymeriaplus_audio/logs/separate
N=${N:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
mkdir -p "$LOG"
cd "$DIR"
rm -f "$LOG/_DONE" "$LOG/_STARTED"
echo "host=$(hostname) N=$N gpus=${GPUS[*]} start=$(date -u +%FT%TZ)" | tee "$LOG/_STARTED"
RERANK=${RERANK:-1}
DESC=${DESC:-speech}
OVW=""; [ "${OVERWRITE:-0}" = "1" ] && OVW="--overwrite"
echo "desc='$DESC' reranking=$RERANK overwrite=${OVERWRITE:-0}" | tee -a "$LOG/_STARTED"
for i in $(seq 0 $((N-1))); do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g python separate.py --shard "$i/$N" \
    --description "$DESC" --reranking "$RERANK" $OVW \
    > "$LOG/run_shard_${i}_of_${N}.log" 2>&1 &
done
wait
echo "ALL $N SEPARATE SHARDS DONE $(date -u +%FT%TZ)" | tee "$LOG/_DONE"
python - <<'PY'
import glob, json, collections
c=collections.Counter()
for f in glob.glob("/weka/jungbin/nymeriaplus_audio/separate/**/*.json", recursive=True):
    c[json.load(open(f)).get("status")]+=1
print("SEPARATE TALLY", dict(c))
PY
