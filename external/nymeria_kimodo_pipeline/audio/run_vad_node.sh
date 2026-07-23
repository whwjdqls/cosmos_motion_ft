#!/usr/bin/env bash
# Stage 2 VAD driver for running directly on a GPU node (e.g. in tmux), no slurm.
# Spawns N shards of vad.py (FireRedVAD), round-robin pinned across the node's GPUs.
set -uo pipefail
source /home/jungbin_cho/miniforge3/etc/profile.d/conda.sh
conda activate audio
DIR=/home/jungbin_cho/nymeria_kimodo_pipeline/audio
LOG=/weka/jungbin/nymeriaplus_audio/logs/vad
N=${N:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
mkdir -p "$LOG"
cd "$DIR"
rm -f "$LOG/_DONE" "$LOG/_STARTED"
echo "host=$(hostname) N=$N gpus=${GPUS[*]} start=$(date -u +%FT%TZ)" | tee "$LOG/_STARTED"
for i in $(seq 0 $((N-1))); do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g python vad.py --backend firered --shard "$i/$N" \
    > "$LOG/run_shard_${i}_of_${N}.log" 2>&1 &
done
wait
echo "ALL $N VAD SHARDS DONE $(date -u +%FT%TZ)" | tee "$LOG/_DONE"
python - <<'PY'
import glob, json, collections
c=collections.Counter(); spk=0
for f in glob.glob("/weka/jungbin/nymeriaplus_audio/vad/**/*.json", recursive=True):
    d=json.load(open(f)); c["total"]+=1; spk+=int(d.get("has_speech",False))
print(f"VAD TALLY total={c['total']} has_speech={spk} no_speech={c['total']-spk}")
PY
