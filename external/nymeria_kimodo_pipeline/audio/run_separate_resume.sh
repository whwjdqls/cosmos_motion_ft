#!/usr/bin/env bash
# Resume stage 3 for a subset listed in resume_list.txt (subj:seq per line).
# Each shard separates its slice (index%N of the list) with the given config.
set -uo pipefail
source /home/jungbin_cho/miniforge3/etc/profile.d/conda.sh
conda activate audio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=/home/jungbin_cho/nymeria_kimodo_pipeline/audio
LOG=/weka/jungbin/nymeriaplus_audio/logs/separate
N=${N:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
RERANK=${RERANK:-3}; DESC=${DESC:-speech}
mapfile -t ONLY < "$LOG/resume_list.txt"
cd "$DIR"
rm -f "$LOG/_RESUME_DONE"
echo "resume host=$(hostname) N=$N items=${#ONLY[@]} rerank=$RERANK start=$(date -u +%FT%TZ)" | tee "$LOG/_RESUME_STARTED"
for i in $(seq 0 $((N-1))); do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g python separate.py --shard "$i/$N" --only "${ONLY[@]}" \
    --description "$DESC" --reranking "$RERANK" --overwrite \
    > "$LOG/resume_shard_${i}_of_${N}.log" 2>&1 &
done
wait
echo "RESUME DONE $(date -u +%FT%TZ)" | tee "$LOG/_RESUME_DONE"
python - <<'PY'
import glob, json
n3=sum(1 for f in glob.glob("/weka/jungbin/nymeriaplus_audio/separate/*/*.json")
       if json.load(open(f)).get("reranking")==3)
print("r3 jsons now:", n3)
PY
