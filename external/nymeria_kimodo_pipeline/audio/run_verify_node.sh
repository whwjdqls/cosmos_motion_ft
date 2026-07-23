#!/usr/bin/env bash
# Stage 4 verify+filter driver for a GPU node (tmux), no slurm.
# N shards of verify_and_filter.py (2nd FireRedVAD pass + silent/broken checks) across GPUs,
# then merges per-shard cleaned.shard*.jsonl into cleaned.jsonl + cleaned_summary.json.
set -uo pipefail
source /home/jungbin_cho/miniforge3/etc/profile.d/conda.sh
conda activate audio
DIR=/home/jungbin_cho/nymeria_kimodo_pipeline/audio
OUT=/weka/jungbin/nymeriaplus_audio
LOG=$OUT/logs/verify
N=${N:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
mkdir -p "$LOG"
cd "$DIR"
rm -f "$LOG/_DONE" "$LOG/_STARTED" "$OUT"/cleaned.shard*_of_*.jsonl
echo "host=$(hostname) N=$N start=$(date -u +%FT%TZ)" | tee "$LOG/_STARTED"
for i in $(seq 0 $((N-1))); do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g python verify_and_filter.py --backend firered --shard "$i/$N" \
    > "$LOG/run_shard_${i}_of_${N}.log" 2>&1 &
done
wait
echo "shards done, merging..."
python - <<PY
import glob, json, collections
rows=[]
for f in sorted(glob.glob("$OUT/cleaned.shard*_of_*.jsonl")):
    rows += [json.loads(l) for l in open(f)]
with open("$OUT/cleaned.jsonl","w") as o:
    for r in rows: o.write(json.dumps(r)+"\n")
kept=sum(r["keep"] for r in rows)
rc=collections.Counter(x for r in rows for x in (r["reasons"] or ["kept"]))
summary={"total":len(rows),"kept":kept,"dropped":len(rows)-kept,"reason_counts":dict(rc)}
json.dump(summary, open("$OUT/cleaned_summary.json","w"), indent=2)
print("MERGED", json.dumps(summary))
PY
echo "ALL $N VERIFY SHARDS DONE $(date -u +%FT%TZ)" | tee "$LOG/_DONE"
