#!/usr/bin/env bash
# Stage 1 extraction driver for running directly on a node (e.g. in tmux), no slurm.
# Spawns N parallel shards of extract_audio.py (CPU/IO-bound) and waits.
set -uo pipefail
PY=/home/jungbin_cho/miniforge3/envs/nymeria_plus/bin/python
DIR=/home/jungbin_cho/nymeria_kimodo_pipeline/audio
LOG=/weka/jungbin/nymeriaplus_audio/logs/extract
N=${N:-32}
mkdir -p "$LOG"
cd "$DIR"
rm -f "$LOG/_DONE" "$LOG/_STARTED"
echo "host=$(hostname) N=$N start=$(date -u +%FT%TZ)" | tee "$LOG/_STARTED"
for i in $(seq 0 $((N-1))); do
  $PY extract_audio.py --shard "$i/$N" > "$LOG/run_shard_${i}_of_${N}.log" 2>&1 &
done
wait
echo "ALL $N SHARDS DONE $(date -u +%FT%TZ)" | tee "$LOG/_DONE"
# tally
$PY - <<'PY'
import glob, json
ok=skip=err=other=0
for f in glob.glob("/weka/jungbin/nymeriaplus_audio/logs/extract/shard_*_of_*.jsonl"):
    for l in open(f):
        s=json.loads(l).get("status")
        if s=="ok": ok+=1
        elif s=="skip_exists": skip+=1
        elif s in ("read_error","no_data_vrs"): err+=1
        else: other+=1
print(f"TALLY ok={ok} skip={skip} err={err} other={other} total={ok+skip+err+other}")
PY
