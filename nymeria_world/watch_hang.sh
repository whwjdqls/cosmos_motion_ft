#!/usr/bin/env bash
# Passive watcher: if full-gen (job 2556 on node 2) deadlocks (sustained 0% GPU util), py-spy all ranks
# BEFORE it's killed, to capture the definitive stack trace. Non-destructive (observe only).
N=a3ultravis-a3ultranodeset-2
OUT=/weka/jungbin/cosmos_motion_ft_runs/fullgen_hang_watch; mkdir -p "$OUT"
zero=0
while squeue -j 2556 -h 2>/dev/null | grep -q 2556; do
  util=$(ssh $N 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | sort -rn | head -1 | tr -dc 0-9')
  iter=$(grep -oE "\] [0-9]+ : iter_speed" /home/jungbin_cho/cosmos_motion_ft/slurm-camworld-2556.out 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  if [ "${util:-9}" = "0" ]; then zero=$((zero+1)); else zero=0; fi
  echo "$(date +%T) iter=${iter:-?} maxutil=${util:-?} zerostreak=$zero" >> "$OUT/watch.log"
  if [ "$zero" -ge 6 ]; then
    echo "$(date +%T) HANG DETECTED at iter ${iter} -> py-spy" >> "$OUT/watch.log"
    ssh $N 'for pid in $(pgrep -f "cosmos_framework.scripts.train"); do echo "===== PID $pid ====="; /home/jungbin_cho/miniforge3/envs/cosmos/bin/py-spy dump --pid $pid 2>&1; echo; done' > "$OUT/pyspy_stacks_iter${iter}.txt" 2>&1
    echo "$(date +%T) STACKS_CAPTURED -> $OUT/pyspy_stacks_iter${iter}.txt" >> "$OUT/watch.log"
    break
  fi
  sleep 30
done
echo "$(date +%T) watcher exit (job gone or hang captured)" >> "$OUT/watch.log"
