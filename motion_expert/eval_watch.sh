#!/usr/bin/env bash
# Auto-eval: wait for each milestone ckpt of a MotionExpert run, then sample (cosmos) + render (kimodo)
# the text-conditioned-vs-null ABLATION. Runs on a chosen GPU (default 2; real training is on GPU 1).
# Usage: bash eval_watch.sh <run_dir> [gpu] [steps_csv]
set -u
RUN="${1:-/weka/jungbin/cosmos_motion_ft_runs/motionexpert_poc_v1}"
GPU="${2:-2}"
STEPS="${3:-5000,20000,50000,100000}"
ME=/home/jungbin_cho/cosmos_motion_ft/motion_expert
KPY=/home/jungbin_cho/miniforge3/envs/kimodo/bin/python

IFS=',' read -ra MS <<< "$STEPS"
for s in "${MS[@]}"; do
  ck="$RUN/ckpt_step$(printf '%06d' "$s").pt"
  echo "[eval] waiting for $ck ..."
  while [ ! -f "$ck" ]; do sleep 120; done
  sleep 20  # let torch.save flush
  out="$RUN/samples_step$((s/1000))k"
  echo "[eval] sampling ckpt step $s -> $out"
  CUDA_VISIBLE_DEVICES="$GPU" bash "$ME/run.sh" sample.py --ckpt "$ck" --out "$out" \
      --ablation both --steps 50 --guidance 2.0 > "$RUN/eval_step${s}_sample.log" 2>&1
  echo "[eval] rendering (kimodo) -> $out"
  unset LD_LIBRARY_PATH
  "$KPY" "$ME/viz.py" --dir "$out" > "$RUN/eval_step${s}_viz.log" 2>&1
  echo "[eval] DONE step $s : $(ls "$out"/*ABLATION.mp4 2>/dev/null | wc -l) ablation clips in $out"
done
echo "[eval] all milestones done"
