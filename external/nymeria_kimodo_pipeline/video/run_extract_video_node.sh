#!/usr/bin/env bash
# Stage A driver: extract ALL egocentric mp4s, run directly on an a3ultra node
# (e.g. inside tmux/srun). VRS decode is CPU-bound; the node has 224 CPUs, so
# ~96 workers fills it. ~732 sequences x ~600 s / 96 ~= 80 min wall.
#
#   srun -p a3ultra -N1 --cpus-per-task=224 --time=6:00:00 --pty bash
#   N=96 SIZE=640 bash run_extract_video_node.sh
#
# Idempotent: sequences with a _done sentinel + mp4 + json are skipped, so it
# resumes after interruption. Then build the manifest (Stage B):
#   /home/jungbin_cho/miniforge3/envs/kimodo/bin/python build_video_manifest.py
set -uo pipefail
PY=/home/jungbin_cho/miniforge3/envs/nymeria_plus/bin/python
DIR=/home/jungbin_cho/nymeria_kimodo_pipeline/video
N=${N:-96}
SIZE=${SIZE:-640}
LOG=/weka/jungbin/nymeriaplus_kimodo_proportional/video/_logs
mkdir -p "$LOG"
cd "$DIR"
echo "host=$(hostname) N=$N SIZE=$SIZE start=$(date -u +%FT%TZ)" | tee "$LOG/_STARTED"
$PY extract_ego_video.py --workers "$N" --size "$SIZE" 2>&1 | tee "$LOG/extract_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "DONE $(date -u +%FT%TZ)" | tee "$LOG/_DONE"
