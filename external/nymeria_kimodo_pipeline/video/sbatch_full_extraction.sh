#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=200
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --job-name=nymeria_ego_extract
#SBATCH --output=/weka/jungbin/nymeriaplus_kimodo_proportional/video/_logs/sbatch-%j.out
# Full egocentric extraction for all 732 NymeriaPlus sequences, CPU-only:
#   phase 1: camera trajectory sidecars  (parallel, ~15-25 min)
#   phase 2: ego mp4 videos              (parallel N=96, ~80 min)
#   phase 3: build the Cosmos video manifest
# Both decode from VRS/MPS (env nymeria_plus); the manifest is numpy (env kimodo).
# Idempotent: existing sidecars/mp4s (and the video _done sentinel) are skipped, so
# this resumes cleanly if re-submitted.
set -uo pipefail
PYN=/home/jungbin_cho/miniforge3/envs/nymeria_plus/bin/python
PYK=/home/jungbin_cho/miniforge3/envs/kimodo/bin/python
PIPE=/home/jungbin_cho/nymeria_kimodo_pipeline
LOG=/weka/jungbin/nymeriaplus_kimodo_proportional/video/_logs
mkdir -p "$LOG"
echo "host=$(hostname) start=$(date -u +%FT%TZ)"

echo "=== phase 1: camera trajectories (N=64) ==="
cd "$PIPE/camera"
$PYN extract_camera_all.py --workers 64 2>&1 | grep -v -iE 'warning|progresslog|multirecord|vrsdata|streamid|timecode|it/s|loaded #|\[0m'

echo "=== phase 2: ego mp4 videos (N=96, size=640) ==="
cd "$PIPE/video"
$PYN extract_ego_video.py --workers 96 --size 640 2>&1 | grep -v -iE 'warning|progresslog|multirecord|vrsdata|streamid|timecode|it/s|loaded #|\[0m|recording_head:'

echo "=== phase 3: build manifest ==="
$PYK build_video_manifest.py

echo "DONE $(date -u +%FT%TZ)"
