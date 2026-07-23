#!/usr/bin/env bash
#SBATCH --job-name=npa_audio
#SBATCH --partition=a3ultra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/weka/jungbin/nymeriaplus_audio/logs/slurm/%x_%A_%a.out
#
# One sbatch for every stage; pick the stage with STAGE=... and the array width N.
# The array index becomes the shard: --shard ${SLURM_ARRAY_TASK_ID}/N.
#
#   STAGE=extract  : env nymeria_plus, CPU-bound (GPU idle but partition is GPU-only)
#   STAGE=vad      : env audio, FireRedVAD
#   STAGE=separate : env audio, SAM-Audio (needs gated weights + hf auth login)
#   STAGE=verify   : env audio, second VAD pass + filtering
#
# Submit (32-way shards):
#   sbatch --array=0-31 --export=ALL,STAGE=extract,N=32 pipeline.sbatch
#   sbatch --array=0-31 --export=ALL,STAGE=vad,N=32      pipeline.sbatch
#   sbatch --array=0-31 --export=ALL,STAGE=separate,N=32 pipeline.sbatch
#   sbatch --array=0-0  --export=ALL,STAGE=verify,N=1    pipeline.sbatch   # N=1 -> writes cleaned.jsonl + summary
set -euo pipefail
cd /home/jungbin_cho/nymeria_kimodo_pipeline/audio
source /home/jungbin_cho/miniforge3/etc/profile.d/conda.sh
mkdir -p /weka/jungbin/nymeriaplus_audio/logs/slurm

SHARD="${SLURM_ARRAY_TASK_ID:-0}/${N:-1}"
case "${STAGE:?set STAGE=extract|vad|separate|verify}" in
  extract)  conda activate nymeria_plus
            python extract_audio.py --shard "$SHARD" ;;
  vad)      conda activate audio
            python vad.py --backend "${BACKEND:-firered}" --shard "$SHARD" ;;
  separate) conda activate audio
            python separate.py --description "${DESC:-speech}" --shard "$SHARD" ;;
  verify)   conda activate audio
            python verify_and_filter.py --backend "${BACKEND:-firered}" --shard "$SHARD" ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
