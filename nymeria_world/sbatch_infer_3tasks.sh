#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --job-name=cam_infer
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-caminfer-%j.out

# Sample the 3 tasks (inverse/forward dynamics + policy) from a MERGED full DCP, on a dedicated GPU.
# Env: MERGED (merged DCP dir), EVAL (input/output root), NUM_FRAMES (default 97).
set -e
MERGED=${MERGED:?set MERGED}
EVAL=${EVAL:-/weka/jungbin/cosmos_motion_ft_runs/nymeria_eval}
NUM_FRAMES=${NUM_FRAMES:-97}
TAG=${TAG:-merged}
RUN=/home/jungbin_cho/cosmos_motion_ft/nymeria_world/run_infer_merged.sh
echo "[infer] node=$(hostname) merged=$MERGED tag=$TAG"

for task in invdyn fd policy; do
  echo "=== $task ==="
  bash "$RUN" "$MERGED" "$EVAL/${task}_input.jsonl" "$EVAL/${task}_out_${TAG}" "$NUM_FRAMES" 0 || echo "[$task FAILED]"
done
echo "[infer] done"
