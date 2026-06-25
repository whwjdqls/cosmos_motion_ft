#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --job-name=bones_export
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-export-%j.out

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate kimodo
export PYTHONPATH=/home/jungbin_cho/kimodo_open:$PYTHONPATH
cd /home/jungbin_cho/kimodo_open

OUT="${1:-/weka/jungbin/seed/cosmos_text_motion_subset}"
MAX="${2:-4000}"

echo "[sbatch_export] host=$(hostname) out=$OUT max=$MAX"
python /home/jungbin_cho/cosmos_motion_ft/export_bones_seed_text_motion.py \
    --out "$OUT" --max-samples "$MAX" --seed 0
echo "[sbatch_export] done"
