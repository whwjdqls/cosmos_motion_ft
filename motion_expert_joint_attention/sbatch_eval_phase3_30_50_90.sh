#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_p3
#SBATCH --array=0-2
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-eval-p3-%A_%a.out

set -euo pipefail

RUN_ROOT=/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_7task
WINDOWS=/weka/jungbin/cosmos_motion_ft_runs/joint_attention/full71_windows.json
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention

STEPS=(030000 050000 090000)
STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
CKPT="$RUN_ROOT/ckpt_step${STEP}.pt"
OUT="$RUN_ROOT/eval_all_step${STEP}_full71"

echo "[eval_p3] node=$(hostname) step=$STEP ckpt=$CKPT out=$OUT"
bash "$D/run_eval.sh" "$CKPT" 71 inverse_dynamics forward_dynamics policy video2motion -- \
  --windows_json "$WINDOWS" \
  --out_dir "$OUT" \
  --motion_viz_limit 5
