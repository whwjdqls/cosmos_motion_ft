#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — ONE command to run the full 7-task eval + viz harness for a
# joint-attention checkpoint (single GPU on a node, cosmos env).
#
# Dispatches (eval_all.py):
#   * camera tasks  (inverse_dynamics/forward_dynamics/policy) -> eval_camera + viz_camera
#   * motion-recon  (video2motion; text2motion/textimg2motion flagged generative)
#                   -> sample -> eval_motion_recon metric -> skeleton mp4
#   * video-gen     (motimg2video) -> sample -> Wan2.2-VAE decode -> GT|gen mp4
# T / latent_root / motion objective are resolved FROM the checkpoint's saved args.
#
# Usage:
#     bash run_eval.sh <ckpt.pt|none> [n_windows=8] [task ...] [-- extra eval_all.py args]
#
# Examples:
#     # ALL 7 tasks on a checkpoint (metrics JSONs + viz gallery + summary table):
#     ssh a3ultravis-a3ultranodeset-0
#     bash run_eval.sh /weka/jungbin/cosmos_motion_ft_runs/ja_7task_full/latest.pt 8
#
#     # BASE-model smoke (no ckpt): proves sample->decode->metric->viz end to end:
#     bash run_eval.sh none 2 inverse_dynamics video2motion motimg2video text2motion
#
#     # motion tasks only, x0-motion run:
#     bash run_eval.sh /weka/.../ja_t2m_x0_T200/ckpt_step010000.pt 8 video2motion text2motion
# =============================================================================
set -euo pipefail

CKPT="${1:?usage: run_eval.sh <ckpt.pt|none> [n] [task ...] [-- extra args]}"
N="${2:-8}"
[[ $# -ge 1 ]] && shift
[[ $# -ge 1 ]] && shift
# Remaining positionals: task names until an optional "--", then extra eval_all.py args.
TASKS=()
EXTRA=()
seen_dd=0
for a in "$@"; do
  if [[ "$a" == "--" ]]; then seen_dd=1; continue; fi
  if [[ $seen_dd -eq 1 ]]; then EXTRA+=("$a"); else TASKS+=("$a"); fi
done

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention

# --- conda: activate the `cosmos` env (cu128) ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos

# --- env hygiene (load-bearing for the cosmos CUDA stack) ---
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH="${WAN_VAE_PATH:-/weka/jungbin/wan22_vae/Wan2.2_VAE.pth}"

# --- cwd: cosmos-framework root (relative QWEN_JSON / asset paths resolve here) ---
cd /home/jungbin_cho/cosmos-framework

# --- import paths: framework + both motion-expert repos + nymeria_world (viz/metric reuse) ---
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/motion_expert:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention:/home/jungbin_cho/cosmos_motion_ft/nymeria_world

PY=~/miniforge3/envs/cosmos/bin/python

# --- "none"/missing ckpt -> BASE-model smoke (eval_all.py handles the fallback) ---
CKPT_ARG=(--ckpt "$CKPT")
if [[ "$CKPT" == "none" || ! -f "$CKPT" ]]; then
  echo "[run_eval] ckpt '$CKPT' missing -> BASE-model smoke (no trained delta)"
  CKPT_ARG=()
fi

TASK_ARG=()
[[ ${#TASKS[@]} -gt 0 ]] && TASK_ARG=(--tasks "${TASKS[@]}")

echo "[run_eval] node=$(hostname) ckpt=$CKPT n=$N tasks=${TASKS[*]:-<all7>} extra=${EXTRA[*]:-}"
"$PY" "$D/eval_all.py" "${CKPT_ARG[@]}" --n "$N" "${TASK_ARG[@]}" "${EXTRA[@]}"

# --- resolve the eval output dir the same way eval_all.py does ---
if [[ -f "$CKPT" ]]; then
  OUT_DIR="$(dirname "$(readlink -f "$CKPT")")/eval_all"
else
  OUT_DIR=/weka/jungbin/cosmos_motion_ft_runs/eval_all_base
fi
echo "[run_eval] DONE. summary: $OUT_DIR/summary.json ; viz: $OUT_DIR/viz ; metrics under $OUT_DIR"
