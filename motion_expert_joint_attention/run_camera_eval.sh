#!/usr/bin/env bash
# =============================================================================
# run_camera_eval.sh — eval + viz the CAMERA tasks of a Phase-1 joint-attention
# checkpoint (inverse_dynamics / forward_dynamics / policy), single GPU on a node.
#
# Mirrors run.sh's invariant env exactly (cosmos conda env, scrubbed
# LD_LIBRARY_PATH, cosmos-framework cwd, PYTHONPATH over framework + both
# motion-expert repos), then adds WAN_VAE_PATH for the Wan2.2-VAE latent->pixel
# decode that eval_camera.py uses for the FD/policy video output.
#
# Usage:
#     bash run_camera_eval.sh <ckpt.pt> [n_windows=8] [extra eval_camera.py args...]
#
# NOTE: --num_frames and the latent root DEFAULT from the CHECKPOINT's args
# (ckpt["args"]["T"] + train.py's per-T cache root, e.g. joint_latents_T97 for
# a T=97 run) -- no need to pass them for non-default-T checkpoints. Explicit
# --num_frames / --latent_root in the extra args still override.
#
# Examples:
#     # Eval a Phase-1 checkpoint (metrics + generated video + viz):
#     ssh a3ultravis-a3ultranodeset-0
#     bash run_camera_eval.sh \
#         /weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/latest.pt 8
#
#     # Smoke on the BASE model (no ckpt yet) -- prove the path runs end to end:
#     bash run_camera_eval.sh none 1
#
#     # inverse-dynamics metrics only (skip the heavy VAE video decode):
#     bash run_camera_eval.sh <ckpt.pt> 8 --tasks inverse_dynamics --no_video
# =============================================================================
set -euo pipefail

CKPT="${1:?usage: run_camera_eval.sh <ckpt.pt|none> [n_windows] [extra args...]}"
N="${2:-8}"
# drop $1 (ckpt) and $2 (n) if present; the rest are passed straight to eval_camera.py.
[[ $# -ge 1 ]] && shift
[[ $# -ge 1 ]] && shift
EXTRA=("$@")

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../restored_env.sh
source "${REPO_ROOT}/restored_env.sh"
D="${REPO_ROOT}/motion_expert_joint_attention"

# --- env hygiene (load-bearing for the cosmos CUDA stack) ---
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH

# --- cwd: cosmos-framework root (relative QWEN_JSON / asset paths resolve here) ---
cd "${COSMOS_FRAMEWORK_ROOT}"

# --- import paths: framework + both motion-expert repos + nymeria_world (viz/metric reuse) ---
export PYTHONPATH

PY="${COSMOS_PYTHON}"

# --- "none"/missing ckpt -> smoke on the BASE model (eval_camera.py handles the fallback) ---
CKPT_ARG=(--ckpt "$CKPT")
if [[ "$CKPT" == "none" || ! -f "$CKPT" ]]; then
  echo "[run_camera_eval] ckpt '$CKPT' missing -> BASE-model smoke (no trained gen delta)"
  CKPT_ARG=()
fi

echo "[run_camera_eval] node=$(hostname) ckpt=$CKPT n=$N extra=${EXTRA[*]:-}"
"$PY" "$D/eval_camera.py" "${CKPT_ARG[@]}" --n "$N" "${EXTRA[@]}"

# --- resolve the eval output dir the same way eval_camera.py does, then viz ---
if [[ -f "$CKPT" ]]; then
  EVAL_DIR="$(dirname "$(readlink -f "$CKPT")")/camera_eval"
else
  EVAL_DIR="${RUN_ROOT}/camera_eval_base"
fi

echo "[run_camera_eval] viz -> $EVAL_DIR/viz"
"$PY" "$D/viz_camera.py" --eval_dir "$EVAL_DIR" --tag "$(basename "$(dirname "$EVAL_DIR")")"

echo "[run_camera_eval] DONE. metrics: $EVAL_DIR/invdyn_metrics.json ; viz: $EVAL_DIR/viz"
