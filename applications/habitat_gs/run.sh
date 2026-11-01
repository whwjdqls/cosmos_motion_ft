#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export REPO_ROOT
export COSMOS_SITE=${COSMOS_SITE:-yonsei}
case "$COSMOS_SITE" in
  yonsei)
    export WEKA_ROOT=${WEKA_ROOT:-/lustre/whwjdqls99/cosmos/weka}
    export COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/lustre/whwjdqls99/cosmos/envs/cosmos}
    export COSMOS_FRAMEWORK_EDGE_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/home/whwjdqls99/cosmos-framework-edge}
    export APP_STORAGE=${APP_STORAGE:-/scratch2/whwjdqls99/cosmos/applications/habitat_gs}
    export CUDA_HOME=${CUDA_HOME:-/opt/ohpc/pub/apps/cuda/12.8}
    export HF_HOME=${HF_HOME:-/scratch2/whwjdqls99/hugginface_cache}
    export TORCH_HOME=${TORCH_HOME:-/lustre/whwjdqls99/cosmos/.cache/torch}
    export COSMOS_FFMPEG_LIBRARY_PATH=${COSMOS_FFMPEG_LIBRARY_PATH:-/home/whwjdqls99/.local/lib/cosmos-ffmpeg8-pyav_20260907:$COSMOS_ENV_ROOT/lib/python3.13/site-packages/av.libs}
    ;;
  grasp|custom)
    : "${WEKA_ROOT:?}" "${COSMOS_ENV_ROOT:?}" "${COSMOS_FRAMEWORK_EDGE_ROOT:?}" "${APP_STORAGE:?}"
    ;;
  *) exit 2;;
esac
export COSMOS_PYTHON="$COSMOS_ENV_ROOT/bin/python"
export EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-$WEKA_ROOT/Cosmos3-Edge}
export BASE_CHECKPOINT_PATH=${BASE_CHECKPOINT_PATH:-$WEKA_ROOT/cosmos3_edge_dcp}
export WAN_VAE_PATH=${WAN_VAE_PATH:-$WEKA_ROOT/wan22_vae/Wan2.2_VAE.pth}
export NATIVEP1_PREFIX_LENGTHS=1 NATIVEP1_EVAL_PREFIX_LENGTHS=1
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$COSMOS_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nymeria_world:$COSMOS_FRAMEWORK_EDGE_ROOT"
NVIDIA_LIB=$COSMOS_ENV_ROOT/lib/python3.13/site-packages/nvidia
export LD_LIBRARY_PATH="${COSMOS_FFMPEG_LIBRARY_PATH:+$COSMOS_FFMPEG_LIBRARY_PATH:}$NVIDIA_LIB/npp/lib:$NVIDIA_LIB/cuda_nvrtc/lib:$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/nvjitlink/lib"
export NATIVEP1_MODEL_FAMILY=edge NATIVEP1_AUTO_EVAL=0 NATIVEP1_AUTO_EVAL_FULL71=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset NATIVEP1_ADAPTATION_MODE NYMERIA_DROP_MODES NATIVEP1_SHIFT_OVERRIDE NATIVEP1_INFERENCE_SHIFT
cd "$REPO_ROOT"
exec "$COSMOS_PYTHON" -m applications.habitat_gs."${1:?Use profile, calibrate, or rollout}" "${@:2}"
