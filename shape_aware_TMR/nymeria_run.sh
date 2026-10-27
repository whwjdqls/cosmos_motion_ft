#!/usr/bin/env bash
# Site-portable environment for Nymeria evaluator preparation, training, and scoring.
set -euo pipefail
: "${SLURM_JOB_ID:?Run on an owned Slurm compute allocation}"
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei or grasp}"
host=$(hostname -s)
case "$host" in *login*|cs-gpu-01) printf 'Compute node required: %s\n' "$host" >&2; exit 2;; esac
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_ROOT
case "$COSMOS_SITE" in
    custom)
        source "$REPO_ROOT/restored_env.sh"
        export COSMOS_FRAMEWORK_ROOT=$COSMOS_FRAMEWORK_EDGE_ROOT
        ;;
    yonsei)
        export PROJECT_ROOT=/lustre/whwjdqls99/cosmos
        export COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-$PROJECT_ROOT/envs/cosmos}
        export COSMOS_PYTHON=${COSMOS_PYTHON:-$COSMOS_ENV_ROOT/bin/python3.13}
        export COSMOS_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_ROOT:-/home/whwjdqls99/cosmos-framework-edge}
        export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/scratch2/whwjdqls99/cosmos/cache/huggingface_nano_viz/hub}
        export NYMERIA_TMR_TEXT_HUB=${NYMERIA_TMR_TEXT_HUB:-/scratch2/whwjdqls99/cosmos/cache/phase2_c45_hf/hub}
        export CUDA_HOME=${CUDA_HOME:-/opt/ohpc/pub/apps/cuda/12.8}
        ;;
    grasp) export PROJECT_ROOT=/mnt/projects/ll/jungbinc;;
    *) printf 'Unsupported COSMOS_SITE=%s\n' "$COSMOS_SITE" >&2; exit 2;;
esac
source "$REPO_ROOT/restored_env.sh"
# Tokenizer/config and model/PEFT loaders consult different HF cache variables.
# Keep all of them on the same existing LLM2Vec cache, after site defaults.
text_hub=${NYMERIA_TMR_TEXT_HUB:-${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}}
export HUGGINGFACE_HUB_CACHE="$text_hub" HF_HUB_CACHE="$text_hub"
export HUGGINGFACE_CACHE_DIR="$text_hub"
unset TRANSFORMERS_CACHE
printf '[nymeria-env] text_hub=%s\n' "$text_hub"
export SHAPE_TMR_BUNDLE=${SHAPE_TMR_BUNDLE:-$WEKA_ROOT/shape_aware_motion_eval_c45_20260715}
export KIMODO_ROOT=${KIMODO_ROOT:-$SHAPE_TMR_BUNDLE/code/kimodo_open}
export PYTHONPATH="$REPO_ROOT/shape_aware_TMR:$KIMODO_ROOT:$REPO_ROOT:$REPO_ROOT/motion_expert_joint_attention:$REPO_ROOT/motion_expert${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
if [[ "$COSMOS_SITE" == yonsei ]]; then
    export COSMOS_MOTION_CACHE_ROOT=${COSMOS_MOTION_CACHE_ROOT:-/scratch2/whwjdqls99/cosmos/cache/uniego_arrays_v1}
    export COSMOS_MOTION_CACHE_SOURCE_ROOT=${COSMOS_MOTION_CACHE_SOURCE_ROOT:-$WEKA_ROOT}
fi
exec "${SHAPE_TMR_PYTHON:-$COSMOS_PYTHON}" "$@"
