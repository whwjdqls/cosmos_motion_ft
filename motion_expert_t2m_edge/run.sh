#!/usr/bin/env bash
set -euo pipefail

# Shared defaults for Grasp and Yonsei; cache locations remain site-specific.
# An explicit zero keeps the synchronous path available for comparisons.
if [[ -n "${COSMOS_EDGE_CONDITION_CACHE_ROOT:-}" ]]; then
  export COSMOS_EDGE_CONDITION_PREFETCH_THREADS="${COSMOS_EDGE_CONDITION_PREFETCH_THREADS:-4}"
  export COSMOS_EDGE_CONDITION_ASYNC_WRITES="${COSMOS_EDGE_CONDITION_ASYNC_WRITES:-1}"
fi

EDGE_T2M_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${EDGE_T2M_DIR}/.." && pwd)}

if [[ "${COSMOS_SITE:-}" == yonsei ]]; then
    export PROJECT_ROOT=${PROJECT_ROOT:-/lustre/whwjdqls99/cosmos}
    export RUN_ROOT=${RUN_ROOT:-/lustre/whwjdqls99/cosmos/weka/cosmos_motion_ft_runs}
    export COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/lustre/whwjdqls99/cosmos/envs/cosmos}
    export COSMOS_FRAMEWORK_EDGE_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/home/whwjdqls99/cosmos-framework-edge}
fi

# shellcheck source=../restored_env.sh
source "${REPO_ROOT}/restored_env.sh"
export COSMOS_FRAMEWORK_EDGE_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
export COSMOS_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT}
export EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-${WEKA_ROOT}/Cosmos3-Edge}
# restored_env.sh exposes the historical Nano base as the repository-wide
# default. This Edge-only wrapper must select the converted Edge DCP instead.
# Accept the Yonsei EDGE_DCP_ROOT alias; EDGE_BASE_CHECKPOINT_PATH takes priority.
export EDGE_BASE_CHECKPOINT_PATH=${EDGE_BASE_CHECKPOINT_PATH:-${EDGE_DCP_ROOT:-${COSMOS3_EDGE_DCP_ROOT}}}
export BASE_CHECKPOINT_PATH=${EDGE_BASE_CHECKPOINT_PATH:-${COSMOS3_EDGE_DCP_ROOT}}

EDGE_MOTION_REPRESENTATION=${EDGE_MOTION_REPRESENTATION:-camera_head_recanonicalization_v1}
case "${EDGE_MOTION_REPRESENTATION}" in
    camera_head_recanonicalization_v1)
        # shellcheck source=../motion_expert_joint_attention/use_camera_head_v1.sh
        source "${REPO_ROOT}/motion_expert_joint_attention/use_camera_head_v1.sh"
        ;;
    legacy_uniego283)
        export NYMERIA_UNIEGO_ROOT="${WEKA_ROOT}/nymeriaplus_kimodo_proportional/uniego_rep"
        export MOTION_STATS_MEAN="${REPO_ROOT}/motion_expert/stats/uniego283_mean.npy"
        export MOTION_STATS_STD="${REPO_ROOT}/motion_expert/stats/uniego283_std.npy"
        unset MOTION_STATS_SUMMARY
        export NYMERIA_MOTION_REPRESENTATION=legacy_uniego283
        ;;
    *)
        echo "[edge-t2m] unsupported EDGE_MOTION_REPRESENTATION=${EDGE_MOTION_REPRESENTATION}" >&2
        exit 2
        ;;
esac
export EDGE_MOTION_REPRESENTATION

export LD_LIBRARY_PATH=
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="${EDGE_T2M_DIR}:${REPO_ROOT}/motion_expert_joint_attention:${REPO_ROOT}/nymeria_world:${REPO_ROOT}:${COSMOS_FRAMEWORK_EDGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for required in \
    "${COSMOS_FRAMEWORK_EDGE_ROOT}/cosmos_framework" \
    "${EDGE_MODEL_ROOT}/model.safetensors.index.json" \
    "${BASE_CHECKPOINT_PATH}/model/.metadata"; do
    if [[ ! -e "${required}" ]]; then
        echo "[edge-t2m] missing required artifact: ${required}" >&2
        exit 1
    fi
done

cd "${COSMOS_FRAMEWORK_EDGE_ROOT}"
exec "${COSMOS_PYTHON}" "$@"
