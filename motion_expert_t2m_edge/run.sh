#!/usr/bin/env bash
set -euo pipefail

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
# restored_env.sh installs a Nano default; select Edge after sourcing it.
# EDGE_DCP_ROOT is the explicit override for an alternative Edge DCP location.
export BASE_CHECKPOINT_PATH=${EDGE_DCP_ROOT:-${WEKA_ROOT}/cosmos3_edge_dcp}

# shellcheck source=../motion_expert_joint_attention/use_camera_head_v1.sh
source "${REPO_ROOT}/motion_expert_joint_attention/use_camera_head_v1.sh"

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
