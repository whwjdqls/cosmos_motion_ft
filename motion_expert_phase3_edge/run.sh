#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${HERE}/.." && pwd)
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei or grasp}"
if [[ "${COSMOS_SITE}" == yonsei ]]; then
    # Match the working Yonsei Edge launchers. Transformer Engine locates
    # libcudart through CUDA_HOME even though the shared wrapper clears LD_LIBRARY_PATH.
    export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == *login* || "$(hostname -s)" == cs-gpu-01 || "$(hostname -s)" == node37 ]]; then
    printf '%s\n' 'Phase-3 compute must run inside an owned allocation, never on login/cs-gpu-01/node37.' >&2
    exit 2
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export EDGE_MOTION_REPRESENTATION=camera_head_recanonicalization_v1
exec bash "${REPO}/motion_expert_t2m_edge/run.sh" "$@"
