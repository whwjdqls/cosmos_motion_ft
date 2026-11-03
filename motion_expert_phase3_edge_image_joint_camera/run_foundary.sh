#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
site_config=${COSMOS_SITE_CONFIG:-$repo/.site.local.sh}
if [[ -f "$site_config" ]]; then source "$site_config"; fi
: "${COSMOS_FRAMEWORK_EDGE_ROOT:?Source the node site configuration}"
export PYTHONPATH="$repo/motion_expert_t2m_edge:$repo/motion_expert_joint_attention:$repo/nymeria_world:$repo:$COSMOS_FRAMEWORK_EDGE_ROOT:$repo/.deps/kimodo_open${PYTHONPATH:+:$PYTHONPATH}"
export EDGE_ARTIFACT_SOURCE_SITE=${EDGE_ARTIFACT_SOURCE_SITE:-yonsei}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=
cd "$COSMOS_FRAMEWORK_EDGE_ROOT"
exec "${COSMOS_PYTHON:-python}" -m motion_expert_phase3_edge_image_joint_camera.launch "$@"
