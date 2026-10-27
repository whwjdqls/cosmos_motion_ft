#!/usr/bin/env bash
# Sourced only for COSMOS_SITE=custom. Does not select a CUDA installation.
_cosmos_custom_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_cosmos_custom_exports=$("${COSMOS_CONFIG_PYTHON:-/usr/bin/python3}" -S "${_cosmos_custom_dir}/custom_paths.py") || { return 2 2>/dev/null || exit 2; }
eval "${_cosmos_custom_exports}"
export PATH="${COSMOS_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${COSMOS_FRAMEWORK_ROOT}:${REPO_ROOT}:${REPO_ROOT}/motion_expert:${REPO_ROOT}/motion_expert_joint_attention:${REPO_ROOT}/nymeria_world:${KIMODO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset _cosmos_custom_dir _cosmos_custom_exports
