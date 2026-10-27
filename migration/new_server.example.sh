#!/usr/bin/env bash
# Copy to .site.local.sh, edit paths, then source it from the repository root.
# No credentials here. Authenticate Hugging Face / W&B separately.
export COSMOS_SITE=custom
export COSMOS_STORAGE_ROOT=/path/to/persistent/cosmos
export COSMOS_ENV_ROOT=/path/to/envs/cosmos
# Optional: split run/checkpoint storage from read-mostly datasets and weights.
export COSMOS_RUNS_ROOT=/path/to/runs/cosmos_motion_ft_runs
# The checkout location is detected automatically; dependencies use REPO/.deps.
# export COSMOS_DEPS_ROOT=/path/to/source/dependencies
# Set CUDA_HOME to this server's installation, not Yonsei's /opt/ohpc path.
# export CUDA_HOME=/usr/local/cuda-12.8
# export WANDB_ENTITY=your-team
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/restored_env.sh"
