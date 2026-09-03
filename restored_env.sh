#!/usr/bin/env bash
# Runtime paths for the restored server. Source this file before training or sampling.

_cosmos_motion_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/projects/ll/jungbinc}"
export REPO_ROOT="${REPO_ROOT:-${_cosmos_motion_repo}}"
export WEKA_ROOT="${WEKA_ROOT:-${PROJECT_ROOT}/weka}"
export RUN_ROOT="${RUN_ROOT:-${WEKA_ROOT}/cosmos_motion_ft_runs}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export COSMOS_FRAMEWORK_ROOT="${COSMOS_FRAMEWORK_ROOT:-${PROJECT_ROOT}/cosmos-framework}"
export COSMOS_ENV_ROOT="${COSMOS_ENV_ROOT:-${PROJECT_ROOT}/miniconda3/envs/cosmos}"
export COSMOS_PYTHON="${COSMOS_PYTHON:-${COSMOS_ENV_ROOT}/bin/python3.13}"

export COSMOS3_NANO_SNAPSHOT="${COSMOS3_NANO_SNAPSHOT:-${HUGGINGFACE_HUB_CACHE}/models--nvidia--Cosmos3-Nano/snapshots/7a312c868bcce8e40b3eb40861300a9d0ba3fde1}"
export COSMOS_TEXT_TOKENIZER_PATH="${COSMOS_TEXT_TOKENIZER_PATH:-${COSMOS3_NANO_SNAPSHOT}/text_tokenizer}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-${WEKA_ROOT}/wan22_vae/Wan2.2_VAE.pth}"
export COSMOS_PHASE1_GEN_INIT="${COSMOS_PHASE1_GEN_INIT:-${RUN_ROOT}/portable/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_iter100000_ema_gen_delta.pt}"

export PATH="${COSMOS_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${COSMOS_FRAMEWORK_ROOT}:${REPO_ROOT}:${REPO_ROOT}/motion_expert:${REPO_ROOT}/motion_expert_joint_attention:${REPO_ROOT}/nymeria_world${PYTHONPATH:+:${PYTHONPATH}}"

unset _cosmos_motion_repo

# BEGIN EXPLICIT COSMOS SITE SELECTOR
# Keep this block last: it overrides historical machine defaults above.
# Select with: export COSMOS_SITE=yonsei  OR  export COSMOS_SITE=grasp
case "${COSMOS_SITE:-}" in
  yonsei)
    _cosmos_repo_root=/home/whwjdqls99/cosmos_motion_ft
    _cosmos_storage_root=/lustre/whwjdqls99/cosmos
    _cosmos_weka_root=/lustre/whwjdqls99/cosmos/weka
    ;;
  grasp)
    _cosmos_repo_root=/home/jungbinc/cosmos_motion_ft
    _cosmos_storage_root=/mnt/projects/ll/jungbinc
    _cosmos_weka_root=/mnt/projects/ll/jungbinc
    ;;
  "")
    printf '%s\n' 'restored_env.sh: COSMOS_SITE must be set to yonsei or grasp' >&2
    return 2 2>/dev/null || exit 2
    ;;
  *)
    printf 'restored_env.sh: unsupported COSMOS_SITE=%s; expected yonsei or grasp\n' "$COSMOS_SITE" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

export COSMOS_REPO_ROOT="$_cosmos_repo_root"
export COSMOS_STORAGE_ROOT="$_cosmos_storage_root"
export WEKA_ROOT="$_cosmos_weka_root"
export COSMOS_RUNS_ROOT="$WEKA_ROOT/cosmos_motion_ft_runs"
export HF_HOME="$COSMOS_STORAGE_ROOT/.cache/huggingface"
export TORCH_HOME="$COSMOS_STORAGE_ROOT/.cache/torch"
export WAN_VAE_PATH="$WEKA_ROOT/wan22_vae/Wan2.2_VAE.pth"
export COSMOS3_EDGE_ROOT="$WEKA_ROOT/Cosmos3-Edge"
export COSMOS3_NANO_ROOT="$WEKA_ROOT/Cosmos3-Nano"
export COSMOS3_EDGE_DCP_ROOT="$WEKA_ROOT/cosmos3_edge_dcp"
export BASE_CHECKPOINT_PATH="$WEKA_ROOT/cosmos3_nano_dcp"

unset _cosmos_repo_root _cosmos_storage_root _cosmos_weka_root
# END EXPLICIT COSMOS SITE SELECTOR
