#!/usr/bin/env bash
# Fresh Edge Phase-1 task ablation, based on the exclude5 Yonsei 20260905 run.
# Only forward_dynamics and inverse_dynamics remain, with original ratio 40:25.
# Forward retains caption + first image + camera; inverse has empty caption.
# Same T97/20fps/256, global32, LoRA16/alpha32, camera heads, EMA and losses.
# Use COSMOS_SITE=yonsei or grasp inside a retained two-GPU Slurm allocation.
# Optional RESUME_CHECKPOINT must belong to this new run, not the baseline.
# Output/config.yaml and native_phase1_contract.json record the resolved recipe.
# Schedule uses the baseline's final 200k LambdaLinear configuration. Its
# historical 100k-to-200k extension is not replayed as a staged LR restart.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export REPO_ROOT
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei or grasp}"
case "${COSMOS_SITE}" in
  custom)
    source "${REPO_ROOT}/restored_env.sh"
    ;;
  yonsei)
    export WEKA_ROOT=${WEKA_ROOT:-/lustre/whwjdqls99/cosmos/weka}
    export COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/lustre/whwjdqls99/cosmos/envs/cosmos}
    export COSMOS_FRAMEWORK_EDGE_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/home/whwjdqls99/cosmos-framework-edge}
    export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs}
    export HF_HOME=${HF_HOME:-/lustre/whwjdqls99/cosmos/.cache/huggingface}
    export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
    ;;
  grasp)
    export WEKA_ROOT=${WEKA_ROOT:-/mnt/projects/ll/jungbinc/weka}
    export COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
    export COSMOS_FRAMEWORK_EDGE_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
    export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-${WEKA_ROOT}/cosmos_motion_ft_runs}
    export HF_HOME=${HF_HOME:-/mnt/projects/ll/jungbinc/.cache/huggingface}
    ;;
  *) printf 'Unsupported COSMOS_SITE=%s\n' "${COSMOS_SITE}" >&2; exit 2 ;;
esac

HOST=$(hostname -s)
"${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/compute_admission.py"
EXECUTION_ID=${SLURM_JOB_ID:-${NATIVEP1_EXECUTION_ID:?Foundary execution ID required}}
NUM_GPUS=${NATIVEP1_NUM_GPUS:-2}
case "$NUM_GPUS" in 1|2) ;; *) printf 'Expected one or two GPUs\n' >&2; exit 2;; esac

export PATH="${COSMOS_ENV_ROOT}/bin:${CUDA_HOME:+${CUDA_HOME}/bin:}${PATH}"
export LD_LIBRARY_PATH=${NATIVEP1_LD_LIBRARY_PATH:-}
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/nymeria_world:${COSMOS_FRAMEWORK_EDGE_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_PROJECT=cosmos3_camera
export WANDB_DATA_DIR="${IMAGINAIRE_OUTPUT_ROOT}/.wandb_runtime/data"
export WANDB_CACHE_DIR="${IMAGINAIRE_OUTPUT_ROOT}/.wandb_runtime/cache"
export TRITON_CACHE_DIR="${IMAGINAIRE_OUTPUT_ROOT}/.triton_phase1_${EXECUTION_ID}"
mkdir -p "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${TRITON_CACHE_DIR}"

export EDGE_MODEL_ROOT="${WEKA_ROOT}/Cosmos3-Edge"
export BASE_CHECKPOINT_PATH="${WEKA_ROOT}/cosmos3_edge_dcp"
export WAN_VAE_PATH="${WEKA_ROOT}/wan22_vae/Wan2.2_VAE.pth"
export NYMERIA_LATENT_ROOT="${WEKA_ROOT}/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256"
export NATIVEP1_MODEL_FAMILY=edge
export NYMERIA_NUM_FRAMES=97 NYMERIA_RESOLUTION=256 NYMERIA_MODE=mixture
export NYMERIA_DROP_MODES=policy,image2video
export NYMERIA_REPLACE_STANDALONE_C=1 NYMERIA_STANDALONE_C_SUBJECT=camera_wearer
export NYMERIA_EXCLUDE_ORIGINAL_VIDEO_NAMES=brittney_goodwin,austin_lopez,kim_wilkins,kevin_davis,devon_norris
export NYMERIA_QUALITY_FILTER=
export NATIVEP1_ADAPTATION_MODE=${NATIVEP1_ADAPTATION_MODE:-global_lora}
case "$NATIVEP1_ADAPTATION_MODE" in
  global_lora) ;;
  action_only)
    case "${NATIVEP1_RUN_NAME:-}" in
      *_action_only_*) ;;
      *) printf 'action_only requires a distinct action_only run name\n' >&2; exit 2 ;;
    esac
    ;;
  action_video_projections)
    case "${NATIVEP1_RUN_NAME:-}" in
      *_action_video_projections_*) ;;
      *) printf 'action_video_projections requires a distinct run name\n' >&2; exit 2 ;;
    esac
    ;;
  *) printf 'Unsupported two-task adaptation: %s\n' "$NATIVEP1_ADAPTATION_MODE" >&2; exit 2 ;;
esac
export NATIVEP1_PREFIX_LENGTHS=${NATIVEP1_PREFIX_LENGTHS:-1}
if [[ "${NATIVEP1_PREFIX_LENGTHS}" != 1 && -z "${NATIVEP1_RUN_NAME:-}" ]]; then
  printf '%s\n' 'A non-default prefix requires a distinct NATIVEP1_RUN_NAME.' >&2
  exit 2
fi
export NATIVEP1_CLIPS_PER_GPU=${NATIVEP1_CLIPS_PER_GPU:-16}
[[ "$NATIVEP1_CLIPS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid clips per GPU\n' >&2; exit 2; }
export NATIVEP1_EXPECTED_LATENT_HW=16 NATIVEP1_EXPECTED_IMAGE_HW=256
export NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=1
export NATIVEP1_LORA_LR=5e-5 NATIVEP1_ACTION_LR_MULT=4
export NATIVEP1_ACTION_LOSS_WEIGHT=10 NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=0
export NATIVEP1_AUTO_EVAL=0 NATIVEP1_AUTO_EVAL_FULL71=0
export COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES=15
unset NYMERIA_FULL_FT NYMERIA_MAX_SAMPLES NATIVEP1_SHIFT_OVERRIDE NATIVEP1_PREFIX_SAMPLING_WEIGHTS
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT

RUN_NAME=${NATIVEP1_RUN_NAME:-edge_phase1_T97_20fps_2gpu_bs16_camera_wearer_lora_exclude5_fd_invdyn_200k_${COSMOS_SITE}_20261015}
RUN_DIR="${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world_edge/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
exec 9>"${RUN_DIR}/training.lock"
flock -n 9 || { printf '%s\n' 'This run already has a launcher.' >&2; exit 2; }
export TB_LOG_DIR="${RUN_DIR}/tensorboard"

LOAD_PATH=${RESUME_CHECKPOINT:-${BASE_CHECKPOINT_PATH}}
RESUME=false
SKIP_KEYS='[net_ema.]'
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  case "${RESUME_CHECKPOINT}" in
    "${RUN_DIR}"/checkpoints/iter_*) ;;
    *) printf '%s\n' 'Resume must reference this ablation, not another model.' >&2; exit 2 ;;
  esac
  RESUME=true
  SKIP_KEYS='[]'
elif [[ -d "${RUN_DIR}/checkpoints" ]]; then
  printf '%s\n' 'Existing checkpoint directory: set RESUME_CHECKPOINT explicitly.' >&2
  exit 2
fi
for REQUIRED in "${LOAD_PATH}/model/.metadata" "${EDGE_MODEL_ROOT}/modular_model_index.json" \
  "${WAN_VAE_PATH}" "${NYMERIA_LATENT_ROOT}/latent_cache_contract.json" \
  "${NYMERIA_LATENT_ROOT}/latent_cache_complete.json"; do
  [[ -e "${REQUIRED}" ]] || { printf 'Missing training artifact: %s\n' "${REQUIRED}" >&2; exit 2; }
done

printf '[phase1-two-task] job=%s node=%s output=%s\n' "${EXECUTION_ID}" "${HOST}" "${RUN_DIR}"
printf '[phase1-two-task] initialize=%s resume=%s global_batch=%s (%sx%s), accum=1\n' "${LOAD_PATH}" "${RESUME}" "$((NUM_GPUS * NATIVEP1_CLIPS_PER_GPU))" "$NUM_GPUS" "$NATIVEP1_CLIPS_PER_GPU"
printf '[phase1-two-task] adaptation=%s; forward:inverse=40:25; exclude5; captions_disabled=%s.\n' "$NATIVEP1_ADAPTATION_MODE" "${NATIVEP1_DISABLE_CAPTIONS:-0}"
printf '[phase1-two-task] forward RGB conditioning prefix=%s; inverse still observes all97 frames.\n' "${NATIVEP1_PREFIX_LENGTHS}"
printf '%s\n' '[phase1-two-task] 200k steps; effective camera LR 8e-5 (LoRA LR 2e-5 when enabled), 500-step plateau then linear decay to zero.'
printf '[phase1-two-task] checkpoint every5k plus rolling15min; in-process fixed W&B previews=%s.\n' "${NATIVEP1_WANDB_PREVIEWS:-0}"
cd "${COSMOS_FRAMEWORK_EDGE_ROOT}"
exec "${COSMOS_ENV_ROOT}/bin/torchrun" --standalone --nproc_per_node="${NUM_GPUS}" \
  "${REPO_ROOT}/native_phase_training/run_latent_train.py" \
  --sft-toml "${REPO_ROOT}/native_phase_training/world_camera_nymeria_latent_edge.toml" \
  job.name="${RUN_NAME}" job.wandb_mode=online \
  trainer.max_iter=200000 trainer.grad_accum_iter=1 trainer.logging_iter=50 \
  scheduler.cycle_lengths='[200000]' scheduler.warm_up_steps='[500]' \
  scheduler.f_start='[0.4]' scheduler.f_max='[0.4]' scheduler.f_min='[0.0]' \
  checkpoint.save_iter=5000 checkpoint.load_path="${LOAD_PATH}" \
  checkpoint.load_training_state="${RESUME}" checkpoint.strict_resume="${RESUME}" \
  checkpoint.keys_to_skip_loading="${SKIP_KEYS}" checkpoint.keys_not_to_resume='[]' \
  model.config.compile.enabled=false model.config.ema.enabled=true \
  model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.parallelism.data_parallel_replicate_degree="${NUM_GPUS}" \
  model.config.parallelism.context_parallel_shard_degree=1 \
  model.config.parallelism.cfg_parallel_shard_degree=1
