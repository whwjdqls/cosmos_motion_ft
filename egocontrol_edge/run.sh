#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export REPO_ROOT
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei or grasp}"
case "$COSMOS_SITE" in
  custom)
    source "$REPO_ROOT/restored_env.sh"
    : "${EGOCONTROL_CACHE:?Set the transferred Phase3 cache explicitly}"
    : "${EGOCONTROL_INITIAL_CHECKPOINT:?Set the matching Phase1 EMA checkpoint explicitly}"
    : "${EGOCONTROL_VALIDATION_CONFIG:?Set the transferred held-out validation configuration explicitly}"
    ;;
  yonsei)
    export WEKA_ROOT=${WEKA_ROOT:-/lustre/whwjdqls99/cosmos/weka}
    export COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/lustre/whwjdqls99/cosmos/envs/cosmos}
    export COSMOS_FRAMEWORK_EDGE_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/home/whwjdqls99/cosmos-framework-edge}
    export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs}
    export EGOCONTROL_CACHE=${EGOCONTROL_CACHE:-/scratch2/whwjdqls99/cosmos/caches/phase3_edge_camhead_T97_v1}
    export HF_HOME=${HF_HOME:-/lustre/whwjdqls99/cosmos/.cache/huggingface}
    export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
    ;;
  grasp)
    export WEKA_ROOT=${WEKA_ROOT:-/mnt/projects/ll/jungbinc/weka}
    export COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
    export COSMOS_FRAMEWORK_EDGE_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
    export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$WEKA_ROOT/cosmos_motion_ft_runs}
    : "${EGOCONTROL_CACHE:?Set the Grasp Phase3 cache explicitly}"
    : "${EGOCONTROL_INITIAL_CHECKPOINT:?Set the matching Grasp Phase1 EMA checkpoint explicitly}"
    ;;
  *) exit 2;;
esac
host=$(hostname -s)
[[ -n "${SLURM_JOB_ID:-}" && "$host" != *login* && "$host" != cs-gpu-01 && "$host" != node37 ]] || exit 2
[[ "${SLURM_JOB_USER:-$(id -un)}" == "$(id -un)" ]] || exit 2
export EGOCONTROL_INITIAL_CHECKPOINT=${EGOCONTROL_INITIAL_CHECKPOINT:-$WEKA_ROOT/cosmos_motion_ft_runs/cosmos3_camera/camera_world_edge/edge_phase1_T97_20fps_4gpu_bs8_camera_wearer_lora_exclude5_yonsei_20260905/checkpoints/iter_000200000}
export EGOCONTROL_MOTION_MEAN=${EGOCONTROL_MOTION_MEAN:-$WEKA_ROOT/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_mean.npy}
export EGOCONTROL_MOTION_STD=${EGOCONTROL_MOTION_STD:-$WEKA_ROOT/cosmos_motion_ft_runs/nymeria_camera_head_recanonicalization_v1/stats/clean_calibrated_uniego283_std.npy}
export EGOCONTROL_CONDITIONING_LAYOUT=${EGOCONTROL_CONDITIONING_LAYOUT:-phase3_seven}
export EGOCONTROL_VALIDATION_CONFIG=${EGOCONTROL_VALIDATION_CONFIG:-$IMAGINAIRE_OUTPUT_ROOT/motion_expert_phase3_edge/edge_p3_p1ema200k_crosscaption200k_bridge7_T97_gbs32_wv10_yonsei_20260924/eval_heldout71_step_000200000_unipc35_20261002/configuration.json}
export EDGE_MODEL_ROOT=$WEKA_ROOT/Cosmos3-Edge BASE_CHECKPOINT_PATH=$WEKA_ROOT/cosmos3_edge_dcp
export WAN_VAE_PATH=$WEKA_ROOT/wan22_vae/Wan2.2_VAE.pth
export NYMERIA_LATENT_ROOT=$WEKA_ROOT/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256
export PATH="$COSMOS_ENV_ROOT/bin:${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
export LD_LIBRARY_PATH=
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nymeria_world:$REPO_ROOT/motion_expert_joint_attention:$COSMOS_FRAMEWORK_EDGE_ROOT${KIMODO_ROOT:+:$KIMODO_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NATIVEP1_MODEL_FAMILY=edge NYMERIA_NUM_FRAMES=97 NYMERIA_RESOLUTION=256 NYMERIA_MODE=image2video
export NYMERIA_DROP_MODES=forward_dynamics,inverse_dynamics,policy
export NYMERIA_EXCLUDE_ORIGINAL_VIDEO_NAMES= NYMERIA_QUALITY_FILTER=
export NATIVEP1_DISABLE_CAPTIONS=1 NATIVEP1_PREFIX_LENGTHS=1 NATIVEP1_ADAPTATION_MODE=global_lora
export NATIVEP1_CLIPS_PER_GPU=${EGOCONTROL_MICROBATCH:-16}
export NATIVEP1_EXPECTED_LATENT_HW=16 NATIVEP1_EXPECTED_IMAGE_HW=256 NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=1
export NATIVEP1_LORA_LR=5e-5 NATIVEP1_ACTION_LR_MULT=4 NATIVEP1_ACTION_LOSS_WEIGHT=10
export NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=0 NATIVEP1_AUTO_EVAL=0 NATIVEP1_AUTO_EVAL_FULL71=0
export COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES=15
export WANDB_ENTITY=jungbinc-upenn WANDB_PROJECT=cosmos3_camera
unset WANDB_API_KEY WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_DISABLED
unset NYMERIA_FULL_FT NATIVEP1_SHIFT_OVERRIDE NATIVEP1_PREFIX_SAMPLING_WEIGHTS
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
mode=${1:-train}
shift || true
cd "$COSMOS_FRAMEWORK_EDGE_ROOT"
if [[ "$mode" == checks ]]; then
  exec "$COSMOS_ENV_ROOT/bin/python" -u -m egocontrol_edge.acceptance "$@"
fi
if [[ "${EGOCONTROL_ACCEPTANCE:-0}" != 1 && -n "${NYMERIA_MAX_SAMPLES:-}" ]]; then
  printf 'Production must use the complete Phase3 cache; unset NYMERIA_MAX_SAMPLES.\n' >&2
  exit 2
fi
if [[ "$EGOCONTROL_CONDITIONING_LAYOUT" != phase3_seven && -z "${EGOCONTROL_RUN_NAME:-}" ]]; then
  printf 'Legacy architecture requires an explicit original EGOCONTROL_RUN_NAME.\n' >&2
  exit 2
fi
name=${EGOCONTROL_RUN_NAME:-egocontrol_edge_7cond_m2v_p1ema200k_phase1recipe_T97_gbs32_200k_${COSMOS_SITE}_20261026}
run=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_camera/egocontrol_edge/$name
mkdir -p "$run"
exec 9>"$run/training.lock"
flock -n 9 || exit 3
export TB_LOG_DIR=$run/tensorboard
export WANDB_DATA_DIR=$IMAGINAIRE_OUTPUT_ROOT/.wandb_runtime/data
export WANDB_CACHE_DIR=$IMAGINAIRE_OUTPUT_ROOT/.wandb_runtime/cache
export TRITON_CACHE_DIR=$IMAGINAIRE_OUTPUT_ROOT/.triton_egocontrol_${SLURM_JOB_ID}
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TRITON_CACHE_DIR"
load=$EGOCONTROL_INITIAL_CHECKPOINT
resume=false
ema=true
if [[ -n "${EGOCONTROL_RESUME:-}" ]]; then
  load=$EGOCONTROL_RESUME
  case "$load" in "$run"/checkpoints/iter_*) ;; *) printf 'Resume belongs to another run\n' >&2; exit 2;; esac
  resume=true
  ema=false
elif [[ -e "$run/checkpoints/latest_checkpoint.txt" ]]; then
  printf 'Existing run: explicitly set EGOCONTROL_RESUME.\n' >&2
  exit 2
fi
[[ -f "$load/model/.metadata" && -f "$EGOCONTROL_CACHE/index.json" ]] || exit 2
printf '[egocontrol] job=%s node=%s output=%s initialize=%s resume=%s\n' "$SLURM_JOB_ID" "$host" "$run" "$load" "$resume"
exec "$COSMOS_ENV_ROOT/bin/torchrun" --standalone --nproc_per_node="${EGOCONTROL_GPUS:-2}" \
  "$REPO_ROOT/egocontrol_edge/train.py" --sft-toml "$REPO_ROOT/egocontrol_edge/train.toml" \
  job.name="$name" job.wandb_mode="${EGOCONTROL_WANDB_MODE:-online}" \
  trainer.max_iter="${EGOCONTROL_STEPS:-200000}" trainer.grad_accum_iter=1 trainer.logging_iter=50 \
  scheduler.cycle_lengths='[200000]' scheduler.warm_up_steps='[500]' \
  scheduler.f_start='[0.4]' scheduler.f_max='[0.4]' scheduler.f_min='[0.0]' \
  checkpoint.save_iter="${EGOCONTROL_SAVE_EVERY:-5000}" checkpoint.load_path="$load" \
  checkpoint.load_training_state="$resume" checkpoint.strict_resume="$resume" \
  checkpoint.load_ema_to_reg="$ema" checkpoint.keys_to_skip_loading='[]' checkpoint.keys_not_to_resume='[]' \
  model.config.compile.enabled=false model.config.ema.enabled=true \
  model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.parallelism.data_parallel_replicate_degree="${EGOCONTROL_GPUS:-2}" \
  model.config.parallelism.context_parallel_shard_degree=1 model.config.parallelism.cfg_parallel_shard_degree=1 "$@"
