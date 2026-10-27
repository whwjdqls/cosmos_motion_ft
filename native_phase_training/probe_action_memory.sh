#!/usr/bin/env bash
# Short, isolated memory measurement for the current two-task, one-frame recipe.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${COSMOS_SITE:?Select yonsei, grasp or custom}"
: "${SLURM_JOB_ID:?Use an owned two-GPU compute step}"
host=$(hostname -s)
case "$host" in *login*|cs-gpu-01|node37) exit 2;; esac
case "$COSMOS_SITE" in
  yonsei)
    export COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/lustre/whwjdqls99/cosmos/envs/cosmos}
    export COSMOS_FRAMEWORK_EDGE_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/home/whwjdqls99/cosmos-framework-edge}
    export CUDA_HOME=${CUDA_HOME:-/opt/ohpc/pub/apps/cuda/12.8}
    ;;
  grasp)
    export COSMOS_FRAMEWORK_EDGE_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
    ;;
  custom) ;;
  *) exit 2;;
esac
source "$repo/restored_env.sh"
export PHASE1_MEMORY_PROBE_DIR=${1:?Provide a new diagnostic output directory}
micro=${2:-16}
shards=${3:-1}
[[ "$micro" =~ ^[1-9][0-9]*$ && "$shards" =~ ^[12]$ ]] || exit 2
(( 16 % micro == 0 )) || { printf 'Microbatch must divide16 to preserve global32\n' >&2; exit 2; }
[[ ! -e "$PHASE1_MEMORY_PROBE_DIR" ]] || { printf 'Refusing existing probe output\n' >&2; exit 2; }
mkdir -p "$PHASE1_MEMORY_PROBE_DIR"
export IMAGINAIRE_OUTPUT_ROOT=$PHASE1_MEMORY_PROBE_DIR/native_output
export NATIVEP1_MODEL_FAMILY=edge NATIVEP1_ADAPTATION_MODE=action_only
export NATIVEP1_CLIPS_PER_GPU=$micro NATIVEP1_PREFIX_LENGTHS=1 NATIVEP1_DISABLE_CAPTIONS=1
export NYMERIA_NUM_FRAMES=97 NYMERIA_RESOLUTION=256 NYMERIA_MODE=mixture
export NYMERIA_DROP_MODES=policy,image2video NYMERIA_QUALITY_FILTER=
export NYMERIA_REPLACE_STANDALONE_C=1 NYMERIA_STANDALONE_C_SUBJECT=camera_wearer
export NYMERIA_EXCLUDE_ORIGINAL_VIDEO_NAMES=brittney_goodwin,austin_lopez,kim_wilkins,kevin_davis,devon_norris
export NATIVEP1_EXPECTED_LATENT_HW=16 NATIVEP1_EXPECTED_IMAGE_HW=256 NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=1
export NATIVEP1_LORA_LR=5e-5 NATIVEP1_ACTION_LR_MULT=4 NATIVEP1_ACTION_LOSS_WEIGHT=10 NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=0
export NATIVEP1_AUTO_EVAL=0 NATIVEP1_AUTO_EVAL_FULL71=0 NATIVEP1_WANDB_PREVIEWS=0 COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES=0
export EDGE_MODEL_ROOT=$WEKA_ROOT/Cosmos3-Edge BASE_CHECKPOINT_PATH=$WEKA_ROOT/cosmos3_edge_dcp
export NYMERIA_LATENT_ROOT=$WEKA_ROOT/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256
export PATH="$COSMOS_ENV_ROOT/bin:${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
export LD_LIBRARY_PATH= PYTHONPATH="$repo:$repo/nymeria_world:$COSMOS_FRAMEWORK_EDGE_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 WANDB_MODE=disabled
unset NYMERIA_FULL_FT NYMERIA_MAX_SAMPLES NATIVEP1_SHIFT_OVERRIDE NATIVEP1_PREFIX_SAMPLING_WEIGHTS
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
cd "$COSMOS_FRAMEWORK_EDGE_ROOT"
exec "$COSMOS_ENV_ROOT/bin/torchrun" --standalone --nproc_per_node=2 \
    -m native_phase_training.probe_action_memory \
    --sft-toml "$repo/native_phase_training/world_camera_nymeria_latent_edge.toml" \
    job.name=action_only_memory job.wandb_mode=disabled trainer.grad_accum_iter="$((16 / micro))" \
    trainer.logging_iter=1 trainer.callbacks.iter_speed.hit_thres=2 \
    scheduler.cycle_lengths='[200000]' scheduler.warm_up_steps='[500]' \
    scheduler.f_start='[0.4]' scheduler.f_max='[0.4]' scheduler.f_min='[0.0]' \
    checkpoint.load_path="$BASE_CHECKPOINT_PATH" checkpoint.load_training_state=false \
    checkpoint.keys_to_skip_loading='[net_ema.]' checkpoint.keys_not_to_resume='[]' \
    model.config.compile.enabled=false model.config.ema.enabled=true \
    model.config.parallelism.data_parallel_shard_degree="$shards" \
    model.config.parallelism.data_parallel_replicate_degree="$((2 / shards))" \
    model.config.parallelism.context_parallel_shard_degree=1 model.config.parallelism.cfg_parallel_shard_degree=1
