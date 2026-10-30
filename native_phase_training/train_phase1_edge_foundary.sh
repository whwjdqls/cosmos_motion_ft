#!/usr/bin/env bash
# User-assigned Foundary GPUs: Edge, two tasks, fixed observed prefix, no accumulation.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/.site.local.sh"
gpu_indices=${FOUNDRY_GPU_INDICES:-${FOUNDRY_GPU_INDEX:-}}
[[ "$gpu_indices" =~ ^[0-7](,[0-7])?$ ]] || { printf 'Select one or two assigned GPUs with FOUNDRY_GPU_INDICES (for example 0,5)\n' >&2; exit 2; }
IFS=, read -r -a selected_indices <<< "$gpu_indices"
gpu_count=${#selected_indices[@]}
if (( gpu_count == 2 )) && [[ "${selected_indices[0]}" == "${selected_indices[1]}" ]]; then
    printf 'GPU indices must be distinct\n' >&2; exit 2
fi
: "${NATIVEP1_RUN_NAME:?Choose a new, distinct run name}"
[[ "$NATIVEP1_RUN_NAME" != */* ]] || exit 2
export NATIVEP1_CLIPS_PER_GPU=${NATIVEP1_CLIPS_PER_GPU:-64}
[[ "$NATIVEP1_CLIPS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { printf 'Batch size must be a positive integer\n' >&2; exit 2; }
selected_uuids=()
for index in "${selected_indices[@]}"; do
    selected_uuids+=("$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader)")
done
export FOUNDRY_GPU_UUIDS
FOUNDRY_GPU_UUIDS=$(IFS=,; printf '%s' "${selected_uuids[*]}")
export CUDA_VISIBLE_DEVICES="$FOUNDRY_GPU_UUIDS"
unset FOUNDRY_GPU_UUID
unset SLURM_JOB_ID SLURM_JOB_USER
"$COSMOS_PYTHON" -m native_phase_training.compute_access

dryrun=()
if [[ "${1:-}" == --dryrun ]]; then dryrun=(--dryrun); shift; fi
export PHASE1_WANDB_MODE=${PHASE1_WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-cosmos3_camera}
if [[ "$PHASE1_WANDB_MODE" == online ]]; then
    export WANDB_ENTITY=${WANDB_ENTITY:-$("$COSMOS_PYTHON" -c 'import wandb; print(wandb.Api(timeout=30).default_entity)')}
fi

export IMAGINAIRE_OUTPUT_ROOT=${FOUNDRY_PHASE1_OUTPUT_ROOT:-$COSMOS_STORAGE_ROOT/runs}
case "$(realpath -m "$IMAGINAIRE_OUTPUT_ROOT")/" in /mnt/local/*/) ;; *) exit 2;; esac
run_dir="$IMAGINAIRE_OUTPUT_ROOT/cosmos3_camera/camera_world_edge/$NATIVEP1_RUN_NAME"
mkdir -p "$run_dir" "$FOUNDRY_LOCAL_ROOT/locks"
exec 9>"$run_dir/training.lock"
flock -n 9 || { printf 'Run already has a launcher\n' >&2; exit 2; }
# Config-only dryruns do not reserve or initialize CUDA devices.
if (( ${#dryrun[@]} == 0 )); then
for gpu_uuid in "${selected_uuids[@]}"; do
    exec {gpu_lock_fd}>"$FOUNDRY_LOCAL_ROOT/locks/$gpu_uuid.lock"
    flock -n "$gpu_lock_fd" || { printf 'GPU %s already has one of your launchers\n' "$gpu_uuid" >&2; exit 2; }
    used=$(nvidia-smi -i "$gpu_uuid" --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( used >= 1024 )); then
        # Only explicitly coordinated resident processes may share this GPU.
        pids=$(nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid --format=csv,noheader,nounits)
        [[ -n "$pids" ]] || { printf 'Unattributed GPU memory on %s; refusing launch\n' "$gpu_uuid" >&2; exit 2; }
        while read -r pid; do
            [[ "$pid" =~ ^[0-9]+$ && ",${FOUNDRY_SHARED_GPU_PIDS:-}," == *",$pid,"* ]] || {
                printf 'GPU %s is occupied by uncoordinated PID %s; refusing launch\n' "$gpu_uuid" "$pid" >&2; exit 2;
            }
            printf '[foundary-phase1] coordinated sharing: gpu=%s resident_pid=%s\n' "$gpu_uuid" "$pid"
        done <<< "$pids"
        free=$(nvidia-smi -i "$gpu_uuid" --query-gpu=memory.free --format=csv,noheader,nounits)
        (( free >= 65536 )) || { printf 'Less than 64 GiB free on shared GPU %s\n' "$gpu_uuid" >&2; exit 2; }
    fi
done
fi

export PATH="$COSMOS_ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH=
export PYTHONPATH="$repo:$repo/nymeria_world:$COSMOS_FRAMEWORK_EDGE_ROOT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NATIVEP1_MODEL_FAMILY=edge NATIVEP1_ADAPTATION_MODE=${NATIVEP1_ADAPTATION_MODE:-global_lora}
export NYMERIA_NUM_FRAMES=97 NYMERIA_RESOLUTION=256 NYMERIA_MODE=mixture
export NYMERIA_DROP_MODES=policy,image2video
export NATIVEP1_DISABLE_CAPTIONS=1 NATIVEP1_PREFIX_LENGTHS=${NATIVEP1_PREFIX_LENGTHS:-17}
[[ "$NATIVEP1_PREFIX_LENGTHS" == 1 || "$NATIVEP1_PREFIX_LENGTHS" == 17 ]] || { printf 'Expected prefix 1 or 17\n' >&2; exit 2; }
export NYMERIA_REPLACE_STANDALONE_C=1 NYMERIA_STANDALONE_C_SUBJECT=camera_wearer
export NYMERIA_EXCLUDE_ORIGINAL_VIDEO_NAMES=brittney_goodwin,austin_lopez,kim_wilkins,kevin_davis,devon_norris
export NYMERIA_QUALITY_FILTER=
export NATIVEP1_EXPECTED_LATENT_HW=16 NATIVEP1_EXPECTED_IMAGE_HW=256
export NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=1
export NATIVEP1_LORA_LR=5e-5 NATIVEP1_ACTION_LR_MULT=4
export NATIVEP1_ACTION_LOSS_WEIGHT=10 NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=0
export NATIVEP1_AUTO_EVAL=0 NATIVEP1_AUTO_EVAL_FULL71=0 NATIVEP1_WANDB_PREVIEWS=${NATIVEP1_WANDB_PREVIEWS:-1}
export COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES=15
export TB_LOG_DIR="$run_dir/tensorboard"
export WANDB_DATA_DIR="$run_dir/.wandb_runtime/data" WANDB_CACHE_DIR="$run_dir/.wandb_runtime/cache"
export TRITON_CACHE_DIR="$run_dir/.triton"
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TRITON_CACHE_DIR"
unset NYMERIA_FULL_FT NYMERIA_MAX_SAMPLES NATIVEP1_SHIFT_OVERRIDE NATIVEP1_PREFIX_SAMPLING_WEIGHTS
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME

load_path=${RESUME_CHECKPOINT:-$BASE_CHECKPOINT_PATH}
resume=false
skip='[net_ema.]'
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    case "$RESUME_CHECKPOINT" in "$run_dir"/checkpoints/iter_*) ;; *) printf 'Resume must belong to this run\n' >&2; exit 2;; esac
    resume=true
    skip='[]'
elif [[ -d "$run_dir/checkpoints" ]]; then
    printf 'Existing checkpoints: specify RESUME_CHECKPOINT explicitly\n' >&2; exit 2
fi
for path in "$load_path/model/.metadata" "$EDGE_MODEL_ROOT/modular_model_index.json" "$WAN_VAE_PATH" \
    "$NYMERIA_LATENT_ROOT/latent_cache_contract.json" "$NYMERIA_LATENT_ROOT/latent_cache_complete.json" \
    "$NATIVEP1_PREVIEW_INPUT_DIR/fd_input.jsonl"; do
    [[ -f "$path" ]] || { printf 'Missing input: %s\n' "$path" >&2; exit 2; }
    case "$(realpath "$path")" in /mnt/local/*) ;; *) printf 'Input must be node-local: %s\n' "$path" >&2; exit 2;; esac
done
printf '[foundary-phase1] host=%s gpus=%s run=%s batch_per_gpu=%s global_batch=%s accumulation=1 prefix=%s adaptation=%s tasks=40:25 captions=off\n' "$(hostname)" "$FOUNDRY_GPU_UUIDS" "$run_dir" "$NATIVEP1_CLIPS_PER_GPU" "$((NATIVEP1_CLIPS_PER_GPU * gpu_count))" "$NATIVEP1_PREFIX_LENGTHS" "$NATIVEP1_ADAPTATION_MODE"
cd "$COSMOS_FRAMEWORK_EDGE_ROOT"
launch_processes=$gpu_count
if (( ${#dryrun[@]} > 0 )); then launch_processes=1; fi
exec "$COSMOS_ENV_ROOT/bin/torchrun" --standalone --nproc_per_node="$launch_processes" \
    "$repo/native_phase_training/run_latent_train.py" \
    --sft-toml "$repo/native_phase_training/world_camera_nymeria_latent_edge.toml" "${dryrun[@]}" \
    job.name="$NATIVEP1_RUN_NAME" job.wandb_mode="$PHASE1_WANDB_MODE" \
    trainer.max_iter=200000 trainer.grad_accum_iter=1 trainer.logging_iter=1 \
    scheduler.cycle_lengths='[200000]' scheduler.warm_up_steps='[500]' \
    scheduler.f_start='[0.4]' scheduler.f_max='[0.4]' scheduler.f_min='[0.0]' \
    checkpoint.save_iter=5000 checkpoint.load_path="$load_path" \
    checkpoint.load_training_state="$resume" checkpoint.strict_resume="$resume" \
    checkpoint.keys_to_skip_loading="$skip" checkpoint.keys_not_to_resume='[]' \
    model.config.compile.enabled=false model.config.ema.enabled=true \
    model.config.parallelism.data_parallel_shard_degree=1 \
    model.config.parallelism.data_parallel_replicate_degree="$gpu_count" \
    model.config.parallelism.context_parallel_shard_degree=1 \
    model.config.parallelism.cfg_parallel_shard_degree=1 "$@"
