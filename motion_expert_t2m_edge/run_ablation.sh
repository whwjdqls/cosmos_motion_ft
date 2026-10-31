#!/usr/bin/env bash
# Central A0--A3 implementation. Account-local scripts only call this file.
set -euo pipefail
variant=${1:?Usage: run_ablation.sh A0|A1|A2|A3 show|check|prepare|smoke|capacity|train [arguments]}
action=${2:-show}
shift "$(( $# >= 2 ? 2 : 1 ))"
case "$variant" in A0|A1|A2|A3) ;; *) printf 'Unknown ablation: %s\n' "$variant" >&2; exit 2;; esac
case "$action" in show|check|prepare|sample|smoke|capacity|train) ;; *) printf 'Unknown action: %s\n' "$action" >&2; exit 2;; esac
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_ROOT COSMOS_REPO_ROOT=$REPO_ROOT
export COSMOS_SITE=custom EDGE_ARTIFACT_SOURCE_SITE=yonsei
export COSMOS_STORAGE_ROOT=${PHASE2_ABLATION_STORAGE_ROOT:-/scratch2/$(id -un)/cosmos}
export WEKA_ROOT=${PHASE2_SHARED_WEKA_ROOT:-/lustre/whwjdqls99/cosmos/weka}
export COSMOS_RUNS_ROOT=$WEKA_ROOT/cosmos_motion_ft_runs RUN_ROOT=$WEKA_ROOT/cosmos_motion_ft_runs
export COSMOS_ENV_ROOT=${PHASE2_ABLATION_ENV:-/lustre/whwjdqls99/cosmos/envs/cosmos}
export COSMOS_FRAMEWORK_EDGE_ROOT=${PHASE2_SHARED_EDGE_ROOT:-/home/whwjdqls99/cosmos-framework-edge}
export KIMODO_ROOT=${PHASE2_SHARED_KIMODO_ROOT:-/home/whwjdqls99/kimodo_open}
export CUDA_HOME=${PHASE2_ABLATION_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/12.8}
export PATH=$CUDA_HOME/bin:$COSMOS_ENV_ROOT/bin:$PATH
source "$REPO_ROOT/restored_env.sh"
export PYTHONPATH=$KIMODO_ROOT:$REPO_ROOT/motion_expert${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR=$COSMOS_STORAGE_ROOT/cache/torch_extensions
export TRITON_CACHE_DIR=$COSMOS_STORAGE_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$COSMOS_STORAGE_ROOT/cache/inductor
export XDG_CACHE_HOME=$COSMOS_STORAGE_ROOT/.cache
export WANDB_DATA_DIR=$COSMOS_STORAGE_ROOT/wandb/data WANDB_CACHE_DIR=$COSMOS_STORAGE_ROOT/wandb/cache
export WANDB_DIR=$COSMOS_STORAGE_ROOT/wandb WANDB_CONFIG_DIR=$COSMOS_STORAGE_ROOT/wandb/config
export COSMOS_MOTION_CACHE_ROOT=$COSMOS_STORAGE_ROOT/cache/uniego_arrays_v1
export COSMOS_MOTION_CACHE_SOURCE_ROOT=$WEKA_ROOT
export COSMOS_REASONER_IMAGE_CACHE_ROOT=$COSMOS_STORAGE_ROOT/cache/phase2_images_v1
export COSMOS_EDGE_CONDITION_CACHE_ROOT=$COSMOS_REASONER_IMAGE_CACHE_ROOT
export COSMOS_EDGE_CONDITION_CACHE_ID=edge-d4599e2e43fbd06168e9884205b9b66c3902d8f6-pinned-weights-processor-v1
export COSMOS_EDGE_CONDITION_PREFETCH_THREADS=4 COSMOS_EDGE_CONDITION_ASYNC_WRITES=1
export PHASE2_KIMODO_CACHE_ROOT=${PHASE2_KIMODO_CACHE_ROOT-$COSMOS_STORAGE_ROOT/cache/kimodo369_windows_v1}
export PHASE2_KIMODO_STATS_DIR=${PHASE2_KIMODO_STATS_DIR:-$COSMOS_STORAGE_ROOT/stats/kimodo369_camhead_v1}
export EDGE_MOTION_REPRESENTATION=camera_head_recanonicalization_v1
export EDGE_BONES_CAPTION_POLICY=main_actor_camera_wearer_v1
export EDGE_MOTION_IO_DTYPE=float32 EDGE_MOTION_NORM_DTYPE=bfloat16
export EDGE_MOTION_SELF_ATTENTION_PER_GAP=0 EDGE_MOTION_SELF_ATTENTION_BEFORE_JOINT=0
export EDGE_MOTION_ATTENTION_MODE=self_cross_final_hidden
export EDGE_REASONER_K_NORM_TRAINABLE=0 EDGE_REASONER_K_NORM_BYPASS=1
export EDGE_REASONER_V_NORM_TRAINABLE=0 EDGE_REASONER_KV_PROJECTION=none
export EDGE_CROSS_ATTENTION_K_NORM=rmsnorm EDGE_CROSS_ATTENTION_V_NORM=none
export EDGE_CROSS_ATTENTION_REASONER_ROPE=1 EDGE_TEXT_PACKING=nano_caption_only
export EDGE_PHASE2_FEATURE_CODEC=uniego283 EDGE_PHASE2_OBJECTIVE=x0
if [[ $variant == A1 ]]; then export EDGE_PHASE2_OBJECTIVE=velocity; fi
if [[ $variant == A2 ]]; then export EDGE_PHASE2_FEATURE_CODEC=kimodo369_v1; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
task_weights='{"text2motion":0.75,"textimg2motion":0.25}'
bones_frac=0.5
ti_dropout=0.1
if [[ $variant == A3 ]]; then
    task_weights='{"text2motion":0.5,"textimg2motion":0.5}'
    bones_frac=0.75
    ti_dropout=0.3
fi
batch_size=${PHASE2_ABLATION_MICROBATCH:-256}
grad_accum=${PHASE2_ABLATION_GRAD_ACCUM:-1}
workers=${PHASE2_ABLATION_WORKERS:-8}
if [[ ! $batch_size =~ ^[1-9][0-9]*$ || ! $grad_accum =~ ^[1-9][0-9]*$ ]] || (( batch_size * grad_accum != 256 )); then
    printf 'Require microbatch x accumulation = 256 on ONE GPU\n' >&2; exit 2
fi
if [[ $action == show ]]; then
    exec /usr/bin/python3 -S "$REPO_ROOT/motion_expert_t2m_edge/ablation_launcher_info.py" \
        --variant "$variant" --task-weights "$task_weights" --bones-frac "$bones_frac" \
        --ti-dropout "$ti_dropout" --batch "$batch_size" --accum "$grad_accum"
fi
: "${SLURM_JOB_ID:?Use an owned compute allocation}"
host=$(hostname -s)
case "$host" in *login*|cs-gpu-01) printf 'Compute node required: %s\n' "$host" >&2; exit 2;; esac
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
if [[ $action == prepare ]]; then
    [[ $variant == A2 ]] || { printf 'Only A2 requires new statistics\n' >&2; exit 2; }
    exec bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" \
        "$REPO_ROOT/motion_expert_t2m_edge/prepare_kimodo_stats.py" --out "$PHASE2_KIMODO_STATS_DIR" "$@"
fi
if [[ $action == check ]]; then
    exec bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" \
        "$REPO_ROOT/motion_expert_t2m_edge/check_ablations.py" "$@"
fi
if [[ $action == sample ]]; then
    exec bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" \
        "$REPO_ROOT/motion_expert_t2m_edge/sample.py" "$@"
fi
if (( $# )); then
    printf 'Training recipes are fixed. Use PHASE2_ABLATION_* variables for paths/batch/resume.\n' >&2
    exit 2
fi
run_name=${PHASE2_ABLATION_RUN_NAME:-edge_selfcross_knorm_captiononly_${variant}_gbs256_20261031}
output_root=$COSMOS_STORAGE_ROOT/cosmos_motion_ft_runs/motion_expert_t2m_edge
out=$output_root/$run_name
resume=${PHASE2_ABLATION_RESUME:-auto}
extra=()
if [[ $action == smoke || $action == capacity ]]; then
    out=$COSMOS_STORAGE_ROOT/operations/phase2_ablation_acceptance/${run_name}_${SLURM_JOB_ID}
    # Always run a fresh preflight, including when retrying after a source fix.
    # Preserve failed evidence and never reuse a completed smoke checkpoint.
    out=${out}_${action}_$(date -u +%Y%m%dT%H%M%S%N)
    extra=(--steps 3 --save-every 3 --wandb-mode offline --viz-samples-per-task 1 --viz-bones-samples 1 --viz-at-start --require-viz)
    if [[ $action == smoke ]]; then extra+=(--smoke); fi
    resume=auto
fi
mkdir -p "$out"
exec 8>"$out/.training.lock"
flock -n 8 || { printf 'Another trainer owns %s\n' "$out" >&2; exit 1; }
exec 9>"$COSMOS_STORAGE_ROOT/.phase2_gpu_${SLURM_JOB_ID}.lock"
flock -n 9 || { printf 'A phase2 trainer already owns this allocation\n' >&2; exit 1; }
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
/usr/bin/python3 -S "$REPO_ROOT/motion_expert_t2m_edge/ablation_launcher_info.py" \
    --variant "$variant" --task-weights "$task_weights" --bones-frac "$bones_frac" \
    --ti-dropout "$ti_dropout" --batch "$batch_size" --accum "$grad_accum" --out "$out"
exec > >(tee -a "$out/train_job${SLURM_JOB_ID}.log") 2>&1
exec bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" "$REPO_ROOT/motion_expert_t2m_edge/train.py" \
    --out "$out" --steps 200000 --T 200 --ti2m-frames 97 \
    --batch-size "$batch_size" --grad-accum "$grad_accum" --num-workers "$workers" \
    --task-weights "$task_weights" --bones-frac "$bones_frac" \
    --cfg-dropout 0.1 --ti2m-cfg-dropout "$ti_dropout" --objective "$EDGE_PHASE2_OBJECTIVE" \
    --reasoner-image-size 256 --seed 0 --lr 2e-4 --warmup 1000 --min-lr-ratio 0 \
    --weight-decay 0 --adam-beta2 0.99 --loss-reduction per_task --loss-guard nano --allow-tf32 0 --grad-clip 1 \
    --w-feat 1 --w-joint 1 --w-smooth 5 --joint-loss-mode centered \
    --w-global-rot 0 --w-global-trans 0 --w-contact 0.05 --w-foot-vel 1 --w-foot-height 10 \
    --save-every 10000 --recovery-save-every 0 --checkpoint-interval-only \
    --wandb-mode online --wandb-project cosmos-motion-ft --wandb-entity "${WANDB_ENTITY:-jungbinc-upenn}" \
    --wandb-run-name "$run_name" --wandb-group phase2-paper-ablations-20261031 --require-wandb \
    --viz-every 5000 --viz-samples-per-task 5 --viz-source nymeria --viz-bones-samples 2 \
    --viz-steps 35 --viz-guidance 2 --viz-frame-stride 2 --viz-at-start --require-viz \
    --resume "$resume" "${extra[@]}" "$@"
