#!/usr/bin/env bash
# Independent mixed-data Projection A + K/V norm ablations, Yonsei or Grasp.
# long300k branches the finished B128 200k checkpoint with optimizer state.
# Usage: bash train_kv_capacity_ablation.sh {lr10,long300k,self_joint}
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an owned GPU allocation}"
: "${COSMOS_SITE:?Select yonsei or grasp}"
variant=${1:?Select lr10, long300k, or self_joint}
host=$(hostname -s)
case "$host" in *login*|cs-gpu-01) printf 'Compute node required: %s\n' "$host" >&2; exit 2;; esac
nproc=1
lr=2e-4
steps=200000
pre_joint=0
case "$variant" in
    lr10) lr=2e-3;;
    long300k) steps=300000;;
    self_joint) nproc=2; pre_joint=1;;
    *) printf 'Unknown ablation: %s\n' "$variant" >&2; exit 2;;
esac
global_batch=$((128 * nproc))
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_ROOT
if [[ "$COSMOS_SITE" == yonsei ]]; then
    export PROJECT_ROOT=/lustre/whwjdqls99/cosmos
    export COSMOS_ENV_ROOT=$PROJECT_ROOT/envs/cosmos
    export COSMOS_PYTHON=$COSMOS_ENV_ROOT/bin/python3.13
    export COSMOS_FRAMEWORK_EDGE_ROOT=/home/whwjdqls99/cosmos-framework-edge
    export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
    export PATH=$CUDA_HOME/bin:$COSMOS_ENV_ROOT/bin:$PATH
fi
source "$REPO_ROOT/restored_env.sh"
case "$COSMOS_SITE" in
    yonsei)
        output_root=${KV_CAPACITY_OUTPUT_ROOT:-/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge}
        export COSMOS_MOTION_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/uniego_arrays_v1
        export COSMOS_MOTION_CACHE_SOURCE_ROOT=$WEKA_ROOT
        export COSMOS_REASONER_IMAGE_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/phase2_images_v1
        export COSMOS_EDGE_CONDITION_CACHE_ROOT=$COSMOS_REASONER_IMAGE_CACHE_ROOT
        export COSMOS_EDGE_CONDITION_CACHE_ID=edge-d4599e2e43fbd06168e9884205b9b66c3902d8f6-pinned-weights-processor-v1
        export WANDB_ENTITY=${WANDB_ENTITY:-scene_motion}
        ;;
    grasp|custom)
        output_root=${KV_CAPACITY_OUTPUT_ROOT:-$RUN_ROOT/motion_expert_t2m_edge}
        export WANDB_ENTITY=${WANDB_ENTITY:-jungbinc-upenn}
        ;;
    *) printf 'Unsupported site: %s\n' "$COSMOS_SITE" >&2; exit 2;;
esac
export COSMOS_EDGE_CONDITION_PREFETCH_THREADS=4 COSMOS_EDGE_CONDITION_ASYNC_WRITES=1
export EDGE_MOTION_REPRESENTATION=camera_head_recanonicalization_v1
export EDGE_BONES_CAPTION_POLICY=main_actor_camera_wearer_v1
export EDGE_TEXT_PACKING=edge_native_generation_prefix
export EDGE_MOTION_IO_DTYPE=float32 EDGE_MOTION_NORM_DTYPE=bfloat16
export EDGE_MOTION_ATTENTION_MODE=joint EDGE_MOTION_SELF_ATTENTION_PER_GAP=0
export EDGE_MOTION_SELF_ATTENTION_BEFORE_JOINT=$pre_joint
export EDGE_REASONER_K_NORM_TRAINABLE=1 EDGE_REASONER_K_NORM_BYPASS=0
export EDGE_REASONER_KV_PROJECTION=linear_identity_v1 EDGE_REASONER_V_NORM_TRAINABLE=1
export EDGE_CROSS_ATTENTION_K_NORM=none EDGE_CROSS_ATTENTION_V_NORM=none
export EDGE_CROSS_ATTENTION_REASONER_ROPE=1

run_name=edge_7layer_reasoner_kv_project_vnorm_${variant}_nymeria_bones50_t2m_ti2m_nanoparity_gbs${global_batch}_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_${COSMOS_SITE}_20260920
export OUT=$output_root/$run_name
baseline_name=edge_7layer_reasoner_kv_project_vnorm_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_${COSMOS_SITE}_20260917
fixed_root=${FIXED_VIZ_ROOT:-$output_root/$baseline_name/visualizations}
export WANDB_MODE=online WANDB_PROJECT=cosmos-motion-ft
export WANDB_RUN_NAME=$run_name WANDB_GROUP=cosmos3-edge-phase2-kv-capacity-ablation
export WANDB_TAGS=cosmos3-edge,phase2,nanoparity,kv-projection,learned-projected-v-norm,$variant,mixed,gbs$global_batch,camhead-stats,checkpoint10k,$COSMOS_SITE
export WANDB_DATA_DIR=$output_root/.wandb_runtime/data WANDB_CACHE_DIR=$output_root/.wandb_runtime/cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
mkdir -p "$OUT" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
exec 8>"$OUT/.training.lock"
flock -n 8 || { printf 'Another trainer owns %s\n' "$OUT" >&2; exit 1; }
exec 9>"$HOME/cosmos_phase2_norm_${SLURM_JOB_ID}.lock"
flock -n 9 || { printf 'Another trainer owns job %s\n' "$SLURM_JOB_ID" >&2; exit 1; }
exec > >(tee -a "$OUT/train_job${SLURM_JOB_ID}.log") 2>&1
printf '[kv-capacity] job=%s node=%s variant=%s out=%s\n' "$SLURM_JOB_ID" "$host" "$variant" "$OUT"
printf '[kv-capacity] separate ablation output; ranks=%s batch/rank=128 global_batch=%s\n' "$nproc" "$global_batch"
printf '[kv-capacity] steps=%s peak_lr=%s warmup=1000 cosine_floor=0 seed=0\n' "$steps" "$lr"
printf '[kv-capacity] projection A with learned K/V norms; pre_joint_self_attention=%s; gap_blocks=0\n' "$pre_joint"
printf '[kv-capacity] losses1/1/5 contact0.05/1/10; original native prompt; BONES50 within T2M; camhead stats\n'
if [[ "$pre_joint" == 1 ]]; then
    printf '[architecture] seven blocks: motion-only SA residual -> joint-attention residual -> ONE SwiGLU FFN residual\n'
    printf '[architecture] shape plus valid motion tokens in SA; reasoner never sees motion; added SA uses activation recomputation\n'
fi
resume=auto
if [[ "$variant" == long300k ]]; then
    if ! compgen -G "$OUT/checkpoints/step_*.pt" > /dev/null && [[ ! -f "$OUT/checkpoints/recovery_latest.pt" ]]; then
        resume=$output_root/$baseline_name/checkpoints/step_000200000.pt
        [[ -f "$resume" ]] || { printf 'Required finished B128 checkpoint missing: %s\n' "$resume" >&2; exit 1; }
    fi
    printf '[schedule] continue from baseline200k with optimizer/RNG, then own checkpoints; resume=%s\n' "$resume"
    printf '[schedule] extend cosine horizon to300k: LR restarts around5e-5 at200k and decays to0 at300k; no new warmup\n'
fi
for suite in fixed_samples fixed_samples_bones; do
    if [[ ! -f "$OUT/visualizations/$suite/manifest.json" ]]; then
        if [[ -f "$fixed_root/$suite/manifest.json" ]]; then
            mkdir -p "$OUT/visualizations"
            cp -a "$fixed_root/$suite" "$OUT/visualizations/$suite"
        elif [[ "$COSMOS_SITE" == yonsei || -n "${FIXED_VIZ_ROOT:-}" ]]; then
            printf 'Required fixed visualization suite missing: %s/%s\n' "$fixed_root" "$suite" >&2
            exit 1
        fi
    fi
done
launch=()
if [[ "$nproc" == 2 ]]; then
    launch=(-m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2)
fi
exec bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" "${launch[@]}" \
    "$REPO_ROOT/motion_expert_t2m_edge/train.py" \
    --out "$OUT" --steps "$steps" --T 200 --ti2m-frames 97 \
    --batch-size 128 --grad-accum 1 --num-workers 8 \
    --task-weights '{"text2motion":0.75,"textimg2motion":0.25}' \
    --bones-frac 0.5 --reasoner-image-size 256 \
    --lr "$lr" --warmup 1000 --min-lr-ratio 0 --weight-decay 0 --adam-beta2 0.99 \
    --loss-reduction per_task --loss-guard nano --allow-tf32 0 --grad-clip 1 \
    --cfg-dropout 0.1 --seed 0 \
    --w-feat 1 --w-joint 1 --w-smooth 5 --joint-loss-mode centered \
    --w-global-rot 0 --w-global-trans 0 --w-contact 0.05 --w-foot-vel 1 --w-foot-height 10 \
    --save-every 10000 --recovery-save-every 0 --checkpoint-interval-only \
    --wandb-mode online --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "$run_name" --wandb-group "$WANDB_GROUP" --require-wandb \
    --wandb-service-wait 300 --wandb-init-attempts 3 --wandb-init-retry-delay 10 \
    --viz-every 5000 --viz-samples-per-task 5 --viz-source nymeria --viz-bones-samples 2 \
    --viz-steps 35 --viz-guidance 2 --viz-frame-stride 2 --viz-at-start --require-viz \
    --resume "$resume"
