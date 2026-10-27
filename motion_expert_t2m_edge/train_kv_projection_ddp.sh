#!/usr/bin/env bash
# Fresh/resumable Projection A mixed-data batch-256 experiment, Yonsei or Grasp.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an owned two-GPU Slurm allocation}"
: "${COSMOS_SITE:?Select yonsei or grasp}"
variant=${1:-reasoner_kv_project}
pre_joint=0
case "$variant" in
    reasoner_kv_project)
        export EDGE_REASONER_V_NORM_TRAINABLE=0
        run_date=20260917
        ;;
    reasoner_kv_project_vnorm)
        export EDGE_REASONER_V_NORM_TRAINABLE=1
        run_date=20260919
        ;;
    reasoner_kv_project_vnorm_captiononly_self_joint)
        export EDGE_REASONER_V_NORM_TRAINABLE=1
        pre_joint=1
        run_date=20260925
        if [[ -n "${PROJECTION_A_TEXT_PACKING:-}" && "$PROJECTION_A_TEXT_PACKING" != nano_caption_only ]]; then
            printf 'The caption-only self/joint ablation requires nano_caption_only text packing.\n' >&2
            exit 2
        fi
        export PROJECTION_A_TEXT_PACKING=nano_caption_only
        ;;
    *) printf 'Expected reasoner_kv_project, reasoner_kv_project_vnorm, or reasoner_kv_project_vnorm_captiononly_self_joint\n' >&2; exit 2;;
esac
host=$(hostname -s)
case "$host" in *login*|cs-gpu-01) printf 'Compute node required: %s\n' "$host" >&2; exit 2;; esac
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
        output_root=${PROJECTION_A_OUTPUT_ROOT:-/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge}
        export COSMOS_MOTION_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/uniego_arrays_v1
        export COSMOS_MOTION_CACHE_SOURCE_ROOT=$WEKA_ROOT
        export COSMOS_REASONER_IMAGE_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/phase2_images_v1
        export COSMOS_EDGE_CONDITION_CACHE_ROOT=$COSMOS_REASONER_IMAGE_CACHE_ROOT
        export COSMOS_EDGE_CONDITION_CACHE_ID=edge-d4599e2e43fbd06168e9884205b9b66c3902d8f6-pinned-weights-processor-v1
        export WANDB_ENTITY=${WANDB_ENTITY:-scene_motion}
        ;;
    grasp|custom)
        output_root=${PROJECTION_A_OUTPUT_ROOT:-$RUN_ROOT/motion_expert_t2m_edge}
        export WANDB_ENTITY=${WANDB_ENTITY:-jungbinc-upenn}
        ;;
    *) printf 'Unsupported COSMOS_SITE: %s\n' "$COSMOS_SITE" >&2; exit 2;;
esac
export COSMOS_EDGE_CONDITION_PREFETCH_THREADS=4 COSMOS_EDGE_CONDITION_ASYNC_WRITES=1
export EDGE_MOTION_REPRESENTATION=camera_head_recanonicalization_v1
export EDGE_BONES_CAPTION_POLICY=main_actor_camera_wearer_v1
export EDGE_MOTION_IO_DTYPE=float32 EDGE_MOTION_SELF_ATTENTION_PER_GAP=0
export EDGE_MOTION_SELF_ATTENTION_BEFORE_JOINT=$pre_joint
export EDGE_MOTION_ATTENTION_MODE=joint EDGE_MOTION_NORM_DTYPE=bfloat16
export EDGE_REASONER_K_NORM_TRAINABLE=1 EDGE_REASONER_K_NORM_BYPASS=0
export EDGE_REASONER_KV_PROJECTION=linear_identity_v1
export EDGE_TEXT_PACKING=${PROJECTION_A_TEXT_PACKING:-edge_native_generation_prefix}

baseline_name=edge_7layer_reasoner_kv_project_nymeria_bones50_t2m_ti2m_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_${COSMOS_SITE}_20260916
if [[ "$pre_joint" == 1 ]]; then
    baseline_name=edge_7layer_reasoner_kv_project_vnorm_captiononly_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_${COSMOS_SITE}_20260922
fi
run_name=${PROJECTION_A_RUN_NAME:-edge_7layer_${variant}_nymeria_bones50_t2m_ti2m_nanoparity_gbs256_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_${COSMOS_SITE}_${run_date}}
export OUT=$output_root/$run_name
fixed_source=${FIXED_VIZ_SOURCE:-$output_root/$baseline_name/visualizations/fixed_samples}
export WANDB_MODE=online WANDB_PROJECT=cosmos-motion-ft
export WANDB_RUN_NAME=$run_name
export WANDB_GROUP=cosmos3-edge-phase2-kv-projection-ablation
export WANDB_TAGS=cosmos3-edge,phase2,nanoparity,kv-projection,ablation-A,$variant,mixed,gbs256,ddp2,camhead-stats,checkpoint10k,$COSMOS_SITE,$EDGE_TEXT_PACKING
if [[ "$pre_joint" == 1 ]]; then
    export WANDB_TAGS=$WANDB_TAGS,self-before-joint,learned-projected-kv-norm
fi
export WANDB_DATA_DIR=$output_root/.wandb_runtime/data
export WANDB_CACHE_DIR=$output_root/.wandb_runtime/cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
mkdir -p "$OUT" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
exec 8>"$OUT/.training.lock"
flock -n 8 || { printf 'Another trainer owns %s\n' "$OUT" >&2; exit 1; }
exec 9>"$HOME/cosmos_phase2_norm_${SLURM_JOB_ID}.lock"
flock -n 9 || { printf 'Another trainer owns job %s\n' "$SLURM_JOB_ID" >&2; exit 1; }
exec > >(tee -a "$OUT/train_job${SLURM_JOB_ID}.log") 2>&1
printf '[projection-A-ddp] job=%s host=%s out=%s\n' "$SLURM_JOB_ID" "$host" "$OUT"
printf '[projection-A-ddp] variant=%s learned_reasoner_v_norm=%s\n' "$variant" "$EDGE_REASONER_V_NORM_TRAINABLE"
printf '[projection-A-ddp] text packing=%s\n' "$EDGE_TEXT_PACKING"
printf '[projection-A-ddp] 2 ranks x batch128 x accum1 = global256; workers8/rank; 200k steps\n'
printf '[projection-A-ddp] LR2e-4 warmup1000 cosine floor0; seed0; losses1/1/5 contact0.05/1/10\n'
printf '[projection-A-ddp] BONES50 within T2M; camera/head stats; projection A with pre_joint_self_attention=%s\n' "$pre_joint"
if [[ "$pre_joint" == 1 ]]; then
    printf '[architecture] seven blocks: motion-only SA residual -> projected-K/V joint-attention residual -> ONE FFN; frozen reasoner unchanged\n'
fi
printf '[projection-A-ddp] only positive 10k checkpoints; resume only this separate run, never batch128\n'
if [[ ! -f "$OUT/visualizations/fixed_samples/manifest.json" ]]; then
    mkdir -p "$OUT/visualizations"
    if [[ -f "$fixed_source/manifest.json" ]]; then
        cp -a "$fixed_source" "$OUT/visualizations/fixed_samples"
    elif [[ "$COSMOS_SITE" == yonsei || -n "${FIXED_VIZ_SOURCE:-}" ]]; then
        printf 'Required existing fixed visualization suite missing: %s\n' "$fixed_source" >&2
        exit 1
    fi
fi
if [[ "$pre_joint" == 1 && ! -f "$OUT/visualizations/fixed_samples_bones/manifest.json" ]]; then
    bones_source=${FIXED_VIZ_BONES_SOURCE:-$(dirname "$fixed_source")/fixed_samples_bones}
    if [[ -f "$bones_source/manifest.json" ]]; then
        cp -a "$bones_source" "$OUT/visualizations/fixed_samples_bones"
    elif [[ "$COSMOS_SITE" == yonsei || -n "${FIXED_VIZ_BONES_SOURCE:-}" ]]; then
        printf 'Required existing BONES visualization suite missing: %s\n' "$bones_source" >&2
        exit 1
    fi
fi
exec bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" \
    -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
    "$REPO_ROOT/motion_expert_t2m_edge/train.py" \
    --out "$OUT" --steps 200000 --T 200 --ti2m-frames 97 \
    --batch-size 128 --grad-accum 1 --num-workers 8 \
    --task-weights '{"text2motion":0.75,"textimg2motion":0.25}' \
    --bones-frac 0.5 --reasoner-image-size 256 \
    --lr 2e-4 --warmup 1000 --min-lr-ratio 0 --weight-decay 0 --adam-beta2 0.99 \
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
    --resume auto
