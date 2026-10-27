#!/usr/bin/env bash
# Site-portable, one-GPU fresh parity reasoner-interface experiments.
# Usage: bash train_norm_ablation.sh {motion_fp32,reasoner_k_fp32,reasoner_kv_as_is,reasoner_kv_project} {mixed,bones}
# Caption-only A+Vnorm: reasoner_kv_project_vnorm_captiononly mixed
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an owned one-GPU Slurm allocation}"
: "${COSMOS_SITE:?Select yonsei or grasp}"
variant=${1:?Select motion_fp32, reasoner_k_fp32, reasoner_kv_as_is, or reasoner_kv_project}
data_mix=${2:?Select mixed or bones}
host=$(hostname -s)
case "$host" in *login*|cs-gpu-01) printf 'Compute node required: %s\n' "$host" >&2; exit 2;; esac
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
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
        output_root=${NORM_ABLATION_OUTPUT_ROOT:-/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_t2m_edge}
        export COSMOS_MOTION_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/uniego_arrays_v1
        export COSMOS_MOTION_CACHE_SOURCE_ROOT=$WEKA_ROOT
        export COSMOS_REASONER_IMAGE_CACHE_ROOT=/scratch2/whwjdqls99/cosmos/cache/phase2_images_v1
        export COSMOS_EDGE_CONDITION_CACHE_ROOT=$COSMOS_REASONER_IMAGE_CACHE_ROOT
        export COSMOS_EDGE_CONDITION_CACHE_ID=edge-d4599e2e43fbd06168e9884205b9b66c3902d8f6-pinned-weights-processor-v1
        export WANDB_ENTITY=${WANDB_ENTITY:-scene_motion}
        ;;
    grasp|custom)
        output_root=${NORM_ABLATION_OUTPUT_ROOT:-$RUN_ROOT/motion_expert_t2m_edge}
        export WANDB_ENTITY=${WANDB_ENTITY:-jungbinc-upenn}
        ;;
esac
export COSMOS_EDGE_CONDITION_PREFETCH_THREADS=4
export COSMOS_EDGE_CONDITION_ASYNC_WRITES=1
export EDGE_MOTION_REPRESENTATION=camera_head_recanonicalization_v1
export EDGE_BONES_CAPTION_POLICY=main_actor_camera_wearer_v1
export EDGE_MOTION_IO_DTYPE=float32
export EDGE_MOTION_SELF_ATTENTION_PER_GAP=0
export EDGE_REASONER_K_NORM_BYPASS=0
export EDGE_REASONER_KV_PROJECTION=none
export EDGE_REASONER_V_NORM_TRAINABLE=0
export EDGE_CROSS_ATTENTION_K_NORM=none
export EDGE_CROSS_ATTENTION_V_NORM=none
export EDGE_CROSS_ATTENTION_REASONER_ROPE=1
export EDGE_MOTION_ATTENTION_MODE=joint
export EDGE_TEXT_PACKING=edge_native_generation_prefix
run_date=20260915
case "$variant" in
    motion_fp32)
        export EDGE_MOTION_NORM_DTYPE=float32
        export EDGE_REASONER_K_NORM_TRAINABLE=0
        ;;
    reasoner_k_fp32)
        export EDGE_MOTION_NORM_DTYPE=bfloat16
        export EDGE_REASONER_K_NORM_TRAINABLE=1
        ;;
    reasoner_kv_as_is)
        export EDGE_MOTION_NORM_DTYPE=bfloat16
        export EDGE_REASONER_K_NORM_TRAINABLE=0
        export EDGE_REASONER_K_NORM_BYPASS=1
        ;;
    reasoner_kv_project|reasoner_kv_project_vnorm|reasoner_kv_project_vnorm_captiononly)
        export EDGE_MOTION_NORM_DTYPE=bfloat16
        export EDGE_REASONER_K_NORM_TRAINABLE=1
        export EDGE_REASONER_KV_PROJECTION=linear_identity_v1
        run_date=20260916
        if [[ "$variant" == reasoner_kv_project_vnorm* ]]; then
            export EDGE_REASONER_V_NORM_TRAINABLE=1
            run_date=20260917
        fi
        if [[ "$variant" == reasoner_kv_project_vnorm_captiononly ]]; then
            export EDGE_TEXT_PACKING=nano_caption_only
            run_date=20260919
        fi
        ;;
    self_cross_final_hidden_knorm_vnorm_nokrope)
        export EDGE_MOTION_NORM_DTYPE=bfloat16
        export EDGE_REASONER_K_NORM_TRAINABLE=0
        export EDGE_REASONER_K_NORM_BYPASS=1
        export EDGE_MOTION_ATTENTION_MODE=self_cross_final_hidden
        export EDGE_CROSS_ATTENTION_K_NORM=rmsnorm
        export EDGE_CROSS_ATTENTION_V_NORM=rmsnorm
        export EDGE_CROSS_ATTENTION_REASONER_ROPE=0
        run_date=20260919
        ;;
    self_cross_final_hidden|self_cross_layer_kv)
        export EDGE_MOTION_NORM_DTYPE=bfloat16
        export EDGE_REASONER_K_NORM_TRAINABLE=0
        export EDGE_REASONER_K_NORM_BYPASS=1
        export EDGE_MOTION_ATTENTION_MODE="$variant"
        run_date=20260916
        ;;
    *) printf 'Unknown normalization ablation: %s\n' "$variant" >&2; exit 2;;
esac
case "$data_mix" in
    mixed)
        cohort=nymeria_bones50_t2m_ti2m
        tasks='{"text2motion":0.75,"textimg2motion":0.25}'
        bones_frac=0.5
        viz_source=nymeria
        fixed_source=$RUN_ROOT/motion_expert_t2m_edge/edge_7layer_nymeria_bones50_t2m_ti2m_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260908/visualizations/fixed_samples
        ;;
    bones)
        cohort=bones100_t2m
        tasks='{"text2motion":1.0}'
        bones_frac=1
        viz_source=bones
        fixed_source=$RUN_ROOT/motion_expert_t2m_edge/edge_7layer_bones100_t2m_camheadstats_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_yonsei_20260912/visualizations/fixed_samples
        ;;
    *) printf 'Unknown data mixture: %s\n' "$data_mix" >&2; exit 2;;
esac
run_name=edge_7layer_${variant}_${cohort}_nanoparity_gbs128_camhead_v1_subjectv1_w1_1_5_contact_c0p05_v1_h10_${COSMOS_SITE}_${run_date}
export OUT=$output_root/$run_name
export WANDB_MODE=online
export WANDB_PROJECT=cosmos-motion-ft
export WANDB_RUN_NAME=$run_name
export WANDB_GROUP=cosmos3-edge-phase2-normalization-ablation
export WANDB_TAGS=cosmos3-edge,phase2,nanoparity,normalization-ablation,$variant,$data_mix,gbs128,camhead-stats,checkpoint10k,$COSMOS_SITE
if [[ "$variant" == reasoner_kv_project* ]]; then
    export WANDB_GROUP=cosmos3-edge-phase2-kv-projection-ablation
    export WANDB_TAGS=cosmos3-edge,phase2,nanoparity,kv-projection,ablation-A,$data_mix,gbs128,camhead-stats,checkpoint10k,$COSMOS_SITE
    if [[ "$variant" == reasoner_kv_project_vnorm* ]]; then
        export WANDB_TAGS=$WANDB_TAGS,learned-projected-v-norm
    fi
    if [[ "$variant" == reasoner_kv_project_vnorm_captiononly ]]; then
        export WANDB_TAGS=$WANDB_TAGS,nano-caption-only,t2m-packing-ablation
    fi
fi
export WANDB_DATA_DIR=$output_root/.wandb_runtime/data
if [[ "$variant" == self_cross_* ]]; then
    export WANDB_GROUP=cosmos3-edge-phase2-self-cross-ablation
    export WANDB_TAGS=cosmos3-edge,phase2,nanoparity,self-cross,$variant,$data_mix,gbs128,camhead-stats,checkpoint10k,$COSMOS_SITE
fi
export WANDB_CACHE_DIR=$output_root/.wandb_runtime/cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# These are independent one-GPU runs, never extra ranks of an inherited torchrun.
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
mkdir -p "$OUT" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
exec 8>"$OUT/.training.lock"
flock -n 8 || { printf 'Another trainer owns %s\n' "$OUT" >&2; exit 1; }
exec 9>"$HOME/cosmos_phase2_norm_${SLURM_JOB_ID}.lock"
flock -n 9 || { printf 'Another normalization trainer owns job %s\n' "$SLURM_JOB_ID" >&2; exit 1; }
exec > >(tee -a "$OUT/train_job${SLURM_JOB_ID}.log") 2>&1
printf '[norm-ablation] job=%s host=%s variant=%s data=%s out=%s\n' "$SLURM_JOB_ID" "$host" "$variant" "$data_mix" "$OUT"
printf '[norm-ablation] frozen reasoner; batch128 accum1; 200k steps; LR2e-4 warmup1k cosine floor0; seed0\n'
printf '[norm-ablation] centered losses1/1/5 contact0.05/1/10; camhead statistics for both data mixtures\n'
printf '[norm-ablation] motion_norm_dtype=%s trainable_reasoner_k_norm=%s bypass_interface_k_norm=%s\n' "$EDGE_MOTION_NORM_DTYPE" "$EDGE_REASONER_K_NORM_TRAINABLE" "$EDGE_REASONER_K_NORM_BYPASS"
printf '[norm-ablation] checkpoints at positive multiples of 10000 only; rolling and signal recovery disabled\n'
printf '[architecture] attention_mode=%s; self/cross variants use SA -> CA -> one FFN; no interface K norm\n' "$EDGE_MOTION_ATTENTION_MODE"
printf '[interface] reasoner_kv_projection=%s; reasoner causal path unchanged\n' "$EDGE_REASONER_KV_PROJECTION"
printf '[prompt] text_packing=%s; TI2M image/chat packing and caption subject policies unchanged\n' "$EDGE_TEXT_PACKING"
if [[ "$variant" == reasoner_kv_project_vnorm_captiononly ]]; then
    printf '[prompt] T2M: caption IDs only, no BOS/EOS/SOG wrapper; empty CFG caption: one Edge EOS; native positions follow actual token count\n'
fi
if [[ "$variant" == self_cross_final_hidden_knorm_vnorm_nokrope ]]; then
    printf '[cross] final post-norm hidden -> per-block learned K/V projections -> independent learned per-head K/V RMSNorm; no added reasoner-key RoPE; V never rotated\n'
    printf '[cross] motion self-attention and cross-query RoPE unchanged; frozen reasoner keeps native positional encoding; BF16 norm gains with native FP32 reductions\n'
fi
if [[ "$variant" == reasoner_kv_project_vnorm* ]]; then
    printf '[interface A+Vnorm] K -> projection -> learned K RMSNorm -> native RoPE; V -> projection -> fresh learned V RMSNorm; 128-D per-head reduction, FP32 gains\n'
fi
if [[ "$variant" == reasoner_kv_project ]]; then
    printf '[interface A] existing pre-RoPE K/V -> independent bias-free 1024x1024 identity-initialized projections; K -> trainable copied RMSNorm -> native RoPE; V has no added norm\n'
    printf '[interface A] seven joint blocks; 14680064 projection parameters plus 896 K-norm gains; no extra attention blocks\n'
fi
if [[ ! -f "$OUT/visualizations/fixed_samples/manifest.json" ]]; then
    mkdir -p "$OUT/visualizations"
    if [[ "$COSMOS_SITE" == yonsei ]]; then
        cp -a "$fixed_source" "$OUT/visualizations/fixed_samples"
    fi
fi
exec bash "$REPO_ROOT/motion_expert_t2m_edge/run.sh" \
    "$REPO_ROOT/motion_expert_t2m_edge/train.py" \
    --out "$OUT" --steps 200000 --T 200 --ti2m-frames 97 \
    --batch-size 128 --grad-accum 1 --num-workers 8 \
    --task-weights "$tasks" --bones-frac "$bones_frac" --reasoner-image-size 256 \
    --lr 2e-4 --warmup 1000 --min-lr-ratio 0 --weight-decay 0 --adam-beta2 0.99 \
    --loss-reduction per_task --loss-guard nano --allow-tf32 0 --grad-clip 1 \
    --cfg-dropout 0.1 --seed 0 \
    --w-feat 1 --w-joint 1 --w-smooth 5 --joint-loss-mode centered \
    --w-global-rot 0 --w-global-trans 0 --w-contact 0.05 --w-foot-vel 1 --w-foot-height 10 \
    --save-every 10000 --recovery-save-every 0 --checkpoint-interval-only \
    --wandb-mode online --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "$run_name" --wandb-group "$WANDB_GROUP" --require-wandb \
    --wandb-service-wait 300 --wandb-init-attempts 3 --wandb-init-retry-delay 10 \
    --viz-every 5000 --viz-samples-per-task 5 --viz-source "$viz_source" \
    --viz-steps 35 --viz-guidance 2 --viz-frame-stride 2 --viz-at-start --require-viz \
    --resume auto
