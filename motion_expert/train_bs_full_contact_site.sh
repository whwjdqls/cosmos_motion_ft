#!/usr/bin/env bash
# Fresh original LLM2Vec POC: prepare exact captions, encode, then train.
# Reuses an owned allocation; never allocates/cancels a Slurm job.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an owned single-GPU compute allocation}"
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei or grasp}"
host=$(hostname -s)
case "$host" in *login*|cs-gpu-01|node31|node33) printf 'Unsupported compute host: %s\n' "$host" >&2; exit 2;; esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$COSMOS_SITE" in
    custom)
        source "$repo/restored_env.sh"
        weka=$WEKA_ROOT
        export BS_RUN_ROOT=${BS_RUN_ROOT:-$RUN_ROOT/motion_expert_poc}
        ;;
    yonsei)
        weka=/lustre/whwjdqls99/cosmos/weka
        export BS_RUN_ROOT=${BS_RUN_ROOT:-/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/motion_expert_poc}
        ;;
    grasp)
        weka=/mnt/projects/ll/jungbinc/weka
        export BS_RUN_ROOT=${BS_RUN_ROOT:-$weka/cosmos_motion_ft_runs/motion_expert_poc}
        ;;
    *) printf 'Unsupported site: %s\n' "$COSMOS_SITE" >&2; exit 2;;
esac
export KIMODO_ROOT=${KIMODO_ROOT:-$HOME/kimodo_open}
export BS_DATA_ROOT=${BS_DATA_ROOT:-$weka/seed/soma_proportional_uniegomotion_20fps}
export BS_NATURAL_CSV=${BS_NATURAL_CSV:-$weka/seed/metadata/seed_metadata_v004.csv}
export BS_TEMPORAL_JSONL=${BS_TEMPORAL_JSONL:-$weka/seed/metadata/seed_metadata_v002_temporal_labels.jsonl}
export BS_MULTI_JSONL=${BS_MULTI_JSONL:-$weka/seed/multi_timeline.jsonl}
export BS_SPLIT_DIR=${BS_SPLIT_DIR:-$weka/Kimodo-Motion-Gen-Benchmark/splits}
export SHAPE_TMR_BUNDLE=${SHAPE_TMR_BUNDLE:-$weka/shape_aware_motion_eval_c45_20260715}
name=${BS_POC_RUN_NAME:-bs_native_full_contact_llm2vec_w1_1_5_c0p05_v1_h10_${COSMOS_SITE}_20260919}
out=$BS_RUN_ROOT/$name
mkdir -p "$out"
exec 9>"$HOME/cosmos_phase2_norm_${SLURM_JOB_ID}.lock"
flock -n 9 || { printf 'Allocation already has an active workload\n' >&2; exit 1; }
exec 8>"$out/.pipeline.lock"
flock -n 8 || { printf 'POC pipeline already active\n' >&2; exit 1; }
exec > >(tee -a "$out/pipeline_job${SLURM_JOB_ID}.log") 2>&1
printf '[poc] job=%s node=%s output=%s\n' "$SLURM_JOB_ID" "$host" "$out"
if [[ -e "$out/latest.pt" ]]; then
    printf 'Refusing to restart an existing run; POC checkpoints are model-only, not exact resumes.\n' >&2
    exit 1
fi
# This runner supplies the same pinned Cosmos/Kimodo/LLM2Vec environment as C45.
# It does not launch evaluator training or change this POC recipe.
runner=$repo/shape_aware_TMR/nymeria_run.sh
train_split=$BS_SPLIT_DIR/train_split_paths.txt
viz_split=$BS_SPLIT_DIR/test_content_split_paths.txt
index=$out/bs_train_index.json
cache=${BS_POC_TEXT_CACHE:-$out/bones_seed_llm2vec.pt}
printf 'preparing_index\n' > "$out/pipeline_stage.txt"
bash "$runner" "$repo/motion_expert/bs_prepare_training_cache.py" \
    --train-split "$train_split" --viz-split "$viz_split" \
    --index "$index" --captions "$out/captions.json"
if [[ -z "${BS_POC_TEXT_CACHE:-}" ]]; then
    printf 'encoding_llm2vec\n' > "$out/pipeline_stage.txt"
    bash "$runner" "$repo/motion_expert_joint_attention/precompute_nymeria_tmr_text.py" \
        --captions-json "$out/captions.json" --out "$cache" \
        --bundle-root "$SHAPE_TMR_BUNDLE" --batch-size 8
fi
printf 'training\n' > "$out/pipeline_stage.txt"
bash "$runner" "$repo/motion_expert/bs_native_train.py" \
    --run_name "$name" --steps 200000 --batch_size 128 --lr 2e-4 \
    --d 512 --layers 8 --heads 8 --ffn 2048 --seed 0 \
    --pred x0 --schedule native --native_shift 3 --native_num_train_timesteps 1000 \
    --cfg_dropout 0.1 --warmup 1000 --grad_clip 1 \
    --w_feat 1 --w_joint 1 --w_smooth 5 \
    --w_contact 0.05 --w_foot_vel 1 --w_foot_height 10 --contact_logit_scale 2 \
    --motion_fps 20 --num_workers 8 --dataloader_rng historical_global \
    --step_indexing completed_updates --log_every 20 --save_every 10000 \
    --viz_every 10000 --viz_n 4 --viz_frames 200 --viz_steps 35 --viz_guidance 2 \
    --inline_eval_every 0 --cache_path "$cache" --index_cache "$index" \
    --train_split "$train_split" --viz_split "$viz_split" \
    --mean "$BS_DATA_ROOT/Mean_uniego.npy" --std "$BS_DATA_ROOT/Std_uniego.npy" \
    --normalization_tag bones_seed_proportional_20fps
printf 'completed\n' > "$out/pipeline_stage.txt"
printf '[poc] completed; allocation retained\n'
