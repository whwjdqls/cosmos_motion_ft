#!/usr/bin/env bash
# Actions: prepare, text, train, eval, all. Does not allocate or cancel GPUs.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an owned single-GPU Slurm allocation}"
: "${COSMOS_SITE:?Set yonsei or grasp}"
host=$(hostname -s)
case "$host" in *login*|cs-gpu-01) printf 'Compute node required: %s\n' "$host" >&2; exit 2;; esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runner="$repo/shape_aware_TMR/nymeria_run.sh"
case "$COSMOS_SITE" in
    custom)
        source "$repo/restored_env.sh"
        weka=$WEKA_ROOT
        root=${NYMERIA_TMR_ROOT:-$RUN_ROOT/shape_tmr/nymeria_c45_camhead_v1_20260918}
        hub=$HUGGINGFACE_HUB_CACHE
        ;;
    yonsei)
        weka=/lustre/whwjdqls99/cosmos/weka
        root=${NYMERIA_TMR_ROOT:-/scratch2/whwjdqls99/cosmos/cosmos_motion_ft_runs/shape_tmr/nymeria_c45_camhead_v1_20260918}
        hub=${HUGGINGFACE_HUB_CACHE:-/scratch2/whwjdqls99/cosmos/cache/huggingface_nano_viz/hub}
        ;;
    grasp)
        weka=/mnt/projects/ll/jungbinc/weka
        root=${NYMERIA_TMR_ROOT:-$weka/cosmos_motion_ft_runs/shape_tmr/nymeria_c45_camhead_v1_20260918}
        hub=${HUGGINGFACE_HUB_CACHE:-/mnt/projects/ll/jungbinc/.cache/huggingface/hub}
        ;;
    *) printf 'Unsupported site: %s\n' "$COSMOS_SITE" >&2; exit 2;;
esac
export SHAPE_TMR_BUNDLE=${SHAPE_TMR_BUNDLE:-$weka/shape_aware_motion_eval_c45_20260715}
official=${OFFICIAL_TMR_ROOT:-$hub/models--nvidia--TMR-SOMA-RP-v1/snapshots/e427752ae3446dedba49e928c93ddc9f0e413401}
stats=${NYMERIA_TMR_STATS:-$official/stats/motion}
data=${NYMERIA_TMR_DATA_DIR:-$root/data}
cache=${NYMERIA_TMR_TEXT_CACHE:-$root/nymeria_llm2vec.pt}
out=${NYMERIA_TMR_OUT:-$root/c45_official30fps_nymeria_10k}
nymeria=$weka/nymeriaplus_kimodo_proportional
action=${1:-all}
mkdir -p "$root"
exec 9>"$root/.pipeline.lock"
flock -n 9 || { printf 'Another process owns %s\n' "$root" >&2; exit 1; }
exec > >(tee -a "$root/${action}_job${SLURM_JOB_ID}.log") 2>&1
printf '[nymeria-c45] action=%s job=%s node=%s root=%s\n' "$action" "$SLURM_JOB_ID" "$host" "$root"

prepare() {
    bash "$runner" "$repo/shape_aware_TMR/prepare_nymeria.py" \
        --manifest "${NYMERIA_TMR_MANIFEST:-$nymeria/video/manifest_video.jsonl}" \
        --split-file "${NYMERIA_TMR_SPLIT_FILE:-$nymeria/train_test_split.json}" \
        --uniego-root "${NYMERIA_TMR_MOTION_ROOT:-$nymeria/uniego_rep_camhead_v1}" \
        --floor-calibration "${NYMERIA_TMR_FLOOR:-$nymeria/metadata/floor_calibration.json}" \
        --representation "${NYMERIA_TMR_REPRESENTATION:-camera_head_recanonicalization_v1}" \
        --out "$data" --seed 0 --data-fps 20 --validation-fraction 0.1
}
text_cache() {
    args=(--captions-json "$data/captions.json" --out "$cache" --bundle-root "$SHAPE_TMR_BUNDLE"
          --batch-size "${NYMERIA_TMR_TEXT_BATCH:-8}")
    if [[ -n "${NYMERIA_TMR_BASE_TEXT_CACHE:-}" ]]; then
        args+=(--base-cache "$NYMERIA_TMR_BASE_TEXT_CACHE")
    fi
    bash "$runner" "$repo/motion_expert_joint_attention/precompute_nymeria_tmr_text.py" "${args[@]}"
}
train() {
    init=()
    if [[ -f "$out/last.pt" ]]; then
        init+=(--resume-from "$out/last.pt")
    else
        for item in text_encoder motion_encoder motion_decoder; do
            [[ -f "$official/last_weights/$item.pt" ]] || {
                printf 'Missing official initialization: %s/last_weights/%s.pt; set OFFICIAL_TMR_ROOT\n' "$official" "$item" >&2
                exit 1
            }
        done
        init+=(--pretrained-text-encoder "$official/last_weights/text_encoder.pt"
               --pretrained-motion-encoder "$official/last_weights/motion_encoder.pt"
               --pretrained-motion-decoder "$official/last_weights/motion_decoder.pt")
    fi
    bash "$runner" "$repo/shape_aware_TMR/st_train.py" \
        --dataset nymeria --nymeria-data-dir "$data" --out-dir "$out" \
        --stats-path "$stats" --text-emb-cache "$cache" --eval-text-cache "$cache" \
        --fps 30 --data-fps 20 --freeze-text-encoder --text-use-mean --motion-use-mean \
        --lambda-recon 0.01 --lambda-kl-motion 3e-6 --lambda-kl-text 0 \
        --lambda-contrastive 1 --lambda-align 0 --weight-decay 0.01 \
        --max-steps 10000 --warmup 1000 --lr 5e-5 --batch 128 --num-workers 6 \
        --ckpt-every 1000 --eval-every 1000 --eval-cases 0 \
        --frozen-dup-filter --text-dup-threshold 0.95 --aug-feat-noise-std 0.02 \
        --aug-time-jitter-sec 0.3 --info-nce-temp 0.1 --queue-size 8192 \
        --latent-dim 256 --ff-size 1024 --enc-layers 6 --dec-layers 4 --num-heads 4 \
        --dropout 0.1 --grad-clip 1 --seed 0 "${init[@]}"
}
evaluate() {
    bash "$runner" "$repo/shape_aware_TMR/nymeria_eval.py" \
        --ckpt "${NYMERIA_TMR_EVAL_CKPT:-$out/best.pt}" \
        --stats-path "$stats" --data-dir "$data" --text-cache "$cache" \
        --split "${NYMERIA_TMR_EVAL_SPLIT:-test}" \
        --out "${NYMERIA_TMR_EVAL_OUT:-$out/test_selected_on_validation.json}"
}
case "$action" in
    prepare) prepare;;
    text) text_cache;;
    train) train;;
    eval) evaluate;;
    all) prepare; text_cache; train; evaluate;;
    *) printf 'Usage: bash %s {prepare|text|train|eval|all}\n' "$0" >&2; exit 2;;
esac
printf '[nymeria-c45] completed action=%s; allocation retained\n' "$action"
