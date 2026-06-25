#!/bin/bash
# Direct (non-slurm) LoRA+DDP+FK training launch on one node, for co-tenant use.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd /home/jungbin_cho/cosmos-framework
NGPU=${1:-8}; BATCH=${2:-32}; STEPS=${3:-200000}; MODE=${4:-lora}; LR=${5:-2e-5}; PORT=${6:-51234}
# MODE=lora -> LoRA+motion heads; MODE=full -> full generator. Same base LR (2e-5),
# constant schedule, no head/backbone LR split (per user). ckpt+viz every 10k.
if [ "$MODE" = "lora" ]; then LORA_FLAG="--lora"; else LORA_FLAG=""; fi
RUN=full_${MODE}_ddp_kimodofk_$(hostname|grep -o '.$')_$(date +%Y%m%d_%H%M%S)
OUT=/weka/jungbin/cosmos_motion_ft_runs/$RUN
mkdir -p "$OUT"
echo "[train] host=$(hostname) ngpu=$NGPU batch=$BATCH steps=$STEPS mode=$MODE lr=$LR out=$OUT start=$(date)"
torchrun --standalone --nproc_per_node="$NGPU" --master_port="$PORT" \
  /home/jungbin_cho/cosmos_motion_ft/train_motion_ft.py \
  --ddp $LORA_FLAG --loss kimodo --data /weka/jungbin/seed/cosmos_text_motion_full \
  --out "$OUT" --steps "$STEPS" --batch_size "$BATCH" --lr "$LR" --max_frames 200 \
  --lr_schedule constant --head_lr_mult 1.0 \
  --save_every 10000 --viz_every 10000 --log_every 20 --grad_clip 1.0 --num_workers 4
echo "[train] DONE $(date) -> $OUT"
