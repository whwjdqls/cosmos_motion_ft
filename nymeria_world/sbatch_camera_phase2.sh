#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=cam_world
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-camworld-%j.out

# Cosmos3-Nano egocentric camera-action world-model SFT (NymeriaPlus, camera-only Phase 2).
# LoRA on generator + frozen reasoner + fine-tuned pretrained camera action heads.
# FSDP2 pure-replicate (dp_shard=1) = DDP-equivalent (no param sharding).
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd /home/jungbin_cho/cosmos-framework

export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft/nymeria_world
export BASE_CHECKPOINT_PATH=/weka/jungbin/cosmos3_nano_dcp
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/weka/jungbin/cosmos_motion_ft_runs
export SAVE_TRAINABLE_ONLY=1   # save only trainable params, not the 16B base
# which trainable keys to save (lora_A/lora_B always kept). Set INTERNALLY (commas can't go through
# sbatch --export, which splits on them). FULL-GEN must also save moe_gen + gen/vae heads.
if [ -n "${NYMERIA_FULL_FT:-}" ]; then
  export SAVE_TRAINABLE_KEYS="moe_gen,time_embedder,vae2llm,llm2vae,action2llm,llm2action,action_modality_embed"
else
  export SAVE_TRAINABLE_KEYS="action2llm,llm2action,action_modality_embed"
fi

MAX_ITER=${MAX_ITER:-100000}
SAVE_ITER=${SAVE_ITER:-5000}
BATCH=${BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}   # effective per-GPU batch = BATCH * GRAD_ACCUM
export NYMERIA_NUM_FRAMES=${NYMERIA_NUM_FRAMES:-97}   # read by the experiment config
RUN_NAME=${RUN_NAME:-world_camera_nymeria}            # separates output dir / TB / job name per run
PORT=${PORT:-51237}                                   # distinct port per concurrent run
export TB_LOG_DIR=/weka/jungbin/cosmos_motion_ft_runs/tensorboard/${RUN_NAME}
mkdir -p "$TB_LOG_DIR"
RUN=/weka/jungbin/cosmos_motion_ft_runs/${RUN_NAME}_sbatch_$(date +%Y%m%d_%H%M%S).log
echo "$RUN" > /weka/jungbin/cosmos_motion_ft_runs/${RUN_NAME}_LATEST.txt
echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi -L | wc -l) name=$RUN_NAME frames=$NYMERIA_NUM_FRAMES iters=$MAX_ITER batch=$BATCH -> $RUN"

torchrun --standalone --nproc_per_node="${NPROC:-8}" --master_port="$PORT" \
    -m cosmos_framework.scripts.train \
    --sft-toml examples/toml/sft_config/world_camera_nymeria_repro.toml \
    job.name="$RUN_NAME" trainer.max_iter="$MAX_ITER" trainer.logging_iter=50 \
    trainer.grad_accum_iter="$GRAD_ACCUM" \
    checkpoint.save_iter="$SAVE_ITER" dataloader_train.max_samples_per_batch="$BATCH" \
    ${EXTRA_OPTS:-} \
    2>&1 | tee "$RUN"
