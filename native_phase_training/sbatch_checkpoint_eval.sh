#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=nativeviz
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-nativeviz-%j.out

set -euo pipefail

: "${CHECKPOINT_PATH:?CHECKPOINT_PATH must point to an iter_XXXXXXXXX DCP checkpoint}"
: "${EVAL_INPUT_DIR:?EVAL_INPUT_DIR must contain the four native JSONL files}"
: "${EVAL_OUTPUT_DIR:?EVAL_OUTPUT_DIR is required}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework:${PYTHONPATH:-}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/weka/jungbin/wan22_vae/Wan2.2_VAE.pth}
export NYMERIA_RESOLUTION=256
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

echo "[nativeviz] node=$(hostname) date=$(date)"
echo "[nativeviz] checkpoint=${CHECKPOINT_PATH}"
echo "[nativeviz] inputs=${EVAL_INPUT_DIR}"
echo "[nativeviz] output=${EVAL_OUTPUT_DIR}"
echo "[nativeviz] visible_gpus=${CUDA_VISIBLE_DEVICES:-all}"

/home/jungbin_cho/miniforge3/envs/cosmos/bin/python - <<'PY'
import os
import sys

import torch

threshold_mib = int(os.environ.get("NATIVEP1_EVAL_MIN_FREE_MIB", "120000"))
free_bytes, _ = torch.cuda.mem_get_info(0)
free_mib = free_bytes // (1024 * 1024)
if free_mib < threshold_mib:
    print(
        f"[nativeviz] ERROR: visible GPU has {free_mib} MiB free; required {threshold_mib} MiB",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"[nativeviz] GPU preflight passed: {free_mib} MiB free")
PY

cd /home/jungbin_cho/cosmos-framework

/home/jungbin_cho/miniforge3/envs/cosmos/bin/torchrun --standalone --nproc_per_node=1 \
  -m cosmos_framework.scripts.inference \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-file native_phase_training/inference_config.py \
  --experiment world_camera_nymeria_latent_nano \
  --sampler unipc \
  --use-ema-weights \
  --parallelism-preset latency \
  --dp-shard-size 1 --dp-replicate-size 1 --cp-size 1 --cfgp-size 1 \
  --no-use-torch-compile --no-use-cuda-graphs --no-guardrails \
  -o "${EVAL_OUTPUT_DIR}" \
  -i "${EVAL_INPUT_DIR}/fd_input.jsonl" \
     "${EVAL_INPUT_DIR}/invdyn_input.jsonl" \
     "${EVAL_INPUT_DIR}/policy_input.jsonl" \
     "${EVAL_INPUT_DIR}/i2v_input.jsonl"

cd /home/jungbin_cho/cosmos_motion_ft

/home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
  native_phase_training/visualize_checkpoint.py \
  --inference-root "${EVAL_OUTPUT_DIR}" \
  --eval-root "${EVAL_INPUT_DIR}"

echo "[nativeviz] done date=$(date)"
