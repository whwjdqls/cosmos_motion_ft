#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=96:00:00
#SBATCH --job-name=t2mtinat
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-t2mtinat-%j.out
#SBATCH --exclusive

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
RUN_NAME=${PHASE2_NATIVE_RUN_NAME:-ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d}

echo "[t2mtinat] node=$(hostname) date=$(date)"
echo "[t2mtinat] run=${RUN_NAME}"
echo "[t2mtinat] output T=200: full/native T2M + 97-valid-frame aligned TI2M; x0; native shift=3"
echo "[t2mtinat] TI2M reasoner image=256x256 (64 visual tokens); viz includes image|GT|generated"
echo "[t2mtinat] solver=euler (POC-proven); official UniPC remains an inference-only ablation"
echo "[t2mtinat] BONES overview rows use content_natural_desc_4; single/multi timeline unchanged"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

export PHASE2_NATIVE_MIN_FREE_MIB=${PHASE2_NATIVE_MIN_FREE_MIB:-120000}
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python -c '
import os
import subprocess
import sys
threshold = int(os.environ["PHASE2_NATIVE_MIN_FREE_MIB"])
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
bad = []
for line in raw.splitlines():
    idx, free = [part.strip() for part in line.split(",")]
    if int(free) < threshold:
        bad.append((idx, int(free)))
if bad:
    print(f"[t2mtinat] ERROR: GPUs below {threshold} MiB free: {bad}", file=sys.stderr)
    sys.exit(1)
print(f"[t2mtinat] GPU memory preflight passed: every GPU >= {threshold} MiB free")
'

bash "$D/run.sh" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$D/train.py" --ddp \
  --tasks text2motion textimg2motion \
  --task_weights '{"text2motion":0.75,"textimg2motion":0.25}' \
  --bones_frac 0.5 \
  --T 200 \
  --ti2m_frames 97 \
  --objective x0 \
  --motion_schedule native \
  --motion_shift 3 \
  --motion_num_train_timesteps 1000 \
  --motion_native_solver euler \
  --motion_mrope cosmos3d \
  --coupling joint \
  --textimg_condition reasoner \
  --reasoner_image_size 256 \
  --viz_n 4 --viz_every 2000 --viz_frame_stride 2 --resume auto \
  --out "$RUN_NAME" \
  --steps 200000 --batch_size 32
