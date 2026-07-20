#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=96:00:00
#SBATCH --job-name=t2mticnt
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-t2mticnt-%j.out
#SBATCH --exclusive

set -euo pipefail

ROOT=/home/jungbin_cho/cosmos_motion_ft
D="$ROOT/motion_expert_joint_attention"
RUN_NAME=${PHASE2_CONTACT_RUN_NAME:-ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35}
NUM_WORKERS=${PHASE2_CONTACT_NUM_WORKERS:-4}
PREFETCH_FACTOR=${PHASE2_CONTACT_PREFETCH_FACTOR:-1}
DATALOADER_TIMEOUT=${PHASE2_CONTACT_DATALOADER_TIMEOUT:-300}
MAX_RESTARTS=${PHASE2_CONTACT_MAX_RESTARTS:-2}
SAVE_EVERY=${PHASE2_CONTACT_SAVE_EVERY:-5000}
RESUME=${PHASE2_CONTACT_RESUME:-auto}
MEAN="$ROOT/motion_expert/stats/uniego283_mean.npy"
STD="$ROOT/motion_expert/stats/uniego283_std.npy"
MEAN_SHA=bd1d6bdc9a3b026fe1e5b28899441655ee36672c69c3e6e6389e9baff4b400d3
STD_SHA=ee069e3aa9f3cd1a1e70135cc00bc751030f8045fae6bbfb7b4f5b32fa65f28c

echo "[t2mticnt] node=$(hostname) date=$(date)"
echo "[t2mticnt] run=${RUN_NAME}"
echo "[t2mticnt] POC-proven loss recipe on the native joint-attention Phase-2 pipeline"
echo "[t2mticnt] losses feat/joint/smooth=1/1/5 contact/foot_vel/foot_height=0.05/1/10 scale=2 fps=20"
echo "[t2mticnt] output T=200: full/native T2M + 97-valid-frame aligned TI2M; x0; native shift=3"
echo "[t2mticnt] motion sampling/viz: official NVIDIA UniPC, 35 steps"
echo "[t2mticnt] TI2M reasoner image=256x256 (64 visual tokens); viz includes image|GT|generated"
echo "[t2mticnt] BONES overview rows use content_natural_desc_4; single/multi timeline unchanged"
echo "[t2mticnt] loader workers=${NUM_WORKERS}/rank prefetch=${PREFETCH_FACTOR} timeout=${DATALOADER_TIMEOUT}s"
echo "[t2mticnt] recovery max_restarts=${MAX_RESTARTS} save_every=${SAVE_EVERY} resume=${RESUME}"
printf '%s  %s\n' "$MEAN_SHA" "$MEAN" | sha256sum -c -
printf '%s  %s\n' "$STD_SHA" "$STD" | sha256sum -c -
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

export PHASE2_CONTACT_MIN_FREE_MIB=${PHASE2_CONTACT_MIN_FREE_MIB:-120000}
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python -c '
import os
import subprocess
import sys
threshold = int(os.environ["PHASE2_CONTACT_MIN_FREE_MIB"])
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
    print(f"[t2mticnt] ERROR: GPUs below {threshold} MiB free: {bad}", file=sys.stderr)
    sys.exit(1)
print(f"[t2mticnt] GPU memory preflight passed: every GPU >= {threshold} MiB free")
'

bash "$D/run.sh" -m torch.distributed.run --standalone --nproc_per_node=8 \
  --max_restarts "$MAX_RESTARTS" --monitor_interval 5 \
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
  --motion_native_solver unipc \
  --motion_mrope cosmos3d \
  --coupling joint \
  --textimg_condition reasoner \
  --reasoner_image_size 256 \
  --w_feat 1 \
  --w_joint 1 \
  --w_smooth 5 \
  --w_contact 0.05 \
  --w_foot_vel 1 \
  --w_foot_height 10 \
  --contact_logit_scale 2 \
  --motion_fps 20 \
  --num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" \
  --dataloader_timeout "$DATALOADER_TIMEOUT" --save_every "$SAVE_EVERY" \
  --viz_n 4 --viz_every 2000 --viz_steps 35 --viz_frame_stride 2 --require_viz \
  --resume "$RESUME" \
  --out "$RUN_NAME" \
  --steps 200000 --batch_size 32
