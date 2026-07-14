#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --job-name=p3brvizsmk
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p3brvizsmk-%j.out
#SBATCH --exclusive

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
PHASE1=/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k/checkpoints/iter_000100000
PHASE2=/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt
BRIDGE_CKPT=/weka/jungbin/cosmos_motion_ft_runs/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/ckpt_step005000.pt
RUN_NAME=smoke_phase3_bridge_ddp_viz_fix_${SLURM_JOB_ID}

echo "[p3brvizsmk] node=$(hostname) date=$(date) run=${RUN_NAME}"
echo "[p3brvizsmk] 8-rank gate: one held-out V2M + one held-out M2V, 2 solver steps"
test -f "${PHASE1}/model/.metadata"
test -f "${PHASE2}"
test -f "${BRIDGE_CKPT}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

bash "${D}/run.sh" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${D}/train.py" --ddp \
  --tasks video2motion motimg2video \
  --task_weights '{"video2motion":0.5,"motimg2video":0.5}' \
  --bones_frac 0.0 \
  --T 97 \
  --objective x0 \
  --motion_schedule native \
  --motion_shift 3 \
  --motion_num_train_timesteps 1000 \
  --motion_native_solver euler \
  --gen_schedule native \
  --gen_shift 3 \
  --gen_num_train_timesteps 1000 \
  --gen_native_solver unipc \
  --gen_packing native \
  --gen_fps 20 \
  --gen_temporal_margin 15000 \
  --motion_mrope cosmos3d \
  --coupling bridge_local \
  --gen_lora --gen_lora_rank 16 --gen_lora_alpha 32 --freeze_gen \
  --freeze_motion \
  --init_gen "${PHASE1}" \
  --init_gen_dcp_weights ema \
  --init_motion "${PHASE2}" \
  --precomputed_latents \
  --batch_size 1 \
  --steps 200000 \
  --lr 2e-4 --warmup 1000 --lr_schedule cosine \
  --save_every 5000 \
  --viz_every 5000 \
  --viz_n 2 \
  --viz_steps 2 \
  --viz_guidance 1.0 \
  --viz_frame_stride 4 \
  --require_viz \
  --viz_only \
  --resume "${BRIDGE_CKPT}" \
  --out "${RUN_NAME}"

echo "[p3brvizsmk] PASS date=$(date)"
