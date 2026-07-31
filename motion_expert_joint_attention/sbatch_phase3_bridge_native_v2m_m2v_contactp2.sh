#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=5-00:00:00
#SBATCH --job-name=p3brp2ct
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p3brp2ct-%j.out
#SBATCH --exclusive

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
PHASE1=${PHASE1_INIT:-/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k/checkpoints/iter_000100000}
PHASE2=${PHASE2_INIT:-/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35/ckpt_step200000.pt}
RUN_NAME=${PHASE3_CONTACT_P2_RUN_NAME:-ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k}

echo "[p3brp2ct] node=$(hostname) date=$(date)"
echo "[p3brp2ct] run=${RUN_NAME}"
echo "[p3brp2ct] Phase 1=${PHASE1} (EMA, LoRA rank16/alpha32, native Waver+shift3+UniPC)"
echo "[p3brp2ct] Phase 2=${PHASE2} (x0, native shift3+UniPC, contact-aware T2M+TI2M)"
echo "[p3brp2ct] tasks=50% video2motion + 50% motimg2video; T=97; batch=4/GPU (global 32)"
echo "[p3brp2ct] frozen specialists; only 12 local directional modality bridges train"
echo "[p3brp2ct] V2M loss: feat/joint/smooth=1/1/5 contact/foot_vel/foot_height=0.05/1/10 scale=2"
echo "[p3brp2ct] no head-camera loss and no joint-target multitask objectives"
echo "[p3brp2ct] causal 4x locality: latent0<->frame0, latent1<->frames1..4, ..., latent24<->93..96"
echo "[p3brp2ct] required viz every 5k + final: 2 V2M and 2 M2V, native UniPC-30 sampling"

if [[ -d "${PHASE1}" ]]; then
  test -f "${PHASE1}/model/.metadata"
else
  test -f "${PHASE1}"
fi
test -f "${PHASE2}"
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
  --motion_native_solver unipc \
  --gen_schedule native \
  --gen_shift 3 \
  --gen_num_train_timesteps 1000 \
  --gen_native_solver unipc \
  --gen_packing native \
  --gen_fps 20 \
  --gen_temporal_margin 15000 \
  --motion_mrope cosmos3d \
  --coupling bridge_local \
  --w_feat 1 \
  --w_joint 1 \
  --w_smooth 5 \
  --w_contact 0.05 \
  --w_foot_vel 1 \
  --w_foot_height 10 \
  --contact_logit_scale 2 \
  --motion_fps 20 \
  --gen_lora --gen_lora_rank 16 --gen_lora_alpha 32 --freeze_gen \
  --freeze_motion \
  --init_gen "${PHASE1}" \
  --init_gen_dcp_weights ema \
  --init_motion "${PHASE2}" \
  --precomputed_latents \
  --batch_size 4 \
  --steps 200000 \
  --lr 2e-4 --warmup 1000 --lr_schedule cosine \
  --save_every 5000 \
  --viz_every 5000 \
  --viz_n 4 \
  --viz_steps 30 \
  --viz_guidance 1.0 \
  --viz_frame_stride 2 \
  --require_viz \
  --resume auto \
  --out "${RUN_NAME}"
