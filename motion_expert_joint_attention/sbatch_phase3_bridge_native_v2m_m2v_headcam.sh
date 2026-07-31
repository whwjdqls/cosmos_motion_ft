#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=5-00:00:00
#SBATCH --job-name=p3brheadcam
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p3brheadcam-%j.out
#SBATCH --exclusive

set -euo pipefail

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
PHASE1=${PHASE1_INIT:-/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k/checkpoints/iter_000100000}
PHASE2=${PHASE2_INIT:-/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt}
CALIBRATION=${PHASE3_HEAD_CAMERA_CALIBRATION:-${D}/head_camera_calibration_train.json}
RUN_NAME=${PHASE3_HEAD_CAMERA_RUN_NAME:-ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam}
W_TRANS=${PHASE3_HEAD_CAMERA_W_TRANS:-0.05}
W_ROT=${PHASE3_HEAD_CAMERA_W_ROT:-0.05}

echo "[p3brheadcam] node=$(hostname) date=$(date)"
echo "[p3brheadcam] run=${RUN_NAME}"
echo "[p3brheadcam] Phase 1=${PHASE1} (EMA, LoRA rank16/alpha32, native Waver+shift3+UniPC)"
echo "[p3brheadcam] Phase 2=${PHASE2} (x0, native shifted logit-normal+UniPC, T2M+TI2M)"
echo "[p3brheadcam] tasks=50% video2motion + 50% motimg2video; T=97; batch=4/GPU (global 32)"
echo "[p3brheadcam] frozen specialists; only 12 local directional modality bridges train"
echo "[p3brheadcam] M2V gets clean camera actions derived only from clean motion; V2M camera is target-only"
echo "[p3brheadcam] relative head-camera loss weights: translation=${W_TRANS} rotation=${W_ROT}"
echo "[p3brheadcam] calibration=${CALIBRATION} (train split only; no absolute translations)"
echo "[p3brheadcam] required viz every 5k + final: 2 V2M and 2 M2V, native UniPC 30-step sampling"

if [[ -d "${PHASE1}" ]]; then
  test -f "${PHASE1}/model/.metadata"
else
  test -f "${PHASE1}"
fi
test -f "${PHASE2}"
test -f "${CALIBRATION}"
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
  --head_camera_alignment \
  --head_camera_calibration "${CALIBRATION}" \
  --w_head_camera_trans "${W_TRANS}" \
  --w_head_camera_rot "${W_ROT}" \
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
