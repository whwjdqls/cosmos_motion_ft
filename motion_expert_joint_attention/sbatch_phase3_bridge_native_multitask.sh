#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=5-00:00:00
#SBATCH --job-name=p3brmulti
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p3brmulti-%j.out
#SBATCH --exclusive

set -euo pipefail

ROOT=/home/jungbin_cho/cosmos_motion_ft
D=${ROOT}/motion_expert_joint_attention
PHASE1=/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k/checkpoints/iter_000100000
PHASE2=/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt
RUN_NAME=${PHASE3_MULTITASK_RUN_NAME:-ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask}

echo "[p3brmulti] node=$(hostname) date=$(date)"
echo "[p3brmulti] run=${RUN_NAME}"
echo "[p3brmulti] Phase 1=${PHASE1} (100k EMA, frozen generator LoRA/action specialist)"
echo "[p3brmulti] Phase 2=${PHASE2} (200k native x0, frozen motion specialist)"
echo "[p3brmulti] tasks=35% V2M + 35% M2V + 15% video->camera&motion + 15% camera&image->video&motion"
echo "[p3brmulti] joint branches are each weighted 0.5; expected total branch budget remains 1/sample"
echo "[p3brmulti] effective branch mass: motion=0.50 video=0.425 camera=0.075 (vanilla motion/video=0.50/0.50)"
echo "[p3brmulti] joint training sigmas are independent native marginals (motion logit-normal, gen Waver)"
echo "[p3brmulti] joint sampling is one common shift-3 UniPC state with exact scheduler-sigma x0 conversion"
echo "[p3brmulti] required viz every 5k + final: two held-out examples for every one of the four tasks"
echo "[p3brmulti] git=$(git -C "${ROOT}" rev-parse --short HEAD)"

test -f "${PHASE1}/model/.metadata"
test -f "${PHASE2}"
test -f "${D}/_verify_phase3_multitask_contracts.py"
bash "${D}/run.sh" "${D}/_verify_phase3_multitask_contracts.py"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

# Slurm exclusivity does not prevent an SSH process from occupying the allocated node. Refuse to
# start a five-day run if any GPU already carries meaningful memory from an external process.
MAX_PREFLIGHT_USED_MIB=2048
while IFS=, read -r gpu used; do
  used=${used// /}
  if (( used > MAX_PREFLIGHT_USED_MIB )); then
    echo "[p3brmulti] ERROR: GPU ${gpu} already uses ${used} MiB (> ${MAX_PREFLIGHT_USED_MIB})"
    nvidia-smi
    exit 1
  fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

bash "${D}/run.sh" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${D}/train.py" --ddp \
  --tasks video2motion motimg2video video2camera_motion camimg2video_motion \
  --task_weights '{"video2motion":0.35,"motimg2video":0.35,"video2camera_motion":0.15,"camimg2video_motion":0.15}' \
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
  --viz_n 8 \
  --viz_steps 30 \
  --viz_guidance 1.0 \
  --viz_frame_stride 2 \
  --require_viz \
  --resume auto \
  --out "${RUN_NAME}"
