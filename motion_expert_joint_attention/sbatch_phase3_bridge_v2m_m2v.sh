#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=p3_bridge
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-p3bridge-%j.out

# Bridge-only Phase 3:
#   generator/video/camera side <- Phase-1 camera LoRA/action heads, then frozen
#   motion expert              <- latest ja_t2m_x0_T200_mrope3d, then frozen
#   trainable params           <- local modality bridges only
#
# Tasks are only frame-aligned gen<->motion translation/control:
#   video2motion and motimg2video. No textimg2motion is used here because both
#   specialists were trained without reasoner-image TI2M.

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
INIT_GEN=/weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/latest.pt
INIT_MOTION=/weka/jungbin/cosmos_motion_ft_runs/ja_t2m_x0_T200_mrope3d/latest.pt

echo "[p3_bridge] node=$(hostname) $(date) - bridge-only Phase 3: v2m + motimg2video"
echo "[p3_bridge] init_gen=${INIT_GEN}"
echo "[p3_bridge] init_motion=${INIT_MOTION}"

bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp \
  --tasks video2motion motimg2video \
  --task_weights '{"video2motion":0.5,"motimg2video":0.5}' \
  --bones_frac 0.0 \
  --gen_lora --freeze_gen \
  --freeze_motion \
  --coupling bridge_local \
  --motion_mrope cosmos3d \
  --precomputed_latents \
  --T 97 \
  --viz_n 8 --viz_every 2000 \
  --init_gen "$INIT_GEN" \
  --init_motion "$INIT_MOTION" \
  --resume auto \
  --out ja_phase3_bridge_v2m_m2v_from_t2m3d_p1cam \
  --steps 200000 --batch_size 4
