#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=ja_p3
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-jap3-%j.out
# =============================================================================
# PHASE 3 of the curriculum: full 7-task, WARM-STARTED from the two specialists.
#   --init_gen    <- ja_phase1_camera (trained gen-LoRA + camera/action heads)
#   --init_motion <- ja_t2m_x0_T200   (trained motion expert; T=200 but per-frame,
#                    so it drops into the T=97 aligned regime unchanged)
# Both are weight-only subset loads by name (strict=False). Motion objective = x0
# (default), vision/camera = velocity. --resume auto lets it continue if preempted.
# =============================================================================
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[ja_p3] node=$(hostname) $(date) — PHASE 3 7-task warm-start (gen<-phase1_camera, motion<-t2m_x0)"
bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp \
  --bones_frac 0.5 \
  --gen_lora \
  --precomputed_latents \
  --T 97 \
  --viz_n 8 --viz_every 2000 \
  --init_gen    /weka/jungbin/cosmos_motion_ft_runs/ja_phase1_camera/ckpt_step200000.pt \
  --init_motion /weka/jungbin/cosmos_motion_ft_runs/ja_t2m_x0_T200/ckpt_step200000.pt \
  --resume auto \
  --out ja_phase3_7task \
  --steps 200000 --batch_size 4
