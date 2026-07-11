#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=ja_p1cam
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-jap1cam-%j.out
# =============================================================================
# PHASE 1 of the 3-phase curriculum: finetune the GENERATOR (gen-LoRA) on the
# Nymeria CAMERA tasks ONLY (forward_dynamics / inverse_dynamics / policy) -- NO
# motion. Uses precomputed Wan-VAE latents. The motion expert is FROZEN
# (--freeze_motion --T 97): it is still BUILT but excluded from the optimizer / grad-clip
# / all-reduce, so ONLY the gen-LoRA trains (no motion tokens ever appear).
#
# Phase 2 pretrains the motion expert (text2motion): --tasks text2motion (runnable today).
# Phase 3 warm-starts BOTH: --init_gen <ja_phase1_camera/latest.pt> --init_motion <phase2/latest.pt>.
#
# Do NOT launch alongside the other running jobs without checking node availability.
# =============================================================================
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[ja_p1cam] node=$(hostname) $(date) — PHASE 1: camera-only (fwd/inv/policy), gen-LoRA, motion FROZEN, precomputed latents"
bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp \
  --tasks forward_dynamics inverse_dynamics policy \
  --bones_frac 0 \
  --gen_lora \
  --freeze_motion --T 97 \
  --precomputed_latents \
  --viz_n 0 \
  --resume auto --out ja_phase1_camera \
  --steps 200000 --batch_size 4
