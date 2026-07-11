#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=t2mti3d
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-t2mti3d-%j.out

D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[t2mti3d] node=$(hostname) $(date) - Phase-2 motion expert pretraining: text2motion + reasoner-image textimg2motion, x0, T=200, motion_mrope=cosmos3d"
echo "[t2mti3d] BONES overview/natural rows use content_natural_desc_4 via active bones_pairs jsonl"

bash $D/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  $D/train.py --ddp \
  --tasks text2motion textimg2motion \
  --task_weights '{"text2motion":0.75,"textimg2motion":0.25}' \
  --bones_frac 0.5 \
  --T 200 \
  --motion_mrope cosmos3d \
  --textimg_condition reasoner \
  --viz_n 8 --viz_every 2000 --resume auto \
  --out ja_t2m_ti2m_reasonerimg_x0_T200_mrope3d \
  --steps 200000 --batch_size 32
