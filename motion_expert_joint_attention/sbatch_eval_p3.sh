#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_p3
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-evalp3-%j.out
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
bash $D/run_eval.sh /weka/jungbin/cosmos_motion_ft_runs/ja_phase3_7task/ckpt_step100000.pt 71 \
  inverse_dynamics video2motion \
  --windows_json /weka/jungbin/cosmos_motion_ft_runs/joint_attention/full71_windows.json --no_video
