#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=ja_motion
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-ja-%j.out
set -uo pipefail
RUN=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[ja] node=$(hostname) $(date)"
echo "===== SMOKE TEST (1 GPU, 3 steps, freeze/grad proof) ====="
CUDA_VISIBLE_DEVICES=0 bash $RUN/run.sh $RUN/train.py --smoke --gen_lora --data_mix nymeria
SMOKE=$?
echo "[ja] smoke exit=$SMOKE"
if [ "$SMOKE" -ne 0 ]; then echo "[ja] SMOKE FAILED -> not launching full training (debug needed)"; exit 1; fi
echo "===== SMOKE PASSED -> full Nymeria training (8-GPU FSDP) ====="
bash $RUN/run.sh -m torch.distributed.run --standalone --nproc_per_node=8 $RUN/train.py \
    --ddp --data_mix nymeria --gen_lora --objective velocity --batch_size 32 --steps 200000 \
    --out ja_nymeria_v1
