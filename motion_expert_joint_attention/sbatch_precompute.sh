#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=precomp7
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-precomp7-%j.out
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
cd /home/jungbin_cho/cosmos-framework
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
echo "[precomp7] node=$(hostname) $(date) — 8 parallel shards (resumable/skip-existing)"
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s python $D/precompute_latents.py --num_shards 8 --shard_id $s \
    > /home/jungbin_cho/cosmos_motion_ft/precomp7_shard$s.log 2>&1 &
done
wait
echo "[precomp7] ALL SHARDS DONE $(date)"
echo "total latents written: $(ls /weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents/*/*.npz 2>/dev/null | wc -l)"
