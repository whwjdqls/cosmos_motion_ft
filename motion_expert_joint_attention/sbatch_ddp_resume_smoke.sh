#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodelist=a3ultravis-a3ultranodeset-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:40:00
#SBATCH --job-name=ddp_resume_smoke
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-ddp-resume-smoke-%j.out
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh && conda activate cosmos
unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH=/weka/jungbin/wan22_vae/Wan2.2_VAE.pth
export PYTHONPATH=/home/jungbin_cho/cosmos-framework:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
cd /home/jungbin_cho/cosmos-framework
D=/home/jungbin_cho/cosmos_motion_ft/motion_expert_joint_attention
OUT=/weka/jungbin/cosmos_motion_ft_runs/ddp_resume_smoke_$SLURM_JOB_ID
RUNPY="python -m torch.distributed.run --standalone --nproc_per_node=2"

echo "[smoke] node=$(hostname) $(date) OUT=$OUT"

echo "===================================================================="
echo "=== PHASE A: PURE-DDP run, 20 steps, save every 10 -> latest.pt   ==="
echo "=== (COSMOS_DDP_PARAM_CHECK=1 -> cross-rank param identity assert) =="
echo "===================================================================="
COSMOS_DDP_PARAM_CHECK=1 CUDA_VISIBLE_DEVICES=0,1 $RUNPY $D/train.py \
  --ddp --gen_lora --tasks text2motion \
  --batch_size 4 --steps 20 --T 33 --num_workers 4 \
  --save_every 10 --log_every 2 --viz_n 0 --viz_every 100000 \
  --lr 2e-4 --warmup 5 --lr_schedule cosine \
  --out $(basename $OUT)
echo "[smoke] phaseA exit=$?"
echo "--- checkpoint contents (optimizer present? step?) ---"
python - <<PY
import torch, glob, os
p = "$OUT/latest.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
opt = ck.get("optimizer")
nstate = len(opt["state"]) if opt else 0
print(f"latest.pt: step={ck['step']} n_model_tensors={len(ck['model'])} "
      f"optimizer_present={opt is not None} optimizer_n_state_entries={nstate}")
PY

echo "===================================================================="
echo "=== PHASE B: relaunch with --resume auto -> must continue at 20   ==="
echo "===================================================================="
CUDA_VISIBLE_DEVICES=0,1 $RUNPY $D/train.py \
  --ddp --gen_lora --tasks text2motion \
  --batch_size 4 --steps 40 --T 33 --num_workers 4 \
  --save_every 10 --log_every 2 --viz_n 0 --viz_every 100000 \
  --lr 2e-4 --warmup 5 --lr_schedule cosine \
  --out $(basename $OUT) --resume auto
echo "[smoke] phaseB exit=$?"

echo "===================================================================="
echo "=== PHASE C: FSDP still works (opt-in) + memory number           ==="
echo "===================================================================="
CUDA_VISIBLE_DEVICES=0,1 $RUNPY $D/train.py \
  --ddp --fsdp --gen_lora --tasks text2motion \
  --batch_size 4 --steps 6 --T 33 --num_workers 4 \
  --save_every 100 --log_every 2 --viz_n 0 --viz_every 100000 \
  --lr 2e-4 --warmup 5 --lr_schedule cosine \
  --out ${OUT##*/}_fsdp
echo "[smoke] phaseC exit=$?"
echo "===== DONE $(date) ====="
