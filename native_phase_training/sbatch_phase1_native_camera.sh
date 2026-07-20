#!/bin/bash
#SBATCH -p a3ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=nativep1
#SBATCH --output=/home/jungbin_cho/cosmos_motion_ft/slurm-nativep1-%j.out
#SBATCH --exclusive

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework:${PYTHONPATH:-}
export BASE_CHECKPOINT_PATH=${BASE_CHECKPOINT_PATH:-/weka/jungbin/cosmos3_nano_dcp}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/weka/jungbin/wan22_vae/Wan2.2_VAE.pth}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/weka/jungbin/cosmos_motion_ft_runs}
export NYMERIA_NUM_FRAMES=${NYMERIA_NUM_FRAMES:-97}
export NYMERIA_RESOLUTION=${NYMERIA_RESOLUTION:-256}
export NYMERIA_LATENT_ROOT=${NYMERIA_LATENT_ROOT:-/weka/jungbin/nymeriaplus_kimodo_proportional/joint_latents_T${NYMERIA_NUM_FRAMES}}
export NATIVEP1_LORA_LR=${NATIVEP1_LORA_LR:-5e-5}
export NATIVEP1_ACTION_LR_MULT=${NATIVEP1_ACTION_LR_MULT:-4.0}
export NATIVEP1_CLIPS_PER_GPU=${NATIVEP1_CLIPS_PER_GPU:-4}
export NATIVEP1_RUN_NAME=${NATIVEP1_RUN_NAME:-native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k}
export NATIVEP1_AUTO_EVAL=${NATIVEP1_AUTO_EVAL:-1}
export NATIVEP1_VIZ_N=${NATIVEP1_VIZ_N:-5}
export NATIVEP1_EVAL_INPUT_DIR=${NATIVEP1_EVAL_INPUT_DIR:-/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_viz5_256_T97_v2}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

cd /home/jungbin_cho/cosmos-framework

echo "[nativep1] node=$(hostname) date=$(date)"
echo "[nativep1] base=${BASE_CHECKPOINT_PATH}"
echo "[nativep1] vae=${WAN_VAE_PATH}"
echo "[nativep1] latents=${NYMERIA_LATENT_ROOT}"
echo "[nativep1] output_root=${IMAGINAIRE_OUTPUT_ROOT}"
echo "[nativep1] lora_lr=${NATIVEP1_LORA_LR} action_lr_mult=${NATIVEP1_ACTION_LR_MULT}"
echo "[nativep1] clips_per_gpu=${NATIVEP1_CLIPS_PER_GPU} (0 means 45056-token budget packing)"
echo "[nativep1] run_name=${NATIVEP1_RUN_NAME}"
echo "[nativep1] tensorboard=${TB_LOG_DIR:-${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world/${NATIVEP1_RUN_NAME}/tensorboard}"
echo "[nativep1] auto_eval=${NATIVEP1_AUTO_EVAL} eval_inputs=${NATIVEP1_EVAL_INPUT_DIR} viz_n=${NATIVEP1_VIZ_N}"
echo "[nativep1] visible_gpus=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits || true
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

MIN_FREE_MIB=${NATIVEP1_MIN_FREE_MIB:-132000}
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python - <<'PY'
import os
import subprocess
import sys

threshold = int(os.environ.get("NATIVEP1_MIN_FREE_MIB", "132000"))
raw = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
bad = []
for line in raw.strip().splitlines():
    idx_s, free_s = [p.strip() for p in line.split(",")]
    free = int(free_s)
    if free < threshold:
        bad.append((idx_s, free))
if bad:
    print(f"[nativep1] ERROR: expected mostly-empty GPUs before launch; threshold={threshold} MiB", file=sys.stderr)
    for idx, free in bad:
        print(f"[nativep1] ERROR: GPU {idx} has only {free} MiB free", file=sys.stderr)
    sys.exit(1)
print(f"[nativep1] GPU memory preflight passed: every GPU has >= {threshold} MiB free")
PY

if [[ "${NATIVEP1_AUTO_EVAL}" == "1" ]]; then
  if [[ ! -s "${NATIVEP1_EVAL_INPUT_DIR}/fd_input.jsonl" || \
        ! -s "${NATIVEP1_EVAL_INPUT_DIR}/invdyn_input.jsonl" || \
        ! -s "${NATIVEP1_EVAL_INPUT_DIR}/policy_input.jsonl" || \
        ! -s "${NATIVEP1_EVAL_INPUT_DIR}/i2v_input.jsonl" ]]; then
    echo "[nativep1] preparing ${NATIVEP1_VIZ_N} held-out samples for automatic four-mode checkpoint evaluation"
    /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
      /home/jungbin_cho/cosmos_motion_ft/native_phase_training/prep_test_eval.py \
      --out "${NATIVEP1_EVAL_INPUT_DIR}" \
      --n "${NATIVEP1_VIZ_N}" \
      --seed 0
  fi
fi

/home/jungbin_cho/miniforge3/envs/cosmos/bin/torchrun --standalone --nproc_per_node=8 \
  /home/jungbin_cho/cosmos_motion_ft/native_phase_training/run_latent_train.py \
  --sft-toml /home/jungbin_cho/cosmos_motion_ft/native_phase_training/world_camera_nymeria_latent.toml \
  job.name=${NATIVEP1_RUN_NAME} \
  trainer.max_iter=100000 \
  trainer.logging_iter=50 \
  checkpoint.save_iter=5000 \
  model.config.ema.enabled=true \
  model.config.compile.enabled=false \
  model.config.parallelism.data_parallel_replicate_degree=8 \
  model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.parallelism.context_parallel_shard_degree=1 \
  model.config.parallelism.cfg_parallel_shard_degree=1

echo "[nativep1] done date=$(date)"
