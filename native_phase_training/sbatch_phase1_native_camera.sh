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
export NYMERIA_QUALITY_FILTER=${NYMERIA_QUALITY_FILTER:-}
export NATIVEP1_QUALITY_FILTER_SHA256=${NATIVEP1_QUALITY_FILTER_SHA256:-}
export NYMERIA_DROP_MODES=${NYMERIA_DROP_MODES:-}
export NYMERIA_REPLACE_STANDALONE_C=${NYMERIA_REPLACE_STANDALONE_C:-0}
export NATIVEP1_LORA_LR=${NATIVEP1_LORA_LR:-5e-5}
export NATIVEP1_ACTION_LR_MULT=${NATIVEP1_ACTION_LR_MULT:-4.0}
export NATIVEP1_ACTION_LOSS_WEIGHT=${NATIVEP1_ACTION_LOSS_WEIGHT:-10.0}
export NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE=${NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE:-0}
export NATIVEP1_SHIFT_OVERRIDE=${NATIVEP1_SHIFT_OVERRIDE:-}
export NATIVEP1_ADAPTATION_MODE=${NATIVEP1_ADAPTATION_MODE:-global_lora}
export NATIVEP1_PREFIX_LENGTHS=${NATIVEP1_PREFIX_LENGTHS:-1}
export NATIVEP1_PREFIX_SAMPLING_WEIGHTS=${NATIVEP1_PREFIX_SAMPLING_WEIGHTS:-}
export NATIVEP1_CLIPS_PER_GPU=${NATIVEP1_CLIPS_PER_GPU:-4}
export NATIVEP1_EXPECTED_LATENT_HW=${NATIVEP1_EXPECTED_LATENT_HW:-}
export NATIVEP1_EXPECTED_IMAGE_HW=${NATIVEP1_EXPECTED_IMAGE_HW:-}
export NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT=${NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT:-0}
export NATIVEP1_RUN_NAME=${NATIVEP1_RUN_NAME:-native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k}
export NATIVEP1_MAX_ITER=${NATIVEP1_MAX_ITER:-100000}
export NATIVEP1_SAVE_ITER=${NATIVEP1_SAVE_ITER:-5000}
export NATIVEP1_AUTO_EVAL=${NATIVEP1_AUTO_EVAL:-1}
export NATIVEP1_AUTO_EVAL_EVERY=${NATIVEP1_AUTO_EVAL_EVERY:-0}
export NATIVEP1_PREFLIGHT_STEPS=${NATIVEP1_PREFLIGHT_STEPS:-0}
export NATIVEP1_VIZ_N=${NATIVEP1_VIZ_N:-5}
export NATIVEP1_EVAL_INPUT_DIR=${NATIVEP1_EVAL_INPUT_DIR:-/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_viz5_256_T97_v2}
export NATIVEP1_EVAL_PREFIX_LENGTHS=${NATIVEP1_EVAL_PREFIX_LENGTHS:-1}
export NATIVEP1_AUTO_EVAL_FULL71=${NATIVEP1_AUTO_EVAL_FULL71:-${NATIVEP1_AUTO_EVAL}}
export NATIVEP1_FULL71_EVAL_EVERY=${NATIVEP1_FULL71_EVAL_EVERY:-${NATIVEP1_AUTO_EVAL_EVERY}}
export NATIVEP1_FULL71_EVAL_INPUT_DIR=${NATIVEP1_FULL71_EVAL_INPUT_DIR:-/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

cd /home/jungbin_cho/cosmos-framework

echo "[nativep1] node=$(hostname) date=$(date)"
echo "[nativep1] base=${BASE_CHECKPOINT_PATH}"
echo "[nativep1] vae=${WAN_VAE_PATH}"
echo "[nativep1] latents=${NYMERIA_LATENT_ROOT}"
echo "[nativep1] quality_filter=${NYMERIA_QUALITY_FILTER:-none}"
echo "[nativep1] drop_modes=${NYMERIA_DROP_MODES:-none}"
echo "[nativep1] replace_standalone_c=${NYMERIA_REPLACE_STANDALONE_C}"
echo "[nativep1] output_root=${IMAGINAIRE_OUTPUT_ROOT}"
echo "[nativep1] lora_lr=${NATIVEP1_LORA_LR} action_lr_mult=${NATIVEP1_ACTION_LR_MULT}"
echo "[nativep1] action_loss_weight=${NATIVEP1_ACTION_LOSS_WEIGHT} normalize_by_active=${NATIVEP1_NORMALIZE_LOSS_BY_ACTIVE}"
echo "[nativep1] resolution=${NYMERIA_RESOLUTION} shift_override=${NATIVEP1_SHIFT_OVERRIDE:-released-config}"
echo "[nativep1] adaptation_mode=${NATIVEP1_ADAPTATION_MODE} prefixes=${NATIVEP1_PREFIX_LENGTHS} prefix_weights=${NATIVEP1_PREFIX_SAMPLING_WEIGHTS:-uniform}"
echo "[nativep1] clips_per_gpu=${NATIVEP1_CLIPS_PER_GPU} (0 means 45056-token budget packing)"
echo "[nativep1] expected_latent_hw=${NATIVEP1_EXPECTED_LATENT_HW:-unchecked} expected_image_hw=${NATIVEP1_EXPECTED_IMAGE_HW:-unchecked} require_cache_contract=${NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT}"
echo "[nativep1] run_name=${NATIVEP1_RUN_NAME}"
echo "[nativep1] max_iter=${NATIVEP1_MAX_ITER} save_iter=${NATIVEP1_SAVE_ITER}"
echo "[nativep1] tensorboard=${TB_LOG_DIR:-${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world/${NATIVEP1_RUN_NAME}/tensorboard}"
echo "[nativep1] auto_eval=${NATIVEP1_AUTO_EVAL} every=${NATIVEP1_AUTO_EVAL_EVERY:-all_saves} eval_inputs=${NATIVEP1_EVAL_INPUT_DIR} viz_n=${NATIVEP1_VIZ_N}"
echo "[nativep1] full71_eval=${NATIVEP1_AUTO_EVAL_FULL71} every=${NATIVEP1_FULL71_EVAL_EVERY:-all_saves} eval_inputs=${NATIVEP1_FULL71_EVAL_INPUT_DIR}"
echo "[nativep1] preflight_steps=${NATIVEP1_PREFLIGHT_STEPS}"
echo "[nativep1] visible_gpus=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits || true
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

if [[ -n "${NYMERIA_QUALITY_FILTER}" ]]; then
  if [[ ! -s "${NYMERIA_QUALITY_FILTER}" ]]; then
    echo "[nativep1] ERROR: missing or empty quality filter: ${NYMERIA_QUALITY_FILTER}" >&2
    exit 1
  fi
  if [[ ! "${NATIVEP1_QUALITY_FILTER_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[nativep1] ERROR: filtered runs require a 64-character NATIVEP1_QUALITY_FILTER_SHA256" >&2
    exit 1
  fi
  actual_quality_filter_sha256=$(sha256sum "${NYMERIA_QUALITY_FILTER}" | awk '{print $1}')
  if [[ "${actual_quality_filter_sha256}" != "${NATIVEP1_QUALITY_FILTER_SHA256}" ]]; then
    echo "[nativep1] ERROR: quality-filter SHA-256 mismatch" >&2
    echo "[nativep1] expected=${NATIVEP1_QUALITY_FILTER_SHA256}" >&2
    echo "[nativep1] actual=${actual_quality_filter_sha256}" >&2
    exit 1
  fi
  echo "[nativep1] quality_filter_sha256=${actual_quality_filter_sha256}"
elif [[ -n "${NATIVEP1_QUALITY_FILTER_SHA256}" ]]; then
  echo "[nativep1] ERROR: filter SHA was set without NYMERIA_QUALITY_FILTER" >&2
  exit 1
fi

if [[ "${NATIVEP1_REQUIRE_LATENT_CACHE_CONTRACT}" == "1" ]]; then
  echo "[nativep1] validating complete latent cache before model construction"
  /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
    /home/jungbin_cho/cosmos_motion_ft/native_phase_training/validate_latent_cache.py \
    --root "${NYMERIA_LATENT_ROOT}" \
    --sample-count "${NATIVEP1_CACHE_VALIDATION_SAMPLES:-256}"
fi

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
    prep_args=(
      --out "${NATIVEP1_EVAL_INPUT_DIR}"
      --n "${NATIVEP1_VIZ_N}"
      --seed 0
      --prefix-lengths "${NATIVEP1_EVAL_PREFIX_LENGTHS}"
    )
    if [[ -n "${NYMERIA_QUALITY_FILTER}" ]]; then
      prep_args+=(--quality-filter "${NYMERIA_QUALITY_FILTER}")
    fi
    if [[ "${NYMERIA_REPLACE_STANDALONE_C}" == "1" ]]; then
      prep_args+=(--replace-standalone-c)
    fi
    /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
      /home/jungbin_cho/cosmos_motion_ft/native_phase_training/prep_test_eval.py \
      "${prep_args[@]}"
  fi
fi

if [[ "${NATIVEP1_AUTO_EVAL_FULL71}" == "1" ]]; then
  for input_name in fd_input.jsonl invdyn_input.jsonl; do
    input_path=${NATIVEP1_FULL71_EVAL_INPUT_DIR}/${input_name}
    if [[ ! -s "${input_path}" ]]; then
      echo "[nativep1] ERROR: missing canonical full-71 input ${input_path}" >&2
      exit 1
    fi
    input_count=$(wc -l < "${input_path}")
    if [[ "${input_count}" -ne 71 ]]; then
      echo "[nativep1] ERROR: expected 71 records in ${input_path}, found ${input_count}" >&2
      exit 1
    fi
  done
fi

run_native_phase1() {
  local max_iter=$1
  local save_iter=$2
  /home/jungbin_cho/miniforge3/envs/cosmos/bin/torchrun --standalone --nproc_per_node=8 \
    /home/jungbin_cho/cosmos_motion_ft/native_phase_training/run_latent_train.py \
    --sft-toml /home/jungbin_cho/cosmos_motion_ft/native_phase_training/world_camera_nymeria_latent.toml \
    job.name=${NATIVEP1_RUN_NAME} \
    trainer.max_iter=${max_iter} \
    trainer.logging_iter=50 \
    checkpoint.save_iter=${save_iter} \
    model.config.ema.enabled=true \
    model.config.compile.enabled=false \
    model.config.parallelism.data_parallel_replicate_degree=8 \
    model.config.parallelism.data_parallel_shard_degree=1 \
    model.config.parallelism.context_parallel_shard_degree=1 \
    model.config.parallelism.cfg_parallel_shard_degree=1
}

run_dir=${IMAGINAIRE_OUTPUT_ROOT}/cosmos3_camera/camera_world/${NATIVEP1_RUN_NAME}
latest_file=${run_dir}/checkpoints/latest_checkpoint.txt
if (( NATIVEP1_PREFLIGHT_STEPS > 0 )) && [[ ! -s "${latest_file}" ]]; then
  echo "[nativep1] starting ${NATIVEP1_PREFLIGHT_STEPS}-step save/resume preflight"
  run_native_phase1 "${NATIVEP1_PREFLIGHT_STEPS}" "${NATIVEP1_PREFLIGHT_STEPS}"
  expected_preflight=$(printf 'iter_%09d' "${NATIVEP1_PREFLIGHT_STEPS}")
  if [[ ! -s "${latest_file}" || "$(<"${latest_file}")" != "${expected_preflight}" ]]; then
    echo "[nativep1] ERROR: preflight did not save ${expected_preflight} in ${run_dir}" >&2
    exit 1
  fi
  echo "[nativep1] preflight checkpoint saved; the full command must resume ${expected_preflight}"
elif (( NATIVEP1_PREFLIGHT_STEPS > 0 )); then
  echo "[nativep1] existing checkpoint $(<"${latest_file}"); skipping first-launch preflight"
fi

run_native_phase1 "${NATIVEP1_MAX_ITER}" "${NATIVEP1_SAVE_ITER}"

echo "[nativep1] done date=$(date)"
