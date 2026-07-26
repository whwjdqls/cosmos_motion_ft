#!/bin/bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k}
EVAL_INPUT_DIR=${EVAL_INPUT_DIR:-/weka/jungbin/cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2}
EVAL_ROOT=${EVAL_ROOT:-${RUN_DIR}/eval_full71_inverse_forward}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
DP_SHARD_SIZE=${DP_SHARD_SIZE:-${NPROC_PER_NODE}}
FULL71_VISUALIZE_LIMIT=${FULL71_VISUALIZE_LIMIT:-0}
FULL71_FORCE=${FULL71_FORCE:-0}
FULL71_ITERATION=${FULL71_ITERATION:-}

source /home/jungbin_cho/miniforge3/etc/profile.d/conda.sh
conda activate cosmos
export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework:${PYTHONPATH:-}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/weka/jungbin/wan22_vae/Wan2.2_VAE.pth}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

for input_name in fd_input.jsonl invdyn_input.jsonl; do
    input_path=${EVAL_INPUT_DIR}/${input_name}
    if [[ ! -s "${input_path}" ]]; then
        echo "[full71] missing input ${input_path}" >&2
        exit 1
    fi
    count=$(wc -l < "${input_path}")
    if [[ "${count}" -ne 71 ]]; then
        echo "[full71] expected 71 records in ${input_path}, found ${count}" >&2
        exit 1
    fi
done

mkdir -p "${EVAL_ROOT}"
echo "[full71] node=$(hostname) output=${EVAL_ROOT}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
if [[ -n "${FULL71_ITERATION}" && \
      ! -s "${RUN_DIR}/checkpoints/${FULL71_ITERATION}/model/.metadata" ]]; then
    echo "[full71] requested checkpoint is missing/incomplete: ${FULL71_ITERATION}" >&2
    exit 1
fi

while true; do
    mapfile -t checkpoint_candidates < <(
        find "${RUN_DIR}/checkpoints" -maxdepth 1 -mindepth 1 -type d -name 'iter_*' | sort -V
    )
    checkpoints=()
    for candidate in "${checkpoint_candidates[@]}"; do
        if [[ -s "${candidate}/model/.metadata" ]]; then
            checkpoints+=("${candidate}")
        fi
    done
    if [[ ${#checkpoints[@]} -eq 0 ]]; then
        echo "[full71] no complete DCP checkpoints found under ${RUN_DIR}/checkpoints" >&2
        exit 1
    fi
    processed=0

    for checkpoint in "${checkpoints[@]}"; do
        iteration=$(basename "${checkpoint}")
        if [[ -n "${FULL71_ITERATION}" && "${iteration}" != "${FULL71_ITERATION}" ]]; then
            continue
        fi
        output_dir=${EVAL_ROOT}/${iteration}
        inference_dir=${output_dir}/inference
        analysis_dir=${output_dir}/analysis
        mkdir -p "${output_dir}"
        /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
          /home/jungbin_cho/cosmos_motion_ft/native_phase_training/run_contract.py \
          --checkpoint-path "${checkpoint}" \
          --output-dir "${output_dir}"
        # Values are validated against the immutable checkpoint contract before
        # experiment.py is imported by official inference.
        source "${output_dir}/resolved_run_contract.env"
        /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
          /home/jungbin_cho/cosmos_motion_ft/native_phase_training/validate_eval_inputs.py \
          --input-dir "${EVAL_INPUT_DIR}" \
          --expected-shift "${NATIVEP1_EFFECTIVE_SHIFT}" \
          --expected-resolution "${NYMERIA_RESOLUTION}" \
          --expected-num-frames "${NYMERIA_NUM_FRAMES}" \
          fd_input.jsonl invdyn_input.jsonl
        if [[ "${FULL71_FORCE}" != 1 && -s "${analysis_dir}/COMPLETE.json" ]]; then
            /home/jungbin_cho/miniforge3/envs/cosmos/bin/python - "${output_dir}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "run_contract": str(root / "resolved_run_contract.json"),
            "metrics_manifest": str(root / "analysis" / "COMPLETE.json"),
        },
        indent=2,
    )
    + "\n"
)
PY
            continue
        fi
        processed=$((processed + 1))

        echo "[full71] inference ${iteration}"
        cd /home/jungbin_cho/cosmos-framework
        /home/jungbin_cho/miniforge3/envs/cosmos/bin/torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
          -m cosmos_framework.scripts.inference \
          --checkpoint-path "${checkpoint}" \
          --config-file native_phase_training/inference_config.py \
          --experiment world_camera_nymeria_latent_nano \
          --sampler unipc \
          --use-ema-weights \
          --parallelism-preset latency \
          --dp-shard-size "${DP_SHARD_SIZE}" --dp-replicate-size 1 --cp-size 1 --cfgp-size 1 \
          --no-use-torch-compile --no-use-cuda-graphs --no-guardrails \
          -o "${inference_dir}" \
          -i "${EVAL_INPUT_DIR}/fd_input.jsonl" "${EVAL_INPUT_DIR}/invdyn_input.jsonl"

        echo "[full71] metrics and visualizations ${iteration}"
        cd /home/jungbin_cho/cosmos_motion_ft
        /home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
          native_phase_training/evaluate_inverse_forward.py \
          --inference-root "${inference_dir}" \
          --eval-root "${EVAL_INPUT_DIR}" \
          --out "${analysis_dir}" \
          --expected-count 71 \
          --visualize-limit "${FULL71_VISUALIZE_LIMIT}" \
          --lpips-device cuda:0 \
          --lpips-batch-size 16 2>&1 | tee "${output_dir}/analysis.log"

        /home/jungbin_cho/miniforge3/envs/cosmos/bin/python - "${output_dir}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = (root / "resolved_run_contract.json", root / "analysis" / "COMPLETE.json")
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"full-71 evaluation is incomplete: {missing}")
(root / "COMPLETE.json").write_text(
    json.dumps(
        {"run_contract": str(required[0]), "metrics_manifest": str(required[1])},
        indent=2,
    )
    + "\n"
)
PY
    done

    if [[ "${processed}" -eq 0 || "${FULL71_FORCE}" == 1 ]]; then
        break
    fi
done

cd /home/jungbin_cho/cosmos_motion_ft
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
  native_phase_training/summarize_inverse_forward.py \
  --eval-root "${EVAL_ROOT}"
echo "[full71] complete: ${EVAL_ROOT}"
