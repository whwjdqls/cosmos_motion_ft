#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT_PATH INPUT_JSONL OUTPUT_DIR" >&2
  exit 2
fi

CHECKPOINT_PATH=$1
INPUT_JSONL=$2
OUTPUT_DIR=$3

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cosmos

export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework:${PYTHONPATH:-}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/weka/jungbin/wan22_vae/Wan2.2_VAE.pth}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NYMERIA_RESOLUTION=720

[[ -f "${INPUT_JSONL}" ]] || {
  echo "missing shard input: ${INPUT_JSONL}" >&2
  exit 1
}
mkdir -p "${OUTPUT_DIR}"

unset NATIVEP1_ADAPTATION_MODE NYMERIA_DROP_MODES
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
  /home/jungbin_cho/cosmos_motion_ft/native_phase_training/run_contract.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}"
source "${OUTPUT_DIR}/resolved_run_contract.env"

echo "[full71-720] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-all}"
echo "[full71-720] checkpoint=${CHECKPOINT_PATH}"
echo "[full71-720] input=${INPUT_JSONL}"
echo "[full71-720] adaptation=${NATIVEP1_ADAPTATION_MODE} drops=${NYMERIA_DROP_MODES:-none}"

cd /home/jungbin_cho/cosmos-framework
/home/jungbin_cho/miniforge3/envs/cosmos/bin/torchrun --standalone --nproc_per_node=1 \
  -m cosmos_framework.scripts.inference \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-file native_phase_training/inference_config.py \
  --experiment world_camera_nymeria_latent_nano \
  --sampler unipc \
  --use-ema-weights \
  --parallelism-preset latency \
  --dp-shard-size 1 --dp-replicate-size 1 --cp-size 1 --cfgp-size 1 \
  --no-use-torch-compile --no-use-cuda-graphs --no-guardrails \
  -o "${OUTPUT_DIR}" \
  -i "${INPUT_JSONL}"

cd /home/jungbin_cho/cosmos_motion_ft
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python - "${INPUT_JSONL}" "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import cv2

input_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
records = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
validated = []
for record in records:
    sample_dir = output_dir / record["name"]
    payload = json.loads((sample_dir / "sample_outputs.json").read_text())
    if payload.get("status") != "success":
        raise RuntimeError(f"failed output: {sample_dir}")
    video_path = sample_dir / "vision.mp4"
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if (width, height, frames) != (640, 640, 97):
        raise RuntimeError(f"{video_path}: expected 640x640x97, got {width}x{height}x{frames}")
    validated.append({"name": record["name"], "video": str(video_path)})

(output_dir / "FULL71_720_SHARD_COMPLETE.json").write_text(
    json.dumps(
        {
            "input": str(input_path),
            "checkpoint_contract": str(output_dir / "resolved_run_contract.json"),
            "count": len(validated),
            "validated_outputs": validated,
        },
        indent=2,
    )
    + "\n"
)
print(f"[full71-720] validated {len(validated)} outputs")
PY
