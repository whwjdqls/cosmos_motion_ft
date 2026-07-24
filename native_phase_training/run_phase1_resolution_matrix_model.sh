#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT_PATH INPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

CHECKPOINT_PATH=$1
INPUT_DIR=$2
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
# High-tier records inherit this default. The explicit 256 records override it.
export NYMERIA_RESOLUTION=720

for input in fd_r256_s3.jsonl fd_r720_s3.jsonl fd_r720_s10.jsonl manifest.json; do
  [[ -f "${INPUT_DIR}/${input}" ]] || {
    echo "missing matrix input: ${INPUT_DIR}/${input}" >&2
    exit 1
  }
done

mkdir -p "${OUTPUT_DIR}"
unset NATIVEP1_ADAPTATION_MODE NYMERIA_DROP_MODES
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
  /home/jungbin_cho/cosmos_motion_ft/native_phase_training/run_contract.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}"
source "${OUTPUT_DIR}/resolved_run_contract.env"

echo "[resolution-matrix] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-all}"
echo "[resolution-matrix] checkpoint=${CHECKPOINT_PATH}"
echo "[resolution-matrix] adaptation=${NATIVEP1_ADAPTATION_MODE} drops=${NYMERIA_DROP_MODES:-none}"

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
  -i "${INPUT_DIR}/fd_r256_s3.jsonl" \
     "${INPUT_DIR}/fd_r720_s3.jsonl" \
     "${INPUT_DIR}/fd_r720_s10.jsonl"

cd /home/jungbin_cho/cosmos_motion_ft
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python - "${INPUT_DIR}" "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import cv2

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
manifest = json.loads((input_dir / "manifest.json").read_text())
expected_sizes = {
    cell: tuple(settings["expected_output_size"]) for cell, settings in manifest["cells"].items()
}

validated = []
for input_path in sorted(input_dir.glob("fd_r*.jsonl")):
    records = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    for record in records:
        cell = record["name"].rsplit("__", 1)[-1]
        sample_dir = output_dir / record["name"]
        payload_path = sample_dir / "sample_outputs.json"
        video_path = sample_dir / "vision.mp4"
        payload = json.loads(payload_path.read_text())
        if payload.get("status") != "success":
            raise RuntimeError(f"failed inference output: {payload_path}")
        capture = cv2.VideoCapture(str(video_path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if (width, height) != expected_sizes[cell] or frames != 97:
            raise RuntimeError(
                f"{video_path}: expected {expected_sizes[cell]}x97, got {(width, height)}x{frames}"
            )
        validated.append(
            {"name": record["name"], "cell": cell, "video": str(video_path), "size": [width, height], "frames": frames}
        )

expected_count = manifest["sample_count"] * len(manifest["cells"])
if len(validated) != expected_count:
    raise RuntimeError(f"expected {expected_count} outputs, validated {len(validated)}")
(output_dir / "RESOLUTION_MATRIX_COMPLETE.json").write_text(
    json.dumps(
        {
            "checkpoint": str(Path(sys.argv[2]) / "resolved_run_contract.json"),
            "input_manifest": str(input_dir / "manifest.json"),
            "validated_outputs": validated,
        },
        indent=2,
    )
    + "\n"
)
print(f"[resolution-matrix] validated {len(validated)} successful outputs")
PY
