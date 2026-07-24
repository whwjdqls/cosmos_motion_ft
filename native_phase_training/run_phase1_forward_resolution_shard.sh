#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 CHECKPOINT INPUT_JSONL OUTPUT_DIR RESOLUTION_TIER WIDTH HEIGHT" >&2
  exit 2
fi

CHECKPOINT_PATH=$1
INPUT_JSONL=$2
OUTPUT_DIR=$3
RESOLUTION_TIER=$4
EXPECTED_WIDTH=$5
EXPECTED_HEIGHT=$6

if [[ ${RESOLUTION_TIER} != 256 && ${RESOLUTION_TIER} != 720 ]]; then
  echo "resolution tier must be 256 or 720, got ${RESOLUTION_TIER}" >&2
  exit 2
fi
[[ ${EXPECTED_WIDTH} =~ ^[1-9][0-9]*$ && ${EXPECTED_HEIGHT} =~ ^[1-9][0-9]*$ ]] || {
  echo "WIDTH and HEIGHT must be positive integers" >&2
  exit 2
}
[[ -d ${CHECKPOINT_PATH}/model && -f ${INPUT_JSONL} ]] || {
  echo "missing checkpoint model directory or input JSONL" >&2
  exit 1
}
mkdir -p "${OUTPUT_DIR}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cosmos

export LD_LIBRARY_PATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/jungbin_cho/cosmos_motion_ft:/home/jungbin_cho/cosmos_motion_ft/nymeria_world:/home/jungbin_cho/cosmos-framework:${PYTHONPATH:-}
export WAN_VAE_PATH=${WAN_VAE_PATH:-/weka/jungbin/wan22_vae/Wan2.2_VAE.pth}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NYMERIA_RESOLUTION=${RESOLUTION_TIER}

/home/jungbin_cho/miniforge3/envs/cosmos/bin/python - \
  "${INPUT_JSONL}" "${RESOLUTION_TIER}" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
tier = int(sys.argv[2])
records = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
if not records:
    raise ValueError(f"empty input JSONL: {input_path}")
if len({record["name"] for record in records}) != len(records):
    raise ValueError(f"duplicate names in {input_path}")
for record in records:
    if record.get("model_mode") != "forward_dynamics":
        raise ValueError(f"{record.get('name')}: expected forward_dynamics")
    expected = {
        "fps": 20,
        "shift": 3 if tier == 256 else 10,
        "seed": 0,
        "action_chunk_size": 96,
        "image_size": 256 if tier == 256 else 480,
        "num_steps": 30,
        "guidance": 1,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"{record.get('name')}: expected {key}={value!r}, got {record.get(key)!r}"
            )
    if tier == 256:
        explicit = {"num_frames": 97, "resolution": "256", "aspect_ratio": "1,1"}
        for key, value in explicit.items():
            if record.get(key) != value:
                raise ValueError(
                    f"{record.get('name')}: expected {key}={value!r}, got {record.get(key)!r}"
                )
    else:
        for key in ("num_frames", "resolution", "aspect_ratio"):
            if key in record:
                raise ValueError(f"{record.get('name')}: high-tier record must omit {key}")
    for key in ("vision_path", "action_path"):
        if not Path(record[key]).is_file():
            raise FileNotFoundError(record[key])
print(f"[phase1-forward-resolution] preflight tier={tier} records={len(records)}")
PY

unset NATIVEP1_ADAPTATION_MODE NYMERIA_DROP_MODES
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python \
  /home/jungbin_cho/cosmos_motion_ft/native_phase_training/run_contract.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}"
source "${OUTPUT_DIR}/resolved_run_contract.env"

echo "[phase1-forward-resolution] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-all}"
echo "[phase1-forward-resolution] checkpoint=${CHECKPOINT_PATH}"
echo "[phase1-forward-resolution] input=${INPUT_JSONL}"
echo "[phase1-forward-resolution] tier=${RESOLUTION_TIER} size=${EXPECTED_WIDTH}x${EXPECTED_HEIGHT}"
echo "[phase1-forward-resolution] adaptation=${NATIVEP1_ADAPTATION_MODE} drops=${NYMERIA_DROP_MODES:-none}"

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
/home/jungbin_cho/miniforge3/envs/cosmos/bin/python - \
  "${INPUT_JSONL}" "${OUTPUT_DIR}" "${RESOLUTION_TIER}" \
  "${EXPECTED_WIDTH}" "${EXPECTED_HEIGHT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import cv2

input_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
tier = int(sys.argv[3])
expected_width = int(sys.argv[4])
expected_height = int(sys.argv[5])
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
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if (width, height, frames) != (expected_width, expected_height, 97) or abs(fps - 20) > 1e-3:
        raise RuntimeError(
            f"{video_path}: expected {expected_width}x{expected_height},97f,20fps; "
            f"got {width}x{height},{frames}f,{fps:g}fps"
        )
    validated.append({"name": record["name"], "video": str(video_path)})

(output_dir / "PHASE1_FORWARD_RESOLUTION_COMPLETE.json").write_text(
    json.dumps(
        {
            "input": str(input_path),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "checkpoint_contract": str(output_dir / "resolved_run_contract.json"),
            "resolution_tier": tier,
            "expected_output_size": [expected_width, expected_height],
            "count": len(validated),
            "validated_outputs": validated,
        },
        indent=2,
    )
    + "\n"
)
print(f"[phase1-forward-resolution] validated tier={tier} outputs={len(validated)}")
PY
