#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MATRIX_ROOT EVAL_ROOT" >&2
  exit 2
fi

MATRIX_ROOT=$1
EVAL_ROOT=$2
VIZ_ROOT=${MATRIX_ROOT}/viz
mkdir -p "${VIZ_ROOT}/by_model" "${VIZ_ROOT}/by_cell"

mapfile -t SOURCES < <(jq -r .name "${MATRIX_ROOT}/inputs/fd_r256_s3.jsonl" | sed 's/__r256_s3$//')
MODELS=(original A B D)
CELLS=(r256_s3 r720_s3 r720_s10)

for source in "${SOURCES[@]}"; do
  gt="${EVAL_ROOT}/samples/${source}/gt_clip.mp4"
  for model in "${MODELS[@]}"; do
    out="${VIZ_ROOT}/by_model/${model}_${source}.mp4"
    ffmpeg -y -loglevel error \
      -i "${gt}" \
      -i "${MATRIX_ROOT}/models/${model}/${source}__r256_s3/vision.mp4" \
      -i "${MATRIX_ROOT}/models/${model}/${source}__r720_s3/vision.mp4" \
      -i "${MATRIX_ROOT}/models/${model}/${source}__r720_s10/vision.mp4" \
      -filter_complex \
      "[0:v]scale=256:256,drawbox=x=0:y=0:w=iw:h=ih:color=green:t=6,drawtext=text='GT':x=10:y=10:fontsize=22:fontcolor=green:box=1:boxcolor=black@0.7[a];\
[1:v]scale=256:256,drawtext=text='256 shift 3':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[b];\
[2:v]scale=256:256,drawtext=text='720 shift 3':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[c];\
[3:v]scale=256:256,drawtext=text='720 shift 10':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[d];\
[a][b][c][d]xstack=inputs=4:layout=0_0|256_0|512_0|768_0[v]" \
      -map "[v]" -an -c:v libx264 -crf 18 -pix_fmt yuv420p -r 20 "${out}"
  done

  for cell in "${CELLS[@]}"; do
    out="${VIZ_ROOT}/by_cell/${cell}_${source}.mp4"
    ffmpeg -y -loglevel error \
      -i "${gt}" \
      -i "${MATRIX_ROOT}/models/original/${source}__${cell}/vision.mp4" \
      -i "${MATRIX_ROOT}/models/A/${source}__${cell}/vision.mp4" \
      -i "${MATRIX_ROOT}/models/B/${source}__${cell}/vision.mp4" \
      -i "${MATRIX_ROOT}/models/D/${source}__${cell}/vision.mp4" \
      -filter_complex \
      "[0:v]scale=256:256,drawbox=x=0:y=0:w=iw:h=ih:color=green:t=6,drawtext=text='GT':x=10:y=10:fontsize=22:fontcolor=green:box=1:boxcolor=black@0.7[a];\
[1:v]scale=256:256,drawtext=text='Original':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[b];\
[2:v]scale=256:256,drawtext=text='A':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[c];\
[3:v]scale=256:256,drawtext=text='B':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[d];\
[4:v]scale=256:256,drawtext=text='D':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[e];\
[a][b][c][d][e]xstack=inputs=5:layout=0_0|256_0|512_0|768_0|1024_0[v]" \
      -map "[v]" -an -c:v libx264 -crf 18 -pix_fmt yuv420p -r 20 "${out}"
  done
done

/home/jungbin_cho/miniforge3/envs/cosmos/bin/python - "${VIZ_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
by_model = sorted(str(path) for path in (root / "by_model").glob("*.mp4"))
by_cell = sorted(str(path) for path in (root / "by_cell").glob("*.mp4"))
if len(by_model) != 20 or len(by_cell) != 15:
    raise RuntimeError(f"expected 20 by-model and 15 by-cell videos, got {len(by_model)} and {len(by_cell)}")
(root / "manifest.json").write_text(
    json.dumps({"gt_border": "green", "by_model": by_model, "by_cell": by_cell}, indent=2) + "\n"
)
print(f"[resolution-viz] complete: {len(by_model) + len(by_cell)} videos")
PY
