#!/bin/bash
# Convert the pinned local renewed Cosmos3-Edge HF artifact to training DCP.
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=edge2dcp
#SBATCH --output=/home/jungbinc/cosmos_motion_ft/slurm-edge2dcp-%j.out

set -euo pipefail

EDGE_FRAMEWORK_ROOT=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
COSMOS_ENV_ROOT=${COSMOS_ENV_ROOT:-/mnt/projects/ll/jungbinc/miniconda3/envs/cosmos}
REPO_ROOT=${REPO_ROOT:-/home/jungbinc/cosmos_motion_ft}
EDGE_MODEL_ROOT=${EDGE_MODEL_ROOT:-/mnt/projects/ll/jungbinc/weka/Cosmos3-Edge}
EDGE_DCP_ROOT=${EDGE_DCP_ROOT:-/mnt/projects/ll/jungbinc/weka/cosmos3_edge_dcp}

[[ -d "${EDGE_FRAMEWORK_ROOT}/cosmos_framework" ]] || { echo "missing Edge framework" >&2; exit 1; }
[[ -s "${EDGE_MODEL_ROOT}/modular_model_index.json" ]] || { echo "missing Edge model" >&2; exit 1; }
if [[ -s "${EDGE_DCP_ROOT}/model/.metadata" ]]; then
  echo "[edge2dcp] existing DCP: ${EDGE_DCP_ROOT}"
  exit 0
fi

export LD_LIBRARY_PATH=
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${REPO_ROOT}:${EDGE_FRAMEWORK_ROOT}:${PYTHONPATH:-}
cd "${EDGE_FRAMEWORK_ROOT}"
"${COSMOS_ENV_ROOT}/bin/python" "${REPO_ROOT}/native_phase_training/convert_edge_to_dcp.py" \
  --checkpoint-path "${EDGE_MODEL_ROOT}" \
  -o "${EDGE_DCP_ROOT}"

[[ -s "${EDGE_DCP_ROOT}/model/.metadata" ]] || { echo "[edge2dcp] conversion produced no model/.metadata" >&2; exit 1; }
echo "[edge2dcp] complete: ${EDGE_DCP_ROOT}"
