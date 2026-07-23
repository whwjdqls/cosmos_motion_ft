#!/usr/bin/env bash
set -euo pipefail

COSMOS_PYTHON="${COSMOS_PYTHON:-/home/jungbin_cho/miniforge3/envs/cosmos/bin/python}"
CDFVD_ROOT="${CDFVD_ROOT:-/home/jungbin_cho/.cache/cosmos_motion_ft/third_party/content-debiased-fvd}"
CDFVD_REVISION="a1e037ab7cb087debd2221d14ae4a001ec054201"

mkdir -p "$(dirname "${CDFVD_ROOT}")"
if [[ ! -d "${CDFVD_ROOT}/.git" ]]; then
  git clone https://github.com/songweige/content-debiased-fvd.git "${CDFVD_ROOT}"
fi
git -C "${CDFVD_ROOT}" fetch origin main
git -C "${CDFVD_ROOT}" checkout --detach "${CDFVD_REVISION}"
"${COSMOS_PYTHON}" -m pip install --no-deps dreamsim==0.2.1

echo "DreamSim 0.2.1 and content-debiased-fvd ${CDFVD_REVISION} are ready."

