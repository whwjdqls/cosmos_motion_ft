#!/bin/bash
# Materialize the clean Edge-capable framework checkout used by all Edge jobs.

set -euo pipefail

TARGET=${COSMOS_FRAMEWORK_EDGE_ROOT:-/mnt/projects/ll/jungbinc/cosmos-framework-edge}
REVISION=d4599e2e43fbd06168e9884205b9b66c3902d8f6
REMOTE=https://github.com/NVIDIA/cosmos-framework.git

if [[ ! -d "${TARGET}/.git" ]]; then
  git clone "${REMOTE}" "${TARGET}"
fi

actual=$(git -C "${TARGET}" rev-parse HEAD)
if [[ "${actual}" != "${REVISION}" ]]; then
  if [[ -n "$(git -C "${TARGET}" status --porcelain)" ]]; then
    echo "[edge-framework] ERROR: ${TARGET} is dirty at ${actual}; refusing to change it" >&2
    exit 1
  fi
  git -C "${TARGET}" fetch origin "${REVISION}"
  git -C "${TARGET}" checkout --detach "${REVISION}"
fi

actual=$(git -C "${TARGET}" rev-parse HEAD)
[[ "${actual}" == "${REVISION}" ]] || { echo "wrong framework revision: ${actual}" >&2; exit 1; }
[[ -z "$(git -C "${TARGET}" status --porcelain)" ]] || { echo "framework checkout is dirty" >&2; exit 1; }
echo "[edge-framework] ready ${TARGET} @ ${actual}"
