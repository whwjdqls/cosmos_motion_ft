#!/usr/bin/env bash
# Optional automated first two-segment smoke after simulator installation.
set -euo pipefail
: "${APP_STORAGE:?}" "${SLURM_JOB_ID:?}"
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export CUDA_VISIBLE_DEVICES=${APP_CUDA_VISIBLE_DEVICES:-0}
while [[ ! -f "$APP_STORAGE/setup_venv.exit" ]]; do sleep 20; done
[[ "$(cat "$APP_STORAGE/setup_venv.exit")" == 0 ]] || {
  printf '%s\n' 'Simulator setup failed; see setup_venv.log. No model rollout started.' >&2; exit 1;
}
cd "$REPO_ROOT"
test -f "$APP_STORAGE/evaluation_profile.json"
test -f "$APP_STORAGE/camera_rays.npz"
test -f "$APP_STORAGE/assets/asset_manifest.json"
if [[ ! -f "$APP_STORAGE/rollout_smoke.json" ]]; then
  cp applications/habitat_gs/rollout.example.json "$APP_STORAGE/rollout_smoke.json"
fi
exec bash applications/habitat_gs/run.sh rollout --config "$APP_STORAGE/rollout_smoke.json"
