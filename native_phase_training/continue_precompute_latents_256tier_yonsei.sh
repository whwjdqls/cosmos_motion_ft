#!/usr/bin/env bash
# Wait for the RTX 4090 smoke, reuse its allocation for shard 0, and submit 1-7.

set -euo pipefail

TMUX_SESSION=${LATENT_TMUX_SESSION:-wan_latents_4090_yonsei}
SMOKE_ROOT=/lustre/whwjdqls99/cosmos/weka/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256_smoke_yonsei_4090
SMOKE_COMPLETE=${SMOKE_ROOT}/smoke_validation_complete.json
SMOKE_LOG=/lustre/whwjdqls99/cosmos/logs/wan_latents_T97_256_smoke_4090.log
STATE=/lustre/whwjdqls99/cosmos/logs/wan_latents_T97_256_submission.txt
LOCK=${STATE}.lock
WORKER=/home/whwjdqls99/cosmos_motion_ft/native_phase_training/run_precompute_latents_256tier_yonsei.sh
ARRAY_SCRIPT=/home/whwjdqls99/cosmos_motion_ft/native_phase_training/sbatch_precompute_latents_256tier_yonsei_array.sh

if [[ -s "${STATE}" ]]; then
  echo "[p1pc256-continue] production was already submitted:"
  cat "${STATE}"
  exit 0
fi
if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "[p1pc256-continue] another continuation supervisor owns ${LOCK}" >&2
  exit 1
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

echo "[p1pc256-continue] waiting for smoke validation: ${SMOKE_COMPLETE}"
while [[ ! -s "${SMOKE_COMPLETE}" ]]; do
  if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "[p1pc256-continue] ERROR: persistent allocation ended before smoke validation" >&2
    exit 1
  fi
  if grep -Eq 'smoke_exit=[1-9][0-9]*' "${SMOKE_LOG}" 2>/dev/null; then
    echo "[p1pc256-continue] ERROR: smoke failed; production will not be submitted" >&2
    exit 1
  fi
  sleep 15
done

echo "[p1pc256-continue] smoke validation passed"
while ! tmux capture-pane -p -t "${TMUX_SESSION}" -S -20 | \
  grep -q 'allocation_kept_open_for_production_shard_0'; do
  if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "[p1pc256-continue] ERROR: allocation ended during smoke handoff" >&2
    exit 1
  fi
  sleep 2
done

shard0_command="TOTAL_SHARDS=4 SHARD_ID=0 bash ${WORKER} shard"
tmux send-keys -t "${TMUX_SESSION}" "${shard0_command}" C-m
array_job=$(sbatch --parsable "${ARRAY_SCRIPT}")

{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "persistent_tmux=${TMUX_SESSION}"
  echo "persistent_srun_shard=0"
  echo "sbatch_array_job=${array_job}"
  echo "sbatch_array_shards=1-3"
  echo "total_shards=4"
} > "${STATE}"

cat "${STATE}"
