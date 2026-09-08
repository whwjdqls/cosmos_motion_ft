#!/usr/bin/env bash
# Validate the completed cache, release the 4090, then launch Pro 6000 smokes.

set -euo pipefail

ROOT=/lustre/whwjdqls99/cosmos
FULL_LATENT_ROOT=${ROOT}/weka/nymeriaplus_kimodo_proportional/joint_latents_T97_256tier_256
WORKER=/home/whwjdqls99/cosmos_motion_ft/native_phase_training/run_precompute_latents_256tier_yonsei.sh
SMOKE_RUNNER=/home/whwjdqls99/cosmos_motion_ft/native_phase_training/run_phase1_edge_smokes_yonsei.sh
LATENT_TMUX=wan_latents_4090_yonsei
PRO_TMUX=phase1_edge_pro6000_yonsei
STATE=${ROOT}/logs/phase1_edge_pro6000_submission.txt

exec > >(tee -a "${ROOT}/logs/phase1_edge_after_latents.log") 2>&1

if [[ -s "${STATE}" ]]; then
  echo "[edge-p1-after-latents] Pro 6000 workflow was already submitted"
  cat "${STATE}"
  exit 0
fi

echo "[edge-p1-after-latents] waiting for production shards 0-3"
while :; do
  complete=0
  for shard in 0 1 2 3; do
    log=${ROOT}/logs/wan_latents_T97_256_shard_${shard}_of_4.log
    if grep -q "production shard ${shard}/4 complete" "${log}" 2>/dev/null; then
      complete=$((complete + 1))
    fi
  done
  echo "[$(date --iso-8601=seconds)] completed_latent_shards=${complete}/4"
  (( complete == 4 )) && break
  sleep 60
done

if [[ ! -s "${FULL_LATENT_ROOT}/latent_cache_complete.json" ]]; then
  echo "[edge-p1-after-latents] running final cache validation in ${LATENT_TMUX}"
  tmux has-session -t "${LATENT_TMUX}" 2>/dev/null || {
    echo "[edge-p1-after-latents] ERROR: persistent 4090 allocation is missing" >&2
    exit 1
  }
  tmux send-keys -t "${LATENT_TMUX}" "bash ${WORKER} validate" C-m
  while [[ ! -s "${FULL_LATENT_ROOT}/latent_cache_complete.json" ]]; do
    tmux has-session -t "${LATENT_TMUX}" 2>/dev/null || {
      echo "[edge-p1-after-latents] ERROR: 4090 allocation ended during validation" >&2
      exit 1
    }
    sleep 30
  done
fi

echo "[edge-p1-after-latents] latent cache complete; releasing persistent 4090"
tmux send-keys -t "${LATENT_TMUX}" exit C-m 2>/dev/null || true

if tmux has-session -t "${PRO_TMUX}" 2>/dev/null; then
  echo "[edge-p1-after-latents] ERROR: tmux session already exists: ${PRO_TMUX}" >&2
  exit 1
fi

tmux new-session -d -s "${PRO_TMUX}" \
  "srun --partition=asus_pro6000,gigabyte_pro6000 --qos=pro6000_qos --nodes=1 --ntasks=1 --gres=gpu:RTXPRO6000:4 --cpus-per-task=32 --mem=512G --time=3-00:00:00 --job-name=edgep1-pro4 bash -lc 'bash ${SMOKE_RUNNER}; rc=\$?; echo [edge-p1-ysdc] smoke_exit=\$rc; echo [edge-p1-ysdc] allocation_kept_open; exec bash'"

{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "tmux_session=${PRO_TMUX}"
  echo "partitions=asus_pro6000,gigabyte_pro6000"
  echo "qos=pro6000_qos"
  echo "gres=gpu:RTXPRO6000:4"
  echo "cpus=32"
  echo "memory=512G"
} > "${STATE}"
cat "${STATE}"
