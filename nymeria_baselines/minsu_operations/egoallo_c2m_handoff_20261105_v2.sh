#!/usr/bin/env bash
set -Eeuo pipefail
umask 0027
OP=/scratch2/minsu1206/cosmos/operations/egoallo_c2m_fulltest_20261105
TRAIN=/scratch2/minsu1206/cosmos/operations/shape_conditioned_baselines_20261101/egoallo/training
REPO=/home/whwjdqls99/cosmos_motion_ft
LABEL=egoallo_final200k_camera_shape
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(id -un) == minsu1206 && $(hostname -s) == node38 ]] || exit 2
if [[ ${1:-} == wait ]]; then
 exec 8>"$OP/handoff.lock";flock -n 8 || exit 3
 exec >>"$OP/handoff.log" 2>&1
 echo "$(date -Is) waiting for final200k and trainer GPU exit"
 while true; do
  state=$(squeue -h -j 2667149 -o '%u %T %N')
  [[ $state == 'minsu1206 RUNNING node38' ]] || { echo "Allocation unavailable: $state";exit 4; }
  if [[ -f $TRAIN/COMPLETE.json && -f $TRAIN/checkpoints_200000/COMPLETE.json ]]; then
   break
  fi
  sleep 10
 done
 echo "$(date -Is) final checkpoint complete and both GPUs free; launching evaluation"
 exec srun --jobid=2667149 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=12 --gres=gpu:RTX4090:2 \
  bash "$here/egoallo_c2m_handoff_20261105_v2.sh" run
fi
[[ ${SLURM_JOB_ID:-} == 2667149 && ${1:-} == run ]] || exit 2
exec 9>"$OP/production.lock";flock -n 9 || exit 3
exec >>"$OP/pipeline.log" 2>&1
trap 'rc=$?; echo "$rc" > "$OP/exit_code"; if ((rc)); then date -Is > "$OP/FAILED"; fi' EXIT
export LD_PRELOAD=/lustre/whwjdqls99/cosmos/envs/cosmos/lib/libstdc++.so.6
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 COSMOS_SITE=yonsei
export PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export XDG_CACHE_HOME=$OP/runtime_cache TORCH_EXTENSIONS_DIR=$OP/runtime_cache/torch_extensions
export MPLCONFIGDIR=$OP/runtime_cache/matplotlib TMPDIR=$(mktemp -d /tmp/minsu-eac2m.XXXXXX)
mkdir -p "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR"
cd "$REPO"
PY=/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015/env_egoallo/bin/python
mapfile -t devices < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ ${#devices[@]} == 2 ]] || exit 5
for device in "${devices[@]}"; do
 while [[ -n $(nvidia-smi -i "$device" --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do sleep 10; done
done
echo SAMPLING > "$OP/stage"
CUDA_VISIBLE_DEVICES=${devices[0]} "$PY" -u "$here/egoallo_c2m_fulltest_20261105.py" --rank 0 > "$OP/rank0.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=${devices[1]} "$PY" -u "$here/egoallo_c2m_fulltest_20261105.py" --rank 1 > "$OP/rank1.log" 2>&1 & p1=$!
trap 'kill -TERM "$p0" "$p1" 2>/dev/null || true; wait || true' TERM INT
rc=0;wait "$p0" || rc=1;wait "$p1" || rc=1
((rc==0)) || exit 7
"$PY" - <<'PY'
import json
from pathlib import Path
from nymeria_baselines.fulltest_v2m import read,save
op=Path('/scratch2/minsu1206/cosmos/operations/egoallo_c2m_fulltest_20261105');label='egoallo_final200k_camera_shape';root=op/'models'/label
rows=read(op/'cases.json');counts=[read(root/f'COMPLETE_rank{i}.json')['cases'] for i in range(2)]
assert sum(counts)==len(rows)==11911
assert all((root/'predictions'/(r['id']+'.npz')).exists() for r in rows)
save(root/'INFERENCE_COMPLETE.json',dict(cases=len(rows),step=200000,shards=counts))
PY
echo SCORING > "$OP/stage"
CUDA_VISIBLE_DEVICES=${devices[0]} bash "$REPO/shape_aware_TMR/nymeria_run.sh" \
 -m nymeria_baselines.fulltest_v2m_score --out "$OP" --model "$LABEL" > "$OP/score.log" 2>&1
echo SCORED_AWAITING_NOTION > "$OP/stage"
date -Is > "$OP/READY_FOR_PUBLICATION"
