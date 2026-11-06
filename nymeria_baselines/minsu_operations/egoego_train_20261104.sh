#!/usr/bin/env bash
# Run the newly authorized local EgoEgo integration; shared trainers are untouched.
set -Eeuo pipefail
umask 0027
[[ ${SLURM_JOB_ID:-} == 2670587 && $(hostname -s) == node39 && $(id -un) == minsu1206 ]] || exit 2
op=/scratch2/minsu1206/cosmos/operations/egoego_shape_20261104
export PYTHONPATH=/home/minsu1206/cosmos_motion_ft PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LD_PRELOAD=/lustre/whwjdqls99/cosmos/envs/cosmos/lib/libstdc++.so.6
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_RAS_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export XDG_CACHE_HOME=$op/runtime MPLCONFIGDIR=$op/runtime/matplotlib
export TMPDIR=$(mktemp -d /tmp/minsu-egoego.XXXXXX)
mkdir -p "$MPLCONFIGDIR" "$op/acceptance"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT
py=/scratch2/minsu1206/cosmos/envs/uem_blackwell/bin/python
exec 9>"$op/worker.lock"
flock -n 9 || exit 3
stage=${1:?smoke or production}
printf '%s\n' "$stage" > "$op/stage"
trap 'code=$?; printf "%s\n" "$code" > "$op/${stage}.exit_code"; if ((code!=0)); then printf "FAILED_%s\n" "$stage" > "$op/stage"; fi' EXIT
launch=("$py" -u -m torch.distributed.run --standalone --nproc_per_node=2 -m nymeria_baselines.egoego_shape.train --stats "$op/data/stats.npz")
if [[ $stage == smoke ]]; then
    "${launch[@]}" --out "$op/smoke" --smoke --stop-after 4 --resume none > "$op/smoke4.log" 2>&1
    cp "$op/smoke/last.ckpt" "$op/acceptance/step4.ckpt"
    "${launch[@]}" --out "$op/smoke" --smoke --stop-after 6 --resume auto > "$op/resume6.log" 2>&1
    "$py" - "$op" <<'PY'
import json,sys,torch
from pathlib import Path
op=Path(sys.argv[1]);a=torch.load(op/'acceptance/step4.ckpt',map_location='cpu',weights_only=False,mmap=True);b=torch.load(op/'smoke/last.ckpt',map_location='cpu',weights_only=False,mmap=True)
assert a['global_step']==4 and b['global_step']==6
assert a['hyper_parameters']==b['hyper_parameters']
assert b['optimizer_states'][0]['current_step']==6
assert len(b['optimizer_states'][0]['ema'])>0
assert len(b['optimizer_states'][0]['opt']['state'])>0
assert a['lr_schedulers'] and b['lr_schedulers']
assert any(not torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items() if v.is_floating_point())
value={'passed':True,'fresh_step':4,'resumed_step':6,'optimizer_steps':b['optimizer_states'][0]['current_step'],'optimizer_and_ema_restored':True,'scheduler_restored':True,'parameters_changed_after_resume':True}
(op/'acceptance/RESUME_ACCEPTED.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
PY
    echo SMOKE_PASSED > "$op/stage"
elif [[ $stage == production ]]; then
    "$py" - "$op" <<'PY'
import json,sys
from pathlib import Path
from nymeria_baselines.egoego_shape.train import contract
op=Path(sys.argv[1]);a=json.loads((op/'acceptance/ACCEPTED.json').read_text());b=json.loads((op/'acceptance/RESUME_ACCEPTED.json').read_text());assert a['passed'] and b['passed'];assert a['contract']==contract(op/'data/stats.npz'),'Source or recipe changed after preflight'
PY
    echo TRAINING > "$op/stage"
    "${launch[@]}" --out "$op/training" --resume auto > "$op/production.log" 2>&1
    echo COMPLETE > "$op/stage"
else
    exit 2
fi
