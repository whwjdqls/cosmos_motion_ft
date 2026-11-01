#!/usr/bin/env bash
# Thin account launcher. All model/trainer implementation remains in whw's repo.
set -euo pipefail
SCRIPT=$(realpath "${BASH_SOURCE[0]}")
REPO=/home/whwjdqls99/cosmos_motion_ft
BASE=/scratch2/whwjdqls99/cosmos/operations/nymeria_official_baselines_20261015
SHARED=/scratch2/whwjdqls99/cosmos/operations/shape_conditioned_baselines_20261101
UPSTREAM=$SHARED/upstream
CACHE=/scratch2/whwjdqls99/cosmos/caches/nymeria_soma_official_baselines_T97_20fps_floorclean_20261016
ROOT=/scratch2/minsu1206/cosmos/operations/shape_conditioned_baselines_20261101
if [[ $(id -un) != minsu1206 ]]; then
    echo 'Run this launcher from the minsu1206 account; it cannot impersonate another user.' >&2
    exit 2
fi
mode=${1:-queue}
name=${2:-}
if [[ $mode == queue ]]; then
    mkdir -p "$ROOT"
    exec 9>"$ROOT/queue.lock"
    flock -n 9 || { echo 'Another queue operation is active.'; exit 3; }
    for item in uem egoallo; do
        [[ -r "$BASE/env_$item/bin/python" && -r "$SHARED/$item/cpu_preflight/ACCEPTED.json" ]]
    done
    [[ -r "$CACHE/index.json" && -r "$CACHE/body_shape16.npz" && -r "$UPSTREAM/source_manifest.json" ]]
    for item in uem egoallo; do
        mkdir -p "$ROOT/$item"
        session="${item}_shape2_minsu_20261101"
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "$session already exists; retaining it."
        elif [[ -n $(squeue -u minsu1206 -h -n "$item-shape2") ]]; then
            echo "A $item-shape2 Slurm job already exists; no duplicate submitted."
        else
            tmux new-session -d -s "$session" bash "$SCRIPT" request "$item"
            echo "Created $session"
        fi
    done
    echo "Logs/checkpoints: $ROOT/{uem,egoallo}"
    exit 0
fi
[[ $name == uem || $name == egoallo ]] || { echo 'Expected uem or egoallo'; exit 2; }
OP=$ROOT/$name
mkdir -p "$OP"
if [[ $mode == request ]]; then
    exec > >(tee -a "$OP/request.log") 2>&1
    exec srun --account=ys_ms --qos=base_qos --partition=suma_rtx4090 \
        --gres=gpu:RTX4090:2 --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G \
        --time=3-00:00:00 --exclude=cs-gpu-01,node37,node29 --job-name="$name-shape2" \
        --pty bash "$SCRIPT" allocation "$name"
elif [[ $mode == allocation ]]; then
    printf '%s %s\n' "$SLURM_JOB_ID" "$(hostname -s)" > "$OP/allocation.txt"
    set +e
    bash "$SCRIPT" run "$name" > "$OP/driver.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$OP/exit_code"
    echo "$name driver exited with $code; retaining allocation. See $OP/driver.log"
    exec bash -i
elif [[ $mode != run ]]; then
    echo 'Modes: queue, request, allocation, run' >&2
    exit 2
fi
[[ -n ${SLURM_JOB_ID:-} && $(hostname -s) != *login* ]]
exec 8>"$OP/run.lock"
flock -n 8 || { echo 'A trainer driver already owns this run.'; exit 3; }
export LD_PRELOAD=/lustre/whwjdqls99/cosmos/envs/cosmos/lib/libstdc++.so.6
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 COSMOS_SITE=yonsei
export PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=$ROOT/runtime_cache/$name
export TORCH_EXTENSIONS_DIR=$XDG_CACHE_HOME/torch_extensions
export MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib
export TMPDIR=$XDG_CACHE_HOME/tmp
mkdir -p "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR" "$TMPDIR"
cd "$REPO"
PY=$BASE/env_$name/bin/python
common=(--upstream "$UPSTREAM" --cache "$CACHE")
stamp=$(date -u +%Y%m%dT%H%M%SZ)
accept=$OP/acceptance_$stamp
"$PY" -u -m nymeria_baselines.shape_conditioned.preflight --model "$name" \
    "${common[@]}" --output "$accept" --device cuda > "$OP/gpu_preflight_$stamp.log" 2>&1
# Fresh acceptance output per attempt; restart it to prove actual optimizer resume.
smoke=$OP/ddp_smoke_$stamp
for stop in 3 5; do
    "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
        -m "nymeria_baselines.shape_conditioned.$name" "${common[@]}" \
        --output "$smoke" --stop-after "$stop" > "$OP/ddp_${stop}_$stamp.log" 2>&1
done
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("step",d.get("global_step"))==5; print("DDP resume accepted",d)' "$smoke/STOPPED.json"
date -Is > "$OP/PRODUCTION_STARTED"
exec "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m "nymeria_baselines.shape_conditioned.$name" "${common[@]}" \
    --output "$OP/training" >> "$OP/training.log" 2>&1
