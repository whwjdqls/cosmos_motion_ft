#!/usr/bin/env bash
# Inference only, inside an existing owned two-GPU allocation.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Use an owned two-GPU allocation}"
case "$(hostname -s)" in *login*|cs-gpu-01|node37) exit 2;; esac
REPO=/home/whwjdqls99/cosmos_motion_ft
OP=${DA3_EVAL_ROOT:-/scratch2/whwjdqls99/cosmos/operations/da3_c2m_fulltest_20261116}
source /scratch2/whwjdqls99/cosmos/operations/egoego_fulltest_20261107/env.sh
cd "$REPO"
mkdir -p "$OP/logs"
exec 9>"$OP/worker.lock"
flock -n 9 || exit 3
trap 'code=$?; printf "%s\n" "$code" > "$OP/worker.exit_code"' EXIT
mapfile -t devices < <("$EGOEGO_PYTHON" -c 'import os;print("\n".join(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))')
[[ ${#devices[@]} == 2 ]]
for attempt in {1..180}; do
    [[ -f "$OP/PREPARED.json" ]] && break
    sleep 5
done
[[ -f "$OP/PREPARED.json" ]]
"$EGOEGO_PYTHON" -m unittest nymeria_baselines.test_predicted_camera > "$OP/logs/geometry_tests_final.log" 2>&1

publish() {
    "$EGOEGO_PYTHON" -u -m nymeria_baselines.da3_c2m_report --out "$OP" >> "$OP/logs/notion.log" 2>&1
}

run_model() {
    local model=$1 label=$2
    local model_python=$EGOEGO_PYTHON model_pythonpath=$PYTHONPATH
    if [[ "$model" == egoallo ]]; then
        model_python=/lustre/whwjdqls99/cosmos/envs/cosmos/bin/python3.13
        model_pythonpath="$OP/egoallo_deps:$PYTHONPATH"
    fi
    [[ ! -f "$OP/scores/$label/COMPLETE.json" ]] || return 0
    local children=() rank child failed=0
    for rank in 0 1; do
        PYTHONPATH="$model_pythonpath" CUDA_VISIBLE_DEVICES="${devices[$rank]}" "$model_python" -u -m nymeria_baselines.da3_c2m_fulltest \
            sample --out "$OP" --model "$model" --rank "$rank" --world 2 --batch 128 \
            > "$OP/logs/${model}_rank${rank}.log" 2>&1 &
        children+=("$!")
    done
    for child in "${children[@]}"; do wait "$child" || failed=1; done
    ((failed==0)) || return 1
    "$EGOEGO_PYTHON" -u -m nymeria_baselines.da3_c2m_fulltest finish --out "$OP" --model "$model" --world 2 \
        > "$OP/logs/${model}_finish.log" 2>&1
    CUDA_VISIBLE_DEVICES="${devices[0]}" bash shape_aware_TMR/nymeria_run.sh \
        -m nymeria_baselines.fulltest_v2m_score --out "$OP" --model "$label" > "$OP/logs/${model}_score.log" 2>&1
    publish || printf 'Notion publication failed; local results retained.\n' >&2
}

weights=/scratch2/minsu1206/cosmos/operations/egoallo_c2m_fulltest_20261105/snapshots/model_200000.safetensors
if [[ ! -r "$weights" ]]; then
    "$EGOEGO_PYTHON" -c 'import json,sys;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps(dict(reason="Final200k checkpoint owned by minsu1206; read permission required",checkpoint=sys.argv[2]),indent=2)+"\n")' "$OP/EGOALLO_BLOCKED.json" "$weights"
fi
publish || printf 'Initial Notion publication failed; retry after inference.\n' >&2
run_model egoego egoego_da3_trainprior
run_model uem uem_da3_trainprior
if [[ -r "$weights" ]]; then
    [[ ! -f "$OP/EGOALLO_BLOCKED.json" ]] || mv "$OP/EGOALLO_BLOCKED.json" "$OP/EGOALLO_BLOCKED.resolved.json"
    if [[ -f "$OP/EGOALLO_EXTERNAL_WORKER.json" ]]; then
        deadline=$((SECONDS+10800))
        while [[ ! -f "$OP/scores/egoallo_da3_trainprior/COMPLETE.json" && ! -f "$OP/egoallo_worker.exit_code" ]]; do
            ((SECONDS<deadline)) || { echo 'External EgoAllo exceeded the evaluation wait budget.' >&2; exit 4; }
            sleep 15
        done
    fi
    if [[ ! -f "$OP/scores/egoallo_da3_trainprior/COMPLETE.json" ]]; then
        run_model egoallo egoallo_da3_trainprior
    fi
fi
publish
