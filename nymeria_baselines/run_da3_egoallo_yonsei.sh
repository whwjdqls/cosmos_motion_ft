#!/usr/bin/env bash
# One-GPU native EgoAllo evaluation, parallel with the two-GPU UEM worker.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Owned allocation required}"
case "$(hostname -s)" in *login*|cs-gpu-01|node37) exit 2;; esac
source /scratch2/whwjdqls99/cosmos/operations/egoego_fulltest_20261107/env.sh
OP=/scratch2/whwjdqls99/cosmos/operations/da3_c2m_fulltest_20261116
cd "$REPO_ROOT"
exec 9>"$OP/egoallo_worker.lock"
flock -n 9 || exit 3
trap 'code=$?; printf "%s\n" "$code" > "$OP/egoallo_worker.exit_code"' EXIT
export PYTHONPATH="$OP/egoallo_deps:$PYTHONPATH"
PY=/lustre/whwjdqls99/cosmos/envs/cosmos/bin/python3.13
"$PY" -u -m nymeria_baselines.da3_c2m_fulltest sample --out "$OP" --model egoallo --world 1 --batch 1 > "$OP/logs/egoallo_rank0.log" 2>&1
"$PY" -u -m nymeria_baselines.da3_c2m_fulltest finish --out "$OP" --model egoallo --world 1 > "$OP/logs/egoallo_finish.log" 2>&1
bash shape_aware_TMR/nymeria_run.sh -m nymeria_baselines.fulltest_v2m_score --out "$OP" --model egoallo_da3_trainprior > "$OP/logs/egoallo_score.log" 2>&1
"$PY" -u -m nymeria_baselines.da3_c2m_report --out "$OP" >> "$OP/logs/notion.log" 2>&1
