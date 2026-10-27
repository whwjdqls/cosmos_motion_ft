#!/usr/bin/env bash
# Lightweight scheduler watcher only. ALL data/model work runs in srun steps.
set -Eeuo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${COSMOS_SITE:?Set COSMOS_SITE=yonsei}"
: "${PREP_JOB:?Set the retained idle compute allocation for cache preparation and preflight}"
: "${CANDIDATE_JOBS:?Space-separated owned 2/4-GPU production allocation IDs}"
: "${PHASE3_OPS:?Set the new phase-3 operations directory}"
mkdir -p "${PHASE3_OPS}"
exec 9>"${PHASE3_OPS}/orchestrator.lock"
flock -n 9
# EXIT also covers explicit non-OOM exits from conditional pipelines, where
# Bash suppresses ERR. Never leave a failed preflight looking live.
trap 'code=$?; if ((code != 0)); then printf "{\"stage\":\"failed\",\"exit_code\":%s}\n" "$code" > "${PHASE3_OPS}/status.json"; fi' EXIT
printf '{"stage":"preparing_cache"}\n' > "${PHASE3_OPS}/status.json"
srun --jobid="${PREP_JOB}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=6 --gres=gpu:0 --cpu-bind=none \
    bash "${HERE}/run.sh" -m motion_expert_phase3_edge.data --workers 4 \
    2>&1 | tee "${PHASE3_OPS}/prepare.log"
printf '{"stage":"4090_preflight"}\n' > "${PHASE3_OPS}/status.json"
if ! srun --jobid="${PREP_JOB}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 --cpu-bind=none \
    bash "${HERE}/run.sh" -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m motion_expert_phase3_edge.train --preflight --microbatch 4 --workers 0 \
    --out "${PHASE3_OPS}/preflight_4090_mb4" 2>&1 | tee "${PHASE3_OPS}/preflight_4090_mb4.log"; then
    if ! rg -qi 'out of memory|OutOfMemoryError' "${PHASE3_OPS}/preflight_4090_mb4.log"; then
        exit 1
    fi
    srun --jobid="${PREP_JOB}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 --cpu-bind=none \
        bash "${HERE}/run.sh" -m torch.distributed.run --standalone --nproc_per_node=2 \
        -m motion_expert_phase3_edge.train --preflight --microbatch 2 --workers 0 \
        --out "${PHASE3_OPS}/preflight_4090_mb2" 2>&1 | tee "${PHASE3_OPS}/preflight_4090_mb2.log"
fi
while true; do
    printf '{"stage":"waiting_for_pro6000"}\n' > "${PHASE3_OPS}/status.json"
    for job in ${CANDIDATE_JOBS}; do
        state=$(squeue -h -j "${job}" -o '%T' 2>/dev/null || true)
        [[ "${state}" == RUNNING ]] || continue
        description=$(scontrol show job "${job}" -o)
        user=$(sed -n 's/.*UserId=\([^ (]*\).*/\1/p' <<< "${description}")
        [[ "${user}" == "$(id -un)" ]] || { printf '%s\n' 'Allocation is not owned by this user'; exit 1; }
        gres=$(squeue -h -j "${job}" -o '%b')
        gpus=${gres##*:}
        [[ "${gpus}" == 2 || "${gpus}" == 4 ]] || { printf 'Unexpected GRES: %s\n' "${gres}"; exit 1; }
        cpus=$((gpus * 6))
        printf '{"stage":"production_node_preflight","job":"%s","gpus":%s}\n' "${job}" "${gpus}" > "${PHASE3_OPS}/status.json"
        srun --jobid="${job}" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${cpus}" --gres="gpu:${gpus}" --cpu-bind=none \
            bash "${HERE}/run.sh" -m torch.distributed.run --standalone --nproc_per_node="${gpus}" \
            -m motion_expert_phase3_edge.train --preflight --microbatch 4 --workers 0 \
            --out "${PHASE3_OPS}/preflight_pro_${job}" 2>&1 | tee "${PHASE3_OPS}/preflight_pro_${job}.log"
        printf '{"stage":"training","job":"%s","gpus":%s}\n' "${job}" "${gpus}" > "${PHASE3_OPS}/status.json"
        srun --jobid="${job}" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${cpus}" --gres="gpu:${gpus}" --cpu-bind=none \
            bash "${HERE}/run.sh" -m torch.distributed.run --standalone --nproc_per_node="${gpus}" \
            -m motion_expert_phase3_edge.train --microbatch 4 --workers 4 \
            --acceptance "${PHASE3_OPS}/preflight_pro_${job}/acceptance.json" \
            2>&1 | tee "${PHASE3_OPS}/training_${job}.log"
        printf '{"stage":"trainer_exited","job":"%s"}\n' "${job}" > "${PHASE3_OPS}/status.json"
        exit 0
    done
    sleep 60
done
