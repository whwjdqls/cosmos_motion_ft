#!/usr/bin/env bash
set -euo pipefail

SECTION=${1:-all}
GCS_ROOT=${GCS_ROOT:-gs://mm-jinhyung_kim/jungbin_cho}
WEKA_ROOT=${WEKA_ROOT:-/weka/jungbin}
RUN_ROOT=${RUN_ROOT:-${WEKA_ROOT}/cosmos_motion_ft_runs}
HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
NANO_REV=fea6e03ac3d7884b4105ed8ee79fc480fca70965

case "${SECTION}" in
    source|data|core|runs) ;;
    all)
        for section in source data core runs; do
            bash "$0" "${section}"
        done
        echo "[migration-restore] complete section=all"
        exit 0
        ;;
    *)
        echo "usage: $0 {source|data|core|runs|all}" >&2
        exit 2
        ;;
esac

run_section() {
    [[ "${SECTION}" == "$1" ]]
}

restore_tree() {
    local source=$1
    local destination=$2
    mkdir -p "${destination}"
    gcloud storage rsync --recursive "${source}" "${destination}"
}

if run_section data; then
    restore_tree "${GCS_ROOT}/nymeriaplus_proportional" \
        "${WEKA_ROOT}/nymeriaplus_kimodo_proportional"
    restore_tree "${GCS_ROOT}/seed" "${WEKA_ROOT}/seed"
fi

if run_section source; then
    restore_tree "${GCS_ROOT}/source" "${WEKA_ROOT}/cosmos_motion_source_bundles"
fi

if run_section core; then
    restore_tree "${GCS_ROOT}/runtime/cosmos3_nano_dcp" "${WEKA_ROOT}/cosmos3_nano_dcp"
    restore_tree "${GCS_ROOT}/runtime/wan22_vae" "${WEKA_ROOT}/wan22_vae"
    restore_tree "${GCS_ROOT}/runtime/model_cache" "${WEKA_ROOT}/model_cache"
    restore_tree "${GCS_ROOT}/runtime/hf/Cosmos3-Nano-${NANO_REV}" \
        "${HF_HOME}/hub/models--nvidia--Cosmos3-Nano/snapshots/${NANO_REV}"
    restore_tree "${GCS_ROOT}/evaluators/shape_aware_motion_eval_c45_20260715" \
        "${WEKA_ROOT}/shape_aware_motion_eval_c45_20260715"
fi

if run_section runs; then
    restore_tree "${GCS_ROOT}/cosmos_motion_ft_runs" "${RUN_ROOT}"
fi

echo "[migration-restore] complete section=${SECTION}"
