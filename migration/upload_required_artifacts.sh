#!/usr/bin/env bash
set -euo pipefail

SECTION=${1:-all}
GCS_ROOT=${GCS_ROOT:-gs://mm-jinhyung_kim/jungbin_cho}
WEKA_ROOT=${WEKA_ROOT:-/weka/jungbin}
RUN_ROOT=${RUN_ROOT:-${WEKA_ROOT}/cosmos_motion_ft_runs}
NANO_REV=fea6e03ac3d7884b4105ed8ee79fc480fca70965
NANO_STAGE=${NANO_STAGE:-${WEKA_ROOT}/cosmos_motion_migration_staging/hf/Cosmos3-Nano-${NANO_REV}}
SOURCE_STAGE=${SOURCE_STAGE:-${WEKA_ROOT}/cosmos_motion_migration_staging/source}
BENCHMARK_STAGE=${BENCHMARK_STAGE:-${WEKA_ROOT}/cosmos_motion_migration_staging/benchmarks}
TORCH_HOME=${TORCH_HOME:-${HOME}/.cache/torch}

case "${SECTION}" in
    source|data|core|eval|phase1|phase2|phase3) ;;
    all)
        for section in source data core eval phase1 phase2 phase3; do
            bash "$0" "${section}"
        done
        echo "[migration-upload] complete section=all root=${GCS_ROOT}"
        exit 0
        ;;
    *)
        echo "usage: $0 {source|data|core|eval|phase1|phase2|phase3|all}" >&2
        exit 2
        ;;
esac

run_section() {
    [[ "${SECTION}" == "$1" ]]
}

sync_tree() {
    local source=$1
    local destination=$2
    test -d "${source}"
    gcloud storage rsync --recursive "${source}" "${destination}"
}

copy_file() {
    local source=$1
    local destination=$2
    test -f "${source}"
    gcloud storage cp "${source}" "${destination}"
}

bundle_repo() {
    local repo=$1
    local name=$2
    local bundle="${SOURCE_STAGE}/${name}.bundle"
    test -d "${repo}/.git"
    mkdir -p "${SOURCE_STAGE}"
    git -C "${repo}" bundle create "${bundle}" --all
    git -C "${repo}" bundle verify "${bundle}"
    copy_file "${bundle}" "${GCS_ROOT}/source/${name}.bundle"
}

sync_selected_joint_run() {
    local run_name=$1
    local checkpoint=$2
    local source="${RUN_ROOT}/${run_name}"
    local destination="${GCS_ROOT}/cosmos_motion_ft_runs/${run_name}"
    test -f "${source}/${checkpoint}"
    copy_file "${source}/${checkpoint}" "${destination}/${checkpoint}"
    for file in config.json train.log; do
        if [[ -f "${source}/${file}" ]]; then
            copy_file "${source}/${file}" "${destination}/${file}"
        fi
    done
    while IFS= read -r eval_dir; do
        sync_tree "${eval_dir}" "${destination}/$(basename "${eval_dir}")"
    done < <(find "${source}" -mindepth 1 -maxdepth 1 -type d -name 'eval*' | sort)
}

sync_native_run() {
    local run_name=$1
    local iteration=$2
    local source="${RUN_ROOT}/cosmos3_camera/camera_world/${run_name}"
    local destination="${GCS_ROOT}/cosmos_motion_ft_runs/cosmos3_camera/camera_world/${run_name}"
    test -f "${source}/checkpoints/${iteration}/model/.metadata"
    gcloud storage rsync --recursive --exclude='^checkpoints/.*$' "${source}" "${destination}"
    sync_tree "${source}/checkpoints/${iteration}" "${destination}/checkpoints/${iteration}"
    if [[ -f "${source}/checkpoints/latest_checkpoint.txt" ]]; then
        copy_file "${source}/checkpoints/latest_checkpoint.txt" "${destination}/checkpoints/latest_checkpoint.txt"
    fi
}

if run_section source; then
    bundle_repo /home/jungbin_cho/cosmos_motion_ft cosmos_motion_ft
    bundle_repo /home/jungbin_cho/cosmos-framework cosmos-framework
    bundle_repo /home/jungbin_cho/kimodo_open kimodo_open
    bundle_repo /home/jungbin_cho/nymeria_kimodo_pipeline nymeria_kimodo_pipeline
fi

if run_section data; then
    proportional="${WEKA_ROOT}/nymeriaplus_kimodo_proportional"
    copy_file "${proportional}/train_test_split.json" \
        "${GCS_ROOT}/nymeriaplus_proportional/train_test_split.json"
    for subdir in camera camera_rgb metadata uniego_rep video joint_latents_T97; do
        sync_tree "${proportional}/${subdir}" "${GCS_ROOT}/nymeriaplus_proportional/${subdir}"
    done
    for subject_dir in "${proportional}"/S*; do
        sync_tree "${subject_dir}" "${GCS_ROOT}/nymeriaplus_proportional/$(basename "${subject_dir}")"
    done
    sync_tree "${WEKA_ROOT}/seed" "${GCS_ROOT}/seed"
fi

if run_section core; then
    sync_tree "${WEKA_ROOT}/cosmos3_nano_dcp" "${GCS_ROOT}/runtime/cosmos3_nano_dcp"
    sync_tree "${WEKA_ROOT}/wan22_vae" "${GCS_ROOT}/runtime/wan22_vae"
    sync_tree "${WEKA_ROOT}/model_cache" "${GCS_ROOT}/runtime/model_cache"
    sync_tree "${TORCH_HOME}/hub/checkpoints" \
        "${GCS_ROOT}/runtime/torch_hub/checkpoints"
    sync_tree "${NANO_STAGE}" "${GCS_ROOT}/runtime/hf/Cosmos3-Nano-${NANO_REV}"
    sync_tree "${WEKA_ROOT}/shape_aware_motion_eval_c45_20260715" \
        "${GCS_ROOT}/evaluators/shape_aware_motion_eval_c45_20260715"
    sync_tree "${WEKA_ROOT}/Kimodo-Motion-Gen-Benchmark/splits" \
        "${GCS_ROOT}/benchmarks/Kimodo-Motion-Gen-Benchmark/splits"
    mkdir -p "${BENCHMARK_STAGE}"
    (
        cd "${WEKA_ROOT}/Kimodo-Motion-Gen-Benchmark-20fps/testsuite"
        find . -type f -name '*.json' -print0 |
            tar --null -T - -czf \
                "${BENCHMARK_STAGE}/Kimodo-Motion-Gen-Benchmark-20fps-testsuite-json.tar.gz"
    )
    copy_file \
        "${BENCHMARK_STAGE}/Kimodo-Motion-Gen-Benchmark-20fps-testsuite-json.tar.gz" \
        "${GCS_ROOT}/benchmarks/Kimodo-Motion-Gen-Benchmark-20fps-testsuite-json.tar.gz"
    copy_file "${WEKA_ROOT}/kimodo_caches/_somaskel77_buffers.npz" \
        "${GCS_ROOT}/runtime/kimodo_caches/_somaskel77_buffers.npz"
fi

if run_section eval; then
    for fixture in \
        native_phase1_eval_inputs_full71_256_T97_v2 \
        native_phase1_eval_inputs_viz5_256_T97_v2 \
        native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1 \
        native_phase1_eval_inputs_vq_prefix1_256_T97_qfilter_person_shift10_v1 \
        joint_attention; do
        sync_tree "${RUN_ROOT}/${fixture}" "${GCS_ROOT}/cosmos_motion_ft_runs/${fixture}"
    done
    sync_tree "${RUN_ROOT}/nymeria_camera_motion_source_audit/final" \
        "${GCS_ROOT}/cosmos_motion_ft_runs/nymeria_camera_motion_source_audit/final"
fi

if run_section phase1; then
    sync_native_run native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k iter_000100000
    sync_native_run native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person iter_000100000
    sync_native_run native_phase1_vq_B_varprefix_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person iter_000100000
    sync_native_run native_phase1_vq_D_varprefix_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person iter_000055000
fi

if run_section phase2; then
    sync_selected_joint_run ja_t2m_ti2m_reasonerimg_x0_T200_mrope3d ckpt_step130000.pt
    sync_selected_joint_run \
        ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d \
        ckpt_step200000.pt
    sync_selected_joint_run \
        ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35 \
        ckpt_step200000.pt
fi

if run_section phase3; then
    sync_selected_joint_run \
        ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k \
        ckpt_step200000.pt
    sync_selected_joint_run \
        ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam \
        ckpt_step115000.pt
    sync_selected_joint_run \
        ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask \
        ckpt_step065000.pt
    sync_selected_joint_run \
        ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k \
        ckpt_step035000.pt
fi

echo "[migration-upload] complete section=${SECTION} root=${GCS_ROOT}"
