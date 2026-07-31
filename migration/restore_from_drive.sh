#!/usr/bin/env bash
# Restore the supplementary Google Drive archive without deleting remote or local files.
set -euo pipefail

SECTION=${1:-all}
DRIVE_REMOTE=${DRIVE_REMOTE:-data:}
WEKA_ROOT=${WEKA_ROOT:-/weka/jungbin}
RUN_ROOT=${RUN_ROOT:-${WEKA_ROOT}/cosmos_motion_ft_runs}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
TORCH_HOME=${TORCH_HOME:-${HOME}/.cache/torch}

case "${SECTION}" in
  data|eval|models|checkpoints|run-metadata) ;;
  all)
    for section in data eval models checkpoints run-metadata; do
      bash "$0" "${section}"
    done
    echo "[drive-restore] complete section=all"
    exit 0
    ;;
  *)
    echo "usage: $0 {data|eval|models|checkpoints|run-metadata|all}" >&2
    exit 2
    ;;
esac

copy_tree() {
  local source=$1
  local destination=$2
  mkdir -p "${destination}"
  rclone copy "${DRIVE_REMOTE}${source}" "${destination}"
}

copy_file() {
  local source=$1
  local destination=$2
  mkdir -p "$(dirname "${destination}")"
  rclone copyto "${DRIVE_REMOTE}${source}" "${destination}"
}

if [[ "${SECTION}" == data ]]; then
  copy_tree cosmos_data/nymeriaplus_kimodo_proportional \
    "${WEKA_ROOT}/nymeriaplus_kimodo_proportional"
  copy_tree cosmos_data/seed "${WEKA_ROOT}/seed"
  copy_tree cosmos_data/benchmarks/Kimodo-Motion-Gen-Benchmark/splits \
    "${WEKA_ROOT}/Kimodo-Motion-Gen-Benchmark/splits"
  copy_tree cosmos_data/benchmarks/Kimodo-Motion-Gen-Benchmark-20fps \
    "${WEKA_ROOT}/Kimodo-Motion-Gen-Benchmark-20fps"
  copy_tree cosmos_data/joint_attention "${RUN_ROOT}/joint_attention"
  copy_file cosmos_data/joint_attention/nymeria_pairs_train.jsonl \
    "${REPO_ROOT}/motion_expert/pairs_train.jsonl"
  copy_file cosmos_data/joint_attention/nymeria_pairs_val.jsonl \
    "${REPO_ROOT}/motion_expert/pairs_val.jsonl"
fi

if [[ "${SECTION}" == eval ]]; then
  copy_tree cosmos_data/eval_fixtures/native_phase1_eval_inputs_full71_256_T97_v2 \
    "${RUN_ROOT}/native_phase1_eval_inputs_full71_256_T97_v2"
  copy_tree cosmos_data/eval_fixtures/native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1 \
    "${RUN_ROOT}/native_phase1_eval_inputs_vq_prefix5_256_T97_qfilter_person_v1"
  copy_tree cosmos_data/eval_fixtures/shape_aware_motion_eval_c45_20260715 \
    "${WEKA_ROOT}/shape_aware_motion_eval_c45_20260715"
fi

if [[ "${SECTION}" == models ]]; then
  copy_tree cosmos_models/evaluation/model_cache "${WEKA_ROOT}/model_cache"
  copy_file cosmos_models/evaluation/torch_hub/checkpoints/alexnet-owt-7be5be79.pth \
    "${TORCH_HOME}/hub/checkpoints/alexnet-owt-7be5be79.pth"
fi

if [[ "${SECTION}" == checkpoints ]]; then
  CAMERA_ROOT="${RUN_ROOT}/cosmos3_camera/camera_world"
  copy_tree cosmos_ckpts/native_phase1_vq_A/iter_000100000 \
    "${CAMERA_ROOT}/native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000"
  copy_tree cosmos_ckpts/native_phase1_vq_B/iter_000100000 \
    "${CAMERA_ROOT}/native_phase1_vq_B_varprefix_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000"
  copy_tree cosmos_ckpts/native_phase1_vq_D/iter_000100000 \
    "${CAMERA_ROOT}/native_phase1_vq_D_varprefix_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/checkpoints/iter_000100000"
  copy_file cosmos_ckpts/native_phase1_baseline/iter_000100000_ema_gen_delta.pt \
    "${RUN_ROOT}/portable/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_iter100000_ema_gen_delta.pt"
  copy_file cosmos_ckpts/ja_phase2_t2m_ti2m_native/ckpt_step200000.pt \
    "${RUN_ROOT}/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/ckpt_step200000.pt"
  copy_file cosmos_ckpts/ja_phase2_t2m_ti2m_contact_unipc35/ckpt_step200000.pt \
    "${RUN_ROOT}/ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35/ckpt_step200000.pt"
  copy_file cosmos_ckpts/ja_phase3_bridge_native/ckpt_step200000.pt \
    "${RUN_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/ckpt_step200000.pt"
  copy_file cosmos_ckpts/ja_phase3_bridge_native_headcam/ckpt_step115000.pt \
    "${RUN_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam/ckpt_step115000.pt"
  copy_file cosmos_ckpts/ja_phase3_bridge_native_multitask/ckpt_step065000.pt \
    "${RUN_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask/ckpt_step065000.pt"
  copy_file cosmos_ckpts/ja_phase3_bridge_native_contact/ckpt_step035000.pt \
    "${RUN_ROOT}/ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k/ckpt_step035000.pt"
fi

if [[ "${SECTION}" == run-metadata ]]; then
  METADATA_STAGING=${RUN_METADATA_STAGING_ROOT:-${WEKA_ROOT}/cosmos_motion_migration_staging/run_metadata}
  copy_tree cosmos_run_metadata_archives "${METADATA_STAGING}"
  mkdir -p "${RUN_ROOT}"
  (
    cd "${METADATA_STAGING}"
    sha256sum -c ./*.tar.zst.sha256
    for archive in ./*.tar.zst; do
      tar --zstd -xf "${archive}" -C "${RUN_ROOT}"
    done
  )
fi

echo "[drive-restore] complete section=${SECTION}"
