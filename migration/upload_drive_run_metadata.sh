#!/usr/bin/env bash
# Archive reproducibility metadata and numeric evaluation outputs without
# duplicating checkpoints or generated visualization media.
set -euo pipefail

RCLONE_REMOTE=${RCLONE_REMOTE:-data:}
WEKA_ROOT=${WEKA_ROOT:-/weka/jungbin}
RUN_ROOT=${RUN_ROOT:-${WEKA_ROOT}/cosmos_motion_ft_runs}
DEST_ROOT=${DRIVE_RUN_METADATA_ROOT:-${RCLONE_REMOTE}cosmos_run_metadata_archives}
STAGING_ROOT=${RUN_METADATA_STAGING_ROOT:-${WEKA_ROOT}/cosmos_motion_migration_staging/run_metadata}

NATIVE_RUNS=(
  native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k
  native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person
  native_phase1_vq_B_varprefix_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person
  native_phase1_vq_D_varprefix_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person
  world_camera_nymeria_97f_cont
  world_camera_nymeria_97f_hung_iter6000
)

JOINT_RUNS=(
  ja_t2m_ti2m_reasonerimg_x0_T200_mrope3d
  ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d
  ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35
  ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k
  ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam
  ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask
  ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k
)

archive_run() {
  local relative=$1
  local source="${RUN_ROOT}/${relative}"
  local archive_name=${relative//\//__}.tar.zst
  local archive="${STAGING_ROOT}/${archive_name}"
  test -d "${source}"
  mkdir -p "${STAGING_ROOT}"
  echo "[drive-run-metadata] archive ${relative}"
  tar --zstd -cf "${archive}" \
    --exclude='*/checkpoints' \
    --exclude='*/checkpoints/*' \
    --exclude='*/ckpt_step*.pt' \
    --exclude='*/latest.pt' \
    --exclude='*/viz*' \
    --exclude='*/viz*/*' \
    --exclude='*/latents' \
    --exclude='*/latents/*' \
    --exclude='*.mp4' \
    --exclude='*.png' \
    --exclude='*.jpg' \
    --exclude='*.jpeg' \
    --exclude='*.gif' \
    --exclude='events.out.tfevents.*' \
    --exclude='*/tensorboard' \
    --exclude='*/tensorboard/*' \
    --exclude='*/DeviceMonitor' \
    --exclude='*/DeviceMonitor/*' \
    --exclude='*/norm_monitor' \
    --exclude='*/norm_monitor/*' \
    -C "${RUN_ROOT}" "${relative}"
  (
    cd "${STAGING_ROOT}"
    sha256sum "${archive_name}" > "${archive_name}.sha256"
  )
  rclone copyto "${archive}" "${DEST_ROOT}/${archive_name}" \
    --drive-chunk-size 64M
  rclone copyto "${archive}.sha256" "${DEST_ROOT}/${archive_name}.sha256"
}

for run in "${NATIVE_RUNS[@]}"; do
  archive_run "cosmos3_camera/camera_world/${run}"
done

for run in "${JOINT_RUNS[@]}"; do
  archive_run "${run}"
done

echo "[drive-run-metadata] complete root=${DEST_ROOT}"
