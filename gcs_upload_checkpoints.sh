#!/usr/bin/env bash
# Upload selected checkpoints + configs + logs + eval METRICS (no heavy viz) to GCS.
# Dest: gs://mm-jinhyung_kim/jungbin/cosmos_motion_ft_checkpoints/<run_name>/...
set -uo pipefail
R=/weka/jungbin/cosmos_motion_ft_runs
DEST=gs://mm-jinhyung_kim/jungbin/cosmos_motion_ft_checkpoints
DRY="${DRY:-0}"                                   # DRY=1 -> just print what would happen

# Exclude (rsync -x, one Python regex vs path relative to src root):
#  - all *.pt (checkpoints copied explicitly), heavy media/arrays, viz_step*/, _smoke*/, checkpoints/ (camera, copied explicitly)
EXCL='(.*\.(pt|mp4|png|jpg|jpeg|gif|webp|npy|npz)$)|(^viz_step.*)|(^_smoke.*)|(.*/viz/.*)|(^checkpoints/.*)'

cp_ckpt(){ # $1=src file/dir  $2=dest suffix
  echo ">> CKPT $1"
  [ "$DRY" = "1" ] && return
  gsutil -o "GSUtil:parallel_composite_upload_threshold=150M" -m cp -r "$1" "$2"
}
sync_rest(){ # $1=src dir  $2=dest dir
  echo ">> META rsync $1 -> $2 (excl heavy/pt/viz)"
  [ "$DRY" = "1" ] && { gsutil -m rsync -r -n -x "$EXCL" "$1" "$2" 2>&1 | grep -iE "would copy" | head -40; return; }
  gsutil -m rsync -r -x "$EXCL" "$1" "$2"
}

run_ckpts(){ # $1=run_name  $2..=ckpt basenames
  local name="$1"; shift
  local src="$R/$name" dst="$DEST/$name"
  for c in "$@"; do
    [ -e "$src/$c" ] && cp_ckpt "$src/$c" "$dst/$c" || echo ">> SKIP (absent) $name/$c"
  done
  sync_rest "$src" "$dst"
}

# ---- Group A: phase3, cross-step union {35000,85000,200000} intersect available ----
run_ckpts ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask ckpt_step035000.pt ckpt_step085000.pt ckpt_step200000.pt
run_ckpts ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam   ckpt_step035000.pt ckpt_step085000.pt ckpt_step200000.pt
run_ckpts ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k           ckpt_step035000.pt ckpt_step085000.pt ckpt_step200000.pt

# ---- Group B: t2m, latest only (step200000) ----
run_ckpts ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_w1_1_5_contact_c0p05_v1_h10_s2_unipc35 ckpt_step200000.pt
run_ckpts ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d                                        ckpt_step200000.pt

# ---- Group C: camera, latest iter dir (model+optim+sched+trainer) + pointer ----
CAM=cosmos3_camera/camera_world/native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k
cp_ckpt "$R/$CAM/checkpoints/iter_000100000" "$DEST/$CAM/checkpoints/iter_000100000"
[ "$DRY" = "1" ] || gsutil -m cp "$R/$CAM/checkpoints/latest_checkpoint.txt" "$DEST/$CAM/checkpoints/latest_checkpoint.txt"
sync_rest "$R/$CAM" "$DEST/$CAM"

echo "ALL UPLOADS SUBMITTED $(date -u +%FT%TZ)"
