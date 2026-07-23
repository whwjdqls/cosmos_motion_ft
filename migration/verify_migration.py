#!/usr/bin/env python
"""Verify required GCS migration objects or a restored local filesystem."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


GCS_ROOT = os.environ.get("GCS_ROOT", "gs://mm-jinhyung_kim/jungbin_cho")
WEKA_ROOT = Path(os.environ.get("WEKA_ROOT", "/weka/jungbin"))
RUN_ROOT = Path(os.environ.get("RUN_ROOT", str(WEKA_ROOT / "cosmos_motion_ft_runs")))
HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
NANO_REV = "fea6e03ac3d7884b4105ed8ee79fc480fca70965"

HASHED_FILES = {
    "nymeriaplus_proportional/train_test_split.json":
        "2e533800db36dc31a6d8c8b87798f54275d9f939a0f0024056491eab4e8d8081",
    "nymeriaplus_proportional/metadata/floor_calibration.json":
        "ad27208f9ad8e472d16bcda14e8fdc765b3496248a8763ed95619a9a2f3693c8",
    "nymeriaplus_proportional/metadata/camera_motion_quality_filter_v1_T97.json":
        "1fd6465890cbf175068db839beb8bb220f6964090ff2c583cbf50d5001989848",
}

GCS_REQUIRED = (
    "runtime/cosmos3_nano_dcp/model/.metadata",
    "runtime/wan22_vae/Wan2.2_VAE.pth",
    f"runtime/hf/Cosmos3-Nano-{NANO_REV}/model.safetensors.index.json",
    f"runtime/hf/Cosmos3-Nano-{NANO_REV}/vision_encoder/model.safetensors",
    f"runtime/hf/Cosmos3-Nano-{NANO_REV}/text_tokenizer/tokenizer.json",
    "runtime/model_cache/cdfvd/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
    "evaluators/shape_aware_motion_eval_c45_20260715/SHA256SUMS",
    "cosmos_motion_ft_runs/joint_attention/full71_windows.json",
    "cosmos_motion_ft_runs/joint_attention/bones_index_train.json",
    "cosmos_motion_ft_runs/native_phase1_eval_inputs_full71_256_T97_v2/fd_input.jsonl",
    "cosmos_motion_ft_runs/cosmos3_camera/camera_world/"
    "native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k/"
    "checkpoints/iter_000100000/model/.metadata",
    "cosmos_motion_ft_runs/cosmos3_camera/camera_world/"
    "native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_qfilterv1/"
    "checkpoints/iter_000035000/model/.metadata",
    "cosmos_motion_ft_runs/cosmos3_camera/camera_world/"
    "native_phase1_camera_json_bs4_lora5e5_action4x_ema_100k_qfilterv1_noi2v/"
    "checkpoints/iter_000035000/model/.metadata",
    "cosmos_motion_ft_runs/cosmos3_camera/camera_world/"
    "native_phase1_vq_A_p1_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/"
    "checkpoints/iter_000100000/model/.metadata",
    "cosmos_motion_ft_runs/cosmos3_camera/camera_world/"
    "native_phase1_vq_B_varprefix_global_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/"
    "checkpoints/iter_000100000/model/.metadata",
    "cosmos_motion_ft_runs/cosmos3_camera/camera_world/"
    "native_phase1_vq_C_varprefix_action_only_aw2_bs4_lr5e5_ema100k_qfilterv1_person/"
    "checkpoints/iter_000065000/model/.metadata",
    "cosmos_motion_ft_runs/cosmos3_camera/camera_world/"
    "native_phase1_vq_D_varprefix_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/"
    "checkpoints/iter_000055000/model/.metadata",
    "cosmos_motion_ft_runs/cosmos3_camera/camera_world/"
    "native_phase1_vq_E_p1_camera_kv_lora_aw2_bs4_lr5e5_ema100k_qfilterv1_person/"
    "checkpoints/iter_000005000/model/.metadata",
    "cosmos_motion_ft_runs/ja_t2m_ti2m_reasonerimg_x0_T200_mrope3d/"
    "ckpt_step130000.pt",
    "cosmos_motion_ft_runs/"
    "ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d/"
    "ckpt_step200000.pt",
    "cosmos_motion_ft_runs/"
    "ja_t2m_ti2m_reasonerimg_x0_native_shift3_T200_ti97_mrope3d_"
    "w1_1_5_contact_c0p05_v1_h10_s2_unipc35/ckpt_step200000.pt",
    "cosmos_motion_ft_runs/"
    "ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k/"
    "ckpt_step200000.pt",
    "cosmos_motion_ft_runs/"
    "ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_headcam/"
    "ckpt_step115000.pt",
    "cosmos_motion_ft_runs/"
    "ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2native200k_multitask/"
    "ckpt_step065000.pt",
    "cosmos_motion_ft_runs/"
    "ja_phase3_bridge_v2m_m2v_native_p1ema100k_p2contact200k/"
    "ckpt_step035000.pt",
)

LOCAL_REQUIRED = (
    WEKA_ROOT / "cosmos3_nano_dcp/model/.metadata",
    WEKA_ROOT / "wan22_vae/Wan2.2_VAE.pth",
    HF_HOME / f"hub/models--nvidia--Cosmos3-Nano/snapshots/{NANO_REV}/vision_encoder/model.safetensors",
    WEKA_ROOT / "model_cache/cdfvd/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
    WEKA_ROOT / "shape_aware_motion_eval_c45_20260715/SHA256SUMS",
    RUN_ROOT / "joint_attention/full71_windows.json",
    RUN_ROOT / "joint_attention/bones_index_train.json",
)

EXPECTED_TREE_COUNTS = {
    "nymeriaplus_proportional/camera": 729,
    "nymeriaplus_proportional/camera_rgb": 735,
    "nymeriaplus_proportional/uniego_rep": 733,
    "nymeriaplus_proportional/video": 1480,
    "nymeriaplus_proportional/joint_latents_T97": 127956,
}


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def gcs_cat(relative: str) -> bytes:
    result = subprocess.run(
        ["gcloud", "storage", "cat", f"{GCS_ROOT}/{relative}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def gcs_exists(relative: str) -> bool:
    result = subprocess.run(
        ["gcloud", "storage", "ls", f"{GCS_ROOT}/{relative}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def gcs_count(relative: str) -> int:
    result = subprocess.run(
        ["gcloud", "storage", "ls", "--recursive", f"{GCS_ROOT}/{relative}/"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return sum(
        1
        for line in result.stdout.splitlines()
        if line.startswith("gs://") and not line.endswith(":")
    )


def verify_gcs() -> None:
    failures: list[str] = []
    for relative in GCS_REQUIRED:
        if not gcs_exists(relative):
            failures.append(f"missing GCS object: {relative}")
    for relative, expected_hash in HASHED_FILES.items():
        actual = sha256_bytes(gcs_cat(relative))
        if actual != expected_hash:
            failures.append(f"hash mismatch {relative}: {actual} != {expected_hash}")
    for relative, expected_count in EXPECTED_TREE_COUNTS.items():
        actual = gcs_count(relative)
        if actual != expected_count:
            failures.append(f"count mismatch {relative}: {actual} != {expected_count}")
        print(f"[verify-gcs] {relative}: {actual} objects", flush=True)
    if failures:
        raise SystemExit("\n".join(failures))
    print("[verify-gcs] required objects, hashes, and counts passed")


def local_path_for_hash(relative: str) -> Path:
    suffix = relative.removeprefix("nymeriaplus_proportional/")
    return WEKA_ROOT / "nymeriaplus_kimodo_proportional" / suffix


def verify_local() -> None:
    failures: list[str] = []
    for path in LOCAL_REQUIRED:
        if not path.is_file():
            failures.append(f"missing local file: {path}")
    for relative, expected_hash in HASHED_FILES.items():
        path = local_path_for_hash(relative)
        if not path.is_file():
            failures.append(f"missing local hashed file: {path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash:
            failures.append(f"hash mismatch {path}: {actual} != {expected_hash}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("[verify-local] required files and hashes passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("gcs", "local"))
    args = parser.parse_args()
    if args.target == "gcs":
        verify_gcs()
    else:
        verify_local()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"migration verification command failed: {error}", file=sys.stderr)
        raise
