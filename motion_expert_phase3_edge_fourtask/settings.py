"""Preserve the two-task specialist recipe, varying motion conditioning only."""
from copy import deepcopy
import os
from pathlib import Path

from motion_expert_phase3_edge.settings import (
    RECIPE as BASE_RECIPE, add_paths as base_add_paths,
    atomic_json, digest_json, require_compute_node, sha256,
)

RECIPE = deepcopy(BASE_RECIPE)
RECIPE.update({
    "tasks": {"motimg2video": 0.5, "video2motion": 0.125,
              "camera2motion": 0.125, "cameratext2motion": 0.125,
              "cameravideo2motion": 0.125},
    "camera_representation": "native_camera_pose_backward_framewise_rot6d_raw9_pad64_domain2",
    "camera_conditioning": "clean_native_action_tokens_in_existing_local_bridges",
    "camera_bridge_interval": "action_i_spans_source_frames_i_and_i_plus_1",
    "camera_task_generator_prompt": "caption_only_or_empty",
    "no_video_policy": "no_vision_tokens_including_no_first_image",
    "initialization": "fresh_bridges_from_original_frozen_phase1_and_phase2",
})
RUN_NAME = "edge_p3_p1ema200k_crosscaption200k_bridge7_camcond3_mix_T97_gbs32_wv10_yonsei_20261004"


def has_camera(mode):
    return mode in {"camera2motion", "cameratext2motion", "cameravideo2motion"}


def has_video(mode):
    return mode in {"motimg2video", "video2motion", "cameravideo2motion"}


def has_text(mode):
    return mode in {"motimg2video", "cameratext2motion"}


def effective_caption(mode, caption):
    return caption if has_text(mode) else ""


def add_paths(parser):
    parser = base_add_paths(parser)
    weka = Path(os.environ.get("WEKA_ROOT", "/lustre/whwjdqls99/cosmos/weka"))
    runs = Path(os.environ.get("RUN_ROOT", weka / "cosmos_motion_ft_runs"))
    root = Path("/scratch2/whwjdqls99/cosmos") if os.environ.get("COSMOS_SITE") == "yonsei" else runs
    output_root = root / "cosmos_motion_ft_runs" if os.environ.get("COSMOS_SITE") == "yonsei" else runs
    parser.set_defaults(out=output_root / "motion_expert_phase3_edge" / RUN_NAME)
    parser.add_argument("--camera-root", type=Path, default=weka / "nymeriaplus_kimodo_proportional/camera_rgb")
    parser.add_argument("--camera-cache", type=Path, default=root / "caches/phase3_edge_camera_conditioning_T97_v1")
    return parser

# Isolated four-task experiment. Never mutate the original camera-mix recipe.
RECIPE = {
    **RECIPE,
    "tasks": {
        "motimg2video": 0.25,
        "video2motion": 0.25,
        "camera2motion": 0.25,
        "cameravideo2motion": 0.25,
    },
    "cfg_dropout": 0.0,
    "experiment": "fourtask_equal_no_caption_v1",
    "caption_policy": "empty_in_all_tasks_and_both_specialist_paths",
}


def has_text(mode):
    return False


def effective_caption(mode, caption):
    return ""


_original_add_paths = add_paths


def add_paths(parser):
    parser = _original_add_paths(parser)
    original_out = Path(parser.get_default("out"))
    parser.set_defaults(out=original_out.parent / "edge_p3_fourtask_equal_no_caption_T97_gbs32")
    return parser
