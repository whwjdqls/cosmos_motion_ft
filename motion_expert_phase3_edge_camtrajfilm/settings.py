"""Keep the four-task recipe; change only the explicitly recorded FiLM design."""
from copy import deepcopy
import os
from pathlib import Path

from motion_expert_phase3_edge_camfilm.settings import (
    RECIPE as REFERENCE_RECIPE, add_paths as reference_paths,
    atomic_json, digest_json, require_compute_node, sha256,
    has_camera, has_video, has_text, effective_caption,
)

ARCHITECTURE = "fourtask_balanced_flooraware18d_motionnoise_camera_film_v1"
RECIPE = deepcopy(REFERENCE_RECIPE)
RECIPE["experiment"] = ARCHITECTURE
RECIPE["tasks"] = {
    "motimg2video": 0.5,
    "video2motion": 1.0 / 6.0,
    "camera2motion": 1.0 / 6.0,
    "cameravideo2motion": 1.0 - 0.5 - 1.0 / 6.0 - 1.0 / 6.0,
}
RECIPE["camera_film"] = {
    "version": 1,
    "placement": "after_each_of_seven_existing_attention_bridges",
    "tasks": ["camera2motion", "cameravideo2motion"],
    "camera_features": "flooraware_uniegomotion_18d_direct_trajectory",
    "camera_feature_dim": 18,
    "camera_standardization": "exact_phase3_training_cohort_only_per_feature_mean_std",
    "floor": "same_total_calibrated_offset_as_motion_targets",
    "first_frame_delta": "identity_rotation_zero_translation_height_preserved",
    "alignment": "97_camera_rows_to_97_motion_frames_one_to_one",
    "motion_context": "current_post_bridge_motion_hidden_state",
    "noise_context": "frozen_motion_time_embedder_of_normalized_motion_sigma",
    "context_dim": 256, "noise_context_dim": 256, "mlp_dim": 256,
    "normalization": "non_affine_layernorm_fp32",
    "formula": "h + gamma(concat(LN(h),camera_context,noise_context)) * LN(h) + beta(context)",
    "final_linear_initialization": "zero_weight_and_bias_no_extra_gate",
    "shape_token": "unchanged", "existing_camera_attention": "retained",
    "ablation_caveat": "changes_source_information_and_explicit_noise_path_not_parameter_matched",
}


def add_paths(parser):
    parser = reference_paths(parser)
    parser.add_argument("--trajectory-cache", type=Path,
        default=Path(os.environ.get("EDGE_PHASE3_TRAJECTORY_CACHE",
            str(Path(parser.get_default("camera_cache")).parent /
                "phase3_edge_camera_trajectory18_T97_camfloor_v1_20261023"))))
    parser.add_argument("--reference-run", type=Path)
    return parser
