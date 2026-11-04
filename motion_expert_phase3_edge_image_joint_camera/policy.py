"""Keep the image-joint recipe and add only the camera-token intervention."""
from motion_expert_phase3_edge_image_joint.policy import configure as base_configure

CAMERA_POLICY = {
    "input": "noise_corrupted_ground_truth",
    "representation": "native_backward_framewise_relative_se3_raw9_padded64",
    "tokens_per_sample": 96,
    "domain_id": 2,
    "clean_camera_tokens": 0,
    "clock": "motion_sigma_for_video2motion_else_video_sigma",
    "noise": "independent_per_global_draw_generator_preserves_baseline_RNG",
    "attention": "native_generator_video_action_attention",
    "bridge_scope": "unchanged_video_motion_bridges_no_direct_camera_motion_edges",
    "loss_weight": 0.0,
    "camera_decoder_gradients": False,
    "trainable_parameters": "same_seven_bridges_only",
}


def configure():
    recipe = base_configure()
    recipe.update(
        experiment="image_joint_402020_bidirectional_camera_prediction_no_camera_loss_v1",
        camera_prediction=CAMERA_POLICY.copy(),
    )
    return recipe
