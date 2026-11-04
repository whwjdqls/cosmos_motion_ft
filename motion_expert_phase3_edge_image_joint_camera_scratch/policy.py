"""Retain the camera ablation recipe; remove learned Phase2 motion weights."""
from motion_expert_phase3_edge_image_joint_camera.policy import configure as camera_configure

INITIALIZATION = {
    'motion': 'fresh_original_phase2_constructor_no_learned_phase2_weights',
    'motion_output': 'original_zero_initialized_projection_trainable',
    'trainable': 'motion_expert_and_seven_bridges',
    'frozen': 'same_phase1_cosmos_reasoner_and_generator',
    'seed': 0,
    'phase2_metadata': 'architecture_and_normalization_contract_only_no_tensors',
}


def configure():
    recipe = camera_configure()
    recipe['experiment'] = 'image_joint_camera_prediction_scratch_motion_trainable_v1'
    recipe['motion_initialization'] = INITIALIZATION.copy()
    recipe['camera_prediction']['trainable_parameters'] = 'scratch_motion_expert_and_same_seven_bridges_only'
    return recipe
