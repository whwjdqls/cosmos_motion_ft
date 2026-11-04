"""Keep the weight-one camera recipe, disabling joint sampling explicitly."""
from motion_expert_phase3_edge_image_joint_camera_wv1.policy import configure as weight_one_configure

EXPERIMENT = 'bidirectional_camera_prediction_no_camera_loss_two_task5050_video_weight1_v1'
TASKS = {'video2motion': .5, 'motimg2video': .5, 'joint': 0.0}

def configure():
    recipe = weight_one_configure()
    recipe.update(experiment=EXPERIMENT, tasks=TASKS.copy())
    return recipe

def check_checkpoint(payload):
    if payload.get('scheduler', {}).get('recipe', {}) != configure():
        raise ValueError('Checkpoint does not match the bidirectional two-task camera recipe')
