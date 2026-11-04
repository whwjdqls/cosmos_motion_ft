"""Change only the experiment identity and native video loss multiplier."""
from motion_expert_phase3_edge_image_joint_camera.policy import configure as camera_configure

EXPERIMENT = 'image_joint_402020_bidirectional_camera_prediction_no_camera_loss_video_weight1_v1'

def configure():
    recipe = camera_configure()
    recipe.update(experiment=EXPERIMENT, video_weight=1.0)
    return recipe

def check_checkpoint(payload):
    expected = configure()
    saved = payload.get('scheduler', {}).get('recipe', {})
    if saved != expected:
        raise ValueError('Checkpoint does not match the video-weight-one camera recipe')
