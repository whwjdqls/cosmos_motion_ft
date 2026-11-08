import os
from motion_expert_phase3_edge_image_joint_camera.policy import configure as camera_configure

TASKS = {'video2motion': .5, 'motimg2video': .5, 'joint': 0.0}
SCOPES = ('temporal_local', 'global')

def configure(scope=None):
    scope = scope or os.environ.get('PHASE3_CAMERA_ATTENTION_SCOPE', 'temporal_local')
    if scope not in SCOPES:
        raise ValueError('Unknown bridge attention scope')
    os.environ['PHASE3_CAMERA_ATTENTION_SCOPE'] = scope
    os.environ['PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION'] = 'directional'
    recipe = camera_configure()
    recipe.update(experiment='no_text_directional_camera_nocamloss_nojoint5050_wv1_'+scope+'_v1',
                  tasks=TASKS.copy(), video_weight=1.0, bridge_attention_scope=scope)
    return recipe

def configure_checkpoint(payload):
    from motion_expert_phase3_edge_joint.adapters import configure_checkpoint as configure_deltas
    saved = payload['scheduler']['recipe']
    configure(saved.get('bridge_attention_scope'))
    expected = configure().copy()
    for key in ['experiment','tasks','video_weight','camera_prediction','caption_policy',
                'conditioning_attention','clean_motion_timestep','bridge_attention_scope']:
        if saved.get(key) != expected[key]:
            raise ValueError('Checkpoint recipe differs: '+key)
    configure_deltas(payload)
