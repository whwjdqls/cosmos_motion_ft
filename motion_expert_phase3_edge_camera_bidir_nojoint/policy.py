import os
from motion_expert_phase3_edge_camera_directional_nojoint.policy import configure as directional_configure

def configure(scope=None):
    recipe=directional_configure(scope)
    os.environ['PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION']='bidirectional'
    recipe['conditioning_attention']='bidirectional'
    recipe['experiment']=recipe['experiment'].replace('directional','bidirectional')
    return recipe

def configure_checkpoint(payload):
    from motion_expert_phase3_edge_joint.adapters import configure_checkpoint as configure_deltas
    saved=payload['scheduler']['recipe'];expected=configure(saved.get('bridge_attention_scope')).copy()
    for key in ['experiment','tasks','video_weight','camera_prediction','caption_policy',
                'conditioning_attention','clean_motion_timestep','bridge_attention_scope']:
        if saved.get(key)!=expected[key]:raise ValueError('Checkpoint recipe differs: '+key)
    configure_deltas(payload)
