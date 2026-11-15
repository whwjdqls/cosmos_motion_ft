from motion_expert_phase3_edge_camera_directional_nojoint.policy import configure as base_configure


def configure(scope=None):
    if scope not in (None, 'temporal_local'):
        raise ValueError('This ablation fixes temporal locality')
    recipe = base_configure('temporal_local')
    recipe.update(experiment='cross_only_directional_camera_nocamloss_5050_wv1_local_v1',
                  bridge_attention_type='cross_only')
    return recipe


def configure_checkpoint(payload):
    from motion_expert_phase3_edge_joint.adapters import configure_checkpoint as configure_deltas
    saved = payload['scheduler']['recipe']
    expected = configure(saved.get('bridge_attention_scope')).copy()
    for key in ['experiment', 'tasks', 'video_weight', 'camera_prediction', 'caption_policy',
                'conditioning_attention', 'clean_motion_timestep', 'bridge_attention_scope', 'bridge_attention_type']:
        if saved.get(key) != expected[key]:
            raise ValueError('Cross-only checkpoint recipe differs: ' + key)
    configure_deltas(payload)
