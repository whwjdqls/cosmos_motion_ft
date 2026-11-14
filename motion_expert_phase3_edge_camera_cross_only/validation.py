from motion_expert_phase3_edge_joint.settings import RECIPE
from motion_expert_phase3_edge_camera_directional_nojoint.validation import configure_validation as base_configure


def configure_validation(args):
    base_configure(args)
    if 'validation' in RECIPE:
        RECIPE['validation']['camera_sampler'] = 'motion_expert_phase3_edge_camera_cross_only.sample'
        RECIPE['validation']['bridge_attention_type'] = 'cross_only'


def install():
    import motion_expert_phase3_edge_joint.validation as validation
    import motion_expert_phase3_edge_joint.sample as sampling
    from .sample import sample
    validation.configure_validation = configure_validation
    sampling.sample = sample
