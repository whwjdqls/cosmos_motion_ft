"""Keep native camera denoising while installing the saved bridge topology."""
from .attention import cross_only_attention
from motion_expert_phase3_edge_camera_directional_nojoint.sample import sample as native_sample


def sample(*args, **kwargs):
    with cross_only_attention():
        return native_sample(*args, **kwargs)
