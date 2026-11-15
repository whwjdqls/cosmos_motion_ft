"""Reuse the empty-GPU launcher and freeze all cross-only adapter sources."""
import inspect
from motion_expert_phase3_edge_camera_directional_nojoint import launch as original
from .policy import configure


def main():
    source = inspect.getsource(original.main)
    old = "original.main(experiment_package='motion_expert_phase3_edge_camera_directional_nojoint')"
    assert source.count(old) == 1
    source = source.replace(old, "original.main(experiment_package='motion_expert_phase3_edge_camera_cross_only')")
    old = "['motion_expert_phase3_edge_image_joint_camera','motion_expert_phase3_edge_camera_directional_nojoint']"
    assert source.count(old) == 1
    source = source.replace(old, "['motion_expert_phase3_edge_image_joint_camera','motion_expert_phase3_edge_camera_directional_nojoint','motion_expert_phase3_edge_camera_cross_only']")
    source = source.replace('from .validation import install',
                            'from motion_expert_phase3_edge_camera_cross_only.validation import install')
    namespace = original.__dict__.copy()
    namespace['configure'] = configure
    exec(compile(source, __file__ + ':guarded_launcher', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
