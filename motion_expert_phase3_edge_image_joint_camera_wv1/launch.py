"""Launch the controlled weight-one variant using the camera resource gates."""
import inspect
from motion_expert_phase3_edge_image_joint_camera import launch as camera_launch

PACKAGE = 'motion_expert_phase3_edge_image_joint_camera_wv1'

def main():
    source = inspect.getsource(camera_launch.main)
    changes = [
        ('from .policy import configure', f'from {PACKAGE}.policy import configure'),
        ("if key != 'experiment' and recipe.get(key) != value", "if key not in {'experiment', 'video_weight'} and recipe.get(key) != value"),
        ("    if mismatches:\n", "    if reference['recipe']['video_weight'] != 10.0 or recipe['video_weight'] != 1.0:\n        raise ValueError('This ablation must compare video weight10 to weight1')\n    if mismatches:\n"),
        ("('image_joint_implementation','motion_expert_phase3_edge_image_joint')]", "('image_joint_implementation','motion_expert_phase3_edge_image_joint'),('camera_prediction_implementation','motion_expert_phase3_edge_image_joint_camera')]"),
        ("        sources += sorted(Path(__file__).parent.glob(\"*.py\"))", "        sources += sorted(Path(__file__).parent.glob(\"*.py\"))\n        sources += sorted((repo/'" + PACKAGE + "').glob('*.py'))"),
        ('"motion_expert_phase3_edge_image_joint_camera.train"', f'"{PACKAGE}.train"'),
        ('"edge-p3-imagejoint402020-camera-prediction-no-camera-loss"', '"edge-p3-imagejoint402020-camera-prediction-video-weight1"'),
        ('"native_camera_prediction_no_camera_loss"', '"native_camera_prediction_video_weight1"'),
    ]
    for before, after in changes:
        if source.count(before) != 1:
            raise RuntimeError('Upstream camera launcher changed; review adapter: ' + before)
        source = source.replace(before, after)
    namespace = camera_launch.__dict__.copy()
    exec(compile(source, __file__ + ':camera_launcher', 'exec'), namespace)
    namespace['main']()

if __name__ == '__main__':
    main()
