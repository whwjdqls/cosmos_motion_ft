"""Keep the camera trainer and its acceptance gates; correct loss and logging."""
import argparse
import inspect
import sys
from pathlib import Path
import torch
from motion_expert_phase3_edge_image_joint_camera import train as camera_train
from motion_expert_phase3_edge_joint.settings import RECIPE, sha256
from .policy import configure
from .objectives import specialist_loss

def prepare_trainer(probe_updates=0):
    source = inspect.getsource(camera_train.prepare_trainer)
    marker = '    source = inspect.getsource(train.main)'
    old = '"video_weighted": 10 * values[2] / max(values[4], 1),'
    new = '"video_weighted": RECIPE["video_weight"] * values[2] / max(values[4], 1),'
    if source.count(marker) != 1:
        raise RuntimeError('Upstream camera trainer changed; review logging adapter')
    source = source.replace(marker, marker + f'\n    old_logging = {old!r}\n    assert source.count(old_logging) == 1\n    source = source.replace(old_logging, {new!r})')
    marker = '    source = inspect.getsource(train.preflight)'
    if source.count(marker) != 1:
        raise RuntimeError('Upstream camera preflight changed')
    source = source.replace(marker, marker + '\n    source = source.replace("native_video_weight10_full_denominator", "native_video_weight1_full_denominator")')
    namespace = camera_train.__dict__.copy()
    namespace.update(configure=configure, specialist_loss=specialist_loss)
    exec(compile(source, __file__ + ':camera_adapter', 'exec'), namespace)
    train = namespace['prepare_trainer'](probe_updates)
    original_identity = train.source_identity

    def source_identity(args, dataset, model):
        contract = original_identity(args, dataset, model)
        contract['camera_video_weight_implementation'] = {
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        contract['runtime_overlays']['video_weight'] = {
            'loss': 'native_formula_video_weight1_motion_weights_unchanged',
            'logging': 'video_weighted_equals_recipe_weight_times_video_loss'}
        return contract

    train.source_identity = source_identity
    return train

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-updates', type=int, default=0)
    args, rest = parser.parse_known_args()
    if args.probe_updates < 0:
        raise ValueError('Probe updates must be nonnegative')
    if '--conditioning-attention' not in rest:
        rest += ['--conditioning-attention', 'bidirectional']
    if '--clean-motion-timestep' not in rest:
        rest += ['--clean-motion-timestep', 'zero']
    train = prepare_trainer(args.probe_updates)
    assert RECIPE['video_weight'] == 1.0
    sys.argv = [sys.argv[0], *rest]
    from scripts.phase3_performance import execution_policy
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with execution_policy(cache_bridge_masks=True, checkpoint_motion=True):
        train.main()

if __name__ == '__main__':
    main()
