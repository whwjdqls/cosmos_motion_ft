"""Native camera preflight and full-state training with cross-only bridge keys."""
import argparse
import inspect
from pathlib import Path
import sys
import torch
from motion_expert_phase3_edge_camera_directional_nojoint import train as original
from motion_expert_phase3_edge_joint.settings import RECIPE, sha256
from .policy import configure
from .data import Phase3Dataset
from .attention import cross_only_attention


def adapt_preflight_source(source):
    old = '''        assert torch.equal(mask[:1600, :1600], meta.gen_frame[:, None] == meta.gen_frame[None, :])
        motion_groups = torch.div(meta.motion_frame[1:] + 3, 4, rounding_mode="floor")
        assert torch.equal(mask[1601:, 1601:], motion_groups[:, None] == motion_groups[None, :])'''
    new = '''        assert not mask[:1600, :1600].any()
        assert not mask[1600:, 1600:].any()
        from motion_expert_phase3_edge_camera_cross_only.attention import cross_attention_mask
        assert torch.equal(mask, cross_attention_mask(meta, bridge.conditioning_attention))'''
    if source.count(old) != 1:
        raise RuntimeError('Review native topology acceptance before adapting cross-only')
    return source.replace(old, new).replace('causal_Wan_intervals_and_', 'cross_only_causal_Wan_intervals_and_')


def prepare_trainer(probe_updates=0):
    namespace = original.__dict__.copy()
    namespace.update(configure=configure, Phase3Dataset=Phase3Dataset, adapt_preflight_source=adapt_preflight_source)
    source = inspect.getsource(original.prepare_trainer)
    source = source.replace('from .validation import install',
                            'from motion_expert_phase3_edge_camera_cross_only.validation import install')
    exec(compile(source, __file__ + ':native_camera_adapter', 'exec'), namespace)
    train = namespace['prepare_trainer'](probe_updates)
    original_identity = train.source_identity
    def source_identity(args, dataset, model):
        contract = original_identity(args, dataset, model)
        contract['camera_cross_only_implementation'] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        contract['runtime_overlays']['bridge_attention_type'] = 'cross_only'
        return contract
    train.source_identity = source_identity
    original_acceptance = train.camera_require_acceptance
    def require_acceptance(receipt, fingerprint, world, microbatch):
        original_acceptance(receipt, fingerprint, world, microbatch)
        if 'cross_only_causal_Wan_intervals_and_directional_conditioning_attention' not in receipt['checks']:
            raise RuntimeError('Missing actual cross-only native/DDP acceptance')
    train.camera_require_acceptance = require_acceptance
    return train


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-updates', type=int, default=0)
    options, rest = parser.parse_known_args()
    if options.probe_updates < 0:
        raise ValueError('Invalid probe count')
    configure()
    if '--conditioning-attention' not in rest:
        rest += ['--conditioning-attention', 'directional']
    elif rest[rest.index('--conditioning-attention') + 1] != 'directional':
        raise ValueError('This ablation uses directional attention')
    if '--clean-motion-timestep' not in rest:
        rest += ['--clean-motion-timestep', 'zero']
    train = prepare_trainer(options.probe_updates)
    sys.argv = [sys.argv[0], *rest]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from scripts.phase3_performance import execution_policy
    with cross_only_attention(), execution_policy(cache_bridge_masks=True, checkpoint_motion=True):
        train.main()


if __name__ == '__main__':
    main()
