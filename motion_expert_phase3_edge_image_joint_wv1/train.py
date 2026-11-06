"""Reuse native joint acceptance and training; pin the weight-one contract."""
import argparse
from pathlib import Path
import sys
import torch
from motion_expert_phase3_edge_image_joint.train import prepare_trainer as base_prepare
from motion_expert_phase3_edge_joint.settings import sha256
from .data import Phase3Dataset
from .policy import configure


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-updates', type=int, default=0)
    args, rest = parser.parse_known_args()
    if args.probe_updates < 0:
        raise ValueError('Probe updates must be nonnegative')
    train = base_prepare(args.probe_updates)
    configure()
    train.Phase3Dataset = Phase3Dataset
    original = train.source_identity

    def source_identity(args, dataset, model):
        identity = original(args, dataset, model)
        identity['joint_weight_one_implementation'] = {
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        return identity

    train.source_identity = source_identity
    for flag, value in (('conditioning-attention', 'bidirectional'), ('clean-motion-timestep', 'zero')):
        if '--' + flag not in rest:
            rest += ['--' + flag, value]
    sys.argv = [sys.argv[0], *rest]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train.main()


if __name__ == '__main__':
    main()
