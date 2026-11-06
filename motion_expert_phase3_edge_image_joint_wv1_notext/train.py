"""Use the native caption-free acceptance gates and pin video weight one."""
import argparse
from pathlib import Path
import sys

import torch

from motion_expert_phase3_edge_image_joint.train import prepare_trainer
from motion_expert_phase3_edge_joint.settings import sha256
from .data import Phase3Dataset
from .policy import configure


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-updates", type=int, default=0)
    parser.add_argument("--conditioning-attention", choices=("bidirectional",), default="bidirectional")
    parser.add_argument("--clean-motion-timestep", choices=("zero",), default="zero")
    args, rest = parser.parse_known_args()
    if args.probe_updates < 0:
        raise ValueError("Probe updates must be nonnegative")
    import os
    os.environ["PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION"] = "bidirectional"
    train = prepare_trainer(args.probe_updates)
    configure()
    train.Phase3Dataset = Phase3Dataset
    original_identity = train.source_identity

    def source_identity(args, dataset, model):
        identity = original_identity(args, dataset, model)
        identity["joint_weight_one_notext_implementation"] = {
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        return identity

    train.source_identity = source_identity
    sys.argv = [sys.argv[0], *rest, "--conditioning-attention", "bidirectional",
                "--clean-motion-timestep", "zero"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train.main()


if __name__ == "__main__":
    main()
