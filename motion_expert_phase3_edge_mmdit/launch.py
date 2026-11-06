"""Launch the requested directional two-task experiment on coordinated GPUs."""
from motion_expert_phase3_edge_image_joint.launch_mmdit import main
import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conditioning-attention", choices=("directional", "bidirectional"), default="directional")
    args, rest = parser.parse_known_args()
    os.environ["PHASE3_MMDIT_CONDITIONING_ATTENTION"] = args.conditioning_attention
    sys.argv = [sys.argv[0], *rest]
    main(experiment_package="motion_expert_phase3_edge_mmdit")
