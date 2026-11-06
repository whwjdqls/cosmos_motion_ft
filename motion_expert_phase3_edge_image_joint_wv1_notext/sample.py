def main():
    import argparse
    from pathlib import Path
    import torch
    from .policy import check_checkpoint
    from motion_expert_phase3_edge_image_joint.model import EdgePhase3
    from motion_expert_phase3_edge_image_joint.sample import sample
    from motion_expert_phase3_edge_joint import sample as base

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args, _ = parser.parse_known_args()
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    check_checkpoint(payload)
    del payload
    base.EdgePhase3, base.sample = EdgePhase3, sample
    base.main()


if __name__ == "__main__":
    main()
