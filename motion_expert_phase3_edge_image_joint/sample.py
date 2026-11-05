"""Image-only coupled inference using distinct native motion/video clocks."""
from __future__ import annotations

from .policy import check_checkpoint, configure
from motion_expert_phase3_edge_joint.sample import sample as conditional_sample
from motion_expert_phase3_edge_joint.sample import sample_joint as coupled_sample


def sample_joint(model, *, caption="", guidance=1.0, **kwargs):
    if guidance != 1.0:
        raise ValueError("Image-only joint sampling has no caption guidance; use guidance=1")
    return coupled_sample(model, caption="", guidance=1.0, **kwargs)


def sample(model, *, mode, caption="", guidance=1.0, **kwargs):
    if mode == "joint":
        return sample_joint(model, caption=caption, guidance=guidance, **kwargs)
    return conditional_sample(model, mode=mode, caption=caption, guidance=guidance, **kwargs)


def main():
    import argparse
    from pathlib import Path
    import torch
    from .model import EdgePhase3
    from motion_expert_phase3_edge_joint import sample as base

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args, _ = parser.parse_known_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    check_checkpoint(payload)
    del payload
    base.EdgePhase3, base.sample = EdgePhase3, sample
    base.main()


if __name__ == "__main__":
    main()
