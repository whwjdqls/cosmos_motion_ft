"""Guarded two-GPU launch using the existing native acceptance/reload probes."""
import argparse
import os
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conditioning-attention", choices=["bidirectional"], default="bidirectional")
    _, rest = parser.parse_known_args()
    os.environ["PHASE3_MMDIT_CONDITIONING_ATTENTION"] = "bidirectional"
    os.environ["PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION"] = "bidirectional"
    from motion_expert_phase3_edge_image_joint import launch
    original_freeze = launch.freeze_sources

    def freeze(repo, snapshot):
        manifest = original_freeze(repo, snapshot)
        from motion_expert_phase3_edge_joint.settings import sha256
        for source in sorted(Path(__file__).parent.glob("*.py")):
            relative = source.relative_to(repo)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest[str(relative)] = sha256(source)
        return manifest

    launch.freeze_sources = freeze
    sys.argv = [sys.argv[0], *rest, "--conditioning-attention", "bidirectional"]
    from motion_expert_phase3_edge_image_joint.launch_mmdit import main as mmdit_main
    mmdit_main(experiment_package="motion_expert_phase3_edge_mmdit_bidir_wv1")


if __name__ == "__main__":
    main()
