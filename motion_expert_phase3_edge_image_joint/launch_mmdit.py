"""Fresh MMDiT bridge run with grouped BF16 projections and fixed71 monitoring."""
import sys


def main(experiment_package="motion_expert_phase3_edge_image_joint"):
    from .launch import main as launch
    names = {arg.split("=", 1)[0] for arg in sys.argv[1:]}
    if "--help" not in names and "-h" not in names and "--validation-config" not in names:
        raise ValueError("Supply the checksum-pinned --validation-config for periodic V2M/M2V metrics")
    for name, value in (("bridge-parameterization", "mmdit"),
                        ("bridge-attention-backend", "grouped"),
                        ("bridge-precision", "bf16_bridge"),
                        ("validation-every", "10000"),
                        ("validation-batch-size", "32"),
                        ("validation-decode-batch-size", "4")):
        if "--" + name not in names:
            sys.argv += ["--" + name, value]
    launch(experiment_package=experiment_package)


if __name__ == "__main__":
    main()
