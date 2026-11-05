"""Launch the requested directional two-task experiment on coordinated GPUs."""
from motion_expert_phase3_edge_image_joint.launch_mmdit import main

if __name__ == "__main__":
    main(experiment_package="motion_expert_phase3_edge_mmdit")
