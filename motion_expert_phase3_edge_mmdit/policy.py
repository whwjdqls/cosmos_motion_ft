"""Requested two-task directional ablation; retain native specialist losses."""
from motion_expert_phase3_edge_image_joint.policy import configure as original_configure
from motion_expert_phase3_edge_joint.settings import configure_conditioning

TASKS = {"video2motion": .5, "motimg2video": .5}
EXPERIMENT = "mmdit_directional_two_task5050_temporal_local_bf16_v1"


def configure():
    recipe = original_configure()
    recipe.update(experiment=EXPERIMENT, tasks=TASKS.copy(),
                  joint_objective="disabled_no_joint_samples",
                  joint_conditioning="disabled", joint_noise="disabled",
                  caption_policy={"video2motion": "empty", "motimg2video": "caption_dropout_0.1"})
    recipe.pop("camera_prediction", None)
    recipe.pop("juice_reference", None)
    configure_conditioning("directional", "zero")
    return recipe
