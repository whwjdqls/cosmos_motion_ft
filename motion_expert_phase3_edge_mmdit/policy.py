"""Requested two-task directional ablation; retain native specialist losses."""
from motion_expert_phase3_edge_image_joint.policy import configure as original_configure
from motion_expert_phase3_edge_joint.settings import configure_conditioning
import os

TASKS = {"video2motion": .5, "motimg2video": .5}
def attention_policy():
    attention = os.environ.get("PHASE3_MMDIT_CONDITIONING_ATTENTION", "directional")
    if attention not in {"directional", "bidirectional"}:
        raise ValueError("Unknown two-task MMDiT attention policy")
    return attention


def configure():
    recipe = original_configure()
    attention = attention_policy()
    recipe.update(experiment=f"mmdit_{attention}_two_task5050_temporal_local_bf16_v1", tasks=TASKS.copy(),
                  joint_objective="disabled_no_joint_samples",
                  joint_conditioning="disabled", joint_noise="disabled",
                  caption_policy={"video2motion": "empty", "motimg2video": "caption_dropout_0.1"})
    recipe.pop("camera_prediction", None)
    recipe.pop("juice_reference", None)
    configure_conditioning(attention, "zero")
    return recipe
