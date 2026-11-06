from motion_expert_phase3_edge_image_joint.policy import configure as base_configure
from motion_expert_phase3_edge_joint.settings import RECIPE

EXPERIMENT = "image_joint_mmdit_bidirectional_wv1_notext_v1"


def configure():
    base_configure("bidirectional")
    RECIPE.update(experiment=EXPERIMENT, video_weight=1.0)
    return RECIPE


def check_checkpoint(payload):
    saved = payload.get("scheduler", {}).get("recipe", {})
    expected = configure().copy()
    for key in ("experiment", "tasks", "joint_conditioning", "joint_noise", "joint_objective",
                "caption_policy", "cfg_dropout", "conditioning_attention", "clean_motion_timestep",
                "motion_weights", "video_weight", "video_normalize_by_active"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"Checkpoint does not match the no-caption/weight-one recipe: {key}")
