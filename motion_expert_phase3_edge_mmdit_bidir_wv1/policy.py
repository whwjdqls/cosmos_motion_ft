"""Fresh caption-free, camera-free, bidirectional two-task weight-one recipe."""
from motion_expert_phase3_edge_mmdit.policy import configure as base_configure
from motion_expert_phase3_edge_joint.settings import configure_conditioning

TASKS = {"video2motion": .5, "motimg2video": .5}


def attention_policy():
    return "bidirectional"


def configure():
    recipe = base_configure()
    recipe.update(experiment="mmdit_bidirectional_two_task5050_temporal_local_bf16_no_text_wv1_v1",
                  tasks=TASKS.copy(), video_weight=1.0)
    configure_conditioning("bidirectional", "zero")
    recipe.pop("camera_prediction", None)
    return recipe


def configure_checkpoint(payload):
    saved = payload["scheduler"]["recipe"]
    expected = configure().copy()
    for key in ["experiment", "tasks", "video_weight", "caption_policy", "cfg_dropout",
                "conditioning_attention", "clean_motion_timestep", "joint_objective",
                "joint_conditioning", "joint_noise", "motion_weights"]:
        if saved.get(key) != expected[key]:
            raise ValueError("Checkpoint recipe differs: " + key)
    if saved.get("camera_prediction") is not None:
        raise ValueError("This variant has no camera prediction")
    from motion_expert_phase3_edge_joint.adapters import configure_checkpoint as configure_deltas
    configure_deltas(payload)
