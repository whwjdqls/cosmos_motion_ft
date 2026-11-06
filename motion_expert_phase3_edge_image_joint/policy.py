"""Explicit experiment policy, separate from the running two-task baselines."""
from __future__ import annotations

import os

from motion_expert_phase3_edge_joint.settings import RECIPE, configure_conditioning

TASKS = {"video2motion": 0.4, "motimg2video": 0.4, "joint": 0.2}
EXPERIMENT = "no_text_joint_native_losses_402020_bidirectional_clean_t0_v2"
JUICE_COMMIT = "1bc0e5f34a2b47d9a4cbb5bc76821091b5994984"
PHASE1_METADATA_SHA256 = "b2b3374b8aa1f688dc73fabb78832889c9d3e4264299f20aa8aefbdcde4ecca8"


def configure(conditioning_attention=None):
    attention = conditioning_attention or os.environ.get("PHASE3_IMAGE_JOINT_CONDITIONING_ATTENTION", "bidirectional")
    if attention not in {"directional", "bidirectional"}:
        raise ValueError("Unknown image-joint conditioning attention")
    RECIPE.update(
        experiment=EXPERIMENT if attention == "bidirectional" else EXPERIMENT.replace("bidirectional", "directional"),
        required_phase1_metadata_sha256=PHASE1_METADATA_SHA256,
        tasks=TASKS.copy(),
        joint_conditioning="first_image_only_plus_supplied_neutral_shape_no_caption",
        joint_noise="independent_specialist_schedules",
        joint_objective="sum_native_video_velocity_and_motion_x0_physical_losses",
        caption_policy={mode: "empty" for mode in TASKS},
        cfg_dropout=0.0,
        video_weight=10.0,
        joint_guidance="none_no_text_condition",
        juice_reference={"repository": "https://github.com/Ced3-han/JUICE-Code", "commit": JUICE_COMMIT,
                         "adapted": "independent_modality_times_and_both_targets",
                         "loss_choice": "native_specialist_losses_not_JUICE_velocity_space_losses"},
    )
    configure_conditioning(attention, "zero")
    return RECIPE


def captions_for_modes(captions, modes):
    if len(captions) != len(modes) or set(modes) - set(TASKS):
        raise ValueError("Invalid image-joint task/caption batch")
    return [""] * len(modes)


def check_checkpoint(payload):
    saved = payload.get("scheduler", {}).get("recipe", {})
    expected_attention = "directional" if saved.get("experiment") == EXPERIMENT.replace("bidirectional", "directional") else "bidirectional"
    expected = configure(expected_attention).copy()
    for key in ("experiment", "tasks", "joint_conditioning", "joint_noise", "joint_objective",
                "caption_policy", "conditioning_attention", "clean_motion_timestep",
                "motion_weights", "video_weight", "video_normalize_by_active"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"Checkpoint does not match the image-only joint policy: {key}")
