"""Phase 3 never conditions either specialist on dataset captions."""


def empty_captions(captions, modes):
    if len(captions) != len(modes):
        raise ValueError("One caption slot is required per Phase3 sample")
    return [""] * len(modes)


def conditioning_captions(captions, modes):
    """Keep newer recipes caption-free; preserve the named Yonsei M2V contract."""
    from motion_expert_phase3_edge_joint.settings import RECIPE
    if RECIPE.get("experiment") == "image_joint_mmdit_bidirectional_wv1_v1":
        from motion_expert_phase3_edge_image_joint_wv1.policy import captions_for_modes
        return captions_for_modes(captions, modes)
    return empty_captions(captions, modes)
