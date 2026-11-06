"""Native, independently noised specialist objectives; no joint caption."""
from motion_expert_phase3_edge_joint.objectives import noise_batch as original_noise_batch
from motion_expert_phase3_edge_joint.objectives import specialist_loss
from .policy import captions_for_modes


def noise_batch(batch, device):
    return original_noise_batch({**batch, "captions": captions_for_modes(batch["captions"], batch["modes"])}, device)
