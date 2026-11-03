"""Enforce image-only joint conditioning before either specialist is packed."""
from motion_expert_phase3_edge_joint.model import EdgePhase3 as OriginalModel
from .policy import PHASE1_METADATA_SHA256, captions_for_modes


class EdgePhase3(OriginalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.provenance["phase1_metadata_sha256"] != PHASE1_METADATA_SHA256:
            raise ValueError("Use the same selected Phase1 video-projection EMA200k specialist as the 50/50 runs")
        if len(self.provenance["phase1_video_projection_tensors"]) != 4:
            raise ValueError("All four trained video projection tensors must be restored")

    def prepare_motion(self, motion, neutral, sigma, captions, modes):
        return super().prepare_motion(motion, neutral, sigma, captions_for_modes(captions, modes), modes)

    def forward(self, *, captions, modes, **kwargs):
        return super().forward(captions=captions_for_modes(captions, modes), modes=modes, **kwargs)
