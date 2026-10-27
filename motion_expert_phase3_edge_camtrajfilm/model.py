"""Fresh bridges plus trajectory FiLM around the unchanged frozen specialists."""
import torch
from torch import nn

from .backbone import EdgePhase3 as Backbone, bridge_meta
from .film import CameraFiLMBridge
from .settings import ARCHITECTURE, RECIPE


class EdgePhase3(Backbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = next(self.bridges.parameters()).device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self.bridges = nn.ModuleDict({
                key: CameraFiLMBridge(block, RECIPE["camera_film"]["context_dim"],
                                     RECIPE["camera_film"]["mlp_dim"])
                for key, block in self.bridges.items()
            })
        self.provenance["phase3_architecture"] = ARCHITECTURE
        self.provenance["trainable_parameters"] = sum(p.numel() for p in self.bridges.parameters())
