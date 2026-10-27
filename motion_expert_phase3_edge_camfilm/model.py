"""Four-task Phase 3 with additive camera FiLM; frozen specialists unchanged."""
import torch
from .backbone import EdgePhase3 as FourTaskEdgePhase3, bridge_meta
from .film import CameraFiLMBridge
from .settings import RECIPE


class EdgePhase3(FourTaskEdgePhase3):
    def __init__(self, phase1, phase2, device="cuda", activation_checkpointing=True):
        super().__init__(phase1, phase2, device, activation_checkpointing)
        dev = torch.device(device)
        devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
        # FiLM initialization must not alter the baseline attention/noise RNG stream.
        with torch.random.fork_rng(devices=devices):
            for key in list(self.bridges):
                self.bridges[key] = CameraFiLMBridge(
                    self.bridges[key], RECIPE["camera_film"]["context_dim"], RECIPE["camera_film"]["mlp_dim"])
        self.provenance["phase3_architecture"] = "fourtask_context_camera_film_v1"
        print(f"[phase3/film] trainable={sum(p.numel() for p in self.bridges.parameters()):,} "
              f"film={sum(p.numel() for b in self.bridges.values() for p in b.film.parameters()):,}", flush=True)
