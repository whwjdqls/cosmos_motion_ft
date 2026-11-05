"""Original fresh motion constructor, with identical native camera attention."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from model import EdgePhase2MotionExpert
from modality_bridge import LocalModalityBridge
from motion_expert_phase3_edge_joint.native import FullFrozenEdge, sha256
from motion_expert_phase3_edge_joint.settings import RECIPE
from motion_expert_phase3_edge_image_joint.policy import PHASE1_METADATA_SHA256
from motion_expert_phase3_edge_image_joint_camera.model import EdgePhase3 as CameraModel
from .policy import INITIALIZATION, configure


def require_metadata_only(path):
    artifact = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if set(artifact) != {'contract', 'scratch_initialization', 'reference_contract_sha256'}:
        raise ValueError('Use the tensor-free architecture artifact, never a trained Phase2 checkpoint')
    def visit(value):
        if isinstance(value, (dict, list, tuple)):
            for item in value.values() if isinstance(value, dict) else value:
                visit(item)
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError('Architecture artifact must contain metadata only, with no tensors')
    visit(artifact)
    if artifact['scratch_initialization'] != INITIALIZATION:
        raise ValueError('Wrong scratch initialization contract')
    return artifact


def state_digest(state):
    result = hashlib.sha256()
    for name, value in state.items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class EdgePhase3(CameraModel):
    def __init__(self, phase1, phase2, device='cuda', activation_checkpointing=True):
        nn.Module.__init__(self)
        configure()
        if RECIPE.get('bridge_parameterization', 'shared') != 'shared':
            raise ValueError('Historical scratch-motion recipe requires the shared bridge')
        artifact = require_metadata_only(phase2)
        edge = FullFrozenEdge(phase1, phase2, device)
        if edge.provenance['phase1_metadata_sha256'] != PHASE1_METADATA_SHA256:
            raise ValueError('Wrong Phase1 LoRA+video projection specialist')
        if len(edge.provenance['phase1_video_projection_tensors']) != 4:
            raise ValueError('Missing trained Phase1 video input/output projections')
        object.__setattr__(self, 'edge', edge)
        object.__setattr__(self, 'native', edge.net)
        # This constructor freshly initializes all registered motion-owned
        # modules. Do not call load_motion_state_dict during construction.
        self.motion = EdgePhase2MotionExpert(device=device, frozen_edge=edge)
        assert all(p.requires_grad for p in self.motion.parameters())
        assert not self.motion.heads.llm2motion.weight.count_nonzero().item()
        assert not self.motion.heads.llm2motion.bias.count_nonzero().item()
        self.bridges = nn.ModuleDict({
            str(index): LocalModalityBridge(2048, 16, 128,
                conditioning_attention=RECIPE['conditioning_attention']).to(device=device, dtype=torch.float32)
            for index in RECIPE['bridge_layers']
        })
        # The shared camera forward includes all registered Phase3 deltas in
        # its DDP zero anchor. This historical scratch recipe has no new LoRA.
        self.phase3_lora = nn.ModuleDict()
        self.activation_checkpointing = bool(activation_checkpointing)
        self.provenance = {
            **edge.provenance, 'phase2_step': None, 'phase2_weights_loaded': False,
            'motion_initialization': INITIALIZATION.copy(),
            'architecture_artifact_sha256': sha256(phase2),
            'reference_camera_contract_sha256': artifact['reference_contract_sha256'],
            'initial_motion_sha256': state_digest(self.motion.motion_state_dict()),
            'initial_bridges_sha256': state_digest(self.bridges.state_dict()),
            'camera_prediction': RECIPE['camera_prediction'].copy(),
            'motion_tensors': len(self.motion.motion_state_dict()),
            'trainable_motion_parameters': sum(p.numel() for p in self.motion.parameters()),
            'trainable_bridge_parameters': sum(p.numel() for p in self.bridges.parameters()),
        }
        self.train()
        print('[phase3/scratch] no learned Phase2 weights; trainable motion='
              f"{self.provenance['trainable_motion_parameters']:,} + bridges="
              f"{self.provenance['trainable_bridge_parameters']:,}", flush=True)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.motion.train(mode)
        self.native.eval()
        return self

    def forward(self, **inputs):
        output = super().forward(**inputs)
        # Homogeneous M2V batches do not supervise the motion readout. Keep
        # mathematically zero edges for the full DDP parameter set.
        output['anchor'] = output['anchor'] + sum(p.flatten()[0]*0. for p in self.motion.parameters())
        return output
