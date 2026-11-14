"""Frozen full Phase3 base; only seven new trajectory FiLMs are trainable.

Camera-only has no generator tokens. The existing bridge returns its inputs
unchanged for an empty generator; therefore camera-only bypasses those frozen
bridges and sends the18D camera condition exclusively into motion FiLM.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .film import TrajectoryFiLM

class CameraOnlyFiLM(nn.Module):
    def __init__(self,base,activation_checkpointing=True):
        super().__init__();self.base=base;self.base.requires_grad_(False).eval();self.base.native.requires_grad_(False).eval()
        self.layers=tuple(sorted(int(k) for k in base.bridges));self.activation_checkpointing=activation_checkpointing
        device=next(base.motion.parameters()).device
        self.film=nn.ModuleDict({str(k):TrajectoryFiLM().to(device=device,dtype=torch.float32) for k in self.layers})
        assert sum(p.numel() for p in self.film.parameters())==15663872
        assert all(not p.requires_grad for p in self.base.parameters())
    def train(self,mode=True):
        super().train(mode);self.base.eval();self.film.train(mode);return self
    def frozen_parameters(self):
        seen=set()
        for module in (self.base.native,self.base.motion,self.base.bridges):
            for p in module.parameters():
                if id(p) not in seen:seen.add(id(p));yield p
    def forward(self,*,motion,neutral,motion_sigma,camera_features,film_enabled=True):
        batch=len(motion)
        if motion.shape!=(batch,97,283) or neutral.shape!=(batch,30,3) or camera_features.shape!=(batch,97,18):raise ValueError('Invalid camera-only shapes')
        with torch.no_grad():
            state,memory,layout,mc,ms,tc,ts=self.base.prepare_motion(motion,neutral,motion_sigma,['']*batch,['video2motion']*batch)
        with torch.no_grad(),torch.autocast(device_type=motion.device.type,enabled=False):
            time=self.base.motion.heads.time_embedder(motion_sigma.float()).float()
        for k in self.layers:
            layer=self.base.motion.layers[k]
            def update(value,block=layer):return block.forward_motion(value,memory,layout,mc,ms,tc,ts)
            state=checkpoint(update,state,use_reentrant=False) if self.activation_checkpointing and self.training and state.requires_grad else update(state)
            if film_enabled:
                hidden=state.reshape(batch,98,2048)
                state=torch.cat((hidden[:,:1],self.film[str(k)](hidden[:,1:],camera_features,time)),1).reshape(-1,2048)
        hidden=self.base.motion.norm_moe_motion(state).reshape(batch,98,2048)
        return torch.stack([self.base.motion.heads.decode(rows[1:]) for rows in hidden])
