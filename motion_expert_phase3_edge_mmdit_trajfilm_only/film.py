"""Zero-initialized FP32 camera18 + frozen motion-time FiLM."""
import torch
from torch import nn

class TrajectoryFiLM(nn.Module):
    def __init__(self,hidden=2048,context=256,mlp=256):
        super().__init__()
        self.motion_norm=nn.LayerNorm(hidden,elementwise_affine=False)
        self.camera_projection=nn.Sequential(nn.Linear(18,context),nn.SiLU())
        self.noise_projection=nn.Sequential(nn.Linear(hidden,context),nn.SiLU())
        self.camera_norm=nn.LayerNorm(context,elementwise_affine=False)
        self.noise_norm=nn.LayerNorm(context,elementwise_affine=False)
        self.mlp=nn.Sequential(nn.Linear(hidden+2*context,mlp),nn.SiLU(),nn.Linear(mlp,2*hidden))
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
    def forward(self,motion,camera,time):
        if motion.ndim!=3 or camera.shape!=(*motion.shape[:2],18) or time.shape!=(motion.shape[0],motion.shape[-1]):
            raise ValueError('Expected Bx97xhidden, Bx97x18 and Bxhidden')
        with torch.autocast(device_type=motion.device.type,enabled=False):
            h=self.motion_norm(motion.float());c=self.camera_norm(self.camera_projection(camera.float()))
            t=self.noise_norm(self.noise_projection(time.float()))[:,None].expand(-1,motion.shape[1],-1)
            gamma,beta=self.mlp(torch.cat((h,c,t),-1)).chunk(2,-1)
            return (motion.float()+gamma*h+beta).to(motion.dtype)
