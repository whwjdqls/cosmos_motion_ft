"""Original EgoEgo transformer; clean shape token and separate camera input.

Only model/conditioning dimensions change. DDPM coefficients, x0 conversion
and posterior remain upstream. MSE, standardization and no output clipping
match our UniEgoMotion recipe (upstream EgoEgo's AMASS trainer used L1/min-max).
"""
import torch
from torch import nn
from torch.nn import functional as F
from .upstream import load

native = load()


class ShapeDenoiser(native.TransformerDiffusionModel):
    def __init__(self):
        super().__init__(d_feats=207,d_model=512,n_dec_layers=4,n_head=4,d_k=256,d_v=256,max_timesteps=99)
        self.shape_encoder = nn.Sequential(nn.Linear(90,768),nn.GELU(),nn.LayerNorm(768),nn.Linear(768,512))

    def forward(self, src, noise_t, padding_mask=None, shape=None):
        if shape is None:
            return super().forward(src,noise_t,padding_mask)
        batch, frames, _ = src.shape
        # The original decoder accepts prefix embeddings; add one clean shape
        # token beside its original diffusion-time token, with no body targets.
        prefix = torch.stack((self.time_mlp(noise_t),self.shape_encoder(shape.reshape(batch,90))),dim=1)
        valid = torch.ones(batch,1,frames+2,device=src.device,dtype=torch.bool)
        positions = torch.arange(1,frames+3,device=src.device)[None,None].expand(batch,1,-1)
        hidden,_ = self.motion_transformer(src.transpose(1,2).detach(),valid,positions,obj_embedding=prefix)
        return self.linear_out(hidden[:,2:])


class CameraShapeDiffusion(native.CondGaussianDiffusion):
    def __init__(self):
        super().__init__(d_feats=207,d_model=512,n_head=4,n_dec_layers=4,d_k=256,d_v=256,
                         max_timesteps=99,out_dim=207,timesteps=1000,loss_type='l2',objective='pred_x0')
        self.denoise_fn = ShapeDenoiser()
        self.seq_len=97

    @staticmethod
    def condition(camera, filler):
        assert camera.shape[-2:]==(97,9) and filler.shape[-2:]==(97,207)
        value=filler.clone()
        # SOMA Head is index6 in the selected body ordering. Its observed input
        # is the camera pose, without replacing the anatomical target Head.
        value[:,:,18:21]=camera[:,:,:3]
        value[:,:,105:111]=camera[:,:,3:]
        return value

    def forward(self, motion, camera, shape):
        t=torch.randint(0,self.num_timesteps,(len(motion),),device=motion.device)
        noisy=self.q_sample(motion,t,noise=torch.randn_like(motion))
        condition=self.condition(camera,torch.randn_like(motion))
        predicted=self.denoise_fn(torch.cat((noisy,condition),dim=-1),t,shape=shape)
        return F.mse_loss(predicted,motion)

    @torch.no_grad()
    def sample(self,camera,shape,seed=62):
        # Inference requires no target motion, head pose, contacts or velocities.
        generator=torch.Generator(device=camera.device).manual_seed(seed)
        size=(len(camera),97,207)
        x=torch.randn(size,device=camera.device,generator=generator)
        condition=self.condition(camera,torch.randn(size,device=camera.device,generator=generator))
        for index in reversed(range(self.num_timesteps)):
            t=torch.full((len(x),),index,device=x.device,dtype=torch.long)
            clean=self.denoise_fn(torch.cat((x,condition),-1),t,shape=shape)
            mean,_,log_variance=self.q_posterior(clean,x,t)
            noise=torch.randn(size,device=x.device,generator=generator)
            x=mean+(index!=0)*(0.5*log_variance).exp()*noise
        if not torch.isfinite(x).all():raise RuntimeError('Non-finite generated motion')
        return x
