"""Batched native UniPC with an independently initialized auxiliary camera state."""
import hashlib
import torch
from motion_expert_phase3_edge_joint.runtime import sampling_execution
from motion_expert_phase3_edge_joint.sample import case_generators,case_noise

def noise_hashes(value):
    return [hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest() for x in value]

class CameraSwitch:
    def __init__(self,model,enabled):self.model=model;self.enabled=enabled;self.bridges=model.bridges
    def eval(self):self.model.eval();return self
    def __call__(self,**inputs):return self.model(**inputs,camera_enabled=self.enabled)

@torch.no_grad()
@sampling_execution
def sample(model,*,mode,motion,neutral,video,seed,steps=35,camera_enabled=True,audit=False):
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    assert mode in ['video2motion','motimg2video']
    device=motion.device;batch=len(motion);seeds=list(seed)
    assert len(seeds)==batch and len(neutral)==len(video)==batch
    def scheduler(shift):
        s=FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.,use_dynamic_shifting=False)
        s.set_timesteps(steps,device=device,shift=shift);return s
    ms,vs=scheduler(3.),scheduler(10.)
    cs=scheduler(3. if mode=='video2motion' else 10.)
    target='motion' if mode=='video2motion' else 'video'
    template=motion if mode=='video2motion' else video
    noisy=case_noise(template,case_generators(seeds,batch,device))
    if mode=='motimg2video':noisy[:,:,0]=video[:,:,0]
    state_m=noisy if mode=='video2motion' else motion
    state_v=noisy if mode=='motimg2video' else video
    state_c=(case_noise(torch.empty(batch,96,64,device=device),
        case_generators([131071+s for s in seeds],batch,device)) if camera_enabled else None)
    receipt=(dict(target=target,target_initial_sha256=noise_hashes(noisy),
                 camera_initial_sha256=noise_hashes(state_c) if camera_enabled else None) if audit else None)
    model.eval()
    for i in range(steps):
        motion_sigma=ms.timesteps[i].float().reshape(1).expand(batch)/1000. if mode=='video2motion' else torch.zeros(batch,device=device)
        video_sigma=vs.timesteps[i].float().reshape(1).expand(batch)/1000. if mode=='motimg2video' else torch.zeros(batch,device=device)
        inputs=dict(motion=state_m,neutral=neutral,video=state_v,motion_sigma=motion_sigma,
                    video_sigma=video_sigma,captions=['']*batch,modes=[mode]*batch)
        if camera_enabled:inputs.update(camera_state=state_c,camera_sigma=cs.timesteps[i].float().reshape(1).expand(batch)/1000.)
        prediction=model(**inputs)
        # Advance both targets from predictions at the same pre-step state.
        if mode=='video2motion':
            sigma=ms.sigmas[i].to(device=device,dtype=torch.float32).clamp_min(1e-6)
            state_m=ms.step((state_m-prediction['motion'].float())/sigma,ms.timesteps[i],state_m,return_dict=False)[0].float()
        else:
            state_v=vs.step(prediction['video'].float(),vs.timesteps[i],state_v,return_dict=False)[0].float()
            state_v[:,:,0]=video[:,:,0]
        if camera_enabled:
            state_c=cs.step(prediction['camera_velocity'].float(),cs.timesteps[i],state_c,return_dict=False)[0].float()
    assert torch.isfinite(state_m).all() and torch.isfinite(state_v).all()
    if camera_enabled:assert torch.isfinite(state_c).all() and state_c.shape==(batch,96,64)
    if mode=='motimg2video':
        torch.testing.assert_close(state_v[:,:,0],video[:,:,0],rtol=0,atol=0)
        torch.testing.assert_close(state_m,motion,rtol=0,atol=0)
    else:torch.testing.assert_close(state_v,video,rtol=0,atol=0)
    return dict(motion=state_m,video=state_v,camera=state_c,audit=receipt)
