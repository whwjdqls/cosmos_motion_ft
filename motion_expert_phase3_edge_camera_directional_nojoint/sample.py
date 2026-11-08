"""Batched native sampler with independently generated camera states, never GT camera."""
import torch
from motion_expert_phase3_edge_joint.sample import case_generators, case_noise
from motion_expert_phase3_edge_joint.runtime import sampling_execution
from motion_expert_phase3_edge_joint.settings import RECIPE
from .attention import attention_scope

@torch.no_grad()
@sampling_execution
def sample(model, *, mode, caption='', motion, neutral, video, seed=0, steps=35, guidance=1., return_all=False):
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    if mode not in {'video2motion','motimg2video'} or guidance != 1. or steps < 1:
        raise ValueError('Two caption-free tasks, native guidance1 only')
    batch, device = len(motion), motion.device
    generators = case_generators(seed, batch, device)
    seeds = [seed+i for i in range(batch)] if isinstance(seed, int) else list(seed)
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.,use_dynamic_shifting=False)
    camera_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.,use_dynamic_shifting=False)
    shift = 3. if mode=='video2motion' else 10.
    for solver in [scheduler,camera_scheduler]:solver.set_timesteps(steps,device=device,shift=shift)
    state = case_noise(motion if mode=='video2motion' else video, generators)
    camera_generators = [torch.Generator(device=device).manual_seed(131071+int(s)) for s in seeds]
    camera = case_noise(torch.empty(batch,96,64,device=device), camera_generators)
    if mode=='motimg2video':state[:,:,0]=video[:,:,0]
    model.eval()
    with attention_scope(RECIPE['bridge_attention_scope']):
        for i,t in enumerate(scheduler.timesteps):
            sigma=t.float().reshape(1).expand(batch)/1000.
            output=model(motion=state if mode=='video2motion' else motion,
                         video=state if mode=='motimg2video' else video,neutral=neutral,
                         motion_sigma=sigma if mode=='video2motion' else sigma*0,
                         video_sigma=sigma if mode=='motimg2video' else sigma*0,
                         captions=['']*batch,modes=[mode]*batch,camera_state=camera,camera_sigma=sigma)
            key='motion' if mode=='video2motion' else 'video'
            prediction=output[key].float()
            if mode=='video2motion':
                prediction=(state-prediction)/scheduler.sigmas[i].to(device).float().clamp_min(1e-6)
            state=scheduler.step(prediction,t,state,return_dict=False)[0].float()
            camera=camera_scheduler.step(output['camera_velocity'].float(),camera_scheduler.timesteps[i],camera,return_dict=False)[0].float()
            if mode=='motimg2video':state[:,:,0]=video[:,:,0]
    if not torch.isfinite(state).all() or not torch.isfinite(camera).all():
        raise FloatingPointError('Nonfinite camera-aware validation output')
    if return_all:
        return dict(motion=state if mode=='video2motion' else motion,
                    video=state if mode=='motimg2video' else video,camera=camera)
    return state
