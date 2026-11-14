"""Frozen-base loading, motion-only noise/loss, complete FiLM state and sampling."""
from pathlib import Path
import hashlib,json,math,os
import numpy as np
import torch
from .model import CameraOnlyFiLM

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x

def load_base(plan,device):
    from motion_expert_phase3_edge_image_joint.train import configure_local_paths
    from motion_expert_phase3_edge_joint.settings import RECIPE,digest_json
    from motion_expert_phase3_edge_mmdit_bidir_wv1.policy import configure_checkpoint
    from motion_expert_phase3_edge_joint.adapters import load_trainable_state
    from motion_expert_phase3_edge_image_joint.model import EdgePhase3
    os.environ['PHASE3_IMAGE_JOINT_PATH_MAP']=json.dumps(plan['path_map']);configure_local_paths()
    p=torch.load(plan['base_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    assert p['step']==p['ema']['num_updates']==200000 and sha(plan['base_checkpoint'])==plan['base_sha256']
    original=read(Path(plan['base_run'])/'contract.json');assert p['fingerprint']==digest_json(original)
    assert 'camera_prediction' not in p['scheduler']['recipe']
    RECIPE.clear();RECIPE.update(p['scheduler']['recipe']);configure_checkpoint(p)
    base=EdgePhase3(plan['phase1'],plan['phase2'],device=device,activation_checkpointing=False)
    assert base.provenance['phase1_metadata_sha256']==p['sources']['phase1_metadata_sha256'] and base.provenance['phase2_contract']==p['sources']['phase2_contract']
    load_trainable_state(base,p,weights='ema')
    for k,v in base.bridges.state_dict().items():torch.testing.assert_close(v.detach().cpu(),p['ema']['state']['bridges'][k],rtol=0,atol=0)
    return CameraOnlyFiLM(base)

def optimizer_for(model,recipe):
    return torch.optim.AdamW(model.film.parameters(),lr=recipe['lr'],betas=(.9,.99),weight_decay=0.,eps=1e-8)
def lr_for(step,recipe):
    if step<=recipe['warmup']:return recipe['lr']*step/recipe['warmup']
    return recipe['lr']*.5*(1+math.cos(math.pi*(step-recipe['warmup'])/(recipe['steps']-recipe['warmup'])))

def noisy(batch,device):
    from cosmos_framework.model.generator.diffusion.rectified_flow import TrainTimeSampler
    target=batch['motion'].to(device).float();sigma=TrainTimeSampler('logitnormal')(len(target),shifts=torch.full((len(target),),3.)).to(device)
    return dict(motion=torch.lerp(target,torch.randn_like(target),sigma[:,None,None]),neutral=batch['neutral'].to(device).float(),camera_features=batch['camera_features'].to(device).float(),motion_sigma=sigma),target

def motion_loss(pred,target,mean,std,weights):
    from losses import t2m_reconstruction_loss
    valid=torch.ones(pred.shape[:2],device=pred.device,dtype=torch.bool)
    return t2m_reconstruction_loss(pred,target,valid,mean,std,**weights,joint_loss_mode='centered',fps=20.,modes=['text2motion']*len(pred),reduction='per_task',decoded_nonfinite='skip')

def frozen_versions(model):return {id(p):p._version for p in model.frozen_parameters()}
def assert_frozen(model,versions):
    for p in model.frozen_parameters():
        if p.requires_grad or p.grad is not None or p._version!=versions[id(p)]:raise RuntimeError('Frozen200k base or specialist changed')

def film_state(model):return {k:v.detach().cpu().clone() for k,v in model.film.state_dict().items()}
def new_ema(model):return {k:v.detach().float().clone() for k,v in model.film.state_dict().items()}
@torch.no_grad()
def update_ema(model,ema,decay=.999):
    for k,v in model.film.state_dict().items():ema[k].lerp_(v.detach().float(),1-decay)

def save_checkpoint(path,model,optimizer,ema,step,recipe,contract,rng):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    value=dict(schema='frozen_mmdit_base_camera_only_trajfilm_v1',step=step,film_state=film_state(model),film_ema=cpu_tree(ema),ema_updates=step,optimizer=optimizer.state_dict(),scheduler=dict(step=step,lr=lr_for(step,recipe),recipe=recipe),rng_by_rank=rng,next_global_draw=step*32,contract=contract)
    with t.open('wb') as f:torch.save(cpu_tree(value),f);f.flush();os.fsync(f.fileno())
    t.replace(p)

def restore(path,model,optimizer,ema,contract,rank,world):
    from checkpoint import restore_rng_state
    p=torch.load(path,map_location='cpu',mmap=True,weights_only=False);assert p['contract']==contract and p['next_global_draw']==p['step']*32 and p['ema_updates']==p['step']
    assert len(p['rng_by_rank'])==world
    model.film.load_state_dict(p['film_state'],strict=True);optimizer.load_state_dict(p['optimizer'])
    for k,v in ema.items():v.copy_(p['film_ema'][k].to(v.device))
    restore_rng_state(p['rng_by_rank'][rank]);return p['step']

@torch.no_grad()
def sample(model,neutral,camera_features,seeds,steps=35):
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    device=neutral.device;batch=len(neutral);scheduler=FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.,use_dynamic_shifting=False);scheduler.set_timesteps(steps,device=device,shift=3.)
    state=torch.cat([torch.randn(1,97,283,device=device,generator=torch.Generator(device=device).manual_seed(int(s))) for s in seeds])
    model.eval()
    for i,t in enumerate(scheduler.timesteps):
        sigma=t.float().reshape(1).expand(batch)/1000.
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):pred=model(motion=state,neutral=neutral,motion_sigma=sigma,camera_features=camera_features)
        velocity=(state-pred.float())/scheduler.sigmas[i].to(device).float().clamp_min(1e-6);state=scheduler.step(velocity,t,state,return_dict=False)[0].float()
    assert torch.isfinite(state).all();return state
