"""Frozen full-test V2M comparison: Phase3 bridges and UniEgoMotion variants.

No training changes. Every eligible floor-clean test annotation is retained.
Models receive observed video and their native shape input, never GT motion.
"""
from pathlib import Path
import argparse
from collections import OrderedDict, Counter
import fcntl
import gc
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

BASE=Path('/scratch2/whwjdqls99/cosmos')
CACHE=BASE/'caches/nymeria_soma_official_baselines_T97_20fps_floorclean_20261016'
OLD=BASE/'operations/nymeria_official_baselines_20261015'
OVERLAY=BASE/'operations/shape_conditioned_baselines_20261101/upstream'
FEATURES=BASE/'caches/nymeria_uem_dinov2_vitl14reg_5fps_20261019'
P3=BASE/'cosmos_motion_ft_runs/motion_expert_phase3_edge'
P3MODELS={
 'p3_baseline':('20260924','motion_expert_phase3_edge', 'bridge7_T97_'),
 'p3_joint':('20261002','motion_expert_phase3_edge_joint','joint3_equal'),
 'p3_camera':('20261004','motion_expert_phase3_edge_camcond','camcond3_mix'),
 'p3_fourtask':('20261020','motion_expert_phase3_edge_fourtask','fourtask_equal_notext_T97'),
 'p3_hidden_film':('20261021','motion_expert_phase3_edge_camfilm','contextcamfilm'),
 'p3_floor18_film':('20261023','motion_expert_phase3_edge_camtrajfilm','floor18d'),
}

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def npz(p,**values):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+'.tmp.npz');np.savez(q,**values);q.replace(p)
def progress(root,stage,**values):
 v=dict(stage=stage,time=time.time(),job=os.environ.get('SLURM_JOB_ID'),**values)
 save(root/'progress.json',v);print(json.dumps(v),flush=True)
def guard():
 if not os.environ.get('SLURM_JOB_ID') or 'login' in os.uname().nodename:raise RuntimeError('Owned compute allocation required')

def snapshot(source,dest):
 import torch
 if not dest.exists():
  before=source.stat();temp=dest.with_suffix('.copying')
  shutil.copyfile(source,temp)
  after=source.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
   temp.unlink();raise RuntimeError('Checkpoint changed during snapshot: '+str(source))
  temp.replace(dest)
 value=torch.load(dest,map_location='cpu',mmap=True,weights_only=False)
 return value

def prepare(a):
 import torch
 from nymeria_baselines.prepare import prepare_window
 from nymeria_baselines.uem_visual_cache import frame_ids
 from nymeria_baselines.body_shape import BODY_JOINTS
 import nymeria_joint_dataset as nj
 from dataset import humanize_caption
 from runtime_paths import resolve_legacy_path
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'snapshots').mkdir(exist_ok=True)
 if (a.out/'configuration.json').exists():
  assert read(a.out/'configuration.json')['source_sha256']==sha(__file__)
  print('Existing frozen configuration retained',flush=True);return
 models=[];contract=None
 for label,(date,module,fragment) in P3MODELS.items():
  runs=[p for p in P3.iterdir() if p.name.endswith(date) and fragment in p.name];assert len(runs)==1,runs
  candidates=list((runs[0]/'checkpoints').glob('step_*.pt'))
  recovery=runs[0]/'checkpoints/recovery_latest.pt'
  candidates+=([recovery] if recovery.exists() else [])
  states=[]
  for p in candidates:
   ck=torch.load(p,map_location='cpu',mmap=True,weights_only=False);states.append((int(ck['step']),p));del ck
  step,path=max(states,key=lambda x:x[0]);dest=a.out/'snapshots'/(label+'.pt')
  ck=snapshot(path,dest);source=ck['sources'];c=source['phase2_contract']
  if contract is None:contract=c
  assert all(c[k]==contract[k] for k in ('motion_mean','motion_std','motion_stats_mean','motion_stats_std'))
  models.append(dict(label=label,family='phase3',module=module,step=step,checkpoint=str(dest),
      checkpoint_sha256=sha(dest),original_checkpoint=str(path),sources=source,
      inputs='video + provided centered SOMA30 shape; empty text; no trajectory',
      sampler=dict(steps=35,shift=3.,guidance=1.,solver='unchanged Phase3 UniPC sampler')))
  del ck;gc.collect()
 for label,variant in [('uem_rgb','RGB_recon'),('uem_recon','recon'),('uem_all','all'),('uem_shape','shape')]:
  run=(BASE/'operations/uem_shape_threeway_ddp_20261104/training' if variant=='shape' else
       OLD/'runs'/f'uniegomotion-{variant}_nymeria_soma_b64_350ep_20261019')
  # Atomic Lightning last.ckpt updates are detected by snapshot's before/after stat.
  dest=a.out/'snapshots'/(label+'.ckpt');ck=snapshot(run/'last.ckpt',dest)
  cfg=ck['hyper_parameters'];assert cfg['DATA']['COND_IMG_FEAT']
  assert Path(cfg['DATA']['DATA_DIR']).resolve()==CACHE.resolve()
  models.append(dict(label=label,family='uem',variant=variant,step=int(ck['global_step']),epoch=int(ck['epoch']),
      checkpoint=str(dest),checkpoint_sha256=sha(dest),original_checkpoint=str(run/'last.ckpt'),hyper_parameters=cfg,
      inputs='RGB features + provided centered SOMA30 shape' if variant=='shape' else 'RGB features; predicted shape; no trajectory',
      sampler=dict(steps=int(cfg['MODEL']['DIFFUSION_STEPS']),guidance=cfg['TRAIN']['COND_SCALE'],
                   solver='unchanged native UEM p_sample_loop',weights='state_dict; same policy as previous comparison')))
  del ck;gc.collect()
 mean,std=np.load(contract['motion_stats_mean']),np.load(contract['motion_stats_std'])
 assert sha(contract['motion_stats_mean'])==contract['motion_mean'] and sha(contract['motion_stats_std'])==contract['motion_std']
 index=read(CACHE/'index.json');test=sorted([r for r in index['rows'] if r['split']=='test'],key=lambda r:(r['uuid'],r['start'],r['caption']))
 assert len({r['uuid'] for r in test})==71
 assert not {r['uuid'] for r in test}&{r['uuid'] for r in index['rows'] if r['split']!='test'}
 nymeria=Path(os.environ['WEKA_ROOT'])/'nymeriaplus_kimodo_proportional'
 videos={}
 with (nymeria/'video/manifest_video.jsonl').open() as f:
  for line in f:
   value=json.loads(line)
   if value.get('uuid'):videos[value['uuid']]=resolve_legacy_path(value['vision_path'])
 rows=[];arrays=OrderedDict();feature_arrays={};latent_root=nymeria/'joint_latents_T97_256tier_256'
 for i,r in enumerate(test):
  key=r['array']
  if key not in arrays:
   arrays[key]=[np.load(CACHE/'records'/f'{key}.{suffix}.npy',mmap_mode='r') for suffix in ('motion','pose_yup','neutral')]
  while len(arrays)>4:arrays.popitem(last=False)
  motion,pose,neutral=arrays[key];s=int(r['start'])
  raw,_=prepare_window(motion[s:s+97],pose[s:s+97],r['ground_offset_y'])
  shape=np.array(neutral,dtype=np.float32,copy=True);shape-=shape.mean(0,keepdims=True)
  norm=(raw-mean)/std
  assert norm.shape==(97,283) and np.isfinite(norm).all() and np.abs(norm).max()<=20
  # Prove all RGB conditions are available before starting any inference.
  ids=np.load(FEATURES/'records'/f'{key}.frame_ids.npy',mmap_mode='r');wanted=frame_ids(s);slots=np.searchsorted(ids,wanted)
  assert (slots<len(ids)).all() and np.array_equal(ids[slots],wanted),(r['uuid'],s)
  uid=f'{i:06d}_'+hashlib.sha256(json.dumps([r['uuid'],s,r['caption']]).encode()).hexdigest()[:12]
  bundle=a.out/'inputs'/(uid+'.npz');npz(bundle,motion=norm.astype(np.float32),raw=raw.astype(np.float32),neutral=shape)
  latent=Path(nj.latent_path(r['uuid'],s,str(latent_root)))
  if not latent.is_file():latent=a.out/'latents'/(hashlib.sha256(json.dumps([r['uuid'],s]).encode()).hexdigest()[:24]+'.npz')
  rows.append(dict(id=uid,uuid=r['uuid'],start=s,caption=humanize_caption(r['caption']),array=key,
      floor_offset=r['ground_offset_y'],bundle=str(bundle),bundle_sha256=sha(bundle),latent=str(latent),
      rgb=videos[r['uuid']],
      seed=int.from_bytes(hashlib.sha256(('fulltest-v2m:'+uid).encode()).digest()[:4],'little')%(2**31)))
  if i%1000==0:progress(a.out,'preparing',completed=i,total=len(test))
 save(a.out/'cases.json',rows)
 ref=read(BASE/'operations/reconstruction_tmr_body23_20261023_axisfix/configuration.json')
 cfg=dict(schema='fulltest_v2m_both_v1',models=models,cases=len(rows),recordings=71,
      physical_windows=len({(r['uuid'],r['start']) for r in rows}),cases_sha256=sha(a.out/'cases.json'),
      source_sha256=sha(__file__),cache=str(CACHE),cache_index_sha256=sha(CACHE/'index.json'),
      phase2_contract=contract,statistics_sha256=sha(CACHE/'train_stats.npz'),features=str(FEATURES),
      visual_identity=read(FEATURES/'READY.json'),body_joints=list(BODY_JOINTS),
      evaluator={k:ref[k] for k in ('evaluator_checkpoint','evaluator_stats','text_cache','adapter_sha256')},
      adapter=str(BASE/'operations/reconstruction_tmr_body23_20261023_axisfix/adapter.json'),
      protocol='All eligible floor-clean test annotation windows; no fixed71 subsampling; no prediction grounding, GT motion or head inpainting. Native samplers, no test-selected guidance.',
      unsupported='Trajectory-only UEM has no trained RGB pathway and is not a V2M baseline',
      notion_page='40dd2a44-f44b-8080-aad3-cda6b5fea7c8')
 save(a.out/'configuration.json',cfg);progress(a.out,'prepared',cases=len(rows),models=len(models))

def load_common(a):
 cfg=read(a.out/'configuration.json');assert cfg['source_sha256']==sha(__file__),'Evaluator changed'
 assert cfg['cases_sha256']==sha(a.out/'cases.json')
 return cfg,read(a.out/'cases.json')

def run_model(a):
 import torch
 from nymeria_baselines.body_shape import BODY_JOINTS,MOTION_INDICES
 from nymeria_baselines.visualize_official_baselines import decode
 cfg,rows=load_common(a);spec=next(x for x in cfg['models'] if x['label']==a.model)
 root=a.out/'models'/a.model;root.mkdir(parents=True,exist_ok=True)
 lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (root/'INFERENCE_COMPLETE.json').exists():return
 assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
 ck=torch.load(spec['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
 if spec['family']=='phase3':
  Model=importlib.import_module(spec['module']+'.model').EdgePhase3
  source=spec['sources'];model=Model(source['phase1'],source['phase2'],device='cuda',activation_checkpointing=False)
  assert model.provenance['phase1_metadata_sha256']==source['phase1_metadata_sha256']
  assert model.provenance['phase2_contract']==source['phase2_contract']
  model.bridges.load_state_dict(ck['bridges'],strict=True)
  from motion_expert_phase3_edge.sample import sample as p3sample
 else:
  sys.path.insert(0,str(OVERLAY/'UniEgoMotion' if spec['variant']=='shape' else OLD/'upstream/UniEgoMotion'))
  from yacs.config import CfgNode
  from module.uem_module import UEM_Module
  hp=CfgNode(ck['hyper_parameters']);hp.TRAIN.ONLY_VALIDATE=True
  model=UEM_Module(hp);model.load_state_dict(ck['state_dict'],strict=True);model.cuda()
  with np.load(CACHE/'train_stats.npz') as z:umean,ustd=z['motion_mean'][MOTION_INDICES],z['motion_std'][MOTION_INDICES]
 model.eval().requires_grad_(False);del ck;gc.collect()
 mean,std=np.load(cfg['phase2_contract']['motion_stats_mean']),np.load(cfg['phase2_contract']['motion_stats_std'])
 feature_arrays=OrderedDict();vae=None;started=time.monotonic();new=0
 for i,r in enumerate(rows):
  dest=root/'predictions'/(r['id']+'.npz')
  if dest.exists():
   with np.load(dest) as z:assert str(z['checkpoint_sha256'])==spec['checkpoint_sha256'] and np.isfinite(z['joints']).all()
   continue
  assert sha(r['bundle'])==r['bundle_sha256']
  with np.load(r['bundle']) as z:gt,neutral=z['motion'].copy(),z['neutral'].copy()
  if spec['family']=='phase3':
   lp=Path(r['latent'])
   if not lp.exists():
    from motion_expert_phase3_edge.visualize import load_vae,make_pipeline,encode_window
    from nymeria_camera_dataset import decode_window_pyav
    if vae is None:vae=load_vae(os.environ['WAN_VAE_PATH'],'256',97,'cuda')
    rgbpath=Path(r['rgb'])
    # Locate the authoritative cached video path rather than guessing directory layout.
    if not rgbpath.is_file():
     raise FileNotFoundError('Missing source RGB for latent encoding: '+str(rgbpath))
    frames=decode_window_pyav(str(rgbpath),r['start'],97,20.)
    latent,_=encode_window(vae,make_pipeline(),'256',frames,'cuda');npz(lp,latents=latent)
   with np.load(lp) as z:video=z['latents'].copy()
   assert video.shape==(48,25,16,16) and np.isfinite(video).all()
   values={k:torch.from_numpy(v)[None].cuda().float() for k,v in dict(motion=np.zeros_like(gt),neutral=neutral,video=video).items()}
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    generated=p3sample(model,mode='video2motion',caption='',seed=r['seed'],steps=35,guidance=1.,**values)[0].float().cpu().numpy()
   raw=generated*std+mean;joints,rotations=decode(raw,30);joints,rotations=joints[:,BODY_JOINTS],rotations[:,BODY_JOINTS]
   neutral23=neutral[list(BODY_JOINTS)]
  else:
   key=r['array']
   if key not in feature_arrays:
    feature_arrays[key]=(np.load(FEATURES/'records'/f'{key}.frame_ids.npy',mmap_mode='r'),np.load(FEATURES/'records'/f'{key}.features.npy',mmap_mode='r'))
   while len(feature_arrays)>4:feature_arrays.popitem(last=False)
   ids,features=feature_arrays[key]
   from nymeria_baselines.uem_visual_cache import frame_ids
   wanted=frame_ids(r['start']);slots=np.searchsorted(ids,wanted)
   image=np.array(features[slots],dtype=np.float32)
   y=dict(valid_frames=torch.ones(1,97,dtype=torch.long,device='cuda'),img_embs=torch.from_numpy(image[None]).cuda(),
          valid_img_embs=torch.ones(1,97,dtype=torch.long,device='cuda'),img_mask=torch.zeros(1,97,dtype=torch.long,device='cuda'))
   if spec['variant']=='shape':y['neutral_joints']=torch.from_numpy(neutral[None]).cuda()
   assert 'traj' not in y
   torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
   with torch.inference_mode():generated=model.sample(y,B=1,cond_scale=hp.TRAIN.COND_SCALE)[0].float().cpu().numpy()
   # Match historical inference: native target uses std+1e-6; the published
   # inverse multiplies by saved std. Preserve that established convention.
   raw=generated[:,:220]*ustd+umean;joints,rotations=decode(raw,23)
   neutral23=neutral[list(BODY_JOINTS)]
   if spec['variant']!='shape':
    from nymeria_baselines.body23_tmr_adapter import neutral_from_pca
    with np.load(CACHE/'body_shape16.npz') as basis:neutral23=neutral_from_pca(generated[:,220:],basis)
  assert joints.shape==(97,23,3) and np.isfinite(joints).all() and np.isfinite(rotations).all()
  npz(dest,generated=generated,joints=joints.astype(np.float32),rotations=rotations.astype(np.float32),neutral23=neutral23.astype(np.float32),
      checkpoint_sha256=np.array(spec['checkpoint_sha256']),case_id=np.array(r['id']),seed=r['seed'])
  new+=1
  if new%10==0 or new<=2:progress(root,'sampling',completed=i+1,total=len(rows),new_cases=new,seconds_per_case=(time.monotonic()-started)/new,step=spec['step'])
  if a.limit and new>=a.limit:return
 save(root/'INFERENCE_COMPLETE.json',dict(cases=len(rows),step=spec['step'],checkpoint_sha256=spec['checkpoint_sha256']))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['prepare','sample'])
 p.add_argument('--out',type=Path,required=True);p.add_argument('--model');p.add_argument('--limit',type=int,default=0)
 a=p.parse_args();guard()
 if a.command=='prepare':prepare(a)
 else:run_model(a)

if __name__=='__main__':main()
