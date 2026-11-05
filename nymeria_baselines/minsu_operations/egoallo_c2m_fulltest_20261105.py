"""Thin full-cohort adapter to the shared supplied-shape native EgoAllo sampler."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import random
import time

OP=Path('/scratch2/minsu1206/cosmos/operations/egoallo_c2m_fulltest_20261105')
TRAIN=Path('/scratch2/minsu1206/cosmos/operations/shape_conditioned_baselines_20261101/egoallo/training')
UP=Path('/scratch2/whwjdqls99/cosmos/operations/shape_conditioned_baselines_20261101/upstream')
LABEL='egoallo_final200k_camera_shape'

def main():
 import numpy as np
 import torch
 from safetensors.torch import load_file
 from nymeria_baselines.fulltest_v2m import read,save,sha,npz,guard,load_common
 from nymeria_baselines.shape_conditioned.egoallo import activate
 from nymeria_baselines.shape_conditioned.sampling import sample_egoallo,egoallo_fk_with_supplied_shape
 from nymeria_baselines.evaluate_egoallo_phase3 import calibrate
 from nymeria_baselines.visualize_official_baselines import Y_FROM_Z,decode
 from nymeria_baselines.body_shape import BODY_JOINTS,PARENTS
 from nymeria_baselines.motion_error_metrics import extra_metrics
 from motion_expert_joint_attention.camera_head_recanonicalization import yaw_rotation_y
 guard();torch.set_num_threads(2)
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,default=2)
 a=p.parse_args();assert 0<=a.rank<a.world
 cfg,rows=load_common(argparse.Namespace(out=OP));assert len(rows)==11911
 assert read(TRAIN/'COMPLETE.json')['step']==200000
 checkpoint=TRAIN/'checkpoints_200000'
 assert read(checkpoint/'COMPLETE.json')['step']==200000
 root=OP/'models'/LABEL;root.mkdir(parents=True,exist_ok=True)
 lock=(root/f'rank{a.rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 activate(UP)
 from egoallo import network
 from egoallo.data.nymeria_soma import NymeriaSomaDataset
 from egoallo.transforms import SE3
 cache=Path(cfg['cache']);dataset=NymeriaSomaDataset(cache,'test')
 assert len(dataset)==len(rows)
 # Match the frozen full-test sort, retaining every caption-duplicate row.
 indices=sorted(range(len(dataset)),key=lambda i:(dataset.rows[i]['uuid'],dataset.rows[i]['start'],dataset.rows[i]['caption']))
 for i,(r,index) in enumerate(zip(rows,indices)):
  d=dataset.rows[index]
  assert (r['uuid'],r['start'],r['array'])==(d['uuid'],d['start'],d['array'])
  assert abs(r['floor_offset']-d['ground_offset_y'])<1e-7
  assert r['id']==f'{i:06d}_'+hashlib.sha256(json.dumps([d['uuid'],d['start'],d['caption']]).encode()).hexdigest()[:12]
 assert sha(cache/'index.json')==cfg['cache_index_sha256']
 snapshot=OP/'snapshots';snapshot.mkdir(exist_ok=True)
 weights=snapshot/'model_200000.safetensors'
 try:os.link(checkpoint/'model.safetensors',weights)
 except FileExistsError:pass
 weight_sha=sha(weights);assert weight_sha==sha(checkpoint/'model.safetensors')
 recipe=read(TRAIN/'experiment.json');model_cfg=network.EgoDenoiserConfig(**recipe['model_config'])
 model=network.EgoDenoiser(model_cfg).eval().cuda();model.load_state_dict(load_file(str(weights)),strict=True)
 assert model.get_d_state()==202 and not model_cfg.include_hands
 calibration=calibrate(cache)
 calibration_path=OP/f'calibration_rank{a.rank}.json';save(calibration_path,calibration)
 extrinsic_r=np.asarray(calibration['head_rotation_in_camera']);extrinsic_p=np.asarray(calibration['head_offset_camera'])
 basis=np.load(cache/'body_shape16.npz')
 protocol=dict(model=LABEL,step=200000,checkpoint=str(weights),checkpoint_sha256=weight_sha,
  cases=len(rows),recordings=len({r['uuid'] for r in rows}),inputs='Measured camera trajectory + supplied train-PCA16 actor shape; no RGB or text',
  sampler='Unmodified shared sample_egoallo, native run_sampling_with_stitching; guidance off, no hands',
  shape='Supplied 16D shape held fixed at every native denoising step and FK',
  placement='Camera + training-only head extrinsics; frame-0 head yaw and camera-calibrated horizontal origin; no test motion fit',
  configuration_sha256=sha(OP/'configuration.json'),cases_sha256=sha(OP/'cases.json'),
  adapter_source_sha256=sha(__file__),sampler_source_sha256=sha(Path(__import__('nymeria_baselines.shape_conditioned.sampling',fromlist=['x']).__file__)),
  shape_basis_sha256=sha(cache/'body_shape16.npz'),training_recipe=recipe,calibration=calibration)
 if a.rank==0:save(OP/'camera_to_motion_protocol.json',protocol)
 selected=list(range(a.rank,len(rows),a.world));started=time.monotonic();durations=[];maximum_gt=0.;maximum_fk=0.
 for n,i in enumerate(selected):
  r=rows[i];dest=root/'predictions'/(r['id']+'.npz')
  if dest.exists():
   with np.load(dest) as z:
    assert str(z['case_id'])==r['id'] and int(z['seed'])==r['seed'] and str(z['checkpoint_sha256'])==weight_sha
    assert np.isfinite(z['joints']).all()
   continue
  data=dataset[indices[i]];camera=SE3(data.T_world_cpf).as_matrix().numpy()
  camera_r=Y_FROM_Z@camera[:,:3,:3];camera_p=np.einsum('ij,tj->ti',Y_FROM_Z,camera[:,:3,3])
  forward=(camera_r[0]@extrinsic_r)[:,2];assert np.linalg.norm(forward[[0,2]])>1e-2
  yaw=yaw_rotation_y(np.asarray(np.arctan2(forward[0],forward[2])))
  head_p=camera_p+np.einsum('tij,j->ti',camera_r,extrinsic_p)
  origin=np.array([head_p[0,0],0.,head_p[0,2]])
  # GT is used only to audit coordinate conversion; never in prediction placement.
  assert sha(r['bundle'])==r['bundle_sha256']
  with np.load(r['bundle']) as z:target,_=decode(z['raw'].copy(),30)
  target=target[:,BODY_JOINTS].astype(np.float32)
  gt_world=camera_p[:,None]+np.einsum('tij,tkj->tki',camera_r,data.joints_wrt_cpf.numpy())
  gt_origin=np.array([gt_world[0,5,0],0.,gt_world[0,5,2]])
  error=float(np.max(np.abs(np.einsum('ij,tkj->tki',yaw.T,gt_world-gt_origin)-target[:,1:])))
  assert error<2e-3,(r['id'],'GT frame mismatch',error);maximum_gt=max(maximum_gt,error)
  cpf=data.T_world_cpf.cuda();delta=data.T_cpf_tm1_cpf_t.cuda();betas=data.betas[None].cuda()
  previous=(SE3(cpf[0])@SE3(delta[0]).inverse()).parameters();transforms=torch.cat([previous[None],cpf])
  def sample():
   random.seed(r['seed']);np.random.seed(r['seed']);torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
   with torch.inference_mode():return sample_egoallo(model,UP,transforms,betas)
  torch.cuda.synchronize();clock=time.monotonic();pred=sample();torch.cuda.synchronize();elapsed=time.monotonic()-clock
  assert torch.isfinite(pred.pack()).all() and torch.equal(pred.betas,betas.expand(1,97,16))
  local=pred.body_rotmats[0].cpu().numpy();assert local.shape==(97,22,3,3)
  supplied=betas.cpu().numpy()[0,0]
  neutral=(basis['mean']+(supplied*basis['scale'])@basis['components']).reshape(23,3);neutral-=neutral[:1].copy()
  fk_r=np.broadcast_to(np.eye(3),(97,23,3,3)).copy()
  for j in range(1,23):fk_r[:,j]=fk_r[:,PARENTS[j]]@local[:,j-1]
  with torch.inference_mode():fk=egoallo_fk_with_supplied_shape(pred.body_rotmats,betas,basis)[0].cpu().numpy()
  audit_fk=np.zeros_like(fk,dtype=np.float64)
  for j in range(1,23):
   par=PARENTS[j];audit_fk[:,j]=audit_fk[:,par]+np.einsum('tij,j->ti',fk_r[:,par],neutral[j]-neutral[par])
  fk_error=float(np.abs(fk-audit_fk).max());assert fk_error<2e-5;maximum_fk=max(maximum_fk,fk_error)
  root_r=camera_r@extrinsic_r@fk_r[:,6].swapaxes(-1,-2)
  root_p=head_p-np.einsum('tij,tj->ti',root_r,fk[:,6])
  joints=np.einsum('ij,tkj->tki',yaw.T,root_p[:,None]+np.einsum('tij,tkj->tki',root_r,fk)-origin).astype(np.float32)
  rotations=(yaw.T@root_r[:,None]@fk_r).astype(np.float32)
  assert np.isfinite(joints).all() and np.isfinite(rotations).all()
  if not durations:
   repeated=sample();difference=float((pred.pack()-repeated.pack()).abs().max())
   assert difference==0.,('Native same-seed replay failed',difference)
   save(root/f'acceptance_rank{a.rank}.json',dict(passed=True,case_id=r['id'],seed=r['seed'],
    replay_max_abs=difference,gt_frame_max_abs_m=error,fk_parity_max_abs_m=fk_error,seconds_per_case=elapsed,
    native_steps='Unchanged native quadratic schedule',supplied_shape_exact=True,checkpoint_sha256=weight_sha))
   deadline=time.monotonic()+180
   for rank in range(a.world):
    acceptance=root/f'acceptance_rank{rank}.json'
    while not acceptance.exists():
     if time.monotonic()>deadline:raise RuntimeError('Peer GPU acceptance did not pass; production blocked')
     time.sleep(1)
    accepted=read(acceptance);assert accepted['passed'] and accepted['checkpoint_sha256']==weight_sha
  npz(dest,case_id=r['id'],seed=r['seed'],checkpoint_sha256=weight_sha,joints=joints,rotations=rotations,
      neutral23=neutral.astype(np.float32),local_rotations=local,betas=supplied,contacts=pred.contacts[0].cpu().numpy(),
      gt_frame_max_abs_m=error,fk_max_abs_m=fk_error,seconds=elapsed,
      gt_extended_metrics=json.dumps(extra_metrics(target,target,fps=20.)))
  durations.append(elapsed)
  if n%10==0 or n+1==len(selected):
   status=dict(stage='sampling',rank=a.rank,completed=n+1,total=len(selected),time=time.time(),
    seconds=time.monotonic()-started,seconds_per_case=float(np.mean(durations[-100:])),last=r['id'],
    gt_frame_max_abs_m=maximum_gt,fk_parity_max_abs_m=maximum_fk)
   save(root/f'progress_rank{a.rank}.json',status);print(json.dumps(status),flush=True)
 gt_values=[]
 for i in selected:
  with np.load(root/'predictions'/(rows[i]['id']+'.npz')) as z:gt_values.append(json.loads(str(z['gt_extended_metrics'])))
 save(root/f'GT_extended_rank{a.rank}.json',dict(cases=len(selected),
  sums={key:float(sum(v[key] for v in gt_values)) for key in gt_values[0]},cases_sha256=sha(OP/'cases.json')))
 save(root/f'COMPLETE_rank{a.rank}.json',dict(cases=len(selected),rank=a.rank,checkpoint_sha256=weight_sha))

if __name__=='__main__':main()
