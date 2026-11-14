from pathlib import Path
from types import SimpleNamespace
import copy,inspect,json
import numpy as np
import pytest
import torch
from torch import nn
from .film import TrajectoryFiLM
from .model import CameraOnlyFiLM
from .data import CameraOnlyDataset
from .runtime import lr_for
ROOT=Path('/mnt/local/jinhyung.kim/cosmos_storage/operations/after_bidir200k_fulltest_c2m_trajfilm_20261114')

def test_zero_film_identity_gradient_and_exact_parameter_budget():
 torch.set_num_threads(1);block=TrajectoryFiLM();assert sum(p.numel() for p in block.parameters())*7==15663872
 h=torch.randn(2,97,2048,dtype=torch.bfloat16);c=torch.randn(2,97,18);t=torch.randn(2,2048)
 out=block(h,c,t);torch.testing.assert_close(out,h,rtol=0,atol=0)
 out.float().square().mean().backward();assert block.mlp[-1].weight.grad.abs().sum()>0
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())

class Block(nn.Module):
 def forward_motion(self,value,*args):return value
class Heads(nn.Module):
 def time_embedder(self,t):return t[:,None].expand(-1,2048)
 def decode(self,h):return h[:,:283]
class Motion(nn.Module):
 def __init__(self):
  super().__init__();self.frozen_weight=nn.Parameter(torch.ones(()));self.layers=nn.ModuleList([Block() for _ in range(28)]);self.heads=Heads();self.norm_moe_motion=nn.Identity()
class Base(nn.Module):
 def __init__(self):
  super().__init__();self.motion=Motion();self.native=nn.Linear(1,1);self.bridges=nn.ModuleDict({str(k):nn.Linear(1,1) for k in [3,7,11,15,19,23,27]})
 def prepare_motion(self,motion,neutral,sigma,captions,modes):
  assert captions==['']*len(motion) and modes==['video2motion']*len(motion)
  hidden=torch.nn.functional.pad(motion,(0,2048-283));hidden=torch.cat((hidden.new_zeros(len(motion),1,2048),hidden),1).reshape(-1,2048)
  return hidden,None,None,None,None,None,None

def test_camera_only_cannot_accept_video_and_never_changes_base():
 model=CameraOnlyFiLM(Base(),activation_checkpointing=False);x=torch.randn(2,97,283);n=torch.zeros(2,30,3);c=torch.randn(2,97,18);s=torch.tensor([.1,.7])
 inputs=dict(motion=x,neutral=n,motion_sigma=s,camera_features=c)
 initial=model(**inputs);torch.testing.assert_close(initial,x,rtol=0,atol=0)
 with pytest.raises(TypeError):model(**inputs,video=torch.randn(2,48,25,16,16))
 optimizer=torch.optim.AdamW(model.film.parameters(),lr=2e-4);versions={id(p):p._version for p in model.frozen_parameters()}
 for _ in range(2):
  optimizer.zero_grad();model(**inputs).square().mean().backward();optimizer.step()
 for p in model.frozen_parameters():assert not p.requires_grad and p.grad is None and p._version==versions[id(p)]
 assert not torch.equal(model(**inputs),model(**{**inputs,'camera_features':c+1}))
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)==15663872

def test_full_training_cache_and_missing_video_invariance():
 p=json.loads((ROOT/'training/plan.json').read_text());ds=CameraOnlyDataset(p['motion_cache'],p['trajectory_cache'],p.get('cache_pairing_receipt'));assert len(ds)==112503
 original=ds[0];ds.rows[0]=copy.deepcopy(ds.rows[0]);ds.rows[0]['latent']='/nonexistent/forbidden/video.npz';ds.rows[0]['caption']='FORBIDDEN CAPTION'
 changed=ds[0]
 for key in ['motion','neutral','camera_features']:torch.testing.assert_close(changed[key],original[key],rtol=0,atol=0)
 assert 'video' not in changed and 'caption' not in changed and 'camera_action' not in changed
 split=json.loads(Path('/mnt/local/jinhyung.kim/cosmos_storage/weka/nymeriaplus_kimodo_proportional/train_test_split.json').read_text());assert not set(split['test']).intersection(r['uuid'] for r in ds.rows)

def test_exact_fixed71_validation_and_50k_schedule():
 p=json.loads((ROOT/'training/plan.json').read_text());rows=json.loads(Path(p['validation_cases']).read_text());assert len(rows)==71 and len({r['uuid'] for r in rows})==71
 old=json.loads(Path('/mnt/local/jinhyung.kim/cosmos_storage/operations/eval_bidirectional_nocamera_wv1_EMA100k_random20_20261112/old_plan.json').read_text());cfg=json.loads(Path(old['validation_config']).read_text());reference=json.loads(Path(cfg['cases']).read_text())
 assert [(r['id'],r['uuid'],r['start'],r['seed']) for r in rows]==[(r['id'],r['uuid'],r['start'],r['seed']) for r in reference]
 for r in rows:
  with np.load(r['bundle']) as z:assert set(z.files)=={'motion','neutral','camera_features'} and z['camera_features'].shape==(97,18) and np.isfinite(z['camera_features']).all()
 recipe=p['recipe'];assert recipe['tasks']=={'camera2motion':1.} and recipe['steps']==50000 and recipe['accumulation']==1
 assert lr_for(1,recipe)==pytest.approx(2e-7) and lr_for(1000,recipe)==2e-4 and lr_for(50000,recipe)==0
