"""GT, raw body physics and common-completion Nymeria TMR for full-test V2M."""
from pathlib import Path
import argparse
from collections import defaultdict
import csv
import json
import os
import time
import numpy as np
import torch

from nymeria_baselines.fulltest_v2m import read,save,sha,npz,guard,load_common,progress


def main(a):
 guard();torch.set_num_threads(2)
 cfg,rows=load_common(a)
 from nymeria_baselines.body23_tmr_adapter import complete_body,KEEP
 from nymeria_baselines.visualize_official_baselines import decode
 from nymeria_baselines.evaluate_uem_phase3 import measure
 from nymeria_baselines.evaluate_reconstruction_tmr import fid,retrieval_details
 from kimodo.metrics.foot_skate import FootSkateFromHeight
 from kimodo.skeleton import SOMASkeleton30
 from nymeria_eval import NymeriaTMREmbedder
 template=read(cfg['adapter']);assert sha(cfg['adapter'])==cfg['evaluator']['adapter_sha256']
 text_cache=a.out/'llm2vec_fulltest_extended.pt'
 if not text_cache.exists():raise FileNotFoundError('Prepare the complete frozen text cache before scoring')
 provenance=read(text_cache.with_suffix('.provenance.json'))
 assert provenance['base_cache_sha256']==sha(cfg['evaluator']['text_cache'])
 scorer=NymeriaTMREmbedder(cfg['evaluator']['evaluator_checkpoint'],cfg['evaluator']['evaluator_stats'],text_cache,'cuda')
 assert all(r['caption'] in scorer.cache for r in rows),'Incomplete full-test caption coverage'
 target=a.out/'scores'/a.model;target.mkdir(parents=True,exist_ok=True)
 if (target/'COMPLETE.json').exists():return
 assert a.model=='GT' or (a.out/'models'/a.model/'INFERENCE_COMPLETE.json').exists()
 from types import SimpleNamespace
 skate=FootSkateFromHeight(skeleton=SimpleNamespace(foot_joint_idx=[17,18,21,22]),fps=20.)
 embeddings=[];texts=[];metrics=[];started=time.monotonic()
 for start in range(0,len(rows),16):
  batch=rows[start:start+16];dest=target/'batches'/f'{start:06d}.npz'
  if dest.exists():
   with np.load(dest) as z:
    assert z['case_id'].tolist()==[r['id'] for r in batch]
    embeddings.append(z['motion'].copy());texts.append(z['text'].copy())
   metrics+=read(dest.with_suffix('.json'));continue
  positions=[];shapes=[];physical=[]
  for r in batch:
   assert sha(r['bundle'])==r['bundle_sha256']
   with np.load(r['bundle']) as z:raw,neutral=z['raw'].copy(),z['neutral'].copy()
   gt,gtrot=decode(raw,30);gt,gtrot=gt[:,KEEP].astype(np.float32),gtrot[:,KEEP]
   if a.model=='GT':joints,rot,rest=gt,gtrot,neutral[KEEP]
   else:
    with np.load(a.out/'models'/a.model/'predictions'/(r['id']+'.npz')) as z:
     joints,rot,rest=z['joints'].copy(),z['rotations'].copy(),z['neutral23'].copy()
   value=measure(joints,gt,skate)
   full,shape=complete_body(joints,rot,rest,template)
   assert np.array_equal(full[:,KEEP],joints)
   positions.append(full);shapes.append(shape)
   physical.append(dict(id=r['id'],uuid=r['uuid'],start=r['start'],metrics=value))
  with torch.inference_mode():
   m=scorer.embed_motion(torch.from_numpy(np.stack(positions)),torch.from_numpy(np.stack(shapes)),lengths=[97]*len(batch),source_fps=20).cpu().numpy()
   t=scorer.embed_text([r['caption'] for r in batch]).cpu().numpy()
  assert m.shape==t.shape==(len(batch),256) and np.isfinite(m).all() and np.isfinite(t).all()
  npz(dest,motion=m,text=t,case_id=np.array([r['id'] for r in batch]));save(dest.with_suffix('.json'),physical)
  embeddings.append(m);texts.append(t);metrics+=physical
  if start%160==0:progress(target,'scoring',completed=start+len(batch),total=len(rows),seconds=time.monotonic()-started)
 motion,text=np.concatenate(embeddings),np.concatenate(texts)
 if a.model=='GT':gt=motion
 else:
  with np.load(a.out/'scores/GT/embeddings.npz') as z:
   gt=z['motion'].copy();assert np.array_equal(text,z['text'])
 ids=[r['id'] for r in rows];retrieval,groups=retrieval_details(motion,text,ids)
 npz(target/'embeddings.npz',motion=motion,text=text,case_id=np.array(ids))
 save(target/'retrieval_groups.json',groups);save(target/'per_case.json',metrics)
 values={k:float(np.mean([r['metrics'][k] for r in metrics])) for k in metrics[0]['metrics']}
 recordings=defaultdict(list)
 for r in metrics:recordings[r['uuid']].append(r['metrics'])
 balanced={k:float(np.mean([np.mean([r[k] for r in v]) for v in recordings.values()])) for k in values}
 result=dict(model=a.model,cases=len(rows),recordings=len(recordings),physical=values,
      recording_balanced_physical=balanced,FID=fid(motion,gt),retrieval=retrieval,
      input_configuration_sha256=sha(a.out/'configuration.json'),source_sha256=sha(__file__),
      evaluator_sha256=sha(cfg['evaluator']['evaluator_checkpoint']),
      text_cache_sha256=sha(text_cache),text_cache_provenance=provenance,completed=time.time())
 save(target/'summary.json',result);save(target/'COMPLETE.json',dict(cases=len(rows),summary_sha256=sha(target/'summary.json')))
 print(json.dumps(result),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--model',required=True)
 main(p.parse_args())
