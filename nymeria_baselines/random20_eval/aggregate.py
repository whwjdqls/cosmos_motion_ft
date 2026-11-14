"""Recompute dataset-level metrics over all1420 cases, never average shard FID/FVD."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,os,sys,time,zipfile
import numpy as np
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--operation',type=Path,required=True);ARGS=parser.parse_args();R=ARGS.operation.resolve();P=json.loads((R/'plan.json').read_text());SNAP=Path(P['run']['snapshot']);STEP=int(P['run']['step'])
sys.path.insert(0,os.environ['COSMOS_FRAMEWORK_EDGE_ROOT'])
sys.path.insert(0,str(Path(os.environ['REPO_ROOT'])/'.deps/kimodo_open'))
for suffix in reversed(('motion_expert_t2m_edge','motion_expert_joint_attention','nymeria_world','')):sys.path.insert(0,str(SNAP/suffix))
sys.path.insert(0,str(SNAP/'shape_aware_TMR'))
from motion_expert_phase3_edge_joint.validation_metrics import frechet
from nymeria_baselines.evaluate_reconstruction_tmr import fid,retrieval_details

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
manifest=read(P['manifest']);assert hashlib.sha256(Path(P['manifest']).read_bytes()).hexdigest()==P['manifest_sha256'];assert (R/'manifest.json').read_bytes()==Path(P['manifest']).read_bytes();ids=[r['id'] for r in manifest['rows']];lookup={r['id']:r for r in manifest['rows']};assert len(ids)==1420 and set(Counter(r['uuid'] for r in manifest['rows']).values())=={20}
chunks=[]
for shard in range(4):
 root=R/'shards'/str(shard);assert read(root/'COMPLETE.json')['windows']==355
 for chunk in sorted((root/'chunks').iterdir()):
  assert (chunk/'COMPLETE.json').exists();chunks.append((chunk,read(chunk/'cases.json')))
records={row['id']:row for _,rows in chunks for row in rows};assert set(records)==set(ids);save(R/'cases.json',[records[k] for k in ids]);summaries={};csv_rows=[];intervals={};all_per_case={}
for cid,row in records.items():assert all(row[k]==lookup[cid][k] for k in ('id','uuid','start','caption','seed'))
for task in P['tasks']:
 per_case={};features={}
 for chunk,rows in chunks:
  folder=chunk/task['id'];assert read(folder/'COMPLETE.json')['windows']==len(rows)
  for row in rows:
   with np.load(folder/task['mode']/(row['id']+'.npz')) as z:
    shape=(48,25,16,16) if task['mode']=='motimg2video' else (97,283);assert z['generated'].shape==shape and z['generated'].dtype==np.float32 and np.isfinite(z['generated']).all();assert int(z['seed'])==row['seed'] and float(z['guidance'])==task['guidance'] and str(z['manifest_sha256'])==P['manifest_sha256'];assert str(z['caption'])==(row['caption'] if task['text'] else '')
    if P['run'].get('camera'):assert z['generated_camera'].shape==(96,64) and np.isfinite(z['generated_camera']).all() and str(z['camera_guidance'])=='conditioned_readout_no_CFG_scaling'
    if task['mode']=='motimg2video':
     with np.load(row['bundle']) as gt:np.testing.assert_array_equal(z['generated'][:,0],gt['video'][:,0].astype(np.float32))
  if task['mode']=='motimg2video':
   native=read(folder/'m2v_per_case.json');assert [x['id'] for x in native]==[x['id'] for x in rows]
   for row,v in zip(rows,native,strict=True):
    with np.load(folder/'video_metrics'/(row['id']+'.npz')) as z:
     a,b=z['fvd_gt'].copy(),z['fvd_generated'].copy();assert a.shape==b.shape==(6,400) and np.isfinite(a).all() and np.isfinite(b).all()
     values={k:float(z[k].astype(np.float64).mean()) for k in ('psnr_db','ssim','lpips_alex','dreamsim')}
     for k in values:assert abs(values[k]-v[k])<1e-5
     per_case[row['id']]=dict(id=row['id'],uuid=row['uuid'],**values);features[row['id']]=(a,b)
  else:
   native=read(folder/'v2m_per_case.json');extra=read(folder/'extra_motion_per_case.json');assert [x['id'] for x in extra]==[x['id'] for x in rows]
   with np.load(folder/'motion_embeddings.npz') as z:
    assert z['ids'].tolist()==[r['id'] for r in rows]
    for i,row in enumerate(rows):per_case[row['id']]=dict(native[i],**extra[i],uuid=row['uuid']);features[row['id']]=tuple(z[k][i].copy() for k in ('pred','gt','text'))
 assert set(per_case)==set(ids) and set(features)==set(ids)
 ordered=[per_case[k] for k in ids]
 for row in ordered:
  if 'upe_m' in row:row.update(upe_cm=row['upe_m']*100.,lpe_cm=row['lpe_m']*100.)
 keys=[k for k in ordered[0] if k not in ('id','uuid','seed')];values={k:float(np.mean([r[k] for r in ordered])) for k in keys}
 if task['mode']=='motimg2video':
  a,b=[np.concatenate([features[k][i] for k in ids]) for i in (0,1)];values.update(fvd=frechet(a,b),gt_self_fvd=frechet(a,a),fvd_segments=len(a));np.savez(R/(task['id']+'_features.npz'),ids=np.array(ids),gt=a,pred=b)
 else:
  pred,gt,text=[np.stack([features[k][i] for k in ids]) for i in (0,1,2)];assert np.isfinite(pred).all() and np.isfinite(gt).all() and np.isfinite(text).all();retrieval,groups=retrieval_details(pred,text,ids);gt_retrieval,_=retrieval_details(gt,text,ids);values.update(fid=fid(pred,gt),**retrieval,matching_distance=float(np.linalg.norm(pred-text,axis=-1).mean()))
  distance_sum=0.;distance_count=0
  for i in range(len(pred)-1):
   d=np.linalg.norm(pred[i+1:]-pred[i],axis=-1);distance_sum+=d.sum(dtype=np.float64);distance_count+=len(d)
  values['diversity']=float(distance_sum/distance_count);np.savez(R/(task['id']+'_embeddings.npz'),ids=np.array(ids),pred=pred,gt=gt,text=text);save(R/(task['id']+'_retrieval_groups.json'),groups);save(R/(task['id']+'_gt_retrieval.json'),gt_retrieval)
 assert all(np.isfinite(v) for v in values.values());summaries[task['id']]=values;save(R/(task['id']+'_per_case.json'),ordered);all_per_case[task['id']]=ordered
 for k,v in values.items():csv_rows.append(dict(model_id=P['run']['canonical_id'],step=STEP,weights='ema',task=task['id'],guidance=task['guidance'],metric=k,value=v,windows=1420,manifest_sha256=P['manifest_sha256']))
 # Recording-block intervals retain all20 windows in each sampled recording.
 uuids=sorted({r['uuid'] for r in ordered});matrix=np.array([[np.mean([r[k] for r in ordered if r['uuid']==uid]) for k in keys] for uid in uuids]);rng=np.random.default_rng(20261110);draws=rng.integers(0,71,(5000,71));means=matrix[draws].mean(1);intervals[task['id']]={k:dict(mean=float(matrix[:,i].mean()),ci95=np.quantile(means[:,i],[.025,.975]).tolist()) for i,k in enumerate(keys)}
save(R/'summary.json',summaries);save(R/'recording_bootstrap_intervals.json',dict(recordings=71,windows_each=20,resamples=5000,seed=20261110,intervals=intervals))
with (R/'metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
save(R/'AUDIT.json',dict(passed=True,windows=1420,recordings=71,tasks=3,predictions=4260,exact20_each=True,manifest_sha256=P['manifest_sha256'],all_ids_shapes_seeds_guidance_captions_finite_and_conditioned_prefixes_verified=True,global_fid_fvd_retrieval_recomputed=True))
text=['# MMDiT EMA checkpoint — Nymeria random20 v1','',f'Manifest SHA256: `{P["manifest_sha256"]}`. 1420 windows,20 per each of71 recordings. Selection seed20261110; fixed before sampling. All3 tasks use the same selected windows/seeds.','', 'Whole-bridge CFG defaults chosen from the earlier fixed71 sweep: M2V1.1; V2M and video+text1.5. Generator captions always empty; text only enters the frozen motion expert in the explicit text condition. Native UniPC35, shifts3/10,97frames/20FPS/256px, supplied shape. EMA weights; exact model/step in plan.json and metrics.csv.','', 'Metrics are recomputed over the complete1420 cases; shard FID/FVD are not averaged. M2V scores frames1–96 and uses8520 native I3D400 segment descriptors. Motion uses the matched91f9d8 Nymeria TMR evaluator and the common Body23 completion. All per-case scores/features/embeddings are retained for later rescores. Native R32 retrieval drops the final12 cases of each shuffled grouping; all1420 enter physical errors/FID.','', '| Task | CFG | MPJPE cm | UPE cm | LPE cm | sJPE | FID | DreamSim | FVD |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for task in P['tasks']:
 v=summaries[task['id']];cells=[f"{v[k]:.5f}" if k in v else '—' for k in ('mpjpe_cm','upe_cm','lpe_cm','sjpe','fid','dreamsim','fvd')];text.append('| '+task['id']+' | '+str(task['guidance'])+' | '+' | '.join(cells)+' |')
text+=['','Full metrics: `metrics.csv`; cohort-level summaries: `summary.json`; recording-block95% intervals: `recording_bootstrap_intervals.json`. These intervals describe recording variation on this fixed subset, not training-seed variation.','', 'This is the random20 benchmark; its distribution scores must not be pooled or directly equated with previous fixed71/full-test metrics. No model is selected by this report.']
(R/'report.md').write_text('\n'.join(text)+'\n');save(R/'SOURCE_MANIFEST_FINAL.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in R.glob('*.py')})
with zipfile.ZipFile(R/'quantitative_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['plan.json','manifest.json','summary.json','metrics.csv','recording_bootstrap_intervals.json','AUDIT.json','SOURCE_MANIFEST_FINAL.json','report.md']+[t['id']+'_per_case.json' for t in P['tasks']]:z.write(R/name,name)
print(json.dumps(dict(complete=True,windows=1420,predictions=4260,tasks=3,manifest_sha256=P['manifest_sha256'])),flush=True)
