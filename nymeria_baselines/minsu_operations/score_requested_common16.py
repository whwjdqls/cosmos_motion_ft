"""Add explicitly scoped common16 position splits and native DisCoRD sJPE."""
from pathlib import Path
import json
import os
import sys
import time
import numpy as np

ROOT=Path(os.environ.get('REVIV_OPERATION','/scratch2/minsu1206/cosmos/operations/reviv_zeroshot_20261105'))
sys.path.insert(0,str(ROOT))
sys.path.insert(1,'/home/whwjdqls99/cosmos_motion_ft')
from protocol import cases,save_json,sha,Y_FROM_Z,JOINT_MAP
from score import fit
from nymeria_baselines.visualize_official_baselines import decode
from nymeria_baselines.motion_error_metrics import sjpe

def main():
 assert os.environ.get('SLURM_JOB_ID')=='2682742' and 'login' not in os.uname().nodename
 output=ROOT/'full/scores_requested_common16';output.mkdir(exist_ok=True)
 rows=cases();previous=ROOT/'full/scores_common16'
 ref={r['id']:r['metrics'] for r in json.loads((previous/'ReViV_per_case.json').read_text())}
 gtref={r['id']:r['metrics'] for r in json.loads((previous/'GT_per_case.json').read_text())}
 source=[r for _,r,_ in JOINT_MAP];target=[s for _,_,s in JOINT_MAP]
 upper=list(range(9,16));lower=list(range(9));end=[7,8,14,15]
 protocol=dict(cases=11911,recordings=71,source_summary_sha256=sha(previous/'summary.json'),
  joints='Existing conservative common16 mapping, unchanged',upper_joint_count=7,lower_joint_count=9,
  position_alignment='One GT-fitted rigid rotation and translation per clip, no scale; preserves prior common16 MPJPE and root ADE',
  pa_mpjpe='Existing independent per-frame similarity alignment, unchanged',
  sjpe='Existing shared DisCoRD four-end-effector mean, native unaligned joints, third differences at20FPS, no smoothing',
  end_effectors=['left_toe','right_toe','left_wrist','right_wrist'],
  unavailable={'R@3':'Validated ReViV21 to SOMA30 TMR adapter required','FID':'Validated ReViV21 to SOMA30 TMR adapter required',
   'Skate':'Camera-relative output has no calibrated floor/contact-height reference',
   'Penetration':'Camera-relative output has no calibrated floor reference'},
  source_sha256=sha(__file__),shared_sjpe_source_sha256=sha(Path('/home/whwjdqls99/cosmos_motion_ft/nymeria_baselines/motion_error_metrics.py')))
 save_json(output/'configuration.json',protocol)
 records=[];truth=[];started=time.monotonic()
 for i,row in enumerate(rows):
  with np.load(ROOT/'full/predictions'/(row['id']+'.npz')) as z:
   assert str(z['case_id'])==row['id'];p=z['joints_reviv'][:,source].astype(np.float64)@Y_FROM_Z.T
  with np.load(row['bundle']) as z:g=decode(z['raw'],30)[0][:,target]
  _,q,t=fit(p,g,False);aligned=p@q+t;dist=np.linalg.norm(aligned-g,axis=-1)
  m=ref[row['id']]
  assert abs(float(dist.mean()*1000)-m['GA_RIGID_MPJPE_mm'])<1e-7
  values=dict(MPJPE_cm=m['GA_RIGID_MPJPE_mm']/10,PA_MPJPE_cm=m['PA_SIM_MPJPE_mm']/10,
   Root_ADE_cm=m['GA_RIGID_ROOT_ADE_mm']/10,UPE_common7_cm=float(dist[:,upper].mean()*100),
   LPE_common9_cm=float(dist[:,lower].mean()*100),**sjpe(p[:,end],g[:,end],fps=20.))
  base=gtref[row['id']]
  gt=dict(MPJPE_cm=base['GA_RIGID_MPJPE_mm']/10,PA_MPJPE_cm=base['PA_SIM_MPJPE_mm']/10,
   Root_ADE_cm=base['GA_RIGID_ROOT_ADE_mm']/10,UPE_common7_cm=float(np.linalg.norm(g-g,axis=-1)[:,upper].mean()),
   LPE_common9_cm=float(np.linalg.norm(g-g,axis=-1)[:,lower].mean()),**sjpe(g[:,end],g[:,end],fps=20.))
  records.append(dict(id=row['id'],uuid=row['uuid'],metrics=values));truth.append(dict(id=row['id'],uuid=row['uuid'],metrics=gt))
  if i%500==0:save_json(output/'progress.json',dict(completed=i+1,total=len(rows),seconds=time.monotonic()-started))
 def aggregate(items):
  keys=list(items[0]['metrics']);uids=sorted({r['uuid'] for r in items})
  return dict(cases=len(items),recordings=len(uids),window_mean={k:float(np.mean([r['metrics'][k] for r in items])) for k in keys},
   recording_balanced_mean={k:float(np.mean([np.mean([r['metrics'][k] for r in items if r['uuid']==u]) for u in uids])) for k in keys})
 summary=dict(ReViV=aggregate(records),GT=aggregate(truth),unavailable=protocol['unavailable'])
 save_json(output/'ReViV_per_case.json',records);save_json(output/'GT_per_case.json',truth)
 save_json(output/'summary.json',summary);save_json(output/'COMPLETE.json',dict(cases=len(rows),summary_sha256=sha(output/'summary.json')))
 print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
